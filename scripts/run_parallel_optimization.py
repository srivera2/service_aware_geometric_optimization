#!/usr/bin/env python3
"""
Parallel Scene Optimization Script
===================================

Runs boresight optimization across multiple scenes in parallel using all available GPUs.

Usage:
    python run_parallel_optimization.py                    # Run all 50 scenes on 8 GPUs
    python run_parallel_optimization.py --workers 4        # Use only 4 workers
    python run_parallel_optimization.py --gpus 0,1,2,3     # Use specific GPUs
    python run_parallel_optimization.py --max-scenes 10    # Process only 10 scenes
    python run_parallel_optimization.py --output results/  # Custom output directory
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

# IMPORTANT: Set multiprocessing start method BEFORE importing torch
# This must happen before any CUDA operations
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


def get_gpu_info():
    """Detect available GPUs and their memory."""
    import torch

    if not torch.cuda.is_available():
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        try:
            free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)
        except:
            free_mem = 0
        gpus.append({
            'id': i,
            'name': props.name,
            'total_gb': props.total_memory / (1024**3),
            'free_gb': free_mem,
        })
    return gpus


def run_parallel_optimization(
    scene_dirs: list,
    parent_folder: str,
    output_dir: str,
    num_workers: int,
    gpu_ids: list,
    validation_thresholds: dict
):
    """
    Run optimization across multiple scenes in parallel.

    Each worker process:
    1. Gets assigned a GPU (round-robin)
    2. Sets CUDA_VISIBLE_DEVICES before importing GPU libraries
    3. Processes scenes independently
    4. Returns serializable results
    """
    # Import worker function here (after spawn method is set)
    # Add src to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(os.path.dirname(script_dir), 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from parallel_worker import process_single_scene

    print("=" * 80)
    print("PARALLEL SCENE OPTIMIZATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Scenes to process: {len(scene_dirs)}")
    print(f"  Parallel workers:  {num_workers}")
    print(f"  GPUs to use:       {gpu_ids}")
    print(f"  Output directory:  {output_dir}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare work items with round-robin GPU assignment
    work_items = [
        (
            scene_name,
            gpu_ids[i % len(gpu_ids)],
            validation_thresholds,
            parent_folder,
            output_dir
        )
        for i, scene_name in enumerate(scene_dirs)
    ]

    results = {}
    start_time = time.time()
    completed = 0

    print(f"Starting {num_workers} worker processes...")
    print("-" * 80)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_scene = {
            executor.submit(process_single_scene, item): item[0]
            for item in work_items
        }

        # Process results as they complete
        for future in as_completed(future_to_scene):
            scene_name = future_to_scene[future]
            completed += 1

            try:
                _, result = future.result(timeout=7200)  # 2 hour timeout per scene
                results[scene_name] = result

                status = result.get('status', 'unknown')
                if status == 'success':
                    improvement = result['stats']['improvement_mean']
                    elapsed = result['elapsed_time']
                    gpu = result.get('gpu_id', '?')
                    print(f"✓ [{completed}/{len(scene_dirs)}] {scene_name}: "
                          f"+{improvement:.2f} dB in {elapsed:.1f}s (GPU {gpu})")
                else:
                    reason = result.get('reason', 'Unknown')[:60]
                    print(f"✗ [{completed}/{len(scene_dirs)}] {scene_name}: {status} - {reason}")

                # Save incremental progress
                progress_file = os.path.join(output_dir, 'results_progress.json')
                with open(progress_file, 'w') as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                print(f"✗ [{completed}/{len(scene_dirs)}] {scene_name}: Exception - {e}")
                results[scene_name] = {'status': 'error', 'reason': str(e)}

            # Progress update
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = len(scene_dirs) - completed
            eta = remaining / rate if rate > 0 else 0
            print(f"   Progress: {100*completed/len(scene_dirs):.0f}% | "
                  f"Rate: {rate*60:.1f} scenes/min | ETA: {eta/60:.1f} min")

    # Final summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results.values() if r.get('status') == 'success')
    failed = sum(1 for r in results.values() if r.get('status') == 'failed')
    errors = sum(1 for r in results.values() if r.get('status') == 'error')

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total time:     {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"Successful:     {successful}/{len(scene_dirs)}")
    print(f"Failed zones:   {failed}")
    print(f"Errors:         {errors}")

    if successful > 0:
        improvements = [
            r['stats']['improvement_mean']
            for r in results.values()
            if r.get('status') == 'success'
        ]
        print(f"\nImprovement Statistics:")
        print(f"  Mean:   {np.mean(improvements):+.2f} dB")
        print(f"  Median: {np.median(improvements):+.2f} dB")
        print(f"  Min:    {np.min(improvements):+.2f} dB")
        print(f"  Max:    {np.max(improvements):+.2f} dB")

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = os.path.join(output_dir, f'results_final_{timestamp}.json')
    with open(final_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {final_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run parallel scene optimization across multiple GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--workers', '-w', type=int, default=None,
        help='Number of parallel workers (default: number of GPUs)'
    )
    parser.add_argument(
        '--gpus', '-g', type=str, default=None,
        help='Comma-separated GPU IDs to use (e.g., "0,1,2,3")'
    )
    parser.add_argument(
        '--max-scenes', '-n', type=int, default=50,
        help='Maximum number of scenes to process (default: 50)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='./parallel_results',
        help='Output directory for results (default: ./parallel_results)'
    )
    parser.add_argument(
        '--scenes-dir', '-s', type=str, default=None,
        help='Directory containing scene subdirectories (default: ../scene/scenes)'
    )

    args = parser.parse_args()

    # Detect GPUs
    print("Detecting GPUs...")
    gpus = get_gpu_info()

    if not gpus:
        print("WARNING: No GPUs detected. Running on CPU may be very slow.")
        gpu_ids = [None]
    else:
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  GPU {gpu['id']}: {gpu['name']} "
                  f"({gpu['free_gb']:.1f} GB free / {gpu['total_gb']:.1f} GB total)")

        if args.gpus:
            gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
        else:
            gpu_ids = [g['id'] for g in gpus]

    # Set number of workers
    num_workers = args.workers if args.workers else len(gpu_ids)

    # Find scenes directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.scenes_dir:
        parent_folder = args.scenes_dir
    else:
        parent_folder = os.path.join(os.path.dirname(script_dir), 'scene', 'scenes')

    if not os.path.isdir(parent_folder):
        print(f"ERROR: Scenes directory not found: {parent_folder}")
        sys.exit(1)

    # Get scene directories
    scene_dirs = sorted([
        d for d in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, d))
    ])[:args.max_scenes]

    if not scene_dirs:
        print(f"ERROR: No scene directories found in {parent_folder}")
        sys.exit(1)

    print(f"\nFound {len(scene_dirs)} scene(s) to process")

    # Zone validation thresholds
    validation_thresholds = {
        'p10_min_dbm': -270.0,
        'p10_max_dbm': -80.0,
        'p90_min_dbm': -130.0,
        'min_percentile_range_db': 15.0,
        'median_max_dbm': -65.0          
    }

    # Run optimization
    results = run_parallel_optimization(
        scene_dirs=scene_dirs,
        parent_folder=parent_folder,
        output_dir=args.output,
        num_workers=num_workers,
        gpu_ids=gpu_ids,
        validation_thresholds=validation_thresholds
    )

    return results


if __name__ == '__main__':
    main()

"""
Batch Validation Script for Boresight Optimization

Runs optimization across multiple scenes with different samplers, frequencies, and LDS methods.
Saves all results to a pickle file for post-processing analysis.

Usage:
    python validation.py
    python validation.py --output results_custom.pkl --max-scenes 10
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless/tmux environments
import matplotlib.pyplot as plt
import numpy as np
import mitsuba as mi
import drjit as dr
import warnings
import sys
import os
os.environ["DRJIT_LIBLLVM_PATH"] = "/usr/lib/x86_64-linux-gnu/libLLVM.so.20.1"
import pickle
import argparse
import gc
import time
import threading
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.abspath('../src'))

from sionna.rt import load_scene, Transmitter, Receiver, Camera, PathSolver, RadioMapSolver
from sionna.rt import AntennaArray, PlanarArray, SceneObject, ITURadioMaterial
from sionna.rt.antenna_pattern import antenna_pattern_registry

from scene_parser import extract_building_info
from tx_placement import TxPlacement
from boresight_pathsolver import create_zone_mask, optimize_boresight_pathsolver, compare_boresight_performance
from angle_utils import azimuth_elevation_to_yaw_pitch
from zone_validator import find_valid_zone

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Scene settings
    'parent_folder': "../scene/scenes",
    'max_scenes': 49,  # Set to None for all scenes

    # Test matrix
    'samplers': ["Rejection", "CDT"],
    'frequencies': [1.0e9, 9.0e9],
    'lds_methods': ["Sobol", "Halton", "Latin"],

    # Map configuration
    'map_config': {
        'center': [0.0, 0.0, 0.0],
        'size': [1000, 1000],
        'cell_size': (1.0, 1.0),
        'ground_height': 0.0,
    },

    # Zone settings
    'zone_params_template': {
        'width': 250.0,
        'height': 250.0
    },
    'zone_search': {
        'min_distance': 50.0,
        'max_distance': 300.0,
        'max_attempts': 200,
    },

    # Validation thresholds
    'validation_thresholds': {
        'p10_min_dbm': -140.0,
        'p10_max_dbm': -90.0,
        'p90_min_dbm': -80.0,
        'min_percentile_range_db': 40.0,
        'median_max_dbm': -60.0
    },

    # Optimization settings
    'optimization': {
        'num_sample_points': 64,
        'learning_rate': 2.0,
        'num_iterations': 100,
    },

    # RadioMapSolver settings for evaluation
    'rm_solver': {
        'max_depth': 5,
        'samples_per_tx': int(6e8),
    },

    # TX placement
    'tx_offset': 5.0,

    # Output
    'output_file': 'validation_results.pkl',
    'save_plots': False,
    'plot_dir': 'validation_plots',
}


# =============================================================================
# CHECKPOINT AND PROGRESS UTILITIES
# =============================================================================

def load_checkpoint(checkpoint_file):
    """Load existing checkpoint file if it exists, for resuming interrupted runs."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            results = checkpoint.get('results', {})
            completed_scenes = checkpoint.get('completed_scenes', set())
            return results, completed_scenes
        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            print(f"WARNING: Corrupt checkpoint file ({e}), starting fresh")
    return {}, set()


def save_checkpoint(results, completed_scenes, checkpoint_file, config):
    """Atomically save checkpoint after each scene completion."""
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results,
        'completed_scenes': completed_scenes,
        'num_results': len(results),
        'num_successful': sum(1 for r in results.values() if r.get('status') == 'success'),
    }
    tmp_file = checkpoint_file + '.tmp'
    with open(tmp_file, 'wb') as f:
        pickle.dump(checkpoint, f)
    os.replace(tmp_file, checkpoint_file)


class OperationWatchdog:
    """Context manager that prints periodic heartbeat messages for long operations.

    Provides visibility into whether the program is still running or has hung,
    without attempting to kill CUDA operations (which would be unsafe).

    Usage:
        with OperationWatchdog("find_valid_zone", "scene_001", interval=60):
            result = find_valid_zone(...)
    """

    def __init__(self, operation_name, scene_name, interval=60):
        self.operation_name = operation_name
        self.scene_name = scene_name
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join(timeout=2)
        elapsed = time.time() - self._start_time
        print(f"  [{self.operation_name}] finished in {elapsed:.1f}s", flush=True)
        return False

    def _heartbeat(self):
        while not self._stop_event.wait(timeout=self.interval):
            elapsed = time.time() - self._start_time
            print(
                f"  [HEARTBEAT] {self.operation_name} for {self.scene_name} "
                f"running for {elapsed:.0f}s...",
                flush=True,
            )


def setup_scene(scene_xml_path):
    """Load and configure a scene with antenna arrays."""
    scene = load_scene(scene_xml_path)
    scene.frequency = 9.99e9  # Initial frequency (will be overwritten per test)

    # gNB antenna: 3GPP TR 38.901 pattern
    gnb_pattern_factory = antenna_pattern_registry.get("tr38901")
    gnb_pattern = gnb_pattern_factory(polarization="V")

    # UE antenna: Isotropic pattern
    ue_pattern_factory = antenna_pattern_registry.get("iso")
    ue_pattern = ue_pattern_factory(polarization="V")

    # SISO: Single antenna element
    single_element = np.array([[0.0, 0.0, 0.0]])

    scene.tx_array = AntennaArray(
        antenna_pattern=gnb_pattern,
        normalized_positions=single_element.T
    )
    scene.rx_array = AntennaArray(
        antenna_pattern=ue_pattern,
        normalized_positions=single_element.T
    )

    # Disable scattering
    for radio_material in scene.radio_materials.values():
        radio_material.scattering_coefficient = 0.4

    return scene


def find_central_building(building_info):
    """Find the building closest to origin (0, 0)."""
    min_distance = float('inf')
    selected_building_id = None

    for building_id, info in building_info.items():
        x_center, y_center = info['center']
        distance = np.sqrt(x_center**2 + y_center**2)
        if distance < min_distance:
            min_distance = distance
            selected_building_id = building_id

    return selected_building_id, min_distance


def evaluate_configuration(scene, tx, zone_mask, map_config, rm_solver, angles):
    """Evaluate a TX configuration and return zone power statistics."""
    yaw, pitch = azimuth_elevation_to_yaw_pitch(angles[0], angles[1])
    tx.orientation = mi.Point3f(float(yaw), float(pitch), 0.0)

    rm = rm_solver(
        scene,
        max_depth=CONFIG['rm_solver']['max_depth'],
        samples_per_tx=CONFIG['rm_solver']['samples_per_tx'],
        cell_size=map_config['cell_size'],
        center=map_config['center'],
        orientation=[0, 0, 0],
        size=map_config['size'],
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        diffraction=True,
        edge_diffraction=True,
        refraction=False,
        stop_threshold=None,
    )

    rss_watts = rm.rss.numpy()[0, :, :]
    signal_strength_dBm = 10.0 * np.log10(rss_watts + 1e-30) + 30.0
    zone_power = signal_strength_dBm[zone_mask == 1.0]

    return zone_power


def run_validation(config=None, checkpoint_file='validation_checkpoint.pkl',
                   scene_timeout=7200):
    """Run the full validation suite with incremental checkpointing and progress tracking.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary. Defaults to CONFIG.
    checkpoint_file : str
        Path to checkpoint pickle file for incremental saves and resume.
    scene_timeout : int
        Max seconds per scene before skipping remaining configs (default 7200 = 2h).
    """
    if config is None:
        config = CONFIG

    parent_folder = config['parent_folder']
    scene_dirs = sorted([d for d in os.listdir(parent_folder)
                         if os.path.isdir(os.path.join(parent_folder, d))])

    max_scenes = config.get('max_scenes')
    if max_scenes:
        scene_dirs = scene_dirs[:max_scenes]

    total_scenes = len(scene_dirs)

    # Load checkpoint for resume
    results, completed_scenes = load_checkpoint(checkpoint_file)
    if completed_scenes:
        print(f"Resuming from checkpoint: {len(completed_scenes)}/{total_scenes} scenes already completed")

    rm_solver = RadioMapSolver()
    run_start_time = time.time()
    scenes_completed_this_run = 0

    print(f"Testing {total_scenes} scenes")
    print(f"Scene timeout: {scene_timeout}s ({scene_timeout/3600:.1f}h)")
    print(f"Validation thresholds: {config['validation_thresholds']}\n")

    for i, scene_name in enumerate(scene_dirs):
        # Skip already-completed scenes
        if scene_name in completed_scenes:
            print(f"[SKIP] Scene {i+1}/{total_scenes}: {scene_name} (already completed)")
            continue

        scene_start_time = time.time()
        print(f"\nScene {i+1}/{total_scenes}: {scene_name}")

        scene_xml_path = os.path.join(parent_folder, scene_name, "scene.xml")
        scene = setup_scene(scene_xml_path)

        # Find and select central building
        building_info = extract_building_info(scene_xml_path, verbose=False)
        selected_building_id, min_distance = find_central_building(building_info)
        print(f"Selected building: {selected_building_id} (distance: {min_distance:.2f}m)")

        # Place transmitter
        tx_placer = TxPlacement(scene, "gnb", scene_xml_path, selected_building_id,
                                offset=config['tx_offset'])
        tx_placer.set_rooftop_center()
        tx = tx_placer.tx
        gnb_position = tx.position.numpy().flatten().tolist()
        print(f"gNB position: {gnb_position}")

        # Find valid zone (with heartbeat monitoring)
        with OperationWatchdog("find_valid_zone", scene_name, interval=60):
            zone_result = find_valid_zone(
                scene=scene,
                tx_name="gnb",
                tx_position=gnb_position,
                map_config=config['map_config'],
                scene_xml_path=scene_xml_path,
                zone_params_template=config['zone_params_template'],
                min_distance=config['zone_search']['min_distance'],
                max_distance=config['zone_search']['max_distance'],
                max_attempts=config['zone_search']['max_attempts'],
                validation_kwargs=config['validation_thresholds'],
                verbose=True
            )

        # find_valid_zone returns 5 values: zone_mask, zone_params, zone_center, validation_stats, attempts
        if zone_result[0] is None:
            attempts = zone_result[-1]
            print(f"Could not find valid zone after {attempts} attempts - skipping\n")
            results[scene_name] = {
                'status': 'failed',
                'reason': 'No valid zone found',
                'attempts': attempts
            }
            completed_scenes.add(scene_name)
            save_checkpoint(results, completed_scenes, checkpoint_file, config)
            print("=" * 80 + "\n")

            del scene, tx, tx_placer
            gc.collect()
            dr.flush_kernel_cache()
            dr.flush_malloc_cache()
            continue

        zone_mask, zone_params, zone_center, validation_stats, attempts = zone_result

        zone_center_x, zone_center_y = zone_center

        # Create zone_stats for compatibility with create_zone_mask if needed elsewhere
        # (zone_params is now returned directly from find_valid_zone)
        _, _, zone_stats = create_zone_mask(
            map_config=config['map_config'],
            zone_type='box',
            origin_point=gnb_position,
            zone_params=zone_params,
            target_height=config['map_config'].get('target_height', 1.5),
            scene_xml_path=scene_xml_path,
            exclude_buildings=True
        )

        print(f"Found valid zone after {attempts} attempt(s)")
        print(f"  Center: ({zone_center_x:.1f}, {zone_center_y:.1f})")
        print(f"  P10: {validation_stats['p10_power_dbm']:.1f} dBm, P90: {validation_stats['p90_power_dbm']:.1f} dBm")

        zone_distance_from_tx = np.sqrt((zone_center_x - gnb_position[0])**2 +
                                        (zone_center_y - gnb_position[1])**2)
        zone_angle_from_tx = np.arctan2(zone_center_y - gnb_position[1],
                                        zone_center_x - gnb_position[0])

        # Test matrix loop
        scene_timed_out = False
        for sampler in config['samplers']:
            if scene_timed_out:
                break
            for freq in config['frequencies']:
                if scene_timed_out:
                    break
                for lds in config['lds_methods']:
                    # Check scene timeout
                    scene_elapsed = time.time() - scene_start_time
                    if scene_elapsed > scene_timeout:
                        print(f"\n  [TIMEOUT] Scene {scene_name} exceeded {scene_timeout}s "
                              f"({scene_elapsed:.0f}s elapsed) - skipping remaining configs")
                        scene_timed_out = True
                        break

                    result_key = f"{scene_name}|{sampler}|{freq}|{lds}"
                    print(f"\n  Testing: {sampler}/{lds} @ {freq/1e9:.1f} GHz")

                    scene.frequency = freq

                    try:
                        # Run optimization (with heartbeat monitoring)
                        with OperationWatchdog("optimize", scene_name, interval=60):
                            best_angles, loss_hist, angle_hist, grad_hist, cov_stats, initial_angles = \
                                optimize_boresight_pathsolver(
                                    scene=scene,
                                    tx_name="gnb",
                                    map_config=config['map_config'],
                                    scene_xml_path=scene_xml_path,
                                    zone_mask=zone_mask,
                                    zone_params=zone_params,
                                    num_sample_points=config['optimization']['num_sample_points'],
                                    learning_rate=config['optimization']['learning_rate'],
                                    num_iterations=config['optimization']['num_iterations'],
                                    verbose=False,
                                    lds=lds,
                                    sampler=sampler
                                )

                        print(f"    Initial: Az={initial_angles[0]:.1f}, El={initial_angles[1]:.1f}")
                        print(f"    Best:    Az={best_angles[0]:.1f}, El={best_angles[1]:.1f}")

                        # Evaluate initial configuration (with heartbeat monitoring)
                        with OperationWatchdog("evaluate_initial", scene_name, interval=30):
                            zone_power_initial = evaluate_configuration(
                                scene, tx, zone_mask, config['map_config'], rm_solver, initial_angles
                            )

                        # Evaluate optimized configuration (with heartbeat monitoring)
                        with OperationWatchdog("evaluate_optimized", scene_name, interval=30):
                            zone_power_optimized = evaluate_configuration(
                                scene, tx, zone_mask, config['map_config'], rm_solver, best_angles
                            )

                        # Store results
                        results[result_key] = {
                            'status': 'success',
                            'scene_name': scene_name,
                            'sampler': sampler,
                            'frequency': freq,
                            'lds': lds,
                            'scene_xml_path': scene_xml_path,
                            'map_config': config['map_config'],
                            'initial_angles': initial_angles,
                            'best_angles': best_angles,
                            'loss_hist': loss_hist,
                            'angle_hist': angle_hist,
                            'grad_hist': grad_hist,
                            'cov_stats': cov_stats,
                            'tx_building_id': selected_building_id,
                            'tx_position': gnb_position,
                            'zone_center': [zone_center_x, zone_center_y],
                            'zone_distance_from_tx': zone_distance_from_tx,
                            'zone_angle_from_tx': zone_angle_from_tx,
                            'zone_attempts': attempts,
                            'validation_stats': validation_stats,
                            'zone_params': zone_params,
                            'zone_stats': zone_stats,
                            'zone_power_initial': zone_power_initial,
                            'zone_power_optimized': zone_power_optimized,
                        }

                        improvement = np.median(zone_power_optimized) - np.median(zone_power_initial)
                        print(f"    Median improvement: {improvement:+.2f} dB")

                    except Exception as e:
                        print(f"    ERROR in {result_key}: {type(e).__name__}: {e}")
                        results[result_key] = {
                            'status': 'error',
                            'scene_name': scene_name,
                            'sampler': sampler,
                            'frequency': freq,
                            'lds': lds,
                            'reason': f"{type(e).__name__}: {e}",
                        }

                    # Flush DrJIT caches between configs to prevent JIT state buildup
                    dr.flush_malloc_cache()

        # Mark scene complete and save checkpoint
        scene_elapsed = time.time() - scene_start_time
        completed_scenes.add(scene_name)
        scenes_completed_this_run += 1

        print(f"\nCompleted {scene_name} in {scene_elapsed/60:.1f} minutes")

        # Release scene resources to prevent memory accumulation
        del scene, tx, tx_placer, zone_mask
        gc.collect()

        # Flush DrJIT's internal JIT kernel cache and malloc cache
        # Without this, compiled LLVM kernels accumulate across scenes and
        # eventually exhaust memory, causing the process to hang
        dr.flush_kernel_cache()
        dr.flush_malloc_cache()

        # Save checkpoint after each scene
        save_checkpoint(results, completed_scenes, checkpoint_file, config)

        # Progress report with ETA
        total_elapsed = time.time() - run_start_time
        remaining_scenes = total_scenes - len(completed_scenes)
        if scenes_completed_this_run > 0:
            avg_per_scene = total_elapsed / scenes_completed_this_run
            eta_seconds = remaining_scenes * avg_per_scene
            print(f"Checkpoint saved. Progress: {len(completed_scenes)}/{total_scenes} "
                  f"({100*len(completed_scenes)/total_scenes:.0f}%)")
            print(f"Avg: {avg_per_scene/60:.1f} min/scene | "
                  f"ETA: {eta_seconds/3600:.1f}h ({remaining_scenes} scenes remaining)")
        print("=" * 80 + "\n")

    # Final summary
    total_elapsed = time.time() - run_start_time
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.1f} minutes)")
    print(f"Scenes completed: {len(completed_scenes)}/{total_scenes}")

    successful = sum(1 for r in results.values() if r.get('status') == 'success')
    failed = sum(1 for r in results.values() if r.get('status') == 'failed')
    errors = sum(1 for r in results.values() if r.get('status') == 'error')

    print(f"Successful configs: {successful}")
    print(f"Failed zones: {failed}")
    print(f"Errors: {errors}")
    print("=" * 80 + "\n")

    return results


def save_results(results, output_file):
    """Save results to pickle file."""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'num_results': len(results),
        'num_successful': sum(1 for r in results.values() if r.get('status') == 'success'),
    }

    output = {
        'metadata': metadata,
        'results': results,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

    print(f"Results saved to {output_file}")
    print(f"  Total entries: {metadata['num_results']}")
    print(f"  Successful: {metadata['num_successful']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run boresight optimization validation")
    parser.add_argument('--output', '-o', default=CONFIG['output_file'],
                        help='Output pickle file path')
    parser.add_argument('--max-scenes', '-n', type=int, default=None,
                        help='Maximum number of scenes to test')
    parser.add_argument('--scene-timeout', type=int, default=7200,
                        help='Max seconds per scene before skipping remaining configs (default: 7200)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignoring any existing checkpoint')
    args = parser.parse_args()

    if args.max_scenes:
        CONFIG['max_scenes'] = args.max_scenes

    checkpoint_file = args.output.replace('.pkl', '_checkpoint.pkl')

    if args.no_resume and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Removed existing checkpoint (--no-resume)")

    results = run_validation(CONFIG, checkpoint_file, args.scene_timeout)
    save_results(results, args.output)

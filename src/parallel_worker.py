"""
Parallel Worker Module for Scene Optimization
=============================================

This module contains the worker function for parallel scene processing.
It must be in a separate file (not a notebook cell) so that spawned
processes can import it.

WHY THIS FILE EXISTS:
--------------------
When Python multiprocessing uses 'spawn' (required for CUDA), it creates
fresh Python interpreters that need to import the worker function.

Jupyter notebooks run as `__main__`, which doesn't exist as an importable
module in child processes. By putting the worker function here, child
processes can do: `from parallel_worker import process_single_scene`
"""

import os
import sys
import gc
import time
import traceback
import numpy as np


def _serialize_value(v):
    """Convert a value to a JSON-serializable type."""
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    elif isinstance(v, np.ndarray):
        return v.tolist()
    elif isinstance(v, (np.floating, np.integer)):
        return float(v)
    elif isinstance(v, dict):
        return _serialize_dict(v)
    elif isinstance(v, (int, float, str, bool, type(None))):
        return v
    else:
        return str(v)


def _serialize_dict(d):
    """Convert a dict to JSON-serializable format."""
    return {k: _serialize_value(v) for k, v in d.items()}


def process_single_scene(args):
    """
    Process a single scene on a specific GPU.

    This function runs in a SEPARATE PROCESS, meaning:
    - It has its own Python interpreter
    - It has its own copy of all imported modules
    - Any data returned must be serializable (picklable)

    Parameters:
    -----------
    args : tuple
        (scene_name, gpu_id, validation_thresholds, parent_folder, output_dir)

    Returns:
    --------
    tuple : (scene_name, result_dict)
    """
    scene_name, gpu_id, validation_thresholds, parent_folder, output_dir = args

    # =========================================================================
    # STEP 0: CPU THREAD LIMITING - Set BEFORE importing any numeric libraries
    # This prevents CPU over-subscription when running multiple workers
    # =========================================================================
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # =========================================================================
    # STEP 1: GPU ISOLATION - Set BEFORE importing CUDA libraries
    # =========================================================================
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # =========================================================================
    # STEP 2: Import GPU-dependent libraries (each process gets own CUDA context)
    # =========================================================================
    import torch
    import mitsuba as mi

    from sionna.rt import load_scene, RadioMapSolver, AntennaArray
    from sionna.rt.antenna_pattern import antenna_pattern_registry

    # Add src to this process's path
    src_path = os.path.dirname(os.path.abspath(__file__))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from scene_parser import extract_building_info
    from tx_placement import TxPlacement
    from boresight_pathsolver import optimize_boresight_pathsolver
    from angle_utils import azimuth_elevation_to_yaw_pitch
    from zone_validator import find_valid_zone

    start_time = time.time()

    try:
        print(f"[GPU {gpu_id}] Starting: {scene_name}", flush=True)

        # =====================================================================
        # STEP 3: Scene processing (same logic as original notebook)
        # =====================================================================
        scene_xml_path = os.path.join(parent_folder, scene_name, "scene.xml")
        scene = load_scene(scene_xml_path)
        scene.frequency = 3.7e9

        # Configure antennas
        gnb_pattern = antenna_pattern_registry.get("tr38901")(polarization="V")
        ue_pattern = antenna_pattern_registry.get("iso")(polarization="V")
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
            radio_material.scattering_coefficient = 0.0

        # Find central building
        building_info = extract_building_info(scene_xml_path, verbose=False)
        min_distance = float('inf')
        selected_building_id = None

        for building_id, info in building_info.items():
            x_center, y_center = info['center']
            distance = np.sqrt(x_center**2 + y_center**2)
            if distance < min_distance:
                min_distance = distance
                selected_building_id = building_id

        # Place transmitter
        tx_placer = TxPlacement(scene, "gnb", scene_xml_path, selected_building_id, offset=5.0)
        tx_placer.set_rooftop_center()
        tx = tx_placer.tx
        gnb_position = tx.position.numpy().flatten().tolist()

        # Map config
        map_config = {
            'center': [0.0, 0.0, 0.0],
            'size': [1000, 1000],
            'cell_size': (1.0, 1.0),
            'ground_height': 0.0,
        }

        # Find valid zone
        zone_params_template = {'width': 250.0, 'height': 250.0}
        zone_mask, zone_stats, zone_center, validation_stats, attempts = find_valid_zone(
            scene=scene,
            tx_name="gnb",
            tx_position=gnb_position,
            map_config=map_config,
            scene_xml_path=scene_xml_path,
            zone_params_template=zone_params_template,
            min_distance=50.0,
            max_distance=400.0,
            max_attempts=20,
            validation_kwargs=validation_thresholds,
            verbose=False
        )

        if zone_mask is None:
            elapsed_time = time.time() - start_time
            print(f"[GPU {gpu_id}] ✗ {scene_name}: No valid zone after {attempts} attempts", flush=True)
            return scene_name, {
                'status': 'failed',
                'reason': 'No valid zone found',
                'attempts': attempts,
                'elapsed_time': elapsed_time
            }

        zone_center_x, zone_center_y = zone_center
        zone_params = zone_stats['zone_params']

        # Run optimization
        best_angles, loss_hist, angle_hist, grad_hist, cov_stats, initial_angles = optimize_boresight_pathsolver(
            scene=scene,
            tx_name="gnb",
            map_config=map_config,
            scene_xml_path=scene_xml_path,
            zone_mask=zone_mask,
            zone_params=zone_params,
            zone_stats=zone_stats,
            num_sample_points=100,
            learning_rate=2.0,
            num_iterations=100,
            verbose=False,
            lds="Halton",
            save_radiomap_frames=False,
            frame_save_interval=10,
            output_dir=output_dir
        )

        # Evaluate with RadioMapSolver
        rm_solver = RadioMapSolver()

        # Initial evaluation
        yaw_initial, pitch_initial = azimuth_elevation_to_yaw_pitch(initial_angles[0], initial_angles[1])
        tx.orientation = mi.Point3f(float(yaw_initial), float(pitch_initial), 0.0)

        rm_initial = rm_solver(
            scene, max_depth=5, samples_per_tx=int(6e8),
            cell_size=map_config['cell_size'], center=map_config['center'],
            orientation=[0, 0, 0], size=map_config['size'],
            los=True, specular_reflection=True, diffuse_reflection=True,
            diffraction=True, edge_diffraction=True, refraction=False, stop_threshold=None,
        )

        rss_initial = rm_initial.rss.numpy()[0, :, :]
        signal_initial = 10.0 * np.log10(rss_initial + 1e-30) + 30.0
        zone_power_initial = signal_initial[zone_mask == 1.0]

        # Optimized evaluation
        yaw_best, pitch_best = azimuth_elevation_to_yaw_pitch(best_angles[0], best_angles[1])
        tx.orientation = mi.Point3f(float(yaw_best), float(pitch_best), 0.0)

        rm_optimized = rm_solver(
            scene, max_depth=5, samples_per_tx=int(6e8),
            cell_size=map_config['cell_size'], center=map_config['center'],
            orientation=[0, 0, 0], size=map_config['size'],
            los=True, specular_reflection=True, diffuse_reflection=True,
            diffraction=True, edge_diffraction=True, refraction=False, stop_threshold=None,
        )

        rss_optimized = rm_optimized.rss.numpy()[0, :, :]
        signal_optimized = 10.0 * np.log10(rss_optimized + 1e-30) + 30.0
        zone_power_optimized = signal_optimized[zone_mask == 1.0]

        # =====================================================================
        # STEP 4: Build serializable result (lists, not numpy arrays!)
        # =====================================================================
        elapsed_time = time.time() - start_time

        improvement_mean = float(np.mean(zone_power_optimized) - np.mean(zone_power_initial))

        result = {
            'status': 'success',
            'scene_xml_path': scene_xml_path,
            'initial_angles': [float(x) for x in initial_angles],
            'best_angles': [float(x) for x in best_angles],
            'loss_hist': [float(x) for x in loss_hist],
            'tx_building_id': selected_building_id,
            'tx_position': gnb_position,
            'zone_center': [float(zone_center_x), float(zone_center_y)],
            'zone_attempts': attempts,
            'validation_stats': _serialize_dict(validation_stats),
            'zone_power_initial': zone_power_initial.tolist(),
            'zone_power_optimized': zone_power_optimized.tolist(),
            'elapsed_time': elapsed_time,
            'gpu_id': gpu_id,
            'stats': {
                'initial_mean': float(np.mean(zone_power_initial)),
                'initial_median': float(np.median(zone_power_initial)),
                'initial_p10': float(np.percentile(zone_power_initial, 10)),
                'optimized_mean': float(np.mean(zone_power_optimized)),
                'optimized_median': float(np.median(zone_power_optimized)),
                'optimized_p10': float(np.percentile(zone_power_optimized, 10)),
                'improvement_mean': improvement_mean,
                'improvement_median': float(np.median(zone_power_optimized) - np.median(zone_power_initial)),
                'improvement_p10': float(np.percentile(zone_power_optimized, 10) - np.percentile(zone_power_initial, 10)),
            }
        }

        print(f"[GPU {gpu_id}] ✓ {scene_name}: +{improvement_mean:.2f} dB in {elapsed_time:.1f}s", flush=True)

        # =====================================================================
        # STEP 5: Cleanup GPU memory
        # =====================================================================
        del scene, rm_solver, rm_initial, rm_optimized
        import torch
        torch.cuda.empty_cache()
        gc.collect()

        return scene_name, result

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[GPU {gpu_id}] ✗ {scene_name}: {error_msg}", flush=True)

        return scene_name, {
            'status': 'error',
            'reason': f"{error_msg}\n{traceback.format_exc()}",
            'elapsed_time': elapsed_time
        }

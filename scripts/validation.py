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
    'max_scenes': 99,  # Set to None for all scenes

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


def run_validation(config=None):
    """Run the full validation suite."""
    if config is None:
        config = CONFIG

    parent_folder = config['parent_folder']
    scene_dirs = sorted([d for d in os.listdir(parent_folder)
                         if os.path.isdir(os.path.join(parent_folder, d))])

    max_scenes = config.get('max_scenes')
    if max_scenes:
        scene_dirs = scene_dirs[:max_scenes]

    results = {}
    rm_solver = RadioMapSolver()

    print(f"Testing {len(scene_dirs)} scenes")
    print(f"Validation thresholds: {config['validation_thresholds']}\n")

    for i, scene_name in enumerate(scene_dirs):
        print(f"Scene {i+1}/{len(scene_dirs)}: {scene_name}")

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

        # Find valid zone
        zone_mask, zone_stats, zone_center, validation_stats, attempts = find_valid_zone(
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

        if zone_mask is None:
            print(f"Could not find valid zone after {attempts} attempts - skipping\n")
            results[scene_name] = {
                'status': 'failed',
                'reason': 'No valid zone found',
                'attempts': attempts
            }
            print("=" * 80 + "\n")
            continue

        zone_center_x, zone_center_y = zone_center
        zone_params = zone_stats['zone_params']

        print(f"Found valid zone after {attempts} attempt(s)")
        print(f"  Center: ({zone_center_x:.1f}, {zone_center_y:.1f})")
        print(f"  P10: {validation_stats['p10_power_dbm']:.1f} dBm, P90: {validation_stats['p90_power_dbm']:.1f} dBm")

        zone_distance_from_tx = np.sqrt((zone_center_x - gnb_position[0])**2 +
                                        (zone_center_y - gnb_position[1])**2)
        zone_angle_from_tx = np.arctan2(zone_center_y - gnb_position[1],
                                        zone_center_x - gnb_position[0])

        # Test matrix loop
        for sampler in config['samplers']:
            for freq in config['frequencies']:
                for lds in config['lds_methods']:

                    result_key = f"{scene_name}|{sampler}|{freq}|{lds}"
                    print(f"\n  Testing: {sampler}/{lds} @ {freq/1e9:.1f} GHz")

                    scene.frequency = freq

                    # Run optimization
                    best_angles, loss_hist, angle_hist, grad_hist, cov_stats, initial_angles = \
                        optimize_boresight_pathsolver(
                            scene=scene,
                            tx_name="gnb",
                            map_config=config['map_config'],
                            scene_xml_path=scene_xml_path,
                            zone_mask=zone_mask,
                            zone_params=zone_params,
                            zone_stats=zone_stats,
                            num_sample_points=config['optimization']['num_sample_points'],
                            learning_rate=config['optimization']['learning_rate'],
                            num_iterations=config['optimization']['num_iterations'],
                            verbose=False,
                            lds=lds,
                            sampler=sampler
                        )

                    print(f"    Initial: Az={initial_angles[0]:.1f}, El={initial_angles[1]:.1f}")
                    print(f"    Best:    Az={best_angles[0]:.1f}, El={best_angles[1]:.1f}")

                    # Evaluate initial configuration
                    zone_power_initial = evaluate_configuration(
                        scene, tx, zone_mask, config['map_config'], rm_solver, initial_angles
                    )

                    # Evaluate optimized configuration
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

                    # Flush DrJIT caches between configs to prevent JIT state buildup
                    dr.flush_malloc_cache()

        print(f"\nCompleted {scene_name}")
        print("=" * 80 + "\n")

        # Release scene resources to prevent memory accumulation
        del scene, tx, tx_placer, zone_mask
        gc.collect()

        # Flush DrJIT's internal JIT kernel cache and malloc cache
        # Without this, compiled LLVM kernels accumulate across scenes and
        # eventually exhaust memory, causing the process to hang
        dr.flush_kernel_cache()
        dr.flush_malloc_cache()

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
    args = parser.parse_args()

    if args.max_scenes:
        CONFIG['max_scenes'] = args.max_scenes

    results = run_validation(CONFIG)
    save_results(results, args.output)

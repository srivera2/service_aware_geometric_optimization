"""
Validation Worker Module for Scene Optimization
================================================

Standalone worker function for parallel scene optimization tasks.
Must be in a separate file so that spawned processes can import it
without triggering top-level CUDA imports.

WHY THIS FILE EXISTS:
--------------------
When Python multiprocessing uses 'spawn' (required for CUDA), it creates
fresh Python interpreters that need to import the worker function.
All GPU-dependent imports (mitsuba, drjit, sionna) are done LAZILY inside
the worker function, AFTER setting CUDA_VISIBLE_DEVICES, so each process
gets an isolated GPU context.
"""

import os
import sys
import gc
import time
import traceback
import numpy as np


def run_optimization_task(scene_name, scene_xml_path, sampler, freq, lds, zone_center, config, gpu_id, building_id):
    """Run a single optimization task on a specific GPU.

    Parameters
    ----------
    scene_name : str
        Name of the scene directory.
    scene_xml_path : str
        Path to the scene XML file.
    sampler : str
        Sampler type ('Rejection' or 'CDT').
    freq : float
        Carrier frequency in Hz.
    lds : str
        Low-discrepancy sequence method ('Sobol', 'Halton', or 'Latin').
    zone_center : tuple
        (x, y) center of the target zone.
    config : dict
        Full configuration dictionary.
    gpu_id : int
        GPU device index to use for this task.
    building_id : str
        Building ID for TX placement on rooftop.

    Returns
    -------
    dict
        Result dictionary with 'status' = 'success' | 'error'.
        On failure, includes 'reason' and 'traceback' for debugging.
    """
    # =========================================================================
    # GPU ISOLATION — must happen BEFORE any CUDA library imports
    # =========================================================================
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["DRJIT_LIBLLVM_PATH"] = "/usr/lib/x86_64-linux-gnu/libLLVM.so.20.1"

    # Add the src directory to the Python path
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(scripts_dir, "..", "src"))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # =========================================================================
    # LAZY CUDA IMPORTS — each process gets its own CUDA context
    # =========================================================================
    import mitsuba as mi
    import drjit as dr
    from sionna.rt import load_scene, RadioMapSolver, AntennaArray
    from sionna.rt.antenna_pattern import antenna_pattern_registry
    from boresight_pathsolver import create_zone_mask, optimize_boresight_pathsolver
    from tx_placement import TxPlacement
    from angle_utils import azimuth_elevation_to_yaw_pitch

    start_time = time.time()
    name_tuple = (scene_name, str(freq), lds, sampler)
    full_name = "_".join(name_tuple)

    try:
        # =================================================================
        # SCENE SETUP
        # =================================================================
        scene = load_scene(scene_xml_path)
        scene.frequency = freq

        # gNB antenna: 3GPP TR 38.901 pattern
        gnb_pattern = antenna_pattern_registry.get("tr38901")(polarization="V")
        ue_pattern = antenna_pattern_registry.get("iso")(polarization="V")
        single_element = np.array([[0.0, 0.0, 0.0]])

        scene.tx_array = AntennaArray(
            antenna_pattern=gnb_pattern, normalized_positions=single_element.T
        )
        scene.rx_array = AntennaArray(
            antenna_pattern=ue_pattern, normalized_positions=single_element.T
        )

        for radio_material in scene.radio_materials.values():
            radio_material.scattering_coefficient = 0.4

        # Place transmitter on the selected building rooftop
        tx_placer = TxPlacement(
            scene, "gnb", scene_xml_path, building_id,
            offset=config["tx_offset"],
        )
        tx_placer.set_rooftop_center()

        # =================================================================
        # ZONE SETUP
        # =================================================================
        rm_solver = RadioMapSolver()
        zone_params = {**config["zone_params_template"], "center": zone_center}

        zone_mask = create_zone_mask(
            config["map_config"],
            zone_type="box",
            origin_point=zone_center,
            zone_params=zone_params,
            target_height=config["map_config"].get("target_height", 1.5),
            scene_xml_path=scene_xml_path,
            exclude_buildings=True,
        )

        # =================================================================
        # OPTIMIZATION
        # =================================================================
        best_angles, loss_hist, angle_hist, grad_hist, cov_stats, initial_angles = (
            optimize_boresight_pathsolver(
                scene=scene,
                tx_name="gnb",
                map_config=config["map_config"],
                scene_xml_path=scene_xml_path,
                zone_mask=zone_mask,
                zone_params=zone_params,
                num_sample_points=config["optimization"]["num_sample_points"],
                learning_rate=config["optimization"]["learning_rate"],
                num_iterations=config["optimization"]["num_iterations"],
                verbose=False,
                lds=lds,
                sampler=sampler,
            )
        )

        print(
            f"    [GPU {gpu_id}] {full_name}: "
            f"Initial Az={initial_angles[0]:.1f}, El={initial_angles[1]:.1f} -> "
            f"Best Az={best_angles[0]:.1f}, El={best_angles[1]:.1f}",
            flush=True,
        )

        # =================================================================
        # EVALUATE INITIAL AND OPTIMIZED CONFIGURATIONS
        # =================================================================
        rm_config = config["rm_solver"]
        map_cfg = config["map_config"]
        tx = scene.get("gnb")

        def _evaluate(angles):
            yaw, pitch = azimuth_elevation_to_yaw_pitch(angles[0], angles[1])
            tx.orientation = mi.Point3f(float(yaw), float(pitch), 0.0)
            rm = rm_solver(
                scene,
                max_depth=rm_config["max_depth"],
                samples_per_tx=rm_config["samples_per_tx"],
                cell_size=map_cfg["cell_size"],
                center=map_cfg["center"],
                orientation=[0, 0, 0],
                size=map_cfg["size"],
                los=True,
                specular_reflection=True,
                diffuse_reflection=True,
                diffraction=True,
                edge_diffraction=True,
                refraction=False,
                stop_threshold=None,
            )
            rss_watts = rm.rss.numpy()[0, :, :]
            signal_dBm = 10.0 * np.log10(rss_watts + 1e-30) + 30.0
            return signal_dBm[zone_mask == 1.0]

        zone_power_initial = _evaluate(initial_angles)
        zone_power_optimized = _evaluate(best_angles)

        improvement = float(np.median(zone_power_optimized) - np.median(zone_power_initial))
        elapsed = time.time() - start_time
        print(f"    [GPU {gpu_id}] {full_name}: Median improvement {improvement:+.2f} dB ({elapsed:.1f}s)", flush=True)

        # =================================================================
        # BUILD RESULT
        # =================================================================
        val = {
            "status": "success",
            "scene_name": full_name,
            "sampler": sampler,
            "frequency": freq,
            "lds": lds,
            "gpu_id": gpu_id,
            "scene_xml_path": scene_xml_path,
            "map_config": config["map_config"],
            "initial_angles": initial_angles,
            "best_angles": best_angles,
            "loss_hist": loss_hist,
            "angle_hist": angle_hist,
            "grad_hist": grad_hist,
            "cov_stats": cov_stats,
            "zone_power_initial": zone_power_initial,
            "zone_power_optimized": zone_power_optimized,
            "elapsed_time": elapsed,
        }

        # =================================================================
        # CLEANUP GPU MEMORY
        # =================================================================
        del scene, zone_mask, rm_solver
        gc.collect()
        dr.flush_kernel_cache()
        dr.flush_malloc_cache()

        return val

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {e}"
        print(f"    [GPU {gpu_id}] FAILED {full_name}: {error_msg} ({elapsed:.1f}s)", flush=True)

        return {
            "status": "error",
            "scene_name": full_name,
            "sampler": sampler,
            "frequency": freq,
            "lds": lds,
            "gpu_id": gpu_id,
            "reason": error_msg,
            "traceback": traceback.format_exc(),
            "elapsed_time": elapsed,
        }

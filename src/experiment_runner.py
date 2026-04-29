"""
experiment_runner.py
====================
Reproducible experiment suite runner for multi-TX boresight optimization.

Orchestrates the gradient-based optimizer (three sampling strategy variants)
and four zeroth-order / empirical baselines from a single call, guaranteeing
identical initial conditions for every method.  Designed for academic
experiments: all hyperparameters and timestamps are stored alongside results,
and the output dict contains everything needed to generate paper figures.

Public API
----------
ExperimentConfig          : flat dataclass for all hyperparameters + which baselines to run
run_experiment_suite()    : main orchestrator — returns a fully structured result dict
compare_all_results()     : builds a pandas DataFrame suitable for LaTeX export
save_results()            : JSON serialization (strips large arrays by default)
load_results()            : JSON deserialization
get_distribution_data()   : extract raw per-cell value arrays for custom plotting
plot_cdf()                : empirical CDF comparison across baselines (requires in-memory run; raw arrays stripped from saved JSON)
plot_loss_curves()        : convergence curves for gradient methods (x_axis="iteration"|"wall_time")
plot_metric_bars()        : grouped bar chart of summary metrics
plot_per_metric_panels()  : comprehensive grid — rows=metrics, cols=TX+All — horizontal bars, legend outside axes
plot_efficiency_frontier(): quality-vs-time scatter showing the efficiency frontier across baselines
plot_zone_overview()      : scene map with zone masks, TX markers, and building labels

Baseline IDs
------------
  Gradient (differ only in sampler / sampling_strata):
    "grad_full_rejection"       rejection sampler,    full zone,         Halton
    "grad_full_triangulated"    triangulated sampler, full zone,         Halton
    "grad_proportional"         triangulated sampler, proportional dead, Halton

  Zeroth-order:
    "random_search"             brute-force random search
    "pso"                       particle swarm optimisation
    "coordinate_descent"        axis-aligned coordinate descent

  Empirical:
    "uma_naive"                 3GPP TR 38.901 UMa fixed-pointing baseline

  First-order (RadioMapSolver-driven):
    "radiomap_gradient"         Adam optimizer with RadioMapSolver as the differentiable engine;
                                SINR loss averaged over radio-map cells within each TX's zone
"""

from __future__ import annotations

import copy
import datetime
import json
import time
import traceback as tb
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import drjit as dr
import mitsuba as mi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from multi_tx_optimizer import TxConfig, optimize_multi_tx, compare_multi_tx_performance
from baseline_optimizers import (
    random_search_multi_tx,
    pso_multi_tx,
    coordinate_descent_multi_tx,
    uma_naive_baseline_multi_tx,
    radiomap_gradient_multi_tx,
    coarse_xy_sweep,
)
from angle_utils import yaw_pitch_to_azimuth_elevation, azimuth_elevation_to_yaw_pitch


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All hyperparameters and run-selection flags for one experiment suite.

    Defaults match the existing optimizer defaults so a bare ``ExperimentConfig()``
    produces a sensible reference run.
    """

    # --- Which baselines to run -------------------------------------------
    # Any subset of the keys in _BASELINE_REGISTRY.
    baselines: list = field(default_factory=lambda: [
        "grad_full_rejection",
        "grad_full_triangulated",
        "grad_proportional",
        "radiomap_gradient",
        "random_search",
        "pso",
        "coordinate_descent",
        "uma_naive",
    ])

    # --- Shared -----------------------------------------------------------
    noise_power: float = 1e-10
    verbose: bool = True

    # --- Gradient optimizer (shared across all three grad baselines) ------
    learning_rate: float = 3.5
    num_iterations: int = 50
    dead_tail_percentile: float = 1.0
    max_dbscan_points: int = 100_000
    grad_lds: str = "Sobol"
    debug_viz: bool = False

    # --- Random search ----------------------------------------------------
    rs_n_candidates: int = 400
    rs_lds: str = "Sobol"
    rs_seed: Optional[int] = None

    # --- PSO --------------------------------------------------------------
    pso_n_particles: int = 20
    pso_num_iterations: int = 30   # raised from 15: must exceed pso_early_stop_window
                                   # by a comfortable margin so PSO can plateau naturally
    pso_w: float = 0.7
    pso_c1: float = 1.5
    pso_c2: float = 1.5
    pso_lds: str = "Sobol"
    pso_seed: Optional[int] = None

    # --- Coordinate descent -----------------------------------------------
    cd_num_cycles: int = 10
    cd_n_line_points: int = 20
    cd_lds: str = "Sobol"

    # --- UMa naive --------------------------------------------------------
    uma_mechanical_downtilt_deg: float = 0.0
    uma_electrical_downtilt_deg: float = 6.0
    uma_lds: str = "Sobol"

    # --- RadioMap gradient -----------------------------------------------
    rmg_samples_per_tx: int = int(1e7)
    rmg_max_depth: int = 8
    rmg_lds: str = "Sobol"

    # --- Loss hyperparameters (shared across gradient + zeroth-order) -----
    # SIR threshold (dB) at the sigmoid centre.  Cells above this are considered
    # "covered"; cells below are pushed up.  Set near the expected operating
    # sir_median so ~50 % of cells lie in the active gradient zone.
    sir_threshold_db: float = 5.0
    # Sigmoid steepness (dB⁻¹).  Lower values widen the active gradient band,
    # making the loss behave more like log-SIR.  Higher values sharpen the
    # threshold but shrink the zone of informative gradient.
    sigmoid_k: float = 0.2

    # --- Early stopping ---------------------------------------------------
    # Shared threshold — all iterative methods stop when improvement < this.
    # With the sigmoid loss bounded in [-1, 0], 1e-2 represents a 1 % change
    # across the full loss range — a defensible stopping point.  Also used as
    # the ReduceLROnPlateau threshold, so the LR schedule tightens consistently.
    early_stop_min_improvement: float = 1e-3

    # Gradient / RadioMap: oscillation window.  These methods can oscillate
    # around a local minimum, so a wider window (10) is needed before declaring
    # convergence.  Additionally requires >= early_stop_min_flips sign-flips in
    # the delta sequence to distinguish oscillation from slow monotone descent.
    early_stop_window: int = 10
    early_stop_min_flips: int = 5

    # PSO: stagnation window.  PSO tracks a monotone global-best, so a shorter
    # window (5) is sufficient and semantically equivalent to coordinate
    # descent's per-cycle no-improvement check — both ask "has anything improved
    # in one representative unit of work?"
    pso_early_stop_window: int = 5

    # --- Coarse warm-start ------------------------------------------------
    # When coarse_xy_steps > 0, a greedy XY sweep is run before any baseline.
    # Each TX's position is swept over a ``coarse_xy_steps × coarse_xy_steps``
    # grid of interior points on its building rooftop polygon; at each
    # candidate the boresight is aimed geometrically toward the zone centroid.
    # The best position becomes the shared starting point for ALL methods.
    # Gradient-free methods (PSO, random search) have the warm-start seeded
    # into their initial populations; all others start from it directly.
    # Set to 0 to disable (default).
    # Recommended: 5 (up to 25 candidates per TX, filtered to polygon interior).
    coarse_xy_steps: int = 0
    coarse_samples_per_tx: int = int(1e7)   # samples per RadioMapSolver call in sweep

    # --- Output / reproducibility -----------------------------------------
    output_path: Optional[str] = None
    # strip_raw_arrays: drop rsrp_values_dbm / sir_values_db before saving to
    # disk.  For a 1400 x 1400 m map at 0.5 m cell size the raw arrays are
    # ~8 M floats per TX per baseline.  They are always kept in the in-memory
    # dict so CDF plots can be made during the notebook session.
    strip_raw_arrays: bool = True
    experiment_tag: str = ""


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------

def _grad_kwargs(cfg: ExperimentConfig, tx_cfgs, scene, map_cfg, xml,
                 sampler: str, sampling_strata: str) -> dict:
    return dict(
        scene=scene, tx_configs=tx_cfgs, map_config=map_cfg, scene_xml_path=xml,
        learning_rate=cfg.learning_rate,
        num_iterations=cfg.num_iterations,
        noise_power=cfg.noise_power,
        dead_tail_percentile=cfg.dead_tail_percentile,
        max_dbscan_points=cfg.max_dbscan_points,
        lds=cfg.grad_lds,
        sampler=sampler,
        sampling_strata=sampling_strata,
        verbose=cfg.verbose,
        debug_viz=cfg.debug_viz,
        early_stop_window=cfg.early_stop_window,
        early_stop_min_improvement=cfg.early_stop_min_improvement,
        early_stop_min_flips=cfg.early_stop_min_flips,
        sir_threshold_db=cfg.sir_threshold_db,
        sigmoid_k=cfg.sigmoid_k,
    )


# Each entry: (optimizer_function, kwargs_builder_lambda)
# kwargs_builder signature: (ExperimentConfig, tx_configs, scene, map_config, scene_xml_path) -> dict
_BASELINE_REGISTRY: dict[str, tuple[Callable, Callable]] = {

    "grad_full_rejection": (
        optimize_multi_tx,
        lambda c, t, s, m, x: _grad_kwargs(c, t, s, m, x,
                                            sampler="rejection",
                                            sampling_strata="full"),
    ),

    "grad_full_triangulated": (
        optimize_multi_tx,
        lambda c, t, s, m, x: _grad_kwargs(c, t, s, m, x,
                                            sampler="triangulated",
                                            sampling_strata="full"),
    ),

    "grad_proportional": (
        optimize_multi_tx,
        lambda c, t, s, m, x: _grad_kwargs(c, t, s, m, x,
                                            sampler="triangulated",
                                            sampling_strata="proportional"),
    ),

    "random_search": (
        random_search_multi_tx,
        lambda c, t, s, m, x: dict(
            scene=s, tx_configs=t, map_config=m, scene_xml_path=x,
            n_candidates=c.rs_n_candidates,
            noise_power=c.noise_power,
            lds=c.rs_lds,
            seed=c.rs_seed,
            verbose=c.verbose,
            sir_threshold_db=c.sir_threshold_db,
            sigmoid_k=c.sigmoid_k,
        ),
    ),

    "pso": (
        pso_multi_tx,
        lambda c, t, s, m, x: dict(
            scene=s, tx_configs=t, map_config=m, scene_xml_path=x,
            n_particles=c.pso_n_particles,
            num_iterations=c.pso_num_iterations,
            noise_power=c.noise_power,
            w=c.pso_w, c1=c.pso_c1, c2=c.pso_c2,
            lds=c.pso_lds,
            seed=c.pso_seed,
            verbose=c.verbose,
            early_stop_window=c.pso_early_stop_window,
            early_stop_min_improvement=c.early_stop_min_improvement,
            sir_threshold_db=c.sir_threshold_db,
            sigmoid_k=c.sigmoid_k,
        ),
    ),

    "coordinate_descent": (
        coordinate_descent_multi_tx,
        lambda c, t, s, m, x: dict(
            scene=s, tx_configs=t, map_config=m, scene_xml_path=x,
            num_cycles=c.cd_num_cycles,
            n_line_points=c.cd_n_line_points,
            noise_power=c.noise_power,
            lds=c.cd_lds,
            verbose=c.verbose,
            early_stop_min_improvement=c.early_stop_min_improvement,
            sir_threshold_db=c.sir_threshold_db,
            sigmoid_k=c.sigmoid_k,
        ),
    ),

    "uma_naive": (
        uma_naive_baseline_multi_tx,
        lambda c, t, s, m, x: dict(
            scene=s, tx_configs=t, map_config=m, scene_xml_path=x,
            mechanical_downtilt_deg=c.uma_mechanical_downtilt_deg,
            electrical_downtilt_deg=c.uma_electrical_downtilt_deg,
            noise_power=c.noise_power,
            lds=c.uma_lds,
            verbose=c.verbose,
            sir_threshold_db=c.sir_threshold_db,
            sigmoid_k=c.sigmoid_k,
        ),
    ),

    "radiomap_gradient": (
        radiomap_gradient_multi_tx,
        lambda c, t, s, m, x: dict(
            scene=s, tx_configs=t, map_config=m, scene_xml_path=x,
            learning_rate=c.learning_rate,
            num_iterations=c.num_iterations,
            noise_power=c.noise_power,
            lds=c.rmg_lds,
            samples_per_tx=c.rmg_samples_per_tx,
            max_depth=c.rmg_max_depth,
            verbose=c.verbose,
            early_stop_window=c.early_stop_window,
            early_stop_min_improvement=c.early_stop_min_improvement,
            early_stop_min_flips=c.early_stop_min_flips,
        ),
    ),
}

# Human-readable display names for plots / tables
_BASELINE_LABELS: dict[str, str] = {
    "grad_full_rejection":    "Grad (Rejection, Full)",
    "grad_full_triangulated": "Grad (Triangulated, Full)",
    "grad_proportional":      "Grad (Triangulated, Prop.)",
    "radiomap_gradient":      "RadioMap Grad (Adam)",
    "random_search":          "Random Search",
    "pso":                    "PSO",
    "coordinate_descent":     "Coord. Descent",
    "uma_naive":              "UMa Naive (3GPP)",
}

# Color palette — Option 1a vibrant scheme
_BASELINE_COLORS: dict[str, str] = {
    "grad_full_rejection":    "#84BCE1",   # steel blue
    "grad_full_triangulated": "#80C080",   # medium green
    "grad_proportional":      "#E68A88",   # warm salmon  ← hero method
    "radiomap_gradient":      "#F6BE80",   # amber orange
    "random_search":          "#FAD7A8",   # light peach
    "pso":                    "#AF93C3",   # soft purple
    "coordinate_descent":     "#CFE5B9",   # sage green
    "uma_naive":              "#C5DBE9",   # pale sky blue
}

# Fixed line-style cycle for greyscale legibility
_BASELINE_LINESTYLES: dict[str, str] = {
    "grad_full_rejection":    "-",
    "grad_full_triangulated": "--",
    "grad_proportional":      "-.",
    "radiomap_gradient":      (0, (4, 2, 1, 2)),
    "random_search":          ":",
    "pso":                    (0, (5, 1)),
    "coordinate_descent":     (0, (3, 1, 1, 1)),
    "uma_naive":              (0, (1, 1)),
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _capture_initial_angles(scene, tx_configs: list) -> dict[str, tuple[float, float]]:
    """Read TX orientations already in the scene and convert to (az, el) degrees.

    Reads from the live Sionna scene rather than re-computing the look-at
    direction, so any manual adjustments made in the notebook are captured.
    """
    angles = {}
    for cfg in tx_configs:
        tx = scene.get(cfg.name)
        yaw_r   = float(dr.detach(tx.orientation[0])[0])
        pitch_r = float(dr.detach(tx.orientation[1])[0])
        az, el  = yaw_pitch_to_azimuth_elevation(yaw_r, pitch_r)
        angles[cfg.name] = (az, el)
    return angles


def _capture_initial_positions(scene, tx_configs: list) -> dict[str, list[float]]:
    """Read TX positions from the scene (set by TxPlacement) for later restoration."""
    positions = {}
    for cfg in tx_configs:
        tx = scene.get(cfg.name)
        positions[cfg.name] = [
            float(dr.detach(tx.position[0])[0]),
            float(dr.detach(tx.position[1])[0]),
            float(dr.detach(tx.position[2])[0]),
        ]
    return positions


def _restore_initial_state(scene, tx_configs: list,
                           initial_positions: dict[str, list[float]],
                           initial_angles: dict[str, tuple[float, float]]) -> None:
    """Reset every TX in the scene to its original placement position and orientation.

    Called before each baseline run so that ``_setup_tx_state`` inside each
    optimizer reads the correct starting position.  Without this, each baseline
    would start from the previous baseline's final (optimised) state.
    """
    for cfg in tx_configs:
        pos = initial_positions[cfg.name]
        az, el = initial_angles[cfg.name]
        yaw_r, pitch_r = azimuth_elevation_to_yaw_pitch(az, el)
        tx = scene.get(cfg.name)
        tx.position    = mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2]))
        tx.orientation = mi.Point3f(float(yaw_r), float(pitch_r), 0.0)


def _inject_initial_angles(tx_configs: list,
                           initial_angles: dict[str, tuple[float, float]]) -> list:
    """Return deep-copied TxConfigs with initial_azimuth_deg / initial_elevation_deg set.

    ``_setup_tx_state`` in both ``optimize_multi_tx`` and the baseline optimizers
    checks these fields before falling back to look-at auto-compute (line 179 of
    multi_tx_optimizer.py), so all optimizers start from identical angles.
    """
    result = []
    for cfg in tx_configs:
        new_cfg = copy.deepcopy(cfg)
        az, el  = initial_angles[cfg.name]
        new_cfg.initial_azimuth_deg   = az
        new_cfg.initial_elevation_deg = el
        result.append(new_cfg)
    return result


def _clean_kwargs(kwargs: dict) -> dict:
    """Remove non-JSON-serializable keys (scene objects, tx_configs) from kwargs."""
    skip = {"scene", "tx_configs", "on_iteration_callback"}
    return {k: v for k, v in kwargs.items() if k not in skip}


def _make_serializable(obj: Any) -> Any:
    """Recursively convert obj to a JSON-serializable form."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    # Anything else that isn't a JSON primitive → stringify
    if not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj


_RAW_ARRAY_KEYS = {
    "rsrp_values_dbm",
    "sir_values_db",
    "dominant_tx_map",
    "boundary_mask",
    "sinr_db_grid",
    "_boundary_sinr_values",
}


def _strip_raw_arrays(comparison_stats: dict) -> dict:
    """Return a shallow copy of comparison_stats with per-cell array keys removed.

    Operates only on the per-TX sub-dicts, not on the top-level dict itself.
    """
    stripped = {}
    for tx_name, tx_stats in comparison_stats.items():
        if not isinstance(tx_stats, dict):
            stripped[tx_name] = tx_stats
            continue
        stripped[tx_name] = {}
        for section, section_data in tx_stats.items():
            if isinstance(section_data, dict):
                stripped[tx_name][section] = {
                    k: v for k, v in section_data.items()
                    if k not in _RAW_ARRAY_KEYS
                }
            else:
                stripped[tx_name][section] = section_data
    return stripped


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_experiment_suite(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    zone_masks: dict,
    exp_config: Optional[ExperimentConfig] = None,
    on_baseline_complete: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    """Run all selected baselines with identical initial conditions.

    Parameters
    ----------
    scene : sionna.rt.Scene
        Sionna scene with all TXs already added and positioned by ``TxPlacement``.
    tx_configs : list[TxConfig]
        One entry per transmitter; order must match scene insertion order.
    map_config : dict
        ``{'center': [x,y,z], 'size': [w,h], 'cell_size': [cw,ch]}``
    scene_xml_path : str
        Path to the Mitsuba scene XML file.
    zone_masks : dict[str, np.ndarray]
        ``{tx_name: 2-D binary mask}`` pre-computed by the notebook via
        ``create_zone_mask``.  Passed directly to ``compare_multi_tx_performance``.
    exp_config : ExperimentConfig or None
        All hyperparameters.  Defaults to ``ExperimentConfig()`` if ``None``.
    on_baseline_complete : callable or None
        Optional hook called after each baseline completes.  Signature::

            on_baseline_complete(baseline_id: str, entry: dict)

        Useful for notebook progress display or live plotting.

    Returns
    -------
    dict
        See module docstring for the full schema.  Key structure::

            {
              "metadata": { timestamp, initial_angles, exp_config, ... },
              "results":  { baseline_id: { optimizer_result,
                                           comparison_stats,
                                           elapsed_s, hyperparams } }
            }
    """
    if exp_config is None:
        exp_config = ExperimentConfig()

    # --- Validation -------------------------------------------------------
    unknown = [b for b in exp_config.baselines if b not in _BASELINE_REGISTRY]
    if unknown:
        valid = list(_BASELINE_REGISTRY.keys())
        raise ValueError(
            f"Unknown baseline ID(s): {unknown}\n"
            f"Valid IDs: {valid}"
        )

    # --- Optional coarse XY warm-start (runs before capturing angles) --------
    xy_sweep_log = None

    if exp_config.coarse_xy_steps > 0:
        xy_result = coarse_xy_sweep(
            scene, tx_configs, map_config, scene_xml_path,
            n_xy_steps=exp_config.coarse_xy_steps,
            samples_per_tx=exp_config.coarse_samples_per_tx,
            noise_power=exp_config.noise_power,
            sir_threshold_db=exp_config.sir_threshold_db,
            sigmoid_k=exp_config.sigmoid_k,
            verbose=exp_config.verbose,
        )
        # Apply warm-start positions and zone-centroid orientations so
        # _capture_initial_angles / _capture_initial_positions pick them up
        # as the shared starting point for all baselines.
        for cfg in tx_configs:
            per_tx = xy_result["meta"]["per_tx"][cfg.name]
            best_x, best_y = per_tx["best_x"], per_tx["best_y"]
            best_az, best_el = per_tx["best_az"], per_tx["best_el"]
            yaw_r, pitch_r = azimuth_elevation_to_yaw_pitch(best_az, best_el)
            tx = scene.get(cfg.name)
            z  = float(dr.detach(tx.position[2])[0])
            tx.position    = mi.Point3f(float(best_x), float(best_y), z)
            tx.orientation = mi.Point3f(float(yaw_r), float(pitch_r), 0.0)
        xy_sweep_log = xy_result

    # --- Capture initial scene state (once) --------------------------------
    initial_angles    = _capture_initial_angles(scene, tx_configs)
    initial_positions = _capture_initial_positions(scene, tx_configs)

    # Flat seed_params vector for gradient-free methods (PSO, random search)
    seed_params: Optional[np.ndarray] = None
    if xy_sweep_log is not None:
        seed_params = np.array(
            [v for cfg in tx_configs
             for v in [initial_angles[cfg.name][0],   # az
                       initial_angles[cfg.name][1],   # el
                       initial_positions[cfg.name][0],  # x
                       initial_positions[cfg.name][1]]],  # y
            dtype=np.float64,
        )

    if exp_config.verbose:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUITE  —  {len(exp_config.baselines)} baseline(s)  "
              f"|  {len(tx_configs)} TX(s)")
        if exp_config.experiment_tag:
            print(f"Tag: {exp_config.experiment_tag}")
        _warm_parts = []
        if xy_sweep_log is not None:
            _warm_parts.append(
                f"xy sweep {exp_config.coarse_xy_steps}×{exp_config.coarse_xy_steps} grid")
        warm_label = (f"  [warm-start: {', '.join(_warm_parts)}]"
                      if _warm_parts else "")
        print(f"Initial angles:{warm_label}")
        for name, (az, el) in initial_angles.items():
            pos = initial_positions[name]
            print(f"  {name}: Az={az:.1f}°  El={el:.1f}°  "
                  f"pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        print(f"{'='*70}\n")

    # --- Metadata ---------------------------------------------------------
    metadata = {
        "timestamp_utc":  datetime.datetime.utcnow().isoformat() + "Z",
        "experiment_tag": exp_config.experiment_tag,
        "baselines_run":  list(exp_config.baselines),
        "tx_names":       [cfg.name for cfg in tx_configs],
        "initial_angles": {n: list(a) for n, a in initial_angles.items()},
        "initial_positions": initial_positions,
        "exp_config":     asdict(exp_config),
        "xy_sweep_log":   xy_sweep_log,
    }

    # --- Run each baseline ------------------------------------------------
    results = {}

    for baseline_id in exp_config.baselines:
        if exp_config.verbose:
            print(f"\n{'#'*70}")
            print(f"  BASELINE: {baseline_id}")
            print(f"{'#'*70}")

        optimizer_fn, kwargs_builder = _BASELINE_REGISTRY[baseline_id]

        # Restore scene to original placement before each run so every
        # optimizer reads the same starting position from the scene.
        _restore_initial_state(scene, tx_configs, initial_positions, initial_angles)

        # Deep-copy TxConfigs with injected angles — guarantees all optimizers
        # start from the same boresight regardless of internal look-at logic.
        tx_configs_copy = _inject_initial_angles(tx_configs, initial_angles)

        kwargs = kwargs_builder(exp_config, tx_configs_copy, scene, map_config, scene_xml_path)

        # Inject warm-start seed into gradient-free methods that support it
        if seed_params is not None and baseline_id in ("random_search", "pso"):
            kwargs["seed_params"] = seed_params

        t0 = time.time()

        # --- Optimizer call -----------------------------------------------
        optimizer_result = None
        opt_error        = None
        try:
            optimizer_result = optimizer_fn(**kwargs)
        except Exception as e:
            opt_error = {"error": str(e), "traceback": tb.format_exc()}
            if exp_config.verbose:
                print(f"  [ERROR] Optimizer failed for '{baseline_id}': {e}")

        # --- Comparison call ----------------------------------------------
        comparison_stats = None
        cmp_error        = None
        if optimizer_result is not None:
            try:
                _, comparison_stats = compare_multi_tx_performance(
                    scene=scene,
                    tx_configs=tx_configs_copy,
                    multi_result=optimizer_result,
                    map_config=map_config,
                    zone_masks=zone_masks,
                    noise_power=exp_config.noise_power,
                    fig=False,
                )
            except Exception as e:
                cmp_error = {"error": str(e), "traceback": tb.format_exc()}
                if exp_config.verbose:
                    print(f"  [ERROR] compare_multi_tx_performance failed "
                          f"for '{baseline_id}': {e}")

        elapsed_s = time.time() - t0

        entry = {
            "baseline_id":      baseline_id,
            "elapsed_s":        elapsed_s,
            "hyperparams":      _clean_kwargs(kwargs),
            "optimizer_result": optimizer_result if opt_error is None else opt_error,
            "comparison_stats": comparison_stats if cmp_error is None else cmp_error,
        }
        if opt_error is not None:
            entry["optimizer_error"] = opt_error
        if cmp_error is not None:
            entry["comparison_error"] = cmp_error

        results[baseline_id] = entry

        if exp_config.verbose:
            status = "OK" if opt_error is None else "FAILED"
            print(f"\n  [{baseline_id}] done — {elapsed_s:.1f}s  status={status}")

        if on_baseline_complete is not None:
            try:
                on_baseline_complete(baseline_id, entry)
            except Exception:
                pass  # never let a callback abort the suite

    # --- Assemble output --------------------------------------------------
    suite_output = {"metadata": metadata, "results": results}
    if exp_config.output_path is not None:
        save_results(suite_output, exp_config.output_path,
                     strip_raw_arrays=exp_config.strip_raw_arrays)

    if exp_config.verbose:
        elapsed_total = sum(
            r["elapsed_s"] for r in results.values()
            if isinstance(r.get("elapsed_s"), (int, float))
        )
        print(f"\n{'='*70}")
        print(f"SUITE COMPLETE — total wall time: {elapsed_total:.1f}s")
        for bid in exp_config.baselines:
            r = results[bid]
            status = "OK" if "optimizer_error" not in r else "FAILED"
            print(f"  {bid:<35s}  {r['elapsed_s']:7.1f}s  [{status}]")
        print(f"{'='*70}\n")

    return suite_output


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

_DEFAULT_METRICS = [
    "rsrp_mean_dbm",
    "rsrp_p10_dbm",
    "rsrp_p90_dbm",
    "sir_median_db",
    "sir_p10_db",
    "coverage_fraction",
]

def compare_all_results(
    suite_output: dict,
    metrics: Optional[list] = None,
    tx_name: Optional[str] = None,
    config_type: str = "optimized",
    print_table: bool = True,
) -> pd.DataFrame:
    """Build a summary DataFrame comparing all baselines on key metrics.

    Parameters
    ----------
    suite_output : dict
        Return value of ``run_experiment_suite``.
    metrics : list[str] or None
        Metric names to include (from ``comparison_stats[tx_name][config_type]``).
        Defaults to ``["rsrp_mean_dbm", "rsrp_p10_dbm", "rsrp_p90_dbm",
        "sir_median_db", "sir_p10_db", "coverage_fraction"]``.
    tx_name : str or None
        If given, restrict to that TX zone.  If ``None``, average the metric
        values across all TX zones.
    config_type : str
        ``"optimized"`` (default) or ``"initial"``.
    print_table : bool
        Print the DataFrame to stdout if ``True``.

    Returns
    -------
    pd.DataFrame
        Indexed by baseline ID.  Call ``.to_latex(float_format="%.2f")`` for
        paper-ready table export.
    """
    if metrics is None:
        metrics = list(_DEFAULT_METRICS)

    tx_names = suite_output["metadata"]["tx_names"]
    target_txs = [tx_name] if tx_name is not None else tx_names

    rows = []
    for baseline_id, entry in suite_output["results"].items():
        row = {"baseline": baseline_id,
               "label":    _BASELINE_LABELS.get(baseline_id, baseline_id)}

        cstats = entry.get("comparison_stats")

        if cstats is None or "error" in entry.get("comparison_stats", {}):
            # Failed run — fill with NaN
            for m in metrics:
                row[m] = float("nan")
            row["elapsed_s"] = entry.get("elapsed_s", float("nan"))
            rows.append(row)
            continue

        # Average scalar metrics across the target TXs
        for m in metrics:
            vals = []
            for tname in target_txs:
                try:
                    vals.append(float(cstats[tname][config_type][m]))
                except (KeyError, TypeError):
                    pass
            row[m] = float(np.mean(vals)) if vals else float("nan")

        row["elapsed_s"] = float(entry.get("elapsed_s", float("nan")))
        rows.append(row)

    df = (pd.DataFrame(rows)
            .set_index("baseline")
            [["label"] + metrics + ["elapsed_s"]])

    if print_table:
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY  ({config_type.upper()})  —  "
              f"TX: {tx_name if tx_name else 'all (avg)'}")
        print(f"{'='*70}")
        float_fmt = lambda x: f"{x:.3f}" if not np.isnan(x) else "FAILED"
        print(df.to_string(float_format=float_fmt))
        print(f"{'='*70}\n")

    return df


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_results(suite_output: dict, path: str,
                 strip_raw_arrays: bool = True) -> None:
    """Save ``suite_output`` to a JSON file.

    Parameters
    ----------
    suite_output : dict
        Return value of ``run_experiment_suite``.
    path : str
        Output file path.  Parent directories are created if needed.
    strip_raw_arrays : bool
        If ``True`` (default), remove ``rsrp_values_dbm`` and ``sir_values_db``
        from each baseline's ``comparison_stats`` before saving.  The in-memory
        dict is NOT modified — stripping only affects the copy written to disk.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Work on a shallow copy so we don't mutate the caller's in-memory dict
    output_copy = dict(suite_output)
    output_copy["results"] = {}
    for bid, entry in suite_output["results"].items():
        entry_copy = dict(entry)
        if strip_raw_arrays and isinstance(entry.get("comparison_stats"), dict):
            entry_copy["comparison_stats"] = _strip_raw_arrays(entry["comparison_stats"])
        output_copy["results"][bid] = entry_copy

    serializable = _make_serializable(output_copy)
    with p.open("w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Results saved to: {p}")


def load_results(path: str) -> dict:
    """Load a previously saved suite result from JSON.

    Note: raw per-cell arrays (``rsrp_values_dbm`` / ``sir_values_db``) are
    not present if the file was saved with ``strip_raw_arrays=True``.
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Charting utilities
# ---------------------------------------------------------------------------

def get_distribution_data(
    suite_output: dict,
    metric: str = "rsrp_values_dbm",
    tx_name: Optional[str] = None,
    config_type: str = "optimized",
) -> dict:
    """Extract per-cell value arrays for each baseline.

    Parameters
    ----------
    suite_output : dict
        Return value of ``run_experiment_suite`` (must have raw arrays in memory).
    metric : str
        ``"rsrp_values_dbm"`` or ``"sir_values_db"``.
    tx_name : str or None
        If ``None``, concatenate arrays across all TX zones.
    config_type : str
        ``"optimized"`` or ``"initial"``.

    Returns
    -------
    dict[str, np.ndarray]
        ``{baseline_id: values_array}`` — ready for custom plotting.
    """
    tx_names = suite_output["metadata"]["tx_names"]
    target_txs = [tx_name] if tx_name is not None else tx_names

    data = {}
    for baseline_id, entry in suite_output["results"].items():
        cstats = entry.get("comparison_stats")
        if not isinstance(cstats, dict):
            continue
        arrays = []
        for tname in target_txs:
            try:
                arr = np.array(cstats[tname][config_type][metric])
                arrays.append(arr)
            except (KeyError, TypeError):
                pass
        if arrays:
            data[baseline_id] = np.concatenate(arrays)
    return data


def plot_cdf(
    suite_output: dict,
    metric: str = "rsrp_values_dbm",
    tx_name: Optional[str] = None,
    ax=None,
    baselines: Optional[list] = None,
    config_type: str = "optimized",
    xlabel: Optional[str] = None,
) -> tuple:
    """Plot empirical CDFs for each baseline on a shared axes.

    Uses a fixed color palette and line-style cycle that remains legible in
    greyscale print.  Returns ``(fig, ax)`` for further customization and saving.

    Parameters
    ----------
    suite_output : dict
    metric : str
        ``"rsrp_values_dbm"`` (default) or ``"sir_values_db"``.
    tx_name : str or None
        Restrict to one TX zone, or concatenate all if ``None``.
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on; creates a new figure if ``None``.
    baselines : list[str] or None
        Subset of baselines to plot; all present baselines if ``None``.
    config_type : str
        ``"optimized"`` or ``"initial"``.
    xlabel : str or None
        X-axis label.  Auto-generated from ``metric`` if ``None``.

    Returns
    -------
    (fig, ax)
    """
    dist_data = get_distribution_data(suite_output, metric, tx_name, config_type)

    if baselines is not None:
        dist_data = {k: v for k, v in dist_data.items() if k in baselines}

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    for bid, values in dist_data.items():
        sorted_vals = np.sort(values)
        cdf         = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(
            sorted_vals, cdf,
            label=_BASELINE_LABELS.get(bid, bid),
            color=_BASELINE_COLORS.get(bid, None),
            linestyle=_BASELINE_LINESTYLES.get(bid, "-"),
            linewidth=1.5,
        )

    _xlabel_defaults = {
        "rsrp_values_dbm": "RSRP (dBm)",
        "sir_values_db":   "SIR (dB)",
    }
    ax.set_xlabel(xlabel or _xlabel_defaults.get(metric, metric))
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    return fig, ax


def plot_loss_curves(
    suite_output: dict,
    ax=None,
    baselines: Optional[list] = None,
    x_axis: str = "iteration",
    normalize: bool = False,
) -> tuple:
    """Plot optimizer loss history (convergence curves) for gradient baselines.

    Baselines with a trivial loss history (len ≤ 1, e.g. zeroth-order / UMa
    naive) are skipped automatically.

    Parameters
    ----------
    suite_output : dict
    ax : matplotlib.axes.Axes or None
    baselines : list[str] or None
        Subset to plot; all if ``None``.
    x_axis : {"iteration", "wall_time"}
        ``"iteration"`` (default) plots loss vs iteration index.
        ``"wall_time"`` plots loss vs cumulative wall-clock seconds, using
        ``joint.iter_time_history`` from each optimizer result.  Baselines
        without timing data fall back to iteration index.
    normalize : bool
        If ``True``, each curve is min-max normalized to [0, 1] across its own
        history before plotting, so baselines with very different loss scales
        can be compared on the same axes.  The y-axis label updates accordingly.

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    else:
        fig = ax.get_figure()

    target = baselines if baselines is not None else list(suite_output["results"].keys())

    plotted = 0
    for bid in target:
        entry = suite_output["results"].get(bid, {})
        opt_result = entry.get("optimizer_result")
        if not isinstance(opt_result, dict):
            continue
        joint = opt_result.get("joint", {})
        history = joint.get("loss_history", [])
        if len(history) <= 1:
            continue

        ys = np.array(history, dtype=float)
        if normalize:
            lo, hi = ys.min(), ys.max()
            if hi > lo:
                ys = (ys - lo) / (hi - lo)
            else:
                ys = np.zeros_like(ys)

        if x_axis == "wall_time":
            iter_times = joint.get("iter_time_history", [])
            if len(iter_times) == len(history):
                xs = list(np.cumsum(iter_times))
            elif len(iter_times) == len(history) - 1:
                # loss_history has a pre-seeded initial entry (t=0); iter_times
                # only covers subsequent iterations — prepend 0 before cumsum.
                xs = list(np.cumsum([0.0] + list(iter_times)))
            else:
                xs = list(range(1, len(history) + 1))
        else:
            xs = list(range(1, len(history) + 1))

        ax.plot(
            xs, ys,
            label=_BASELINE_LABELS.get(bid, bid),
            color=_BASELINE_COLORS.get(bid, None),
            linestyle=_BASELINE_LINESTYLES.get(bid, "-"),
            linewidth=1.5,
        )
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No convergence data\n(zeroth-order baselines only)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
                color="grey")

    ax.set_xlabel("Wall time (s)" if x_axis == "wall_time" else "Iteration")
    ax.set_ylabel("Normalized SIR Loss" if normalize else "SIR Loss")
    ax.grid(True, alpha=0.3)
    if plotted > 0:
        ax.legend(fontsize=7)

    return fig, ax


def plot_metric_bars(
    suite_output: dict,
    metrics: tuple = ("rsrp_mean_dbm", "sir_median_db"),
    tx_name: Optional[str] = None,
    ax=None,
    config_type: str = "optimized",
) -> tuple:
    """Grouped bar chart comparing scalar summary metrics across baselines.

    One bar group per metric, one bar per baseline.  Suitable for a
    single-column figure in a two-column paper layout.

    Parameters
    ----------
    suite_output : dict
    metrics : tuple[str]
        Which scalar metrics to plot (from ``comparison_stats[tx][config_type]``).
    tx_name : str or None
    ax : matplotlib.axes.Axes or None
    config_type : str

    Returns
    -------
    (fig, ax)
    """
    df = compare_all_results(suite_output, metrics=list(metrics),
                             tx_name=tx_name, config_type=config_type,
                             print_table=False)

    baselines_present = [b for b in df.index if not df.loc[b, metrics[0]] != df.loc[b, metrics[0]]]
    n_metrics   = len(metrics)
    n_baselines = len(df)
    x           = np.arange(n_metrics)
    width       = 0.8 / max(n_baselines, 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, n_metrics * 2), 4))
    else:
        fig = ax.get_figure()

    for i, (bid, row) in enumerate(df.iterrows()):
        offsets = (i - n_baselines / 2 + 0.5) * width
        vals    = [row.get(m, float("nan")) for m in metrics]
        ax.bar(
            x + offsets, vals,
            width=width,
            label=_BASELINE_LABELS.get(bid, bid),
            color=_BASELINE_COLORS.get(bid, None),
            edgecolor="white",
            linewidth=0.5,
        )

    _metric_labels = {
        "rsrp_mean_dbm":   "RSRP mean (dBm)",
        "rsrp_p10_dbm":    "RSRP P10 (dBm)",
        "rsrp_p90_dbm":    "RSRP P90 (dBm)",
        "sir_median_db":   "SIR median (dB)",
        "sir_p10_db":      "SIR P10 (dB)",
        "coverage_fraction": "Coverage",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([_metric_labels.get(m, m) for m in metrics], fontsize=8)
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    return fig, ax


# ---------------------------------------------------------------------------
# Metric labels shared across new plot helpers
# ---------------------------------------------------------------------------

_METRIC_DISPLAY: dict[str, str] = {
    "rsrp_mean_dbm":     "RSRP Mean (dBm)",
    "rsrp_p10_dbm":      "RSRP P10 (dBm)",
    "rsrp_p90_dbm":      "RSRP P90 (dBm)",
    "rsrp_median_dbm":   "RSRP Median (dBm)",
    "sir_mean_db":       "SIR Mean (dB)",
    "sir_median_db":     "SIR Median (dB)",
    "sir_p10_db":        "SIR P10 (dB)",
    "sir_p90_db":        "SIR P90 (dB)",
    "coverage_fraction": "Coverage Fraction",
}

_BASELINE_SHORT: dict[str, str] = {
    "grad_full_rejection":    "Grad-Rejection",
    "grad_full_triangulated": "Grad-Triangulated",
    "grad_proportional":      "Grad-Proportional",
    "radiomap_gradient":      "RadioMap-Grad",
    "random_search":          "Random Search",
    "pso":                    "PSO",
    "coordinate_descent":     "Coord. Descent",
    "uma_naive":              "UMa Naive",
}


def plot_per_metric_panels(
    suite_output: dict,
    metrics: tuple = (
        "rsrp_mean_dbm", "rsrp_p10_dbm",
        "sir_median_db", "sir_p10_db",
    ),
    config_type: str = "optimized",
    show_initial_marker: bool = True,
) -> tuple:
    """Comprehensive per-metric comparison across all TX zones.

    Layout: rows = metrics, cols = individual TX zones + "All TXs" aggregate.
    Each cell is a horizontal bar chart — one bar per baseline — so baselines
    are visually separated from each other and from the metric axis labels.
    The legend is placed outside the grid (right of the figure) rather than
    inside any axes panel, preventing overlap with data.

    Parameters
    ----------
    suite_output : dict
        Return value of ``run_experiment_suite``.
    metrics : tuple[str]
        Metrics to plot (rows).  Defaults to four key RF metrics.
    config_type : str
        ``"optimized"`` (default) or ``"initial"``.
    show_initial_marker : bool
        If ``True``, overlays a thin vertical line for the initial (pre-opt)
        value of each metric so the improvement magnitude is visible.

    Returns
    -------
    (fig, axes)  — axes shape is (n_metrics, n_cols).
    """
    tx_names = suite_output["metadata"]["tx_names"]
    col_names = list(tx_names) + ["All TXs"]
    n_metrics      = len(metrics)
    n_data_cols    = len(col_names)          # TX cols + "All TXs"
    n_display_cols = n_data_cols + 1         # +1 for the Wins tally column

    baselines_present = list(suite_output["results"].keys())
    n_baselines       = len(baselines_present)
    bar_h             = 0.65

    # Wins column is narrower than data columns
    width_ratios = [1.0] * n_data_cols + [0.55]
    fig, axes = plt.subplots(
        n_metrics, n_display_cols,
        figsize=(3.8 * n_data_cols + 2.2, 2.2 * n_metrics),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )

    def _get_scalar(baseline_id, tx, ctype, metric):
        try:
            cstats = suite_output["results"][baseline_id]["comparison_stats"]
            if tx == "All TXs":
                vals = [float(cstats[t][ctype][metric]) for t in tx_names
                        if t in cstats and ctype in cstats[t]
                        and metric in cstats[t][ctype]]
                return float(np.mean(vals)) if vals else float("nan")
            return float(cstats[tx][ctype][metric])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    # wins_per_metric[metric][bid] = # columns where bid was best for that metric
    wins_per_metric = {m: {bid: 0 for bid in baselines_present} for m in metrics}

    for row_i, metric in enumerate(metrics):
        for col_i, tx_label in enumerate(col_names):
            ax = axes[row_i][col_i]

            values   = []
            colors   = []
            labels   = []
            initials = []

            for bid in baselines_present:
                val = _get_scalar(bid, tx_label, config_type, metric)
                values.append(val)
                colors.append(_BASELINE_COLORS.get(bid, "#999999"))
                labels.append(_BASELINE_SHORT.get(bid, bid))

                if show_initial_marker:
                    initials.append(_get_scalar(bid, tx_label, "initial", metric))

            # Identify winner for this panel (highest = best for all dB metrics)
            valid_pairs = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
            best_idx = max(valid_pairs, key=lambda p: p[1])[0] if valid_pairs else None
            if best_idx is not None:
                wins_per_metric[metric][baselines_present[best_idx]] += 1

            y_pos = np.arange(n_baselines)
            bars = ax.barh(
                y_pos, values,
                height=bar_h,
                color=colors,
                edgecolor="white",
                linewidth=0.6,
            )

            # Highlight best bar: gold outline + star annotation
            if best_idx is not None:
                bars[best_idx].set_edgecolor("#FFD700")
                bars[best_idx].set_linewidth(2.2)
                best_val = values[best_idx]
                ax.annotate(
                    "★",
                    xy=(best_val, best_idx),
                    xytext=(4, 0),
                    textcoords="offset points",
                    fontsize=8, color="#B8860B",
                    va="center", ha="left",
                    fontweight="bold",
                    annotation_clip=False,
                )

            if show_initial_marker and any(not np.isnan(v) for v in initials):
                for yi, iv in zip(y_pos, initials):
                    if not np.isnan(iv):
                        ax.vlines(iv, yi - bar_h / 2, yi + bar_h / 2,
                                  colors="#333333", linewidth=1.2,
                                  linestyles="--", alpha=0.6)

            ax.set_yticks(y_pos)
            if col_i == 0:
                ax.set_yticklabels(labels, fontsize=7.5)
            else:
                ax.set_yticklabels([""] * n_baselines)

            ax.invert_yaxis()
            ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)

            if row_i == 0:
                ax.set_title(tx_label, fontsize=9, fontweight="bold", pad=6)

            if col_i == n_data_cols - 1:
                ax.set_xlabel(
                    _METRIC_DISPLAY.get(metric, metric),
                    fontsize=8, labelpad=4,
                )
            else:
                ax.tick_params(axis="x", labelsize=7)

        # ── Wins tally column (rightmost) ────────────────────────────────────
        ax_w = axes[row_i][n_data_cols]
        win_vals   = [wins_per_metric[metric][bid] for bid in baselines_present]
        win_colors = [_BASELINE_COLORS.get(bid, "#999999") for bid in baselines_present]
        y_pos      = np.arange(n_baselines)

        win_bars = ax_w.barh(
            y_pos, win_vals,
            height=bar_h,
            color=win_colors,
            edgecolor="white",
            linewidth=0.6,
        )
        # Gold outline on the overall winner in this metric
        if win_vals:
            top_win_idx = int(np.argmax(win_vals))
            if win_vals[top_win_idx] > 0:
                win_bars[top_win_idx].set_edgecolor("#FFD700")
                win_bars[top_win_idx].set_linewidth(2.2)

        ax_w.set_yticks(y_pos)
        ax_w.set_yticklabels([""] * n_baselines)
        ax_w.set_xlim(0, n_data_cols + 0.5)
        ax_w.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_w.invert_yaxis()
        ax_w.grid(True, axis="x", alpha=0.25, linewidth=0.6)
        ax_w.spines[["top", "right"]].set_visible(False)
        ax_w.tick_params(axis="x", labelsize=7)

        if row_i == 0:
            ax_w.set_title("Wins", fontsize=9, fontweight="bold", pad=6)
        ax_w.set_xlabel(f"/ {n_data_cols}", fontsize=7, labelpad=4)

    # ── shared figure-level legend ───────────────────────────────────────────
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    legend_handles = []
    if show_initial_marker:
        legend_handles.append(
            mlines.Line2D([], [], color="#333333", linestyle="--", linewidth=1.2,
                          label="Initial (pre-opt)")
        )
    legend_handles.append(
        mpatches.Patch(facecolor="none", edgecolor="#FFD700", linewidth=2.0,
                       label="★ Best in column")
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig, axes


def plot_final_configurations(suite_output: dict) -> tuple:
    """Scatter plots of final TX position and boresight for each baseline.

    Layout: 2 rows × n_tx columns.

    - Row 0: Top-down XY position (metres).  Each baseline is a coloured dot;
             the initial config is a gray diamond.
    - Row 1: Boresight angle space — Azimuth (x) vs Elevation (y) in degrees.
             Same marker convention.

    Each dot carries a short inline tag so the reader can identify solvers
    without hunting through the legend.

    Returns
    -------
    (fig, axes) — axes shape is (2, n_tx).
    """
    tx_names          = suite_output["metadata"]["tx_names"]
    baselines_present = list(suite_output["results"].keys())
    n_tx              = len(tx_names)

    # Short tags used as inline dot labels
    _TAGS: dict[str, str] = {
        "grad_full_rejection":    "GRej",
        "grad_full_triangulated": "GTri",
        "grad_proportional":      "GProp",
        "random_search":          "RS",
        "pso":                    "PSO",
        "coordinate_descent":     "CD",
        "uma_naive":              "UMa",
    }

    fig, axes = plt.subplots(
        2, n_tx,
        figsize=(4.2 * n_tx, 7.0),
        squeeze=False,
    )

    def _td(bid, tx):
        return (suite_output["results"]
                .get(bid, {})
                .get("optimizer_result", {})
                .get(tx, {}))

    for col_i, tx in enumerate(tx_names):
        ax_pos = axes[0][col_i]
        ax_ang = axes[1][col_i]

        # Initial config (shared across baselines — read from first entry)
        init   = _td(baselines_present[0], tx)
        i_pos  = init.get("initial_position", [None, None, None])
        i_az, i_el = (init.get("initial_angles") or [None, None])

        # ── per-baseline dots ────────────────────────────────────────────
        for bid in baselines_present:
            td     = _td(bid, tx)
            pos    = td.get("final_position",  [None, None, None])
            angles = td.get("best_angles") or  [None, None]
            az, el = angles[0], angles[1]
            color  = _BASELINE_COLORS.get(bid, "#999999")
            tag    = _TAGS.get(bid, bid[:5])

            if None not in (pos[0], pos[1]):
                ax_pos.scatter(pos[0], pos[1], color=color, s=90, zorder=5,
                               edgecolors="white", linewidths=0.8)
                ax_pos.annotate(tag, xy=(pos[0], pos[1]),
                                xytext=(5, 4), textcoords="offset points",
                                fontsize=6.5, color=color, zorder=6,
                                annotation_clip=False)

            if None not in (az, el):
                ax_ang.scatter(az, el, color=color, s=90, zorder=5,
                               edgecolors="white", linewidths=0.8)
                ax_ang.annotate(tag, xy=(az, el),
                                xytext=(5, 4), textcoords="offset points",
                                fontsize=6.5, color=color, zorder=6,
                                annotation_clip=False)

        # ── initial config marker ────────────────────────────────────────
        if None not in (i_pos[0], i_pos[1]):
            ax_pos.scatter(i_pos[0], i_pos[1], color="#555555", marker="D",
                           s=65, zorder=4, edgecolors="white", linewidths=0.8)
            ax_pos.annotate("Init", xy=(i_pos[0], i_pos[1]),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=6.5, color="#555555", zorder=6,
                            annotation_clip=False)

        if None not in (i_az, i_el):
            ax_ang.scatter(i_az, i_el, color="#555555", marker="D",
                           s=65, zorder=4, edgecolors="white", linewidths=0.8)
            ax_ang.annotate("Init", xy=(i_az, i_el),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=6.5, color="#555555", zorder=6,
                            annotation_clip=False)

        # ── formatting ───────────────────────────────────────────────────
        ax_pos.set_aspect("equal", adjustable="datalim")
        ax_pos.margins(0.28)
        ax_pos.set_title(tx, fontsize=10, fontweight="bold", pad=6)
        ax_pos.set_xlabel("X (m)", fontsize=8)
        ax_pos.set_ylabel("Y (m)" if col_i == 0 else "", fontsize=8)

        ax_ang.margins(0.28)
        ax_ang.set_xlabel("Azimuth (°)", fontsize=8)
        ax_ang.set_ylabel("Elevation (°)" if col_i == 0 else "", fontsize=8)

        for ax in (ax_pos, ax_ang):
            ax.grid(True, alpha=0.2, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7.5)

    # ── figure-level legend ───────────────────────────────────────────────
    handles = [
        plt.scatter([], [], color=_BASELINE_COLORS.get(bid, "#999999"), s=65,
                    edgecolors="white", linewidths=0.8,
                    label=_BASELINE_SHORT.get(bid, bid))
        for bid in baselines_present
    ]
    handles.append(
        plt.scatter([], [], color="#555555", marker="D", s=55,
                    edgecolors="white", linewidths=0.8, label="Initial")
    )
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Final TX Configurations — Position & Boresight",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    return fig, axes


def plot_efficiency_frontier(
    suite_output: dict,
    metrics: tuple = ("sir_median_db", "rsrp_median_dbm"),
    config_type: str = "optimized",
    log_time: bool = True,
) -> tuple:
    """Quality-vs-time scatter plot — the efficiency frontier.

    Plots final optimized metric value (y) against total elapsed wall time (x)
    for every baseline, making it easy to see which methods achieve the best
    result for the time they consume.

    Each point is annotated with a short baseline label.  A horizontal dashed
    line marks the best achieved value so the gap to the frontier is visible.

    Parameters
    ----------
    suite_output : dict
        Return value of ``run_experiment_suite``.
    metrics : tuple[str]
        One subplot per metric.  Defaults to SIR median and RSRP mean.
    config_type : str
        ``"optimized"`` (default) or ``"initial"``.
    log_time : bool
        Use a log scale on the time axis (recommended when methods span orders
        of magnitude in elapsed time, e.g. 4 s UMa vs 600 s coord-descent).

    Returns
    -------
    (fig, axes)  — axes shape is (1, n_metrics).
    """
    tx_names          = suite_output["metadata"]["tx_names"]
    baselines_present = list(suite_output["results"].keys())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(5.5 * n_metrics, 4.5),
        squeeze=False,
    )

    for col_i, metric in enumerate(metrics):
        ax = axes[0][col_i]

        xs, ys, colors, short_labels = [], [], [], []

        for bid in baselines_present:
            entry = suite_output["results"].get(bid, {})
            elapsed = entry.get("elapsed_s", float("nan"))
            if np.isnan(elapsed):
                continue

            # Average metric across all TX zones
            try:
                cstats = entry["comparison_stats"]
                vals   = [float(cstats[t][config_type][metric])
                          for t in tx_names
                          if t in cstats
                          and config_type in cstats[t]
                          and metric in cstats[t][config_type]]
                metric_val = float(np.mean(vals)) if vals else float("nan")
            except (KeyError, TypeError):
                metric_val = float("nan")

            if np.isnan(metric_val):
                continue

            xs.append(elapsed)
            ys.append(metric_val)
            colors.append(_BASELINE_COLORS.get(bid, "#999999"))
            short_labels.append(_BASELINE_SHORT.get(bid, bid))

        xs = np.array(xs)
        ys = np.array(ys)

        ax.scatter(xs, ys, c=colors, s=90, zorder=3,
                   edgecolors="#444444", linewidths=0.7)

        # Annotate each point with short baseline name
        for xi, yi, lbl, c in zip(xs, ys, short_labels, colors):
            ax.annotate(
                lbl, xy=(xi, yi),
                xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5,
                color="#222222",
            )

        # Best-value dashed reference line
        if len(ys):
            best_y = np.nanmax(ys)
            ax.axhline(best_y, color="#888888", linewidth=0.9,
                       linestyle="--", alpha=0.7, zorder=1)
            ax.annotate(
                f"best: {best_y:.2f}",
                xy=(ax.get_xlim()[0] if not log_time else xs.min() * 0.8, best_y),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="#666666", va="bottom",
            )

        if log_time:
            ax.set_xscale("log")

        ax.set_xlabel("Elapsed time (s)", fontsize=9)
        ax.set_ylabel(_METRIC_DISPLAY.get(metric, metric), fontsize=9)
        ax.set_title(
            f"Efficiency Frontier — {_METRIC_DISPLAY.get(metric, metric)}",
            fontsize=10, fontweight="bold",
        )
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_zone_overview(
    building_info: dict,
    map_config: dict,
    zone_entries: list,
    title: str = "",
    figsize: tuple = (9, 9),
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    show_building_ids: bool = True,
    building_label_fontsize: int = 6,
    ax=None,
) -> tuple:
    """Scene map showing zone masks, TX positions, and building footprints.

    Reusable across all experiment notebooks — pass one ``zone_entry`` dict per
    TX/zone pair.  Building IDs are annotated at each building's centroid.

    Parameters
    ----------
    building_info : dict
        Output of ``extract_building_info()``, keyed by integer building ID.
        Each value must have ``'vertices'`` (Nx3 array) and ``'center'``
        (x_center, y_center).
    map_config : dict
        Standard ``MAP_CONFIG`` dict with ``'center'`` and ``'size'`` keys.
    zone_entries : list[dict]
        One dict per TX/zone pair.  Recognised keys:

        * ``mask``        – 2-D numpy array (zone coverage mask)
        * ``cmap``        – matplotlib colormap name (e.g. ``'Blues'``)
        * ``color``       – TX marker and centroid color
        * ``tx_pos``      – [x, y, z] or [x, y] TX position
        * ``tx_name``     – TX name string (legend)
        * ``building_id`` – int building ID the TX sits on (legend)
        * ``centroid_xy`` – [x, y] zone centroid
        * ``zone_label``  – short string used in the centroid legend entry
        * ``marker``      – TX marker shape (default ``'^'``)
        * ``show_arrow``  – draw a dashed arrow TX→centroid (default ``True``)

    title : str
    figsize : tuple
    xlim : tuple or None
        (x_min, x_max) axis limits; derived from ``map_config`` if ``None``.
    ylim : tuple or None
        (y_min, y_max) axis limits; derived from ``map_config`` if ``None``.
    show_building_ids : bool
        Annotate each building footprint with its integer ID (default ``True``).
    building_label_fontsize : int
        Font size for building ID annotations (default ``6``).
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    (fig, ax)
    """
    from matplotlib.patches import Polygon as MplPolygon  # local import — avoids hard dep at module level

    cx, cy, _ = map_config["center"]
    w, h = map_config["size"]
    extent = [cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # ── zone masks ────────────────────────────────────────────────────────────
    for entry in zone_entries:
        mask = entry["mask"]
        ax.imshow(
            np.ma.masked_where(mask == 0, mask),
            origin="lower", extent=extent,
            cmap=entry.get("cmap", "Blues"),
            vmin=0, vmax=1, alpha=0.45,
        )

    # ── building footprints + optional ID labels ──────────────────────────────
    _xlim = xlim if xlim is not None else (extent[0], extent[1])
    _ylim = ylim if ylim is not None else (extent[2], extent[3])

    for bid, bdata in building_info.items():
        verts = bdata["vertices"][:, :2]
        ax.add_patch(MplPolygon(
            verts, closed=True,
            facecolor="gray", edgecolor="black", linewidth=0.8, alpha=0.4,
        ))
        if show_building_ids:
            bx, by = bdata["center"]
            if _xlim[0] <= bx <= _xlim[1] and _ylim[0] <= by <= _ylim[1]:
                ax.text(
                    bx, by, str(bid),
                    ha="center", va="center",
                    fontsize=building_label_fontsize,
                    color="black", alpha=0.7,
                    clip_on=True,
                )

    # ── TX markers and zone centroids ─────────────────────────────────────────
    for entry in zone_entries:
        color = entry.get("color", "steelblue")
        marker = entry.get("marker", "^")
        tx_pos = entry["tx_pos"]
        tx_name = entry.get("tx_name", "TX")
        bid = entry.get("building_id", "?")
        centroid = entry.get("centroid_xy")
        zone_label = entry.get("zone_label", "Zone centroid")

        ax.plot(
            tx_pos[0], tx_pos[1], marker,
            color=color, markersize=13, markeredgecolor="black",
            label=f"{tx_name} (bldg {bid})", zorder=5,
        )

        if centroid is not None:
            ax.plot(
                *centroid, "o",
                color=color, markersize=8, markeredgecolor="k",
                label=zone_label, zorder=5,
            )
            if entry.get("show_arrow", True):
                ax.annotate(
                    "", xy=centroid, xytext=tx_pos[:2],
                    arrowprops=dict(
                        arrowstyle="->", color=color, lw=1.5, linestyle="dashed",
                    ),
                )

    # ── axes cosmetics ────────────────────────────────────────────────────────
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# Service Zone Boundary Visualisation
# ---------------------------------------------------------------------------

_BOUNDARY_TX_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
                        "#9B5DE5", "#F15BB5", "#00BBF9"]


def plot_service_boundaries(
    stats: dict,
    map_config: dict,
    tx_names: list,
    figsize: tuple = (14, 18),
    sinr_vmin: float = -20.0,
    sinr_vmax: float = 20.0,
    title: str | None = None,
) -> tuple:
    """Visualise service-zone boundary KPI: dominant TX map, SINR heatmap,
    and before/after comparison statistics.

    Parameters
    ----------
    stats : dict
        Return value of ``compare_multi_tx_performance()``.  Must contain
        ``stats["boundary"]["initial"]`` and ``stats["boundary"]["optimized"]``
        with array fields (``dominant_tx_map``, ``boundary_mask``,
        ``sinr_db_grid``, ``_boundary_sinr_values``) still populated
        (i.e. called *before* ``_strip_raw_arrays``).
    map_config : dict
        Same map grid config passed to the optimiser: keys ``center``,
        ``size``, ``cell_size``.
    tx_names : list[str]
        TX identifiers in the same order used by the optimiser.
    figsize : tuple
        Overall figure size ``(width, height)`` in inches.
    sinr_vmin, sinr_vmax : float
        Colormap range for the SINR heatmap rows (dB).
    title : str or None
        Optional super-title.

    Returns
    -------
    fig, axes : matplotlib Figure and 2-D axes array (3 rows × 2 columns)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    from scipy.ndimage import binary_dilation

    cx, cy = map_config["center"][0], map_config["center"][1]
    sw, sh = map_config["size"][0], map_config["size"][1]
    extent = [cx - sw / 2, cx + sw / 2, cy - sh / 2, cy + sh / 2]

    N = len(tx_names)
    tx_colors = _BOUNDARY_TX_COLORS[:N]

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    config_types = ("initial", "optimized")

    for col, ct in enumerate(config_types):
        bnd = stats["boundary"][ct]
        dom_map   = np.asarray(bnd["dominant_tx_map"])
        bnd_mask  = np.asarray(bnd["boundary_mask"]).astype(bool)
        sinr_grid = np.asarray(bnd["sinr_db_grid"])
        sinr_vals = np.asarray(bnd["_boundary_sinr_values"])
        r         = bnd["boundary_radius_cells"]

        # ----------------------------------------------------------------
        # Row 0: dominant TX map + boundary contour
        # ----------------------------------------------------------------
        ax0 = axes[0, col]
        cmap_vals = ["#CCCCCC"] + tx_colors          # index 0 = no-signal gray
        cmap_dom  = ListedColormap(cmap_vals)
        dom_display = (dom_map + 1).astype(float)    # shift -1→0, 0..N-1→1..N
        ax0.imshow(
            dom_display,
            origin="lower",
            extent=extent,
            cmap=cmap_dom,
            vmin=-0.5,
            vmax=N + 0.5,
            interpolation="nearest",
            aspect="auto",
        )
        if bnd_mask.any():
            ax0.contour(
                bnd_mask.astype(float),
                levels=[0.5],
                colors=["black"],
                linewidths=[0.6],
                origin="lower",
                extent=extent,
            )
        handles = [mpatches.Patch(color=tx_colors[i], label=tx_names[i]) for i in range(N)]
        handles.append(mpatches.Patch(color="#CCCCCC", label="No signal"))
        ax0.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
        mode_label = "Single-TX mode: coverage-edge boundaries" if bnd["single_tx_mode"] else ""
        ax0.set_title(
            f"{ct.capitalize()} — Dominant TX Map"
            + (f"\n({mode_label})" if mode_label else ""),
            fontsize=9,
        )
        ax0.set_xlabel("X (m)", fontsize=8)
        ax0.set_ylabel("Y (m)", fontsize=8)

        # ----------------------------------------------------------------
        # Row 1: SINR heatmap masked to boundary neighbourhood
        # ----------------------------------------------------------------
        ax1 = axes[1, col]
        struct = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
        bnd_region = binary_dilation(bnd_mask, structure=struct)
        sinr_display = np.where(
            bnd_region & (dom_map >= 0), sinr_grid, np.nan
        )
        im = ax1.imshow(
            sinr_display,
            origin="lower",
            extent=extent,
            cmap="RdYlGn",
            vmin=sinr_vmin,
            vmax=sinr_vmax,
            interpolation="nearest",
            aspect="auto",
        )
        if bnd_mask.any():
            ax1.contour(
                bnd_mask.astype(float),
                levels=[0.5],
                colors=["black"],
                linewidths=[0.6],
                origin="lower",
                extent=extent,
            )
        plt.colorbar(im, ax=ax1, shrink=0.7, label="SINR (dB)")
        ax1.set_title(
            f"{ct.capitalize()} — SINR at Boundary Region (±{r} m)", fontsize=9
        )
        ax1.set_xlabel("X (m)", fontsize=8)
        ax1.set_ylabel("Y (m)", fontsize=8)

    # ----------------------------------------------------------------
    # Row 2 left: CDF of boundary SINR — initial vs optimised overlaid
    # ----------------------------------------------------------------
    ax2l = axes[2, 0]
    colors_ct  = {"initial": "darkorange", "optimized": "steelblue"}
    labels_ct  = {"initial": "Initial",    "optimized": "Optimised"}
    for ct in config_types:
        raw = np.asarray(stats["boundary"][ct]["_boundary_sinr_values"])
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            continue
        xs  = np.sort(finite)
        cdf = np.arange(1, len(xs) + 1) / len(xs)
        ax2l.plot(xs, cdf, color=colors_ct[ct], label=labels_ct[ct], linewidth=1.8)
    ax2l.axvline(0.0, color="#888888", linewidth=0.8, linestyle="--", label="0 dB")
    delta_p10 = stats["boundary"]["improvement"]["boundary_sinr_p10_db"]
    ax2l.set_title(f"Boundary SINR CDF  (ΔP10 = {delta_p10:+.1f} dB)", fontsize=9)
    ax2l.set_xlabel("Boundary SINR (dB)", fontsize=8)
    ax2l.set_ylabel("CDF", fontsize=8)
    ax2l.legend(fontsize=8)
    ax2l.grid(True, alpha=0.3)

    # ----------------------------------------------------------------
    # Row 2 right: improvement bar chart
    # ----------------------------------------------------------------
    ax2r = axes[2, 1]
    imp = stats["boundary"]["improvement"]
    metric_pairs = [
        ("boundary_sinr_mean_db",   "Mean SINR"),
        ("boundary_sinr_median_db", "Median SINR"),
        ("boundary_sinr_p10_db",    "P10 SINR"),
    ]
    labels  = [lbl for _, lbl in metric_pairs]
    deltas  = [imp[key] for key, _ in metric_pairs]
    bar_colors = ["#2CA02C" if d >= 0 else "#D62728" for d in deltas]
    bars = ax2r.barh(
        labels, deltas, color=bar_colors, edgecolor="black", linewidth=0.5
    )
    ax2r.axvline(0.0, color="black", linewidth=0.8)
    ax2r.set_xlabel("Improvement (dB)", fontsize=8)
    ax2r.set_title("Boundary SINR Improvement\n(Optimised − Initial)", fontsize=9)
    for bar, val in zip(bars, deltas):
        ha  = "left" if val >= 0 else "right"
        xoff = 0.05 if val >= 0 else -0.05
        ax2r.text(
            val + xoff,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f} dB",
            va="center",
            ha=ha,
            fontsize=8,
        )
    ax2r.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.show()
    return fig, axes

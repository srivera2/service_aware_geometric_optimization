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
plot_cdf()                : empirical CDF comparison across baselines
plot_loss_curves()        : convergence curves for gradient methods
plot_metric_bars()        : grouped bar chart of summary metrics
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
    coarse_az_sweep,
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
    learning_rate: float = 3.0
    num_iterations: int = 50
    dead_tail_percentile: float = 1.0
    max_dbscan_points: int = 100_000
    grad_lds: str = "Halton"
    debug_viz: bool = False

    # --- Random search ----------------------------------------------------
    rs_n_candidates: int = 200
    rs_lds: str = "Halton"
    rs_seed: Optional[int] = None

    # --- PSO --------------------------------------------------------------
    pso_n_particles: int = 20
    pso_num_iterations: int = 30
    pso_w: float = 0.7
    pso_c1: float = 1.5
    pso_c2: float = 1.5
    pso_lds: str = "Halton"
    pso_seed: Optional[int] = None

    # --- Coordinate descent -----------------------------------------------
    cd_num_cycles: int = 10
    cd_n_line_points: int = 20
    cd_lds: str = "Halton"

    # --- UMa naive --------------------------------------------------------
    uma_mechanical_downtilt_deg: float = 0.0
    uma_electrical_downtilt_deg: float = 6.0
    uma_lds: str = "Halton"

    # --- RadioMap gradient -----------------------------------------------
    rmg_samples_per_tx: int = int(1e7)
    rmg_max_depth: int = 8
    rmg_lds: str = "Halton"

    # --- Early stopping ---------------------------------------------------
    early_stop_window: int = 10
    early_stop_min_improvement: float = 1e-3
    # Minimum sign-flips in Δloss over the window to declare oscillation
    # (used by gradient and radiomap_gradient methods only).
    early_stop_min_flips: int = 4

    # --- Coarse azimuth warm-start ----------------------------------------
    # When coarse_az_steps > 0, a greedy azimuth sweep is run once before any
    # baseline.  Each TX's azimuth is swept over ``coarse_az_steps`` uniformly
    # spaced points using cheap RadioMapSolver evaluations; the best per-TX
    # azimuth becomes the shared starting point for ALL methods.
    # Gradient-free methods (PSO, random search) have the warm-start seeded
    # into their initial populations; all others start from it directly.
    # Set to 0 to disable (default — preserves old behaviour).
    # Recommended: 24 (15° spacing, full 360° coverage).
    coarse_az_steps: int = 0
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
            early_stop_window=c.early_stop_window,
            early_stop_min_improvement=c.early_stop_min_improvement,
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

# Fixed color palette (color-blind friendly, print-safe)
_BASELINE_COLORS: dict[str, str] = {
    "grad_full_rejection":    "#1f77b4",
    "grad_full_triangulated": "#ff7f0e",
    "grad_proportional":      "#2ca02c",
    "radiomap_gradient":      "#e377c2",
    "random_search":          "#d62728",
    "pso":                    "#9467bd",
    "coordinate_descent":     "#8c564b",
    "uma_naive":              "#7f7f7f",
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


_RAW_ARRAY_KEYS = {"rsrp_values_dbm", "sir_values_db"}


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

    # --- Optional coarse azimuth warm-start (runs before capturing angles) --
    sweep_log = None
    if exp_config.coarse_az_steps > 0:
        sweep_result = coarse_az_sweep(
            scene, tx_configs, map_config, scene_xml_path,
            n_az_steps=exp_config.coarse_az_steps,
            samples_per_tx=exp_config.coarse_samples_per_tx,
            noise_power=exp_config.noise_power,
            verbose=exp_config.verbose,
        )
        # Apply warm-start orientations to scene so _capture_initial_angles
        # picks them up as the shared starting point for all baselines.
        for cfg in tx_configs:
            best_az, best_el = sweep_result[cfg.name]
            yaw_r, pitch_r = azimuth_elevation_to_yaw_pitch(best_az, best_el)
            tx = scene.get(cfg.name)
            tx.orientation = mi.Point3f(float(yaw_r), float(pitch_r), 0.0)
        sweep_log = sweep_result

    # --- Capture initial scene state (once) --------------------------------
    initial_angles    = _capture_initial_angles(scene, tx_configs)
    initial_positions = _capture_initial_positions(scene, tx_configs)

    # Flat seed_params vector for gradient-free methods (PSO, random search)
    seed_params: Optional[np.ndarray] = None
    if sweep_log is not None:
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
        warm_label = (f"  [warm-start from coarse sweep, "
                      f"{exp_config.coarse_az_steps} steps]"
                      if sweep_log is not None else "")
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
        "sweep_log":      sweep_log,
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
    if sweep_log is not None:
        suite_output["sweep_log"] = sweep_log

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
        history = opt_result.get("joint", {}).get("loss_history", [])
        if len(history) <= 1:
            continue
        ax.plot(
            range(1, len(history) + 1), history,
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

    ax.set_xlabel("Iteration")
    ax.set_ylabel("SIR Loss")
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

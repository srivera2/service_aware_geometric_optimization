"""
baseline_optimizers.py
======================
Zeroth-order baseline optimisers for multi-TX boresight optimisation.

All three baselines share the same public interface as ``optimize_multi_tx``
and return a result dict compatible with ``compare_multi_tx_performance``.

Public API
----------
random_search_multi_tx        : brute-force random search over angle/position space
pso_multi_tx                  : particle swarm optimisation
coordinate_descent_multi_tx   : axis-aligned coordinate descent with 1-D line search

Evaluation
----------
All baselines use ``RadioMapSolver`` (no PathSolver / AD needed).  The
objective mirrors the gradient-based method: area-weighted negative mean
sigmoid soft-coverage, where soft coverage is a sigmoid indicator of each
zone cell's SIR (dB) exceeding ``sir_threshold_db`` (default −3 dB), with
steepness ``sigmoid_k`` (default 0.5) — identical hyperparameters to
``_sir_loss_body`` in ``multi_tx_optimizer``.  A shared ``_evaluate_config``
helper applies a parameter vector to the scene and returns the scalar loss.

Parameter vector layout (per TX, concatenated)
-----------------------------------------------
  [az_0, el_0, x_0, y_0,  az_1, el_1, x_1, y_1, ...]

  az   : azimuth   [0, 360)  degrees
  el   : elevation [0,  90]  degrees  (downward-facing boresight)
  x, y : TX position, constrained to building rooftop polygon
"""

from __future__ import annotations

import gc
import time
import warnings
from typing import Optional

import drjit as dr
from drjit.auto import Float, TensorXf
import mitsuba as mi
import numpy as np
import torch
from sionna.rt import RadioMapSolver

from angle_utils import (
    azimuth_elevation_to_yaw_pitch,
    yaw_pitch_to_azimuth_elevation,
    compute_initial_angles_from_position,
)
from multi_tx_optimizer import TxConfig, _setup_tx_state, _make_qrand
from tx_placement import TxPlacement


# ---------------------------------------------------------------------------
# Angle / parameter bounds
# ---------------------------------------------------------------------------

AZ_MIN, AZ_MAX =   0.0, 360.0
EL_MIN, EL_MAX = -90.0,   0.0   # elevation ≤ 0 for downward-pointing (rooftop antennas)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _param_bounds(tx_configs: list[TxConfig], tx_states: list[dict]):
    """Return (lo, hi) arrays of shape (n_params,) for uniform sampling / clipping."""
    lo, hi = [], []
    for _, state in zip(tx_configs, tx_states):
        lo += [AZ_MIN, EL_MIN]
        hi += [AZ_MAX, EL_MAX]
        # Position bounds from bounding box of building polygon
        verts = state["tx_placement"].building["vertices"][:, :2]
        lo += [float(verts[:, 0].min()), float(verts[:, 1].min())]
        hi += [float(verts[:, 0].max()), float(verts[:, 1].max())]
    return np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64)


def _apply_params(scene, tx_configs: list[TxConfig], tx_states: list[dict],
                  params: np.ndarray):
    """Write a flat parameter vector into the Sionna scene (no grad tracking)."""
    stride = 4  # az, el, x, y per TX
    for i, (cfg, state) in enumerate(zip(tx_configs, tx_states)):
        b = i * stride
        az, el, x, y = params[b], params[b + 1], params[b + 2], params[b + 3]

        # Project position to valid building polygon location
        x, y = state["tx_placement"].project_to_polygon(x, y)

        yaw_r, pitch_r = azimuth_elevation_to_yaw_pitch(float(az), float(el))
        tx = scene.get(cfg.name)
        tx.orientation = mi.Point3f(float(yaw_r), float(pitch_r), 0.0)
        tx.position    = mi.Point3f(float(x), float(y), float(state["tx_height"]))


def _run_radiomap(scene, map_config: dict,
                  samples_per_tx: int = int(1e8)) -> object:
    """Run RadioMapSolver and return the radio map."""
    solver = RadioMapSolver()
    rm = solver(
        scene,
        max_depth=8,
        samples_per_tx=samples_per_tx,
        cell_size=list(map_config["cell_size"]),
        center=map_config["center"],
        orientation=[0, 0, 0],
        size=map_config["size"],
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        diffraction=True,
        refraction=False,
        stop_threshold=None,
    )
    del solver
    return rm


def _sir_loss_from_radiomap(rm, tx_configs: list[TxConfig], tx_states: list[dict],
                             zone_masks: list[np.ndarray],
                             noise_power: float,
                             sir_threshold_db: float = -3.0,
                             sigmoid_k: float = 0.5,
                             epsilon: float = 1.0) -> float:
    """Compute the scalar SIR loss from a RadioMapSolver result.

    Mirrors the gradient-based loss in ``_sir_loss_body`` (multi_tx_optimizer):
    area-weighted negative mean sigmoid soft-coverage over each TX zone, where
    soft coverage is a sigmoid indicator of SIR (dB) exceeding
    ``sir_threshold_db``.  ``epsilon`` is added to the linear SIR metric before
    the dB conversion (matching the ``epsilon=1`` default in the gradient path).
    """
    rss_np = np.array(dr.detach(rm.rss))   # shape (N_tx, H, W)
    N = len(tx_configs)

    # Area weights proportional to zone geographic area (matches gradient optimizer)
    zone_areas   = [state["zone_polygon"].area for state in tx_states]
    total_area   = sum(zone_areas) or 1.0
    area_weights = [a / total_area for a in zone_areas]

    log10_scale = 10.0 / np.log(10.0)   # converts natural log → dB
    total_loss  = 0.0

    for i in range(N):
        mask = zone_masks[i]
        sig  = rss_np[i][mask > 0]

        interf = np.zeros_like(sig)
        for j in range(N):
            if j != i:
                interf += rss_np[j][mask > 0]

        metric   = sig / (interf + noise_power)
        sir_db   = log10_scale * np.log(metric + epsilon)
        soft_cov = 1.0 / (1.0 + np.exp(-sigmoid_k * (sir_db - sir_threshold_db)))
        loss_i   = -area_weights[i] * float(np.mean(soft_cov))
        total_loss += loss_i

    return total_loss


def _evaluate_config(scene, tx_configs: list[TxConfig], tx_states: list[dict],
                     params: np.ndarray, map_config: dict,
                     zone_masks: list[np.ndarray], noise_power: float,
                     samples_per_tx: int = int(1e8),
                     sir_threshold_db: float = -3.0,
                     sigmoid_k: float = 0.5) -> float:
    """Apply params to scene, run RadioMapSolver, return scalar SIR loss."""
    _apply_params(scene, tx_configs, tx_states, params)
    rm = _run_radiomap(scene, map_config, samples_per_tx=samples_per_tx)
    loss = _sir_loss_from_radiomap(rm, tx_configs, tx_states, zone_masks, noise_power,
                                   sir_threshold_db=sir_threshold_db,
                                   sigmoid_k=sigmoid_k)
    del rm
    gc.collect()
    dr.flush_kernel_cache()
    dr.flush_malloc_cache()
    return loss


def _zone_masks_from_states(tx_states: list[dict], map_config: dict) -> list[np.ndarray]:
    """Build 2-D binary zone masks (H x W) for each TX from their zone polygons."""
    width_m, height_m = map_config["size"]
    cell_w,  cell_h   = map_config["cell_size"]
    cx, cy, _         = map_config["center"]

    n_x = int(width_m / cell_w)
    n_y = int(height_m / cell_h)

    x = np.linspace(cx - width_m / 2, cx + width_m / 2, n_x, endpoint=False) + cell_w / 2
    y = np.linspace(cy - height_m / 2, cy + height_m / 2, n_y, endpoint=False) + cell_h / 2
    X, Y = np.meshgrid(x, y)
    pts   = np.stack([X.ravel(), Y.ravel()], axis=1)

    masks = []
    for state in tx_states:
        zone_poly = state["zone_polygon"]
        from shapely.vectorized import contains
        inside = contains(zone_poly, pts[:, 0], pts[:, 1])
        masks.append(inside.reshape(n_y, n_x).astype(np.float32))
    return masks


def _initial_params(tx_configs: list[TxConfig], tx_states: list[dict]) -> np.ndarray:
    """Pack initial per-TX angles and positions into a flat numpy array."""
    params = []
    for state in tx_states:
        params += [
            state["initial_azimuth"],
            state["initial_elevation"],
            state["tx_position"][0],
            state["tx_position"][1],
        ]
    return np.array(params, dtype=np.float64)


def _build_result(tx_configs, tx_states, best_params, loss_history,
                  initial_params, elapsed, method_name,
                  iter_time_history=None, converged_iter=None) -> dict:
    """Assemble the result dict in the same format as ``optimize_multi_tx``."""
    stride = 4
    result = {}
    for i, (cfg, state) in enumerate(zip(tx_configs, tx_states)):
        b  = i * stride
        result[cfg.name] = {
            "best_angles":      [float(best_params[b]), float(best_params[b + 1])],
            "final_position":   [float(best_params[b + 2]), float(best_params[b + 3]),
                                 float(state["tx_height"])],
            "initial_angles":   [state["initial_azimuth"], state["initial_elevation"]],
            "initial_position": state["tx_position"],
            "az_history":       [],
            "el_history":       [],
        }

    result["joint"] = {
        "loss_history":        loss_history,
        "iter_time_history":   iter_time_history if iter_time_history is not None else [],
        "converged_iter":      converged_iter,
        "elapsed_time_s":      elapsed,
        "num_iterations":      len(loss_history),
        "noise_power":         None,     # filled in by caller
        "sampler":             method_name,
        "sampling_strata":     "n/a",
        "lds":                 "n/a",
        "discrepancy_history": [],
        "mean_discrepancy":    None,
        "median_discrepancy":  None,
    }
    return result


# ---------------------------------------------------------------------------
# Warm-start initialisation
# ---------------------------------------------------------------------------

def coarse_xy_sweep(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    n_xy_steps: int = 5,
    samples_per_tx: int = int(1e7),
    noise_power: float = 1e-10,
    sir_threshold_db: float = -3.0,
    sigmoid_k: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Coarse XY position grid search used to warm-start TX placement.

    Sweeps each TX's position over a coarse grid of interior points on its
    building rooftop polygon.  At each candidate (x, y) the boresight is
    aimed geometrically toward the zone centroid (elevation computed from the
    TX height to a 1.5 m receiver target), so the orientation always faces the
    coverage zone regardless of any prior azimuth warm-start.

    **Algorithm — greedy sequential sweep:**

    For TX_i in order:
      - Previous TXs are held at their already-chosen best (x, y, az, el).
      - Remaining TXs stay at their current scene position.
      - XY is swept over a ``n_xy_steps × n_xy_steps`` grid clipped to the
        building polygon interior (points outside are skipped).
      - At each candidate position the az/el pointing to the zone centroid is
        computed and included in the evaluation.
      - The (x, y) — and its corresponding az/el — with the lowest loss is
        committed for TX_i.

    Total cost: up to ``n_xy_steps² × N_tx`` RadioMapSolver evaluations
    (typically fewer because grid points outside the polygon are filtered out).

    Parameters
    ----------
    scene : sionna.rt.Scene
    tx_configs : list[TxConfig]
    map_config : dict
    scene_xml_path : str
    n_xy_steps : int
        Grid divisions along each axis (default 5 → up to 25 candidates per TX,
        filtered to polygon interior).
    samples_per_tx : int
        RadioMapSolver rays per TX per evaluation (default int(1e7)).
    noise_power : float
    verbose : bool

    Returns
    -------
    dict
        ``{tx_name: (best_x, best_y)}`` — one entry per TX.
        Also contains a ``"meta"`` key:
        ``{"xy_steps": N, "evaluations": K, "elapsed_s": T,
           "per_tx": {tx_name: {"best_x": float, "best_y": float,
                                "best_az": float, "best_el": float,
                                "best_loss": float, "n_candidates": int}}}``.
    """
    from shapely.geometry import Point, Polygon as ShapelyPolygon

    if verbose:
        print(f"\n{'='*60}")
        print(f"COARSE XY SWEEP  ({n_xy_steps}×{n_xy_steps} grid × {len(tx_configs)} TX, "
              f"{samples_per_tx:.0e} samples/TX)")
        print(f"{'='*60}")

    t_start   = time.time()
    qrand     = _make_qrand("Halton")
    tx_states = [_setup_tx_state(scene, cfg, scene_xml_path, qrand)
                 for cfg in tx_configs]
    zone_masks = _zone_masks_from_states(tx_states, map_config)

    # Initialise current_params from the scene so previously committed TX
    # positions/angles are held fixed while sweeping TX_i.
    stride         = 4
    current_params = np.empty(len(tx_configs) * stride, dtype=np.float64)
    for i, cfg in enumerate(tx_configs):
        tx  = scene.get(cfg.name)
        b   = i * stride
        yaw_r   = float(dr.detach(tx.orientation[0])[0])
        pitch_r = float(dr.detach(tx.orientation[1])[0])
        az, el  = yaw_pitch_to_azimuth_elevation(yaw_r, pitch_r)
        current_params[b]     = az
        current_params[b + 1] = el
        current_params[b + 2] = float(dr.detach(tx.position[0])[0])
        current_params[b + 3] = float(dr.detach(tx.position[1])[0])

    per_tx_log = {}
    n_evals    = 0

    for i, (cfg, state) in enumerate(zip(tx_configs, tx_states)):
        b     = i * stride
        x_idx = b + 2
        y_idx = b + 3

        # Zone centroid at receiver height — used to re-aim at each candidate.
        zone_c    = state["box_polygon"].centroid
        look_at   = [float(zone_c.x), float(zone_c.y), 1.5]
        tx_height = state["tx_height"]

        # Build candidate set: regular grid over bounding box, interior only.
        verts_2d = state["tx_placement"].building["vertices"][:, :2]
        poly     = ShapelyPolygon(verts_2d)
        minx, miny, maxx, maxy = poly.bounds
        xs = np.linspace(minx, maxx, n_xy_steps)
        ys = np.linspace(miny, maxy, n_xy_steps)
        candidates = [(float(x), float(y))
                      for x in xs for y in ys
                      if poly.contains(Point(x, y))]
        if not candidates:
            # Polygon smaller than grid spacing — use centroid only.
            c = poly.centroid
            candidates = [(float(c.x), float(c.y))]

        best_x    = current_params[x_idx]
        best_y    = current_params[y_idx]
        best_az   = current_params[b]
        best_el   = current_params[b + 1]
        best_loss = np.inf

        for cx, cy in candidates:
            # Re-aim toward zone centroid from this candidate position.
            az_cand, el_cand = compute_initial_angles_from_position(
                [cx, cy, tx_height], look_at, verbose=False
            )
            probe        = current_params.copy()
            probe[b]     = az_cand
            probe[b + 1] = el_cand
            probe[x_idx] = cx
            probe[y_idx] = cy
            loss = _evaluate_config(scene, tx_configs, tx_states, probe,
                                    map_config, zone_masks, noise_power,
                                    samples_per_tx=samples_per_tx,
                                    sir_threshold_db=sir_threshold_db,
                                    sigmoid_k=sigmoid_k)
            n_evals += 1
            if loss < best_loss:
                best_loss = loss
                best_x, best_y = cx, cy
                best_az, best_el = az_cand, el_cand

        # Commit best position + corresponding orientation for TX_i.
        current_params[b]     = best_az
        current_params[b + 1] = best_el
        current_params[x_idx] = best_x
        current_params[y_idx] = best_y
        per_tx_log[cfg.name]  = {
            "best_x":       float(best_x),
            "best_y":       float(best_y),
            "best_az":      float(best_az),
            "best_el":      float(best_el),
            "best_loss":    float(best_loss),
            "n_candidates": len(candidates),
        }

        if verbose:
            print(f"  {cfg.name}: best x={best_x:.1f}  y={best_y:.1f}  "
                  f"az={best_az:.1f}°  el={best_el:.1f}°  "
                  f"loss={best_loss:.4f}  ({len(candidates)} candidates)")

    # Restore scene to the best joint configuration.
    _apply_params(scene, tx_configs, tx_states, current_params)

    elapsed = time.time() - t_start
    if verbose:
        print(f"\nXY sweep complete: {n_evals} evals in {elapsed:.1f}s")
        print(f"{'='*60}\n")

    result = {cfg.name: (per_tx_log[cfg.name]["best_x"],
                          per_tx_log[cfg.name]["best_y"])
              for cfg in tx_configs}
    result["meta"] = {
        "xy_steps":    n_xy_steps,
        "evaluations": n_evals,
        "elapsed_s":   elapsed,
        "per_tx":      per_tx_log,
    }
    return result


# ---------------------------------------------------------------------------
# 1. Random Search
# ---------------------------------------------------------------------------

def random_search_multi_tx(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    n_candidates: int = 200,
    noise_power: float = 1e-10,
    lds: str = "Sobol",
    seed: Optional[int] = None,
    verbose: bool = True,
    seed_params: Optional[np.ndarray] = None,
    sir_threshold_db: float = -3.0,
    sigmoid_k: float = 0.5,
) -> dict:
    """Brute-force random search over the joint angle/position space.

    Samples ``n_candidates`` configurations independently at random (or via a
    low-discrepancy sequence), evaluates each with RadioMapSolver, and returns
    the best.

    If ``seed_params`` is provided it is inserted as the first candidate so the
    warm-start configuration is always evaluated.  Total evaluations remain
    ``n_candidates`` (the last random candidate is dropped to preserve budget).

    Parameters
    ----------
    scene : sionna.rt.Scene
    tx_configs : list[TxConfig]
    map_config : dict
    scene_xml_path : str
    n_candidates : int
        Total number of random configurations to evaluate.
    noise_power : float
        Thermal noise floor (Watts).
    lds : str
        "Halton" | "Sobol" | "Uniform"  — sequence used to draw candidates.
        "Uniform" → pure Monte Carlo.
    seed : int or None
        RNG seed for reproducibility (used for Uniform only).
    verbose : bool

    Returns
    -------
    dict  (same schema as ``optimize_multi_tx``)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"RANDOM SEARCH BASELINE  ({n_candidates} candidates, {len(tx_configs)} TX)")
        print(f"{'='*60}")

    start_time = time.time()
    rng = np.random.default_rng(seed)

    qrand = _make_qrand(lds)
    tx_states = [_setup_tx_state(scene, cfg, scene_xml_path, qrand)
                 for cfg in tx_configs]

    zone_masks = _zone_masks_from_states(tx_states, map_config)
    lo, hi     = _param_bounds(tx_configs, tx_states)
    n_params   = len(lo)

    # Low-discrepancy sampling of the unit hypercube, then scale
    if lds != "Uniform":
        import scipy.stats.qmc
        if lds == "Halton":
            sampler = scipy.stats.qmc.Halton(d=n_params, scramble=True, seed=seed)
        elif lds == "Sobol":
            sampler = scipy.stats.qmc.Sobol(d=n_params, scramble=True, seed=seed)
        else:
            warnings.warn(f"Unknown LDS '{lds}' for random search. Falling back to Halton.")
            sampler = scipy.stats.qmc.Halton(d=n_params, scramble=True, seed=seed)
        unit_samples = sampler.random(n_candidates)
        candidates   = scipy.stats.qmc.scale(unit_samples, lo, hi)
    else:
        candidates = rng.uniform(lo, hi, size=(n_candidates, n_params))

    # Prepend warm-start point so it is always evaluated; drop the last random
    # candidate to keep total evaluations = n_candidates.
    if seed_params is not None:
        candidates = np.vstack([
            np.clip(seed_params, lo, hi).reshape(1, -1),
            candidates[:-1],
        ])

    best_loss        = np.inf
    best_params      = _initial_params(tx_configs, tx_states)
    loss_history     = []
    iter_time_history = []

    for k, params in enumerate(candidates):
        t_k  = time.time()
        loss = _evaluate_config(scene, tx_configs, tx_states, params,
                                map_config, zone_masks, noise_power,
                                sir_threshold_db=sir_threshold_db,
                                sigmoid_k=sigmoid_k)
        iter_time_history.append(time.time() - t_k)
        loss_history.append(float(loss))

        if loss < best_loss:
            best_loss   = loss
            best_params = params.copy()

        if verbose and (k % max(1, n_candidates // 10) == 0 or k == n_candidates - 1):
            print(f"  [{k+1:4d}/{n_candidates}]  loss={loss:.4f}  best={best_loss:.4f}")

    # Apply best config to scene
    _apply_params(scene, tx_configs, tx_states, best_params)

    elapsed = time.time() - start_time
    result  = _build_result(tx_configs, tx_states, best_params, loss_history,
                            best_params, elapsed, "random_search",
                            iter_time_history=iter_time_history)
    result["joint"]["noise_power"] = noise_power

    if verbose:
        print(f"\nBest loss: {best_loss:.4f}  ({elapsed:.1f}s)")
        for i, cfg in enumerate(tx_configs):
            b = i * 4
            print(f"  {cfg.name}: Az={best_params[b]:.1f}°  El={best_params[b+1]:.1f}°"
                  f"  pos=({best_params[b+2]:.1f}, {best_params[b+3]:.1f})")
        print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# 2. Particle Swarm Optimisation
# ---------------------------------------------------------------------------

def pso_multi_tx(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    n_particles: int = 20,
    num_iterations: int = 30,
    noise_power: float = 1e-10,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    lds: str = "Sobol",
    seed: Optional[int] = None,
    verbose: bool = True,
    early_stop_window: int = 5,
    early_stop_min_improvement: float = 1e-3,
    seed_params: Optional[np.ndarray] = None,
    sir_threshold_db: float = -3.0,
    sigmoid_k: float = 0.5,
) -> dict:
    """Particle swarm optimisation over the joint angle/position space.

    Each particle is a full parameter vector [az_0, el_0, x_0, y_0, ...].
    Standard PSO velocity update with inertia weight ``w``, cognitive weight
    ``c1``, and social weight ``c2``.

    If ``seed_params`` is provided, particle 0 is initialised at the warm-start
    position instead of a random point.  Its velocity is still random, so it
    retains freedom to move away.

    Parameters
    ----------
    scene : sionna.rt.Scene
    tx_configs : list[TxConfig]
    map_config : dict
    scene_xml_path : str
    n_particles : int
        Swarm size.  More particles → better exploration, more evaluations.
    num_iterations : int
        Number of PSO iterations (each costs n_particles evaluations).
    noise_power : float
        Thermal noise floor (Watts).
    w : float
        Inertia weight — controls how much of the previous velocity is kept.
    c1 : float
        Cognitive (personal-best) acceleration coefficient.
    c2 : float
        Social (global-best) acceleration coefficient.
    lds : str
        LDS used during tx_state setup ("Halton" | "Sobol" | "Uniform").
    seed : int or None
        RNG seed for reproducibility.
    early_stop_window : int
        Stop if global-best has not improved by ``early_stop_min_improvement``
        over the last ``early_stop_window`` iterations.  PSO tracks a monotone
        global-best, so a window of 5 is semantically equivalent to coordinate
        descent's per-cycle no-improvement check (both ask: "did anything
        improve in one representative unit of work?").  ``num_iterations``
        should comfortably exceed this value so the criterion has room to fire.
    early_stop_min_improvement : float
        Minimum improvement threshold shared across all iterative methods.
    verbose : bool
    seed_params : np.ndarray or None
        Optional warm-start flat parameter vector.  When provided, particle 0
        is placed here instead of a random uniform location.

    Returns
    -------
    dict  (same schema as ``optimize_multi_tx``)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"PSO BASELINE  ({n_particles} particles × {num_iterations} iters, "
              f"{len(tx_configs)} TX)")
        print(f"{'='*60}")

    start_time = time.time()
    rng = np.random.default_rng(seed)

    qrand = _make_qrand(lds)
    tx_states = [_setup_tx_state(scene, cfg, scene_xml_path, qrand)
                 for cfg in tx_configs]

    zone_masks = _zone_masks_from_states(tx_states, map_config)
    lo, hi     = _param_bounds(tx_configs, tx_states)
    span       = hi - lo
    n_params   = len(lo)

    # Initialise positions uniformly, velocities to small random values
    positions  = rng.uniform(lo, hi, size=(n_particles, n_params))
    velocities = rng.uniform(-span * 0.1, span * 0.1, size=(n_particles, n_params))

    # Seed particle 0 at the warm-start point if provided
    if seed_params is not None:
        positions[0] = np.clip(seed_params, lo, hi)

    personal_best_pos  = positions.copy()
    personal_best_loss = np.full(n_particles, np.inf)

    global_best_pos  = positions[0].copy()
    global_best_loss = np.inf

    loss_history      = []   # best loss at each iteration
    iter_time_history = []
    converged_iter    = None

    for iteration in range(num_iterations):
        iter_start  = time.time()
        iter_losses = []
        for k in range(n_particles):
            loss = _evaluate_config(scene, tx_configs, tx_states, positions[k],
                                    map_config, zone_masks, noise_power,
                                    sir_threshold_db=sir_threshold_db,
                                    sigmoid_k=sigmoid_k)
            iter_losses.append(loss)

            if loss < personal_best_loss[k]:
                personal_best_loss[k] = loss
                personal_best_pos[k]  = positions[k].copy()

            if loss < global_best_loss:
                global_best_loss = loss
                global_best_pos  = positions[k].copy()

        loss_history.append(float(global_best_loss))
        iter_time_history.append(time.time() - iter_start)

        if verbose:
            best_iter = min(iter_losses)
            print(f"  Iter {iteration+1:3d}/{num_iterations}  "
                  f"best_iter={best_iter:.4f}  global_best={global_best_loss:.4f}  "
                  f"({iter_time_history[-1]:.1f}s)")

        # No-improvement early stopping
        if len(loss_history) >= early_stop_window:
            if (loss_history[-early_stop_window] - loss_history[-1]
                    < early_stop_min_improvement):
                converged_iter = iteration + 1
                if verbose:
                    print(f"  [early stop] no improvement over last "
                          f"{early_stop_window} iters at iter {converged_iter}")
                break

        # Velocity and position update
        r1 = rng.uniform(0.0, 1.0, size=(n_particles, n_params))
        r2 = rng.uniform(0.0, 1.0, size=(n_particles, n_params))

        velocities = (
            w  * velocities
            + c1 * r1 * (personal_best_pos - positions)
            + c2 * r2 * (global_best_pos   - positions)
        )

        # Clamp velocity to ±20% of parameter range to prevent explosion
        v_max = 0.2 * span
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities
        positions = np.clip(positions, lo, hi)

        # Project positions to valid building polygons
        stride = 4
        for k in range(n_particles):
            for i, state in enumerate(tx_states):
                b = i * stride
                px, py = state["tx_placement"].project_to_polygon(
                    positions[k, b + 2], positions[k, b + 3]
                )
                positions[k, b + 2] = px
                positions[k, b + 3] = py

    # Apply global best to scene
    _apply_params(scene, tx_configs, tx_states, global_best_pos)

    elapsed = time.time() - start_time
    result  = _build_result(tx_configs, tx_states, global_best_pos, loss_history,
                            _initial_params(tx_configs, tx_states), elapsed, "pso",
                            iter_time_history=iter_time_history,
                            converged_iter=converged_iter)
    result["joint"]["noise_power"] = noise_power

    if verbose:
        print(f"\nBest loss: {global_best_loss:.4f}  ({elapsed:.1f}s)")
        for i, cfg in enumerate(tx_configs):
            b = i * 4
            print(f"  {cfg.name}: Az={global_best_pos[b]:.1f}°  El={global_best_pos[b+1]:.1f}°"
                  f"  pos=({global_best_pos[b+2]:.1f}, {global_best_pos[b+3]:.1f})")
        print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# 3. Coordinate Descent
# ---------------------------------------------------------------------------

def coordinate_descent_multi_tx(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    num_cycles: int = 10,
    n_line_points: int = 20,
    noise_power: float = 1e-10,
    lds: str = "Sobol",
    verbose: bool = True,
    early_stop_min_improvement: float = 1e-3,
    sir_threshold_db: float = -3.0,
    sigmoid_k: float = 0.5,
) -> dict:
    """Coordinate descent with 1-D line search over the joint angle/position space.

    In each cycle, every parameter is optimised in turn by evaluating
    ``n_line_points`` values along that axis while holding the others fixed.
    The best value found is immediately committed before moving to the next
    parameter (Gauss-Seidel / greedy-update style).

    Parameters
    ----------
    scene : sionna.rt.Scene
    tx_configs : list[TxConfig]
    map_config : dict
    scene_xml_path : str
    num_cycles : int
        Number of full passes through all parameters.
    n_line_points : int
        Grid points evaluated per parameter per cycle.
    noise_power : float
        Thermal noise floor (Watts).
    lds : str
        LDS used during tx_state setup.
    verbose : bool

    Returns
    -------
    dict  (same schema as ``optimize_multi_tx``)
    """
    param_names = ["az", "el", "x", "y"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"COORDINATE DESCENT BASELINE  ({num_cycles} cycles × "
              f"{len(tx_configs) * 4} params × {n_line_points} points, "
              f"{len(tx_configs)} TX)")
        print(f"{'='*60}")

    start_time = time.time()

    qrand = _make_qrand(lds)
    tx_states = [_setup_tx_state(scene, cfg, scene_xml_path, qrand)
                 for cfg in tx_configs]

    zone_masks = _zone_masks_from_states(tx_states, map_config)
    lo, hi     = _param_bounds(tx_configs, tx_states)
    n_params   = len(lo)

    # Initialise at the same starting point as the gradient-based method
    current = _initial_params(tx_configs, tx_states).copy()

    # Evaluate initial config
    best_loss        = _evaluate_config(scene, tx_configs, tx_states, current,
                                        map_config, zone_masks, noise_power,
                                        sir_threshold_db=sir_threshold_db,
                                        sigmoid_k=sigmoid_k)
    loss_history      = [float(best_loss)]
    iter_time_history = []
    converged_iter    = None

    for cycle in range(num_cycles):
        cycle_start      = time.time()
        cycle_loss_start = best_loss

        for d in range(n_params):
            # Build a 1-D sweep along dimension d
            sweep_values = np.linspace(lo[d], hi[d], n_line_points)
            best_d_loss  = np.inf
            best_d_val   = current[d]

            for val in sweep_values:
                candidate      = current.copy()
                candidate[d]   = val

                # For position coordinates: project to polygon before eval
                # Determine which TX this parameter belongs to
                tx_idx  = d // 4
                param_i = d %  4
                if param_i in (2, 3):   # x or y
                    stride = 4
                    b      = tx_idx * stride
                    px, py = tx_states[tx_idx]["tx_placement"].project_to_polygon(
                        candidate[b + 2], candidate[b + 3]
                    )
                    candidate[b + 2] = px
                    candidate[b + 3] = py

                loss = _evaluate_config(scene, tx_configs, tx_states, candidate,
                                        map_config, zone_masks, noise_power,
                                        sir_threshold_db=sir_threshold_db,
                                        sigmoid_k=sigmoid_k)
                if loss < best_d_loss:
                    best_d_loss = loss
                    best_d_val  = candidate[d]

            # Commit the best value for this coordinate
            current[d] = best_d_val
            if best_d_loss < best_loss:
                best_loss = best_d_loss

        loss_history.append(float(best_loss))
        cycle_dur = time.time() - cycle_start
        iter_time_history.append(cycle_dur)

        improvement = cycle_loss_start - best_loss
        if verbose:
            print(f"  Cycle {cycle+1:3d}/{num_cycles}  "
                  f"loss={best_loss:.4f}  Δ={improvement:+.4f}  ({cycle_dur:.1f}s)")

        # No-improvement early stopping (per-cycle)
        if improvement < early_stop_min_improvement:
            converged_iter = cycle + 1
            if verbose:
                print(f"  [early stop] no improvement (Δ={improvement:.4f} < "
                      f"{early_stop_min_improvement}) at cycle {converged_iter}")
            break

    # Apply final config to scene
    _apply_params(scene, tx_configs, tx_states, current)

    elapsed = time.time() - start_time
    result  = _build_result(tx_configs, tx_states, current, loss_history,
                            _initial_params(tx_configs, tx_states), elapsed,
                            "coordinate_descent",
                            iter_time_history=iter_time_history,
                            converged_iter=converged_iter)
    result["joint"]["noise_power"] = noise_power

    if verbose:
        print(f"\nFinal loss: {best_loss:.4f}  ({elapsed:.1f}s)")
        for i, cfg in enumerate(tx_configs):
            b = i * 4
            print(f"  {cfg.name}: Az={current[b]:.1f}°  El={current[b+1]:.1f}°"
                  f"  pos=({current[b+2]:.1f}, {current[b+3]:.1f})")
        print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# 4. 3GPP TR 38.901 UMa Naive Empirical Baseline
# ---------------------------------------------------------------------------

# 3GPP TR 38.901 / TR 38.802 UMa (Urban Macro) antenna defaults.
# UMa places the BS above surrounding rooftop height (h_BS = 25 m nominal),
# making it the appropriate model when antennas are mounted on building rooftops.
_UMA_MECHANICAL_TILT_DEG = 0.0    # physical bracket tilt; 0° typical for UMa
_UMA_ELECTRICAL_TILT_DEG = 6.0    # electrical beam steering per 3GPP calibration


def uma_naive_baseline_multi_tx(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    mechanical_downtilt_deg: float = _UMA_MECHANICAL_TILT_DEG,
    electrical_downtilt_deg: float = _UMA_ELECTRICAL_TILT_DEG,
    noise_power: float = 1e-10,
    lds: str = "Sobol",
    verbose: bool = True,
    sir_threshold_db: float = -3.0,
    sigmoid_k: float = 0.5,
) -> dict:
    """3GPP TR 38.901 UMa naive empirical baseline.

    Sets each transmitter's boresight using the standard 3GPP Urban Macro (UMa)
    antenna pointing convention — no optimisation is performed.  UMa is the
    appropriate reference scenario when base stations are mounted on building
    rooftops (BS above surrounding clutter height), as opposed to UMi which
    models street-level or below-rooftop deployments.

    **TX height** is taken directly from the scene (i.e. whatever ``TxPlacement``
    set before this function is called — ``building["z_height"] + offset``).
    This matches the exact height used in the gradient-based experiments.

    Pointing rule (per TX)
    ----------------------
    1. **Azimuth** — geometric look-at: direction from the TX position to the
       centroid of its assigned coverage zone.  This is the natural sector
       orientation a network engineer would choose from a site survey.

    2. **Elevation** — fixed downtilt composed of two 3GPP components:

       * ``mechanical_downtilt_deg`` — physical tilt of the antenna bracket.
         Default 0° per TR 38.901 UMa (antennas typically flush-mounted on
         the building facade or rooftop edge).

       * ``electrical_downtilt_deg`` — electrical beam steering applied by the
         antenna array.  Default 6° per TR 38.802 Table A.2-2 UMa calibration
         assumptions.

       Applied elevation = -(mechanical + electrical)
       (negative = downward in this codebase's angle convention).

    3. **Position** — unchanged from whatever ``TxPlacement`` set in the scene.

    Reference
    ---------
    3GPP TR 38.901 v17.0.0, Table 7.1-1 (UMa)
    3GPP TR 38.802, Table A.2-2 (UMa simulation assumptions)

    Parameters
    ----------
    scene : sionna.rt.Scene
    tx_configs : list[TxConfig]
    map_config : dict
    scene_xml_path : str
    mechanical_downtilt_deg : float
        Physical bracket tilt (positive = tilted downward). Default 0°.
    electrical_downtilt_deg : float
        Array-steered beam tilt (positive = tilted downward). Default 6°.
    noise_power : float
        Thermal noise floor (Watts).
    lds : str
        LDS used during tx_state setup ("Halton" | "Sobol" | "Uniform").
    verbose : bool

    Returns
    -------
    dict  (same schema as ``optimize_multi_tx``)
        ``best_angles`` holds the fixed UMa pointing angles.
        ``az_history`` / ``el_history`` are empty (no iteration).
        ``joint.loss_history`` has a single entry (one evaluation).
        ``joint.sampler`` is ``"uma_naive_3gpp_38901"``.
    """
    total_downtilt  = mechanical_downtilt_deg + electrical_downtilt_deg
    fixed_elevation = -total_downtilt   # negative = downward-pointing

    if verbose:
        print(f"\n{'='*60}")
        print(f"3GPP TR 38.901 UMa NAIVE BASELINE  ({len(tx_configs)} TX)")
        print(f"  Mechanical downtilt : {mechanical_downtilt_deg:.1f}°")
        print(f"  Electrical downtilt : {electrical_downtilt_deg:.1f}°")
        print(f"  Total downtilt      : {total_downtilt:.1f}°  "
              f"(elevation = {fixed_elevation:.1f}°)")
        print(f"  Azimuth             : geometric look-at (TX → zone centroid)")
        print(f"  TX height           : from scene (set by TxPlacement)")
        print(f"{'='*60}")

    start_time = time.time()

    qrand = _make_qrand(lds)
    tx_states = [_setup_tx_state(scene, cfg, scene_xml_path, qrand)
                 for cfg in tx_configs]

    zone_masks = _zone_masks_from_states(tx_states, map_config)

    # Build the naive parameter vector.
    # - azimuth  : geometric look-at direction (TX → zone centroid)
    # - elevation: fixed 3GPP UMi total downtilt (replaces geometric elevation)
    # - x, y    : current TX position as placed by TxPlacement (no movement)
    params = []
    for cfg, state in zip(tx_configs, tx_states):
        tx_pos   = state["tx_position"]           # [x, y, z] — from scene
        zone_cxy = state["zone_polygon"].centroid  # Shapely centroid of zone
        look_at  = [zone_cxy.x, zone_cxy.y, map_config["center"][2]]

        az, _ = compute_initial_angles_from_position(tx_pos, look_at, verbose=False)
        params += [az, fixed_elevation, tx_pos[0], tx_pos[1]]

    params = np.array(params, dtype=np.float64)

    if verbose:
        for i, cfg in enumerate(tx_configs):
            b = i * 4
            print(f"  {cfg.name}: Az={params[b]:.1f}°  El={params[b+1]:.1f}°  "
                  f"pos=({params[b+2]:.1f}, {params[b+3]:.1f})  "
                  f"height={tx_states[i]['tx_height']:.1f}m")

    # Single evaluation — no iteration
    loss = _evaluate_config(scene, tx_configs, tx_states, params,
                            map_config, zone_masks, noise_power,
                            sir_threshold_db=sir_threshold_db,
                            sigmoid_k=sigmoid_k)
    loss_history = [float(loss)]

    elapsed = time.time() - start_time
    result  = _build_result(tx_configs, tx_states, params, loss_history,
                            params, elapsed, "uma_naive_3gpp_38901",
                            iter_time_history=[elapsed])
    result["joint"]["noise_power"]         = noise_power
    result["joint"]["mechanical_downtilt"] = mechanical_downtilt_deg
    result["joint"]["electrical_downtilt"] = electrical_downtilt_deg

    if verbose:
        print(f"\nUMa naive loss: {loss:.4f}  ({elapsed:.1f}s)")
        print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# 5. RadioMap Gradient Optimizer (first-order baseline)
# ---------------------------------------------------------------------------

def _radiomap_sir_loss_body(
    all_params, N, tx_configs, tx_states, scene, rm_solver,
    map_config, noise_power, zone_masks_np, n_zone_cells, rm_kwargs,
    epsilon=1e-30,
):
    """Core SINR loss driven through RadioMapSolver (inside @dr.wrap context).

    all_params : list of DrJIT Float scalars ordered as:
        [az_0, el_0, x_0, y_0,  az_1, el_1, x_1, y_1, ...]

    Zone masks (numpy H×W float32 arrays) are converted to TensorXf inside this
    function so they live in the right DrJIT context.  They carry no grad.
    """
    deg2rad = Float(float(np.pi / 180.0))
    dr.disable_grad(deg2rad)

    # Set TX orientations and positions with grad enabled.
    # dr.wrap converts 0-dim torch scalars to TensorXf, not Float. Use .array
    # to extract the underlying Float view so mi.Point3f accepts it.
    for i, cfg in enumerate(tx_configs):
        b    = i * 4
        az_i = all_params[b].array;     dr.enable_grad(az_i)
        el_i = all_params[b + 1].array; dr.enable_grad(el_i)
        x_i  = all_params[b + 2].array; dr.enable_grad(x_i)
        y_i  = all_params[b + 3].array; dr.enable_grad(y_i)

        yaw   = az_i * deg2rad
        pitch = -(el_i * deg2rad)
        roll  = Float(0.0); dr.disable_grad(roll)

        scene.get(cfg.name).orientation = [yaw, pitch, roll]
        scene.get(cfg.name).position = [
            x_i,
            y_i,
            Float(float(tx_states[i]["tx_height"])),
        ]

    # Differentiable RadioMapSolver pass
    rm = rm_solver(scene, **rm_kwargs)

    eps_f   = Float(float(epsilon));     dr.disable_grad(eps_f)
    noise_f = Float(float(noise_power)); dr.disable_grad(noise_f)
    total_loss = Float(0.0);             dr.disable_grad(total_loss)

    for i in range(N):
        # Build constant zone mask as TensorXf (same H×W shape as rm.rss[i])
        mask_tf = TensorXf(zone_masks_np[i])
        dr.disable_grad(mask_tf)

        n_cells_f = Float(float(n_zone_cells[i])); dr.disable_grad(n_cells_f)

        # Signal power for TX i, zeroed outside zone
        rss_i = rm.rss[i] * mask_tf   # TensorXf (H, W), grad-tracked

        # Interference: sum of all other TX powers within zone i
        p_int = None
        for j in range(N):
            if j == i:
                continue
            p_j   = rm.rss[j] * mask_tf
            p_int = p_j if p_int is None else (p_int + p_j)

        if p_int is None:
            # Single-TX scene: maximise raw log-power
            metric = rss_i + eps_f
        else:
            metric = (rss_i + eps_f) / (p_int + noise_f + eps_f)

        # Masked log-mean: multiply by mask so out-of-zone terms vanish,
        # then normalise by number of in-zone cells (not total grid cells).
        log_metric = dr.log(metric) * mask_tf
        loss_i     = -dr.sum(log_metric) / n_cells_f
        total_loss = total_loss + loss_i

    return total_loss


def _make_radiomap_sir_loss(
    N, tx_configs, tx_states, scene, rm_solver,
    map_config, noise_power, zone_masks_np, n_zone_cells, rm_kwargs,
):
    """Build and return the @dr.wrap-decorated RadioMap SINR loss function.

    Uses exec() to produce a function with a fixed positional signature
    (required by @dr.wrap): [az_0, el_0, x_0, y_0, ..., az_N, el_N, x_N, y_N].
    """
    arg_names = []
    for i in range(N):
        b = i * 4
        arg_names += [f"p{b}", f"p{b+1}", f"p{b+2}", f"p{b+3}"]

    arg_str  = ", ".join(arg_names)
    list_str = "[" + ", ".join(arg_names) + "]"

    func_code = (
        f"def _inner({arg_str}):\n"
        f"    return _body({list_str}, _N, _cfgs, _states, _scene, _rmsolver,\n"
        f"                 _mcfg, _noise, _masks, _ncells, _rmkw)\n"
    )

    globs = {
        "_body":     _radiomap_sir_loss_body,
        "_N":        N,
        "_cfgs":     tx_configs,
        "_states":   tx_states,
        "_scene":    scene,
        "_rmsolver": rm_solver,
        "_mcfg":     map_config,
        "_noise":    noise_power,
        "_masks":    zone_masks_np,
        "_ncells":   n_zone_cells,
        "_rmkw":     rm_kwargs,
    }
    exec(func_code, globs)
    inner_fn = globs["_inner"]
    return dr.wrap(source="torch", target="drjit")(inner_fn)


def radiomap_gradient_multi_tx(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    learning_rate: float = 3.0,
    num_iterations: int = 50,
    noise_power: float = 1e-10,
    lds: str = "Sobol",
    samples_per_tx: int = int(1e7),
    max_depth: int = 8,
    verbose: bool = True,
    early_stop_window: int = 10,
    early_stop_min_improvement: float = 1e-3,
    early_stop_min_flips: int = 4,
) -> dict:
    """First-order (Adam) optimizer using RadioMapSolver as the differentiable engine.

    Instead of PathSolver + zone-sampled point receivers, this method drives
    gradients directly through ``RadioMapSolver.rss``.  The SINR objective is
    the average over all radio-map cells that fall inside each TX's user-defined
    coverage zone — no receiver sampling noise.

    This baseline demonstrates that RadioMapSolver gradients are large enough to
    move transmitters across the full angle/position parameter space, providing
    a fair first-order comparison against ``optimize_multi_tx``.

    Loss (same functional form as PathSolver version)
    -------------------------------------------------
    L = Σ_i  -mean_{c ∈ zone_i} log(SINR_{i,c} + ε)

    where SINR_{i,c} = rss[i,c] / (Σ_{j≠i} rss[j,c] + noise).

    Parameters
    ----------
    scene : sionna.rt.Scene
        All TxConfig.name transmitters must already be added.
    tx_configs : list[TxConfig]
    map_config : dict
        {'center', 'size', 'cell_size'} — same format as optimize_multi_tx.
    scene_xml_path : str
    learning_rate : float
        Adam learning rate (default 3.0, same as PathSolver baseline).
    num_iterations : int
    noise_power : float
        Thermal noise floor (Watts).
    lds : str
        Low-discrepancy sequence for tx_state setup.
    samples_per_tx : int
        Rays per TX per RadioMapSolver call (trade-off: accuracy vs. speed).
    max_depth : int
        Maximum path depth for RadioMapSolver.
    verbose : bool

    Returns
    -------
    dict  (same schema as optimize_multi_tx / compare_multi_tx_performance)
    """
    N = len(tx_configs)
    assert N >= 1, "tx_configs must contain at least one TxConfig"

    if verbose:
        print(f"\n{'='*70}")
        print(f"RADIOMAP GRADIENT BASELINE  ({N} TX, {num_iterations} iters, "
              f"lr={learning_rate})")
        print(f"{'='*70}")

    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Build per-TX state dicts (geometry, zone polygons, initial angles)
    # ------------------------------------------------------------------
    qrand     = _make_qrand(lds)
    tx_states = [_setup_tx_state(scene, cfg, scene_xml_path, qrand)
                 for cfg in tx_configs]

    # ------------------------------------------------------------------
    # 2. Build binary zone masks aligned to the RadioMap grid
    # ------------------------------------------------------------------
    zone_masks_np = _zone_masks_from_states(tx_states, map_config)   # list of H×W float32
    n_zone_cells  = [max(1, int(m.sum())) for m in zone_masks_np]

    # ------------------------------------------------------------------
    # 3. RadioMapSolver fixed call kwargs (same for every iteration)
    # ------------------------------------------------------------------
    rm_kwargs = dict(
        max_depth=max_depth,
        samples_per_tx=samples_per_tx,
        cell_size=list(map_config["cell_size"]),
        center=map_config["center"],
        orientation=[0, 0, 0],
        size=map_config["size"],
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        diffraction=True,
        refraction=False,
        stop_threshold=None,
    )

    # ------------------------------------------------------------------
    # 4. Build @dr.wrap loss closure
    # ------------------------------------------------------------------
    rm_solver             = RadioMapSolver()
    rm_solver.loop_mode   = "evaluated"   # required for AD through dr.while_loop
    compute_loss = _make_radiomap_sir_loss(
        N, tx_configs, tx_states, scene, rm_solver,
        map_config, noise_power, zone_masks_np, n_zone_cells, rm_kwargs,
    )

    # ------------------------------------------------------------------
    # 5. Initialise PyTorch parameters and Adam optimiser
    # ------------------------------------------------------------------
    params = []
    for state in tx_states:
        params.append(torch.tensor(state["initial_azimuth"],   device="cuda",
                                   dtype=torch.float32, requires_grad=True))
        params.append(torch.tensor(state["initial_elevation"], device="cuda",
                                   dtype=torch.float32, requires_grad=True))
        params.append(torch.tensor(state["tx_position"][0],   device="cuda",
                                   dtype=torch.float32, requires_grad=True))
        params.append(torch.tensor(state["tx_position"][1],   device="cuda",
                                   dtype=torch.float32, requires_grad=True))

    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))

    # ------------------------------------------------------------------
    # 6. Optimisation loop
    # ------------------------------------------------------------------
    loss_history      = []
    iter_time_history = []
    converged_iter    = None
    final_bufs        = {i: {"az": [], "el": []} for i in range(N)}

    for iteration in range(num_iterations):
        iter_start = time.time()

        loss = compute_loss(*params)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        dr.flush_kernel_cache()
        dr.flush_malloc_cache()

        # Per-TX post-step constraints (no-grad)
        with torch.no_grad():
            for i, state in enumerate(tx_states):
                b    = i * 4
                az_t = params[b]
                el_t = params[b + 1]
                x_t  = params[b + 2]
                y_t  = params[b + 3]

                az_t.clamp_(AZ_MIN, AZ_MAX)
                if az_t.item() >= 360.0:
                    az_t.fill_(az_t.item() % 360.0)

                el_t.clamp_(EL_MIN, EL_MAX)

                proj_x, proj_y = state["tx_placement"].project_to_polygon(
                    x_t.item(), y_t.item()
                )
                x_t.data.fill_(proj_x)
                y_t.data.fill_(proj_y)

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        dur = time.time() - iter_start
        iter_time_history.append(dur)

        for i in range(N):
            b = i * 4
            tx_states[i]["az_history"].append(float(params[b].item()))
            tx_states[i]["el_history"].append(float(params[b + 1].item()))

        window_start = max(0, num_iterations - 10)
        if iteration >= window_start:
            for i in range(N):
                b = i * 4
                final_bufs[i]["az"].append(float(params[b].item()))
                final_bufs[i]["el"].append(float(params[b + 1].item()))

        if verbose:
            print(f"  Iter {iteration+1:3d}/{num_iterations}  "
                  f"loss={loss_val:.4f}  ({dur:.1f}s)")

        # --- Oscillation-based early stopping ----------------------------
        if len(loss_history) >= early_stop_window:
            recent  = loss_history[-early_stop_window:]
            deltas  = [recent[k + 1] - recent[k] for k in range(len(recent) - 1)]
            n_flips = sum(
                1 for k in range(len(deltas) - 1)
                if deltas[k] * deltas[k + 1] < 0
            )
            net_change = abs(recent[-1] - recent[0])
            if n_flips >= early_stop_min_flips and net_change < early_stop_min_improvement:
                converged_iter = iteration + 1
                if verbose:
                    print(f"  [early stop] oscillating at iter {converged_iter} "
                          f"({n_flips} sign flips, net Δloss={net_change:.4f})")
                break

    # ------------------------------------------------------------------
    # 7. Apply best (last-10 average) parameters back to scene
    # ------------------------------------------------------------------
    for i, (cfg, state) in enumerate(zip(tx_configs, tx_states)):
        b       = i * 4
        best_az = (float(np.mean(final_bufs[i]["az"])) if final_bufs[i]["az"]
                   else float(params[b].item()))
        best_el = (float(np.mean(final_bufs[i]["el"])) if final_bufs[i]["el"]
                   else float(params[b + 1].item()))
        final_x = float(params[b + 2].item())
        final_y = float(params[b + 3].item())

        yaw_r, pitch_r = azimuth_elevation_to_yaw_pitch(best_az, best_el)
        scene.get(cfg.name).orientation = mi.Point3f(float(yaw_r), float(pitch_r), 0.0)
        scene.get(cfg.name).position    = mi.Point3f(float(final_x), float(final_y),
                                                      float(state["tx_height"]))
        state["best_angles"]    = [best_az, best_el]
        state["final_position"] = [final_x, final_y, state["tx_height"]]

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # 8. Build result dict (same schema as optimize_multi_tx)
    # ------------------------------------------------------------------
    result = {}
    for cfg, state in zip(tx_configs, tx_states):
        result[cfg.name] = {
            "best_angles":      state["best_angles"],
            "final_position":   state["final_position"],
            "initial_angles":   [state["initial_azimuth"], state["initial_elevation"]],
            "initial_position": state["tx_position"],
            "az_history":       state["az_history"],
            "el_history":       state["el_history"],
        }

    result["joint"] = {
        "loss_history":      loss_history,
        "iter_time_history": iter_time_history,
        "converged_iter":    converged_iter,
        "elapsed_time_s":    elapsed,
        "num_iterations":    num_iterations,
        "noise_power":       noise_power,
        "sampler":           "radiomap_gradient",
        "sampling_strata":   "radiomap_zone_cells",
        "lds":               lds,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"RADIOMAP GRADIENT BASELINE COMPLETE  ({elapsed:.1f}s)")
        for cfg in tx_configs:
            r = result[cfg.name]
            print(f"  {cfg.name}: Az={r['best_angles'][0]:.1f}°, "
                  f"El={r['best_angles'][1]:.1f}°, "
                  f"pos=({r['final_position'][0]:.1f}, {r['final_position'][1]:.1f})")
        print(f"{'='*70}\n")

    return result

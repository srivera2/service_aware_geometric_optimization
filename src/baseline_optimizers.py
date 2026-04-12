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
objective mirrors the gradient-based method: negative mean log-SIR summed
across all TX zones.  A shared ``_evaluate_config`` helper applies a parameter
vector to the scene and returns the scalar loss value.

Parameter vector layout (per TX, concatenated)
-----------------------------------------------
  [az_0, el_0, x_0, y_0,  az_1, el_1, x_1, y_1, ...]

  az   : azimuth   [0, 360)  degrees
  el   : elevation [0,  90]  degrees  (downward-facing boresight)
  x, y : TX position, constrained to building rooftop polygon
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import mitsuba as mi
import numpy as np
from sionna.rt import RadioMapSolver

from angle_utils import (
    azimuth_elevation_to_yaw_pitch,
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


def _run_radiomap(scene, map_config: dict) -> object:
    """Run RadioMapSolver and return the radio map."""
    solver = RadioMapSolver()
    return solver(
        scene,
        max_depth=8,
        samples_per_tx=int(1e8),   # lighter than gradient optimizer for speed
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


def _sir_loss_from_radiomap(rm, tx_configs: list[TxConfig], zone_masks: list[np.ndarray],
                             noise_power: float) -> float:
    """Compute the scalar SIR loss from a RadioMapSolver result.

    Mirrors the gradient-based loss: negative mean log-SIR summed over zones.
    Zone masks are passed as a list aligned to tx_configs order.
    """
    import drjit as dr
    rss_np = np.array(dr.detach(rm.rss))   # shape (N_tx, H, W)
    N = len(tx_configs)
    total_loss = 0.0
    eps = 1e-30

    for i in range(N):
        mask = zone_masks[i]
        sig  = rss_np[i][mask > 0]

        interf = np.zeros_like(sig)
        for j in range(N):
            if j != i:
                interf += rss_np[j][mask > 0]

        sir    = sig / (interf + noise_power)
        loss_i = -float(np.mean(np.log(sir + eps)))
        total_loss += loss_i

    return total_loss


def _evaluate_config(scene, tx_configs: list[TxConfig], tx_states: list[dict],
                     params: np.ndarray, map_config: dict,
                     zone_masks: list[np.ndarray], noise_power: float) -> float:
    """Apply params to scene, run RadioMapSolver, return scalar SIR loss."""
    _apply_params(scene, tx_configs, tx_states, params)
    rm = _run_radiomap(scene, map_config)
    return _sir_loss_from_radiomap(rm, tx_configs, zone_masks, noise_power)


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
                  initial_params, elapsed, method_name) -> dict:
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
        "loss_history":    loss_history,
        "elapsed_time_s":  elapsed,
        "num_iterations":  len(loss_history),
        "noise_power":     None,     # filled in by caller
        "sampler":         method_name,
        "sampling_strata": "n/a",
        "lds":             "n/a",
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
    lds: str = "Halton",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Brute-force random search over the joint angle/position space.

    Samples ``n_candidates`` configurations independently at random (or via a
    low-discrepancy sequence), evaluates each with RadioMapSolver, and returns
    the best.

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

    best_loss   = np.inf
    best_params = _initial_params(tx_configs, tx_states)
    loss_history = []

    for k, params in enumerate(candidates):
        loss = _evaluate_config(scene, tx_configs, tx_states, params,
                                map_config, zone_masks, noise_power)
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
                            best_params, elapsed, "random_search")
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
    lds: str = "Halton",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Particle swarm optimisation over the joint angle/position space.

    Each particle is a full parameter vector [az_0, el_0, x_0, y_0, ...].
    Standard PSO velocity update with inertia weight ``w``, cognitive weight
    ``c1``, and social weight ``c2``.

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
    verbose : bool

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

    personal_best_pos  = positions.copy()
    personal_best_loss = np.full(n_particles, np.inf)

    global_best_pos  = positions[0].copy()
    global_best_loss = np.inf

    loss_history = []   # best loss at each iteration

    for iteration in range(num_iterations):
        iter_losses = []
        for k in range(n_particles):
            loss = _evaluate_config(scene, tx_configs, tx_states, positions[k],
                                    map_config, zone_masks, noise_power)
            iter_losses.append(loss)

            if loss < personal_best_loss[k]:
                personal_best_loss[k] = loss
                personal_best_pos[k]  = positions[k].copy()

            if loss < global_best_loss:
                global_best_loss = loss
                global_best_pos  = positions[k].copy()

        loss_history.append(float(global_best_loss))

        if verbose:
            best_iter = min(iter_losses)
            print(f"  Iter {iteration+1:3d}/{num_iterations}  "
                  f"best_iter={best_iter:.4f}  global_best={global_best_loss:.4f}")

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
                            _initial_params(tx_configs, tx_states), elapsed, "pso")
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
    lds: str = "Halton",
    verbose: bool = True,
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
    best_loss    = _evaluate_config(scene, tx_configs, tx_states, current,
                                    map_config, zone_masks, noise_power)
    loss_history = [float(best_loss)]

    for cycle in range(num_cycles):
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
                                        map_config, zone_masks, noise_power)
                if loss < best_d_loss:
                    best_d_loss = loss
                    best_d_val  = candidate[d]

            # Commit the best value for this coordinate
            current[d] = best_d_val
            if best_d_loss < best_loss:
                best_loss = best_d_loss

        loss_history.append(float(best_loss))

        if verbose:
            improvement = cycle_loss_start - best_loss
            print(f"  Cycle {cycle+1:3d}/{num_cycles}  "
                  f"loss={best_loss:.4f}  Δ={improvement:+.4f}")

    # Apply final config to scene
    _apply_params(scene, tx_configs, tx_states, current)

    elapsed = time.time() - start_time
    result  = _build_result(tx_configs, tx_states, current, loss_history,
                            _initial_params(tx_configs, tx_states), elapsed,
                            "coordinate_descent")
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
    lds: str = "Halton",
    verbose: bool = True,
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
                            map_config, zone_masks, noise_power)
    loss_history = [float(loss)]

    elapsed = time.time() - start_time
    result  = _build_result(tx_configs, tx_states, params, loss_history,
                            params, elapsed, "uma_naive_3gpp_38901")
    result["joint"]["noise_power"]         = noise_power
    result["joint"]["mechanical_downtilt"] = mechanical_downtilt_deg
    result["joint"]["electrical_downtilt"] = electrical_downtilt_deg

    if verbose:
        print(f"\nUMa naive loss: {loss:.4f}  ({elapsed:.1f}s)")
        print(f"{'='*60}\n")

    return result

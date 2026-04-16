"""
motivating_baselines.py
=======================
Four baselines for the paper's motivating example.  Each is isolated here for
independent debugging and reveals a distinct weakness that the proposed method
addresses.

Public API
----------
naive_edge_center_baseline_multi_tx     : geometric look-at + edge placement
empirical_pso_baseline_multi_tx         : PSO driven by 3GPP UMa (no ray tracing)
radiomap_gradient_baseline_multi_tx     : Adam via RadioMapSolver (near-zero grads)
dense_pathsolver_gradient_baseline_multi_tx : Adam via PathSolver dense grid (slow)

All functions share the same result-dict schema as ``optimize_multi_tx``.
Additional keys are documented per-function.
"""

from __future__ import annotations

import gc
import time
from typing import Optional

import drjit as dr
from drjit.auto import Float
import mitsuba as mi
import numpy as np
import torch
from shapely.vectorized import contains
from sionna.rt import PathSolver, RadioMapSolver, Receiver, cpx_abs_square

from angle_utils import (
    azimuth_elevation_to_yaw_pitch,
    compute_initial_angles_from_position,
)
from baseline_optimizers import (
    AZ_MIN, AZ_MAX, EL_MIN, EL_MAX,
    _zone_masks_from_states,
    _apply_params,
    _evaluate_config,
    _build_result,
    _param_bounds,
    _initial_params,
    _make_radiomap_sir_loss,
)
from multi_tx_optimizer import TxConfig, _setup_tx_state, _make_qrand
from tx_placement import TxPlacement


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_grad_norm(params: list) -> float:
    """Mean absolute gradient across all params that have .grad set."""
    grads = [p.grad.abs().item() for p in params if p.grad is not None]
    return float(np.mean(grads)) if grads else 0.0


def _sample_zone_pts_numpy(zone_polygon, n_pts: int,
                            ground_z: float = 1.5,
                            rng=None) -> np.ndarray:
    """Sample n_pts points uniformly inside zone_polygon at height ground_z."""
    if rng is None:
        rng = np.random.default_rng()
    minx, miny, maxx, maxy = zone_polygon.bounds
    pts: list = []
    while len(pts) < n_pts:
        batch = rng.uniform([minx, miny], [maxx, maxy], size=(n_pts * 4, 2))
        inside = contains(zone_polygon, batch[:, 0], batch[:, 1])
        for c in batch[inside]:
            pts.append([c[0], c[1], ground_z])
            if len(pts) >= n_pts:
                break
    return np.array(pts[:n_pts], dtype=np.float64)


def _empirical_sir_loss(
    params_np: np.ndarray,
    tx_configs: list,
    tx_states: list,
    zone_pts_list: list,
    noise_power: float,
    frequency_hz: float,
    sir_threshold_db: float = -3.0,
    sigmoid_k: float = 0.5,
) -> float:
    """3GPP TR 38.901 UMa empirical SIR loss (pure numpy, no GPU).

    Parameters
    ----------
    params_np : flat array [az_0, el_0, x_0, y_0, az_1, ...]
    zone_pts_list : list of (n_pts, 3) arrays — evaluation points per zone
    """
    stride  = 4
    N       = len(tx_configs)
    f_GHz   = frequency_hz / 1e9

    # Extract per-TX pose from params
    tx_positions  = []
    tx_azimuths   = []
    tx_elevations = []
    for i, state in enumerate(tx_states):
        b = i * stride
        tx_positions.append(np.array([float(params_np[b + 2]),
                                       float(params_np[b + 3]),
                                       state["tx_height"]]))
        tx_azimuths.append(float(params_np[b]))
        tx_elevations.append(float(params_np[b + 1]))

    # Received power: all_rx_powers[tx_i][zone_j] = power array at zone j's pts
    all_rx_powers: dict = {}
    for i in range(N):
        tx_pos = tx_positions[i]
        tx_az  = tx_azimuths[i]
        tx_el  = tx_elevations[i]
        h_tx   = tx_pos[2]
        all_rx_powers[i] = {}

        for j in range(N):
            pts  = zone_pts_list[j]          # (n_pts, 3)
            h_rx = pts[:, 2]
            rx_xy = pts[:, :2]
            tx_xy = tx_pos[:2]

            d_2d = np.linalg.norm(rx_xy - tx_xy, axis=1).clip(1.0)
            d_3d = np.sqrt(d_2d**2 + (h_tx - h_rx)**2).clip(1.0)

            # 3GPP UMa path loss (TR 38.901 Table 7.4.1-1)
            pl_los  = 28.0 + 22.0 * np.log10(d_3d) + 20.0 * np.log10(f_GHz)
            pl_nlos = np.maximum(
                pl_los,
                13.54 + 39.08 * np.log10(d_3d) + 20.0 * np.log10(f_GHz)
                - 0.6 * (h_rx - 1.5),
            )

            # LOS probability (UMa, h_rx <= 13 m)
            p_los = (np.minimum(18.0 / d_2d, 1.0)
                     * (1.0 - np.exp(-d_2d / 63.0))
                     + np.exp(-d_2d / 63.0))

            g = (p_los       * 10.0 ** (-pl_los  / 10.0)
                 + (1 - p_los) * 10.0 ** (-pl_nlos / 10.0))

            # Antenna element gain (TR 38.901 §7.3, simplified 2D pattern)
            rx_az_rel  = np.degrees(np.arctan2(rx_xy[:, 1] - tx_pos[1],
                                                rx_xy[:, 0] - tx_pos[0]))
            phi_off    = ((rx_az_rel - tx_az) + 180.0) % 360.0 - 180.0
            rx_el_geom = np.degrees(np.arctan2(h_tx - h_rx, d_2d))
            theta_off  = -rx_el_geom - tx_el

            A_h = -np.minimum(12.0 * (phi_off   / 65.0) ** 2, 30.0)
            A_v = -np.minimum(12.0 * (theta_off / 65.0) ** 2, 30.0)
            G_db  = np.maximum(A_h + A_v, -30.0) + 8.0
            G_lin = 10.0 ** (G_db / 10.0)

            all_rx_powers[i][j] = G_lin * g   # unit tx power

    # SIR soft-coverage loss
    zone_areas   = [s["zone_polygon"].area for s in tx_states]
    total_area   = sum(zone_areas) or 1.0
    area_weights = [a / total_area for a in zone_areas]
    log10_scale  = 10.0 / np.log(10.0)
    total_loss   = 0.0

    for j in range(N):
        sig   = all_rx_powers[j][j]
        interf = sum(all_rx_powers[i][j] for i in range(N) if i != j)
        metric   = sig / (interf + noise_power)
        sir_db   = log10_scale * np.log(metric + 1.0)
        soft_cov = 1.0 / (1.0 + np.exp(-sigmoid_k * (sir_db - sir_threshold_db)))
        total_loss += -area_weights[j] * float(np.mean(soft_cov))

    return total_loss


# ---------------------------------------------------------------------------
# Baseline 1 — Naive Edge-Center
# ---------------------------------------------------------------------------

def naive_edge_center_baseline_multi_tx(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    noise_power: float = 1e-10,
    frequency_hz: float = 3.5e9,
    lds: str = "Halton",
    n_empirical_pts: int = 500,
    rx_height: float = 1.5,
    verbose: bool = True,
    sir_threshold_db: float = -3.0,
    sigmoid_k: float = 0.5,
) -> dict:
    """Naive geometric baseline: place TX on the building edge facing the zone.

    For each TX, projects the zone centroid onto the building rooftop polygon
    boundary to find the closest edge point, then aims geometrically toward the
    zone centroid.  No optimisation is performed.

    Runs a dual evaluation:
    - ``joint["empirical_loss"]``  : 3GPP UMa model prediction
    - ``joint["raytraced_loss"]``  : RadioMapSolver (Sionna ray tracing)

    This exposes the empirical-vs-ray-traced gap that motivates using a
    differentiable ray tracer instead of a closed-form channel model.

    Returns
    -------
    dict  (same schema as ``optimize_multi_tx``)
        Extra keys: ``joint["empirical_loss"]``, ``joint["raytraced_loss"]``
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"NAIVE EDGE-CENTER BASELINE  ({len(tx_configs)} TX)")
        print(f"{'='*60}")

    start_time = time.time()
    qrand      = _make_qrand(lds)
    tx_states  = [_setup_tx_state(scene, cfg, scene_xml_path, qrand)
                  for cfg in tx_configs]

    params = []
    for cfg, state in zip(tx_configs, tx_states):
        zone_c  = state["zone_polygon"].centroid
        zone_cx, zone_cy = zone_c.x, zone_c.y

        # Project zone centroid onto building polygon boundary
        edge_x, edge_y = state["tx_placement"].project_to_polygon(zone_cx, zone_cy)
        tx_pos = [edge_x, edge_y, state["tx_height"]]

        az, el = compute_initial_angles_from_position(
            tx_pos, [zone_cx, zone_cy, rx_height], verbose=False
        )
        params += [az, el, edge_x, edge_y]

        yaw_r, pitch_r = azimuth_elevation_to_yaw_pitch(az, el)
        scene.get(cfg.name).orientation = mi.Point3f(float(yaw_r), float(pitch_r), 0.0)
        scene.get(cfg.name).position    = mi.Point3f(float(edge_x), float(edge_y),
                                                      float(state["tx_height"]))
        if verbose:
            print(f"  {cfg.name}: Az={az:.1f}°  El={el:.1f}°  "
                  f"pos=({edge_x:.1f}, {edge_y:.1f})")

    params_np  = np.array(params, dtype=np.float64)
    zone_masks = _zone_masks_from_states(tx_states, map_config)

    # Ray-traced evaluation
    raytraced_loss = _evaluate_config(
        scene, tx_configs, tx_states, params_np,
        map_config, zone_masks, noise_power,
        sir_threshold_db=sir_threshold_db, sigmoid_k=sigmoid_k,
    )

    # Empirical evaluation
    rng = np.random.default_rng(42)
    zone_pts_list = [
        _sample_zone_pts_numpy(s["zone_polygon"], n_empirical_pts,
                               ground_z=rx_height, rng=rng)
        for s in tx_states
    ]
    empirical_loss = _empirical_sir_loss(
        params_np, tx_configs, tx_states, zone_pts_list, noise_power, frequency_hz,
        sir_threshold_db=sir_threshold_db, sigmoid_k=sigmoid_k,
    )

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n  Ray-traced loss : {raytraced_loss:.4f}")
        print(f"  Empirical loss  : {empirical_loss:.4f}  ({elapsed:.1f}s)")
        print(f"{'='*60}\n")

    result = _build_result(tx_configs, tx_states, params_np, [raytraced_loss],
                           params_np, elapsed, "naive_edge_center",
                           iter_time_history=[elapsed])
    result["joint"]["noise_power"]     = noise_power
    result["joint"]["empirical_loss"]  = float(empirical_loss)
    result["joint"]["raytraced_loss"]  = float(raytraced_loss)
    return result


# ---------------------------------------------------------------------------
# Baseline 2 — Empirical PSO
# ---------------------------------------------------------------------------

def empirical_pso_baseline_multi_tx(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    n_particles: int = 20,
    num_iterations: int = 30,
    noise_power: float = 1e-10,
    frequency_hz: float = 3.5e9,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    lds: str = "Halton",
    seed: Optional[int] = None,
    verbose: bool = True,
    n_eval_pts: int = 300,
    rx_height: float = 1.5,
    early_stop_window: int = 5,
    early_stop_min_improvement: float = 1e-3,
    sir_threshold_db: float = -3.0,
    sigmoid_k: float = 0.5,
) -> dict:
    """PSO driven entirely by the 3GPP UMa empirical channel model.

    No RadioMapSolver or PathSolver calls during optimisation — every function
    evaluation is pure numpy on CPU.  This demonstrates that a state-of-the-art
    empirical model cannot compete with ray-tracing because it assigns the same
    expected path gain to two points at equal distance regardless of blockage.

    Returns
    -------
    dict  (same schema as ``optimize_multi_tx``)
        Extra keys: ``joint["frequency_hz"]``, ``joint["n_eval_pts"]``
    """
    if verbose:
        print(f"\n{'='*65}")
        print(f"EMPIRICAL PSO BASELINE  ({n_particles} particles × {num_iterations} iters, "
              f"{len(tx_configs)} TX)")
        print(f"  Channel model: 3GPP TR 38.901 UMa (no ray tracing)")
        print(f"{'='*65}")

    start_time = time.time()
    rng        = np.random.default_rng(seed)
    qrand      = _make_qrand(lds)
    tx_states  = [_setup_tx_state(scene, cfg, scene_xml_path, qrand)
                  for cfg in tx_configs]

    lo, hi   = _param_bounds(tx_configs, tx_states)
    span     = hi - lo
    n_params = len(lo)

    # Pre-sample fixed evaluation points per zone
    eval_rng = np.random.default_rng(0)
    zone_pts_list = [
        _sample_zone_pts_numpy(s["zone_polygon"], n_eval_pts,
                               ground_z=rx_height, rng=eval_rng)
        for s in tx_states
    ]

    def _eval(params):
        return _empirical_sir_loss(
            params, tx_configs, tx_states, zone_pts_list,
            noise_power, frequency_hz,
            sir_threshold_db=sir_threshold_db, sigmoid_k=sigmoid_k,
        )

    positions  = rng.uniform(lo, hi, size=(n_particles, n_params))
    velocities = rng.uniform(-span * 0.1, span * 0.1, size=(n_particles, n_params))

    personal_best_pos  = positions.copy()
    personal_best_loss = np.full(n_particles, np.inf)
    global_best_pos    = positions[0].copy()
    global_best_loss   = np.inf

    loss_history      = []
    iter_time_history = []
    converged_iter    = None

    for iteration in range(num_iterations):
        iter_start = time.time()
        for k in range(n_particles):
            loss = _eval(positions[k])
            if loss < personal_best_loss[k]:
                personal_best_loss[k] = loss
                personal_best_pos[k]  = positions[k].copy()
            if loss < global_best_loss:
                global_best_loss = loss
                global_best_pos  = positions[k].copy()

        loss_history.append(float(global_best_loss))
        iter_time_history.append(time.time() - iter_start)

        if verbose:
            print(f"  Iter {iteration+1:3d}/{num_iterations}  "
                  f"global_best={global_best_loss:.4f}  ({iter_time_history[-1]:.1f}s)")

        if len(loss_history) >= early_stop_window:
            if (loss_history[-early_stop_window] - loss_history[-1]
                    < early_stop_min_improvement):
                converged_iter = iteration + 1
                if verbose:
                    print(f"  [early stop] no improvement over last "
                          f"{early_stop_window} iters at iter {converged_iter}")
                break

        r1 = rng.uniform(0.0, 1.0, size=(n_particles, n_params))
        r2 = rng.uniform(0.0, 1.0, size=(n_particles, n_params))
        velocities = (w  * velocities
                      + c1 * r1 * (personal_best_pos - positions)
                      + c2 * r2 * (global_best_pos   - positions))
        velocities = np.clip(velocities, -0.2 * span, 0.2 * span)
        positions  = np.clip(positions + velocities, lo, hi)

        stride = 4
        for k in range(n_particles):
            for i, state in enumerate(tx_states):
                b = i * stride
                px, py = state["tx_placement"].project_to_polygon(
                    positions[k, b + 2], positions[k, b + 3])
                positions[k, b + 2] = px
                positions[k, b + 3] = py

    _apply_params(scene, tx_configs, tx_states, global_best_pos)

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n  Best loss (empirical): {global_best_loss:.4f}  ({elapsed:.1f}s)")
        for i, cfg in enumerate(tx_configs):
            b = i * 4
            print(f"  {cfg.name}: Az={global_best_pos[b]:.1f}°  "
                  f"El={global_best_pos[b+1]:.1f}°  "
                  f"pos=({global_best_pos[b+2]:.1f}, {global_best_pos[b+3]:.1f})")
        print(f"{'='*65}\n")

    result = _build_result(tx_configs, tx_states, global_best_pos, loss_history,
                           _initial_params(tx_configs, tx_states), elapsed,
                           "empirical_pso_uma",
                           iter_time_history=iter_time_history,
                           converged_iter=converged_iter)
    result["joint"]["noise_power"]  = noise_power
    result["joint"]["frequency_hz"] = frequency_hz
    result["joint"]["n_eval_pts"]   = n_eval_pts
    return result


# ---------------------------------------------------------------------------
# Baseline 3 — RadioMap Gradient
# ---------------------------------------------------------------------------

def radiomap_gradient_baseline_multi_tx(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    learning_rate: float = 3.0,
    num_iterations: int = 50,
    noise_power: float = 1e-10,
    frequency_hz: float = 3.5e9,
    lds: str = "Halton",
    samples_per_tx: int = int(1e7),
    max_depth: int = 8,
    n_empirical_pts: int = 500,
    rx_height: float = 1.5,
    verbose: bool = True,
    early_stop_window: int = 10,
    early_stop_min_improvement: float = 1e-3,
    early_stop_min_flips: int = 4,
) -> dict:
    """First-order (Adam) optimisation driven by RadioMapSolver gradients.

    Ported from ``baseline_optimizers.radiomap_gradient_multi_tx`` with two
    additions that quantify its gradient pathology:

    1. **Gradient-magnitude logging** — ``joint["grad_norm_history"]`` records
       the mean absolute gradient per iteration.  Expect near-zero values due
       to the solid-angle ray-tube binning approximation.

    2. **Post-convergence dual evaluation** — after the loop, both the UMa
       empirical model and a RadioMapSolver evaluation are run on the final
       configuration:
       - ``joint["final_empirical_loss"]``
       - ``joint["final_raytraced_loss"]``

    Returns
    -------
    dict  (same schema as ``optimize_multi_tx``) with additional keys:
        joint["grad_norm_history"]    — list of per-iter mean |∇param|
        joint["final_empirical_loss"] — UMa loss at converged configuration
        joint["final_raytraced_loss"] — RadioMapSolver loss at converged config
    """
    N = len(tx_configs)
    if verbose:
        print(f"\n{'='*70}")
        print(f"RADIOMAP GRADIENT BASELINE  ({N} TX, {num_iterations} iters, "
              f"lr={learning_rate})")
        print(f"  Logging grad norms + post-convergence dual evaluation")
        print(f"{'='*70}")

    start_time    = time.time()
    qrand         = _make_qrand(lds)
    tx_states     = [_setup_tx_state(scene, cfg, scene_xml_path, qrand)
                     for cfg in tx_configs]

    zone_masks_np = _zone_masks_from_states(tx_states, map_config)
    n_zone_cells  = [max(1, int(m.sum())) for m in zone_masks_np]

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

    rm_solver           = RadioMapSolver()
    rm_solver.loop_mode = "evaluated"   # required for AD through dr.while_loop
    compute_loss = _make_radiomap_sir_loss(
        N, tx_configs, tx_states, scene, rm_solver,
        map_config, noise_power, zone_masks_np, n_zone_cells, rm_kwargs,
    )

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

    loss_history      = []
    iter_time_history = []
    grad_norm_history = []
    converged_iter    = None
    final_bufs        = {i: {"az": [], "el": []} for i in range(N)}

    for iteration in range(num_iterations):
        iter_start = time.time()

        loss = compute_loss(*params)
        loss.backward()

        grad_norm = _log_grad_norm(params)
        grad_norm_history.append(grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        dr.flush_kernel_cache()
        dr.flush_malloc_cache()

        with torch.no_grad():
            for i, state in enumerate(tx_states):
                b    = i * 4
                az_t = params[b];     el_t = params[b + 1]
                x_t  = params[b + 2]; y_t  = params[b + 3]

                az_t.clamp_(AZ_MIN, AZ_MAX)
                if az_t.item() >= 360.0:
                    az_t.fill_(az_t.item() % 360.0)
                el_t.clamp_(EL_MIN, EL_MAX)

                proj_x, proj_y = state["tx_placement"].project_to_polygon(
                    x_t.item(), y_t.item())
                x_t.data.fill_(proj_x)
                y_t.data.fill_(proj_y)

        loss_val = float(loss.item())
        loss_history.append(loss_val)
        dur = time.time() - iter_start
        iter_time_history.append(dur)

        window_start = max(0, num_iterations - 10)
        if iteration >= window_start:
            for i in range(N):
                b = i * 4
                final_bufs[i]["az"].append(float(params[b].item()))
                final_bufs[i]["el"].append(float(params[b + 1].item()))

        for i in range(N):
            b = i * 4
            tx_states[i].setdefault("az_history", []).append(float(params[b].item()))
            tx_states[i].setdefault("el_history", []).append(float(params[b + 1].item()))

        if verbose:
            print(f"  Iter {iteration+1:3d}/{num_iterations}  "
                  f"loss={loss_val:.4f}  |∇|={grad_norm:.2e}  ({dur:.1f}s)")

        if len(loss_history) >= early_stop_window:
            recent  = loss_history[-early_stop_window:]
            deltas  = [recent[k + 1] - recent[k] for k in range(len(recent) - 1)]
            n_flips = sum(1 for k in range(len(deltas) - 1)
                          if deltas[k] * deltas[k + 1] < 0)
            net_change = abs(recent[-1] - recent[0])
            if n_flips >= early_stop_min_flips and net_change < early_stop_min_improvement:
                converged_iter = iteration + 1
                if verbose:
                    print(f"  [early stop] oscillating at iter {converged_iter} "
                          f"({n_flips} sign flips, net Δloss={net_change:.4f})")
                break

    # Apply best (last-10 average) parameters back to scene
    best_params_flat = []
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
        best_params_flat += [best_az, best_el, final_x, final_y]

    best_params_np = np.array(best_params_flat, dtype=np.float64)

    # Post-convergence dual evaluation
    rng = np.random.default_rng(42)
    zone_pts_list = [
        _sample_zone_pts_numpy(s["zone_polygon"], n_empirical_pts,
                               ground_z=rx_height, rng=rng)
        for s in tx_states
    ]
    final_empirical_loss  = _empirical_sir_loss(
        best_params_np, tx_configs, tx_states,
        zone_pts_list, noise_power, frequency_hz,
    )
    zone_masks = _zone_masks_from_states(tx_states, map_config)
    final_raytraced_loss  = _evaluate_config(
        scene, tx_configs, tx_states, best_params_np,
        map_config, zone_masks, noise_power,
    )

    elapsed = time.time() - start_time

    result = {}
    for cfg, state in zip(tx_configs, tx_states):
        result[cfg.name] = {
            "best_angles":      state["best_angles"],
            "final_position":   state["final_position"],
            "initial_angles":   [state["initial_azimuth"], state["initial_elevation"]],
            "initial_position": state["tx_position"],
            "az_history":       state.get("az_history", []),
            "el_history":       state.get("el_history", []),
        }
    result["joint"] = {
        "loss_history":         loss_history,
        "iter_time_history":    iter_time_history,
        "converged_iter":       converged_iter,
        "elapsed_time_s":       elapsed,
        "num_iterations":       num_iterations,
        "noise_power":          noise_power,
        "frequency_hz":         frequency_hz,
        "sampler":              "radiomap_gradient",
        "sampling_strata":      "radiomap_zone_cells",
        "lds":                  lds,
        "grad_norm_history":    grad_norm_history,
        "final_empirical_loss": float(final_empirical_loss),
        "final_raytraced_loss": float(final_raytraced_loss),
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"RADIOMAP GRADIENT BASELINE COMPLETE  ({elapsed:.1f}s)")
        for cfg in tx_configs:
            r = result[cfg.name]
            print(f"  {cfg.name}: Az={r['best_angles'][0]:.1f}°  "
                  f"El={r['best_angles'][1]:.1f}°  "
                  f"pos=({r['final_position'][0]:.1f}, {r['final_position'][1]:.1f})")
        print(f"  Final empirical loss:   {final_empirical_loss:.4f}")
        print(f"  Final ray-traced loss:  {final_raytraced_loss:.4f}")
        print(f"{'='*70}\n")

    return result


# ---------------------------------------------------------------------------
# Baseline 4 — Dense PathSolver Gradient
# ---------------------------------------------------------------------------

def dense_pathsolver_gradient_baseline_multi_tx(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    learning_rate: float = 3.5,
    num_iterations: int = 30,
    noise_power: float = 1e-10,
    lds: str = "Halton",
    grid_spacing_m: float = 4.0,
    max_depth: int = 8,
    verbose: bool = True,
    early_stop_window: int = 10,
    early_stop_min_improvement: float = 1e-3,
    early_stop_min_flips: int = 4,
) -> dict:
    """First-order (Adam) optimisation driven by PathSolver over a dense receiver grid.

    Instead of sampling a small random set of zone points per iteration, this
    baseline uses ALL grid cells in the zone at a fixed ``grid_spacing_m``
    resolution — e.g. a 200×200 m zone at 4 m spacing yields ~2500 receivers.
    This produces very accurate gradients (exact Image Method paths) but at a
    per-iteration cost that scales quadratically with zone size.

    Per-iteration gradient norms are logged in ``joint["grad_norm_history"]``
    for direct comparison against Baseline 3 (RadioMapSolver).  Expected to be
    significantly larger than RadioMapSolver's near-zero gradients.

    Returns
    -------
    dict  (same schema as ``optimize_multi_tx``) with additional keys:
        joint["grad_norm_history"]  — list of per-iter mean |∇param|
        joint["n_receivers"]        — total receivers added to the scene
    """
    N = len(tx_configs)
    if verbose:
        print(f"\n{'='*70}")
        print(f"DENSE PATHSOLVER GRADIENT BASELINE  ({N} TX, {num_iterations} iters, "
              f"lr={learning_rate}, spacing={grid_spacing_m}m)")
        print(f"{'='*70}")

    start_time = time.time()
    qrand      = _make_qrand(lds)
    tx_states  = [_setup_tx_state(scene, cfg, scene_xml_path, qrand)
                  for cfg in tx_configs]

    # ------------------------------------------------------------------
    # Build dense receiver grid for each zone
    # ------------------------------------------------------------------
    ground_z = map_config["center"][2] if len(map_config["center"]) > 2 else 1.5

    zone_rx_pts: list[np.ndarray] = []
    for state in tx_states:
        minx, miny, maxx, maxy = state["zone_polygon"].bounds
        xs = np.arange(minx + grid_spacing_m / 2, maxx, grid_spacing_m)
        ys = np.arange(miny + grid_spacing_m / 2, maxy, grid_spacing_m)
        XX, YY = np.meshgrid(xs, ys)
        pts_2d = np.stack([XX.ravel(), YY.ravel()], axis=1)
        inside = contains(state["zone_polygon"], pts_2d[:, 0], pts_2d[:, 1])
        pts_in = pts_2d[inside]
        pts_3d = np.column_stack([pts_in, np.full(len(pts_in), ground_z + 1.5)])
        zone_rx_pts.append(pts_3d)
        if verbose:
            print(f"  Zone {tx_configs[zone_rx_pts.index(pts_3d)].name}: "
                  f"{len(pts_3d)} receivers at {grid_spacing_m}m spacing")

    # ------------------------------------------------------------------
    # Add receivers to scene (pre-created, fixed positions)
    # ------------------------------------------------------------------
    rx_objects: dict[str, Receiver] = {}
    rx_slices:  list[slice]         = []
    offset = 0
    for i, pts in enumerate(zone_rx_pts):
        sl = slice(offset, offset + len(pts))
        rx_slices.append(sl)
        for k, pos in enumerate(pts):
            rx_name = f"dense_rx_{offset + k}"
            rx = Receiver(name=rx_name,
                          position=[float(pos[0]), float(pos[1]), float(pos[2])])
            scene.add(rx)
            rx_objects[rx_name] = rx
        offset += len(pts)

    total_rx = sum(len(p) for p in zone_rx_pts)
    if verbose:
        print(f"  Total receivers in scene: {total_rx}")

    # ------------------------------------------------------------------
    # Build differentiable loss with @dr.wrap
    # ------------------------------------------------------------------
    p_solver = PathSolver()
    p_solver.loop_mode = "evaluated"

    N_params_per_tx = 4
    arg_names = [f"p{i * N_params_per_tx + j}"
                 for i in range(N) for j in range(N_params_per_tx)]
    arg_str  = ", ".join(arg_names)
    list_str = "[" + arg_str + "]"

    def _loss_body(all_params):
        deg2rad = Float(float(np.pi / 180.0))
        dr.disable_grad(deg2rad)

        for i, cfg in enumerate(tx_configs):
            b    = i * N_params_per_tx
            az_i = all_params[b].array;     dr.enable_grad(az_i)
            el_i = all_params[b + 1].array; dr.enable_grad(el_i)
            x_i  = all_params[b + 2].array; dr.enable_grad(x_i)
            y_i  = all_params[b + 3].array; dr.enable_grad(y_i)

            yaw   = az_i * deg2rad
            pitch = -(el_i * deg2rad)
            roll  = Float(0.0); dr.disable_grad(roll)

            scene.get(cfg.name).orientation = [yaw, pitch, roll]
            scene.get(cfg.name).position    = [
                x_i, y_i,
                Float(float(tx_states[i]["tx_height"])),
            ]

        paths  = p_solver(scene, los=True, refraction=False,
                          specular_reflection=True, diffuse_reflection=False,
                          diffraction=True)
        h_real, h_imag = paths.a
        # h shape: (total_rx, N_tx, max_paths, rx_ant, tx_ant)
        pwr = cpx_abs_square(h_real, h_imag)   # sum over paths
        # pwr shape: (total_rx, N_tx, rx_ant, tx_ant) → reduce to (total_rx, N_tx)
        pwr_per_rx_tx = dr.sum(pwr, axis=(2, 3)) if pwr.ndim > 2 else pwr

        noise_f = Float(float(noise_power)); dr.disable_grad(noise_f)
        total_loss = Float(0.0); dr.disable_grad(total_loss)

        zone_areas   = [s["zone_polygon"].area for s in tx_states]
        total_area   = sum(zone_areas) or 1.0
        area_weights = [a / total_area for a in zone_areas]

        for i in range(N):
            sl    = rx_slices[i]
            n_rx_i = sl.stop - sl.start
            sig   = pwr_per_rx_tx[sl.start:sl.stop, i]
            interf = dr.zeros(Float, n_rx_i)
            for j in range(N):
                if j != i:
                    interf = interf + pwr_per_rx_tx[sl.start:sl.stop, j]

            metric   = (sig + Float(1e-30)) / (interf + noise_f + Float(1e-30))
            log_m    = dr.log(metric)
            loss_i   = -Float(float(area_weights[i])) * dr.sum(log_m) / Float(float(n_rx_i))
            total_loss = total_loss + loss_i

        return total_loss

    func_code = (
        f"def _inner({arg_str}):\n"
        f"    return _body({list_str})\n"
    )
    globs = {"_body": _loss_body}
    exec(func_code, globs)
    compute_loss = dr.wrap(source="torch", target="drjit")(globs["_inner"])

    # ------------------------------------------------------------------
    # PyTorch params + Adam
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

    loss_history      = []
    iter_time_history = []
    grad_norm_history = []
    converged_iter    = None
    final_bufs        = {i: {"az": [], "el": []} for i in range(N)}

    for iteration in range(num_iterations):
        iter_start = time.time()

        loss = compute_loss(*params)
        loss.backward()

        grad_norm = _log_grad_norm(params)
        grad_norm_history.append(grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        dr.flush_kernel_cache()
        dr.flush_malloc_cache()

        with torch.no_grad():
            for i, state in enumerate(tx_states):
                b    = i * 4
                az_t = params[b];     el_t = params[b + 1]
                x_t  = params[b + 2]; y_t  = params[b + 3]

                az_t.clamp_(AZ_MIN, AZ_MAX)
                if az_t.item() >= 360.0:
                    az_t.fill_(az_t.item() % 360.0)
                el_t.clamp_(EL_MIN, EL_MAX)

                proj_x, proj_y = state["tx_placement"].project_to_polygon(
                    x_t.item(), y_t.item())
                x_t.data.fill_(proj_x)
                y_t.data.fill_(proj_y)

        loss_val = float(loss.item())
        loss_history.append(loss_val)
        dur = time.time() - iter_start
        iter_time_history.append(dur)

        window_start = max(0, num_iterations - 10)
        if iteration >= window_start:
            for i in range(N):
                b = i * 4
                final_bufs[i]["az"].append(float(params[b].item()))
                final_bufs[i]["el"].append(float(params[b + 1].item()))

        for i in range(N):
            b = i * 4
            tx_states[i].setdefault("az_history", []).append(float(params[b].item()))
            tx_states[i].setdefault("el_history", []).append(float(params[b + 1].item()))

        if verbose:
            print(f"  Iter {iteration+1:3d}/{num_iterations}  "
                  f"loss={loss_val:.4f}  |∇|={grad_norm:.2e}  ({dur:.1f}s)")

        if len(loss_history) >= early_stop_window:
            recent  = loss_history[-early_stop_window:]
            deltas  = [recent[k + 1] - recent[k] for k in range(len(recent) - 1)]
            n_flips = sum(1 for k in range(len(deltas) - 1)
                          if deltas[k] * deltas[k + 1] < 0)
            net_change = abs(recent[-1] - recent[0])
            if n_flips >= early_stop_min_flips and net_change < early_stop_min_improvement:
                converged_iter = iteration + 1
                if verbose:
                    print(f"  [early stop] oscillating at iter {converged_iter} "
                          f"({n_flips} sign flips, net Δloss={net_change:.4f})")
                break

    # Remove dense receivers from scene
    for rx_name in rx_objects:
        try:
            scene.remove(rx_name)
        except Exception:
            pass
    gc.collect()

    # Apply best params to scene
    best_params_flat = []
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
        best_params_flat += [best_az, best_el, final_x, final_y]

    elapsed = time.time() - start_time

    result = {}
    for cfg, state in zip(tx_configs, tx_states):
        result[cfg.name] = {
            "best_angles":      state["best_angles"],
            "final_position":   state["final_position"],
            "initial_angles":   [state["initial_azimuth"], state["initial_elevation"]],
            "initial_position": state["tx_position"],
            "az_history":       state.get("az_history", []),
            "el_history":       state.get("el_history", []),
        }
    result["joint"] = {
        "loss_history":      loss_history,
        "iter_time_history": iter_time_history,
        "converged_iter":    converged_iter,
        "elapsed_time_s":    elapsed,
        "num_iterations":    num_iterations,
        "noise_power":       noise_power,
        "sampler":           "dense_pathsolver_gradient",
        "sampling_strata":   f"dense_grid_{grid_spacing_m}m",
        "lds":               lds,
        "grad_norm_history": grad_norm_history,
        "n_receivers":       total_rx,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"DENSE PATHSOLVER GRADIENT BASELINE COMPLETE  ({elapsed:.1f}s)")
        for cfg in tx_configs:
            r = result[cfg.name]
            print(f"  {cfg.name}: Az={r['best_angles'][0]:.1f}°  "
                  f"El={r['best_angles'][1]:.1f}°  "
                  f"pos=({r['final_position'][0]:.1f}, {r['final_position'][1]:.1f})")
        print(f"  Mean iter time: {np.mean(iter_time_history):.1f}s")
        print(f"{'='*70}\n")

    return result

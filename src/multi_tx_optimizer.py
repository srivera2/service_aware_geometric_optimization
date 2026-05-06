"""
multi_tx_optimizer.py
=====================
Multi-transmitter SIR optimization for Sionna RT.

Jointly optimizes N transmitter boresight angles, positions, and optionally
transmit powers to maximise the Signal-to-Interference Ratio (SIR) in each
transmitter's target coverage zone.

Public API
----------
TxConfig                       : dataclass describing one transmitter
optimize_multi_tx()            : run the joint optimization
compare_multi_tx_performance() : evaluate optimised BS-only SINR containment within target zone
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional
import time

import alphashape
import matplotlib.pyplot as plt
import mitsuba as mi
from IPython.display import display
import numpy as np
import scipy.stats.qmc
import shapely
import shapely.ops
import torch
import drjit as dr
import random
from drjit.auto import Float
from shapely.affinity import scale as shapely_scale
from shapely.geometry import Polygon as ShapelyPolygon
from sklearn.cluster import HDBSCAN, DBSCAN
from sionna.rt import load_scene, PathSolver, RadioMapSolver, Transmitter, Receiver, cpx_abs_square, AntennaArray

from angle_utils import (
    azimuth_elevation_to_yaw_pitch,
    compute_initial_angles_from_position,
)
from boresight_pathsolver import filter_and_append, sample_grid_points
from triangulate import (
    get_zone_polygon_with_exclusions,
    triangulate_zone,
    prepare_triangulated_sampler,
    sample_from_prepared,
    sample_triangulated_zone,
    sample_dead_zones,
    visualize_triangulation
)
from tx_placement import TxPlacement

# Initialize random number generator
random.seed()

# ---------------------------------------------------------------------------
# TxConfig
# ---------------------------------------------------------------------------

@dataclass
class TxConfig:
    """Configuration for one transmitter in the joint optimisation.

    Parameters
    ----------
    name : str
        Sionna scene name (must already be added to the scene, e.g. "gnb", "gnb2").
    building_id : int
        Building ID from extract_building_info(); TX position is constrained to
        this building's rooftop polygon.
    zone_params : dict
        Coverage-zone description in one of two formats:
            {'center': [x, y], 'width': w, 'height': h}   # box
            {'vertices': [(x1,y1), ...]}                   # polygon
    tx_height_offset : float
        Metres above the rooftop surface (default 10 m).
    optimize_power : bool
        If True, tx.power_dbm is an optimisable parameter.
    initial_power_dbm : float or None
        Override initial power (dBm). None -> read from scene TX.
    power_dbm_bounds : tuple[float, float]
        (min, max) dBm clamp range for power optimisation.
    initial_azimuth_deg : float or None
        Override initial azimuth; None -> auto-compute from TX -> zone centroid.
    initial_elevation_deg : float or None
        Override initial elevation; None -> auto-compute.
    """
    name: str
    on_building: bool
    building_id: int
    zone_params: dict
    tx_height_offset: float = 10.0
    optimize_power: bool = False
    initial_power_dbm: Optional[float] = None
    power_dbm_bounds: tuple = (0.0, 50.0)
    initial_azimuth_deg: Optional[float] = None
    initial_elevation_deg: Optional[float] = None


@dataclass
class JammerConfig:
    """Configuration for one friendly jammer in the joint optimisation.

    Parameters
    ----------
    name : str
        Unique name registered in jam_scene (e.g. "jam_0").
    initial_position : list[float] or None
        [x, y] starting position in metres.  None -> sampled uniformly from
        the map bounds at startup.
    initial_power_dbm : float
        Starting transmit power [dBm].
    power_dbm_bounds : tuple[float, float]
        (min, max) dBm clamp applied after each gradient step.
    """
    name: str
    initial_position: Optional[list] = None   # [x, y]; None = random
    initial_power_dbm: float = 23.0
    power_dbm_bounds: tuple = (0.0, 40.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_qrand(lds: str):
    """Return a scipy QMC sampler (or None for pure uniform)."""
    if lds == "Sobol":
        return scipy.stats.qmc.Sobol(d=3, scramble=False, seed=None)
    elif lds == "Halton":
        return scipy.stats.qmc.Halton(d=3, scramble=False, seed=None)
    elif lds == "Latin":
        return scipy.stats.qmc.LatinHypercube(d=3, scramble=False, seed=None)
    elif lds == "Uniform":
        return None
    else:
        warnings.warn(f"Unknown LDS '{lds}'. Falling back to Halton.")
        return scipy.stats.qmc.Halton(d=3, scramble=False, seed=None)


def _setup_tx_state(scene, cfg: TxConfig, scene_xml_path: str, qrand) -> dict:
    """Initialise geometry and sampling infrastructure for one TX."""
    tx = scene.get(cfg.name)

    tx_x = float(dr.detach(tx.position[0])[0])
    tx_y = float(dr.detach(tx.position[1])[0])
    tx_z = float(dr.detach(tx.position[2])[0])
    tx_position = [tx_x, tx_y, tx_z]
    tx_power_dbm = float(tx.power_dbm[0])

    if cfg.on_building:
        tx_placement = TxPlacement(
            scene, cfg.name, scene_xml_path, cfg.building_id, create_if_missing=False
        )
    else:
        tx_placement = TxPlacement(
            scene, cfg.name, scene_xml_path, None, create_if_missing=False
        )

    # Zone geometry --------------------------------------------------------
    zone_type = "polygon" if "vertices" in cfg.zone_params else "box"
    target_zone, building_exclusions, _ = get_zone_polygon_with_exclusions(
        zone_type=zone_type,
        zone_params=cfg.zone_params,
        scene_xml_path=scene_xml_path,
        exclude_buildings=True,
    )

    if zone_type == "box":
        bx, by = cfg.zone_params["center"][0], cfg.zone_params["center"][1]
        bw, bh = cfg.zone_params["width"], cfg.zone_params["height"]
        box_polygon = ShapelyPolygon([
            (bx - bw / 2, by - bh / 2), (bx + bw / 2, by - bh / 2),
            (bx + bw / 2, by + bh / 2), (bx - bw / 2, by + bh / 2),
        ])
    else:
        box_polygon = ShapelyPolygon(cfg.zone_params["vertices"])

    cached_building_polygons = []
    for bcoords in building_exclusions:
        try:
            p = ShapelyPolygon(bcoords)
            if p.is_valid:
                cached_building_polygons.append(p)
        except Exception:
            pass

    zone_polygon = (
        box_polygon.difference(shapely.ops.unary_union(cached_building_polygons))
        if cached_building_polygons else box_polygon
    )

    tri_verts_full, _ = triangulate_zone(target_zone, building_exclusions)
    tri_full_prepared = prepare_triangulated_sampler(tri_verts_full)

    # Initial angles -------------------------------------------------------
    target_z = 1.5
    if "center" in cfg.zone_params:
        look_at_xyz = list(cfg.zone_params["center"])[:2] + [target_z]
    else:
        c = box_polygon.centroid
        look_at_xyz = [c.x, c.y, target_z]

    if cfg.initial_azimuth_deg is not None and cfg.initial_elevation_deg is not None:
        initial_azimuth = cfg.initial_azimuth_deg
        initial_elevation = cfg.initial_elevation_deg
    else:
        initial_azimuth, initial_elevation = compute_initial_angles_from_position(
            tx_position, look_at_xyz, verbose=False
        )

    initial_power_dbm = cfg.initial_power_dbm if cfg.initial_power_dbm is not None else tx_power_dbm

    return {
        "tx_placement":           tx_placement,
        "tx_position":            tx_position,
        "tx_height":              tx_z,
        "tx_ref_power_dbm":       tx_power_dbm,
        "box_polygon":            box_polygon,
        "zone_polygon":           zone_polygon,
        "target_zone":            target_zone,
        "building_exclusions":    building_exclusions,
        "cached_building_polygons": cached_building_polygons,
        "tri_verts_full":         tri_verts_full,
        "tri_full_prepared":      tri_full_prepared,
        "qrand":                  qrand,
        "dead_zones":             [],
        "dead_buffs":             [],
        "dead_points":            np.zeros((0, 3)),
        "initial_azimuth":        initial_azimuth,
        "initial_elevation":      initial_elevation,
        "initial_power_dbm":      initial_power_dbm,
        "rx_slice":               None,   # filled after all states created
        "rx_indices_drjit":       None,   # filled after all states created
        "az_history":             [],
        "el_history":             [],
        "power_history":          [],
        "current_sample_points":  None,
        "_alive_tri_cache_key":   None,
        "_alive_tri_verts":       None,
        "_alive_tri_prepared":    None,
    }


def _param_strides(tx_configs):
    """Return (strides, offsets) for the gNB flat parameter list."""
    strides = [5 if cfg.optimize_power else 4 for cfg in tx_configs]
    offsets = [sum(strides[:k]) for k in range(len(strides))]
    return strides, offsets


def _jam_param_strides(jam_configs):
    """Return (strides, offsets) for the jammer flat parameter list.

    Each jammer always contributes exactly 3 params: [x, y, power_dbm].
    """
    strides = [3] * len(jam_configs)
    offsets = [sum(strides[:k]) for k in range(len(strides))]
    return strides, offsets


def _sample_zone_points(state: dict, cfg: TxConfig, n: int,
                        sampler: str, sampling_strata: str,
                        ground_z: float) -> np.ndarray:
    """Sample n receiver points in this TX's coverage zone.

    sampler         : "triangulated" | "rejection"
    sampling_strata : "dead_only"    | "proportional"   | "full"

    In "proportional" and "dead_only" modes the alive zone and each dead zone
    stratum are always sampled independently.  Each dead zone stratum receives
    a point budget proportional to its area (largest-remainder allocation).
    The "full" strata must be chosen explicitly; it never fires as a fallback.
    """
    dead_zones     = state["dead_zones"]
    zone_polygon   = state["zone_polygon"]
    tri_verts_full = state["tri_verts_full"]
    qrand          = state["qrand"]
    cached_bldgs   = state["cached_building_polygons"]

    def _sample_full_zone(n_pts):
        if sampler == "rejection":
            pts, *_ = sample_grid_points(
                zone_polygon, n_pts, qrand,
                building_polygons=cached_bldgs, ground_z=ground_z,
            )
        else:
            pts = sample_from_prepared(state["tri_full_prepared"], n_pts, ground_z=ground_z)
        return pts

    def _sample_dead_strata(n_pts):
        """Sample n_pts across dead zone strata, each stratum independently.

        Point budget per stratum is proportional to area (largest-remainder).
        Returns an (m, 3) array or None if all strata fail.
        """
        areas = [dz.area for dz in dead_zones]
        total = sum(areas) or 1.0
        exact = [n_pts * a / total for a in areas]
        floors = [int(e) for e in exact]
        remainder = n_pts - sum(floors)
        order = sorted(range(len(dead_zones)), key=lambda i: -(exact[i] - floors[i]))
        for i in order[:remainder]:
            floors[i] += 1

        if sampler == "rejection":
            parts = []
            for dz, k in zip(dead_zones, floors):
                if k > 0:
                    pts, *_ = sample_grid_points(
                        dz, k, qrand,
                        building_polygons=cached_bldgs, ground_z=ground_z,
                    )
                    parts.append(pts)
            return np.vstack(parts) if parts else None
        else:
            # sample_dead_zones triangulates each stratum and allocates
            # proportionally by area — pass the pre-computed allocations
            # by sampling each stratum independently and stacking.
            parts = []
            for dz, k in zip(dead_zones, floors):
                if k > 0:
                    pts = sample_dead_zones([dz], k, building_exclusions=[state["building_exclusions"]])
                    if pts is not None and len(pts) > 0:
                        pts[:, 2] = ground_z
                        parts.append(pts)
            return np.vstack(parts) if parts else None

    if sampling_strata == "proportional":
        if not dead_zones:
            # No dead zones — entire zone is alive
            return _sample_full_zone(n)[:n]

        dead_union = shapely.ops.unary_union(dead_zones)
        alive_zone = zone_polygon.difference(dead_union)
        if alive_zone.is_empty:
            alive_zone = zone_polygon

        alive_area = alive_zone.area
        dead_area  = dead_union.intersection(zone_polygon).area
        total_area = alive_area + dead_area or 1.0
        n_alive = max(1, round(n * alive_area / total_area))
        n_dead  = max(0, n - n_alive)

        # Sample alive zone independently
        if sampler == "rejection":
            alive_pts, *_ = sample_grid_points(
                alive_zone, n_alive, qrand,
                building_polygons=cached_bldgs, ground_z=ground_z,
            )
        else:
            cache_key = id(dead_zones)
            if state["_alive_tri_cache_key"] != cache_key:
                try:
                    alive_tri, _ = triangulate_zone(
                        alive_zone, state["building_exclusions"]
                    )
                    state["_alive_tri_verts"]    = alive_tri
                    state["_alive_tri_prepared"] = prepare_triangulated_sampler(alive_tri)
                    state["_alive_tri_cache_key"] = cache_key
                except Exception as e:
                    warnings.warn(
                        f"alive-zone triangulation failed, falling back to full zone: {e}"
                    )
                    state["_alive_tri_verts"]    = tri_verts_full
                    state["_alive_tri_prepared"] = state["tri_full_prepared"]
                    state["_alive_tri_cache_key"] = cache_key
            alive_pts = sample_from_prepared(
                state["_alive_tri_prepared"], n_alive, ground_z=ground_z
            )

        # Sample each dead zone stratum independently
        if n_dead > 0:
            dead_pts = _sample_dead_strata(n_dead)
            if dead_pts is not None and len(dead_pts) > 0:
                return np.vstack([alive_pts, dead_pts[:n_dead]])[:n]

        return alive_pts[:n]

    elif sampling_strata == "dead_only":
        if dead_zones:
            pts = _sample_dead_strata(n)
            if pts is not None and len(pts) > 0:
                return pts[:n]
        return np.zeros((0, 3), dtype=np.float32)

    elif sampling_strata == "full":
        return _sample_full_zone(n)[:n]

    else:
        raise ValueError(f"Unknown sampling_strata: {sampling_strata!r}")


def _sample_outside_zone(state: dict, n: int, ground_z: float) -> np.ndarray:
    """Sample n points in the ring between the zone and its 2x-scaled version.

    Scale the raw box polygon (no building exclusions applied yet), subtract the
    original box, then punch out building footprints from the ring so that the
    outer region geometry is consistent with the inside zone.
    """
    from shapely import contains_xy as _cxy
    import shapely.ops

    box_poly = state["box_polygon"]
    centroid = box_poly.centroid
    outer_poly = shapely_scale(box_poly, xfact=2.0, yfact=2.0, origin=centroid)
    outer_ring = outer_poly.difference(box_poly)

    cached_bldgs = state.get("cached_building_polygons", [])
    if cached_bldgs:
        bldg_union = shapely.ops.unary_union(cached_bldgs)
        outer_ring = outer_ring.difference(bldg_union)

    minx, miny, maxx, maxy = outer_ring.bounds
    rng = np.random.default_rng()
    pts: list = []
    while len(pts) < n:
        batch = max((n - len(pts)) * 4, 512)
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        mask = _cxy(outer_ring, xs, ys)
        for x, y in zip(xs[mask], ys[mask]):
            pts.append([x, y, ground_z])
            if len(pts) >= n:
                break
    return np.array(pts[:n], dtype=np.float32)


def _accumulate_dead_zones(
    state, cfg, tx_idx, rm,
    map_config, dead_tail_percentile, max_dbscan_points,
    debug_viz=False, iteration=0
):
    """Rebuild dead-zone polygons for one TX using cell-aggregated RSS.

    Uses rm.rss[tx_idx] (shape H×W, watts) so multipath fades average out
    across the cell and only spatially coherent weak areas are identified.
    """
    from shapely import contains_xy as _cxy

    state["dead_points"] = np.zeros((0, 3))
    state["dead_zones"]  = []
    state["dead_buffs"]  = []

    # Per-TX cell-aggregated RSS (watts), shape (H, W)
    rss_np = rm.rss.numpy()[tx_idx]
    #print(f"RSS Size from RMS: {rss_np}")

    # Compute 1 m cell centres from map_config geometry
    cx, cy = map_config["center"][0], map_config["center"][1]
    #print(f"Center Values: {cx}, {cy}")
    sx, sy = map_config["size"][0],   map_config["size"][1]
    #print(f"Zone Size Values: {sx}, {sy}")
    H, W   = rss_np.shape
    xs = cx - sx / 2.0 + sx / W * (np.arange(W) + 0.5)
    #print(f"Grid Axis X: {xs}")
    ys = cy - sy / 2.0 + sy / H * (np.arange(H) + 0.5)
    #print(f"Grid Axis Y: {ys}")
    xx, yy = np.meshgrid(xs, ys)

    pos_x = xx.ravel()
    pos_y = yy.ravel()
    power = rss_np.ravel()

    # Zone Check
    in_zone  = _cxy(state["zone_polygon"], pos_x, pos_y)
    pos_x_z  = pos_x[in_zone]
    pos_y_z  = pos_y[in_zone]
    power_z  = power[in_zone]

    if len(pos_x_z) == 0:
        return
    
    # Fill rx_data 
    rx_data  = np.column_stack([pos_x_z, pos_y_z, power_z])
    dead_pts = filter_and_append(rx_data, state["dead_points"], dead_tail_percentile)

    if debug_viz:
        import matplotlib.pyplot as _plt
        import os
        os.makedirs("debug_viz", exist_ok=True)

        fig, ax = _plt.subplots(figsize=(8, 8))

        log_pwr_z = np.log10(power_z + 1e-30)
        sc = ax.scatter(pos_x_z, pos_y_z, c=log_pwr_z,
                        cmap="viridis", s=4.0, alpha=0.8, rasterized=True)

        # Overlay dead-tail boundary
        if len(dead_pts):
            thresh_log = np.log10(dead_pts[:, 2].max() + 1e-30)
            ax.scatter(dead_pts[:, 0], dead_pts[:, 1],
                       edgecolors="black", facecolors="none",
                       s=6.0, linewidths=0.3, alpha=0.6, rasterized=True,
                       label=f"dead tail ({dead_tail_percentile}th %ile)")

        zone_geoms = (list(state["zone_polygon"].geoms)
                      if state["zone_polygon"].geom_type == "MultiPolygon"
                      else [state["zone_polygon"]])
        for gi, geom in enumerate(zone_geoms):
            ax.plot(*geom.exterior.xy, "w-", linewidth=1.5,
                    label="zone" if gi == 0 else None)

        _plt.colorbar(sc, ax=ax, label="log₁₀(RSS) — all in-zone cells")
        ax.set_title(f"{cfg.name} (tx_idx={tx_idx}) — iter {iteration} "
                     f"— {len(dead_pts)} dead cells / {int(in_zone.sum())} in-zone")
        ax.set_aspect("equal")
        out_path = os.path.join("debug_viz", f"{cfg.name}_iter{iteration:03d}.png")
        fig.savefig(out_path, dpi=120)
        _plt.show()
        _plt.close(fig)
        print(f"[debug_viz] {out_path}")

    if len(dead_pts) == 0:
        return

    if len(dead_pts) > max_dbscan_points:
        rng      = np.random.default_rng(42)
        dead_pts = dead_pts[rng.choice(len(dead_pts), max_dbscan_points, replace=False)]

    #clusters      = HDBSCAN(min_samples=5, copy=False).fit(dead_pts[:, :2])
    
    clusters      = DBSCAN(min_samples=5, eps=5.0).fit(dead_pts[:, :2])
    labels        = clusters.labels_
    
    unique_labels = set(labels) - {-1}

    for cid in sorted(unique_labels):
        pts = dead_pts[labels == cid, :2]
        if len(pts) < 3:
            continue
        # Delaunay (used by alphashape) lifts 2-D points onto a 3-D paraboloid;
        # if the points are (nearly) collinear the lifted simplex is flat and
        # Qhull raises QH6013/QH6154.  Check via the smaller singular value of
        # the centred point matrix — if it's essentially zero the cluster is
        # degenerate and contributes nothing useful as a dead-zone polygon.
        centered = pts - pts.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        if s[-1] < 1e-6:
            continue

        shape = alphashape.alphashape(pts, alpha=0.01)
        shape = shapely.make_valid(shape)
        state["dead_zones"].append(shape)
    
    state["dead_buffs"] = state["dead_zones"]
    state["dead_points"] = dead_pts

    if debug_viz and len(state["dead_zones"]) > 0:
        _visualize_dead_zone_pipeline(
            dead_pts, labels, state["dead_zones"],
            state["zone_polygon"], cfg.name, iteration,
            building_exclusions=state["building_exclusions"],
        )


def _visualize_dead_zone_pipeline(
    dead_pts, labels, dead_zones, zone_polygon,
    tx_name, iteration, building_exclusions=None,
):
    """Three-panel figure showing the dead-zone processing pipeline:
    cluster labels → alphashapes → triangulated strata.
    """
    from IPython.display import display
    from matplotlib.collections import PolyCollection

    n_clusters = len(dead_zones)
    cmap = plt.cm.get_cmap("tab10" if n_clusters <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(max(n_clusters, 1))]

    unique_cluster_ids = sorted(set(labels) - {-1})
    label_to_color = {cid: colors[i % len(colors)] for i, cid in enumerate(unique_cluster_ids)}

    # Pre-build building polygons once for drawing on all panels
    bldg_polys = []
    if building_exclusions:
        for bcoords in building_exclusions:
            try:
                p = ShapelyPolygon(bcoords)
                if p.is_valid and not p.is_empty:
                    bldg_polys.append(p)
            except Exception:
                pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def _draw_zone(ax):
        geoms = (list(zone_polygon.geoms)
                 if zone_polygon.geom_type == "MultiPolygon"
                 else [zone_polygon])
        for geom in geoms:
            ax.plot(*geom.exterior.xy, "k-", linewidth=1.5)
            # Interior rings are buildings fully inside the zone — draw them too
            for interior in geom.interiors:
                ix, iy = interior.xy
                ax.fill(ix, iy, color="#999999", alpha=0.7, zorder=2)
                ax.plot(ix, iy, "k-", linewidth=0.8, zorder=3)
        # Buildings touching the boundary become MultiPolygon splits rather than
        # interior rings, so draw all building footprints directly to catch those.
        for bp in bldg_polys:
            bx, by = bp.exterior.xy
            ax.fill(bx, by, color="#999999", alpha=0.7, zorder=2)
            ax.plot(bx, by, "k-", linewidth=0.8, zorder=3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # ── Panel 1: DBSCAN cluster labels ────────────────────────────────────
    ax = axes[0]
    _draw_zone(ax)
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(dead_pts[noise_mask, 0], dead_pts[noise_mask, 1],
                   c="lightgray", s=2, alpha=0.4, rasterized=True, label="noise")
    for cid in unique_cluster_ids:
        mask = labels == cid
        ax.scatter(dead_pts[mask, 0], dead_pts[mask, 1],
                   color=label_to_color[cid], s=3, alpha=0.7,
                   rasterized=True, label=f"cluster {cid}")
    ax.set_title("1. DBSCAN Cluster Labels")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    if n_clusters <= 10:
        ax.legend(markerscale=3, fontsize=7, loc="best")

    # ── Panel 2: alphashapes ──────────────────────────────────────────────
    ax = axes[1]
    _draw_zone(ax)
    for i, shape in enumerate(dead_zones):
        color = colors[i % len(colors)]
        geoms = (list(shape.geoms)
                 if shape.geom_type in ("MultiPolygon", "GeometryCollection")
                 else [shape])
        for geom in geoms:
            if geom.geom_type != "Polygon":
                continue
            x, y = geom.exterior.xy
            ax.fill(x, y, color=color, alpha=0.45)
            ax.plot(x, y, color=color, linewidth=1.0)
    ax.set_title("2. Alphashape Strata")
    ax.set_xlabel("X (m)")

    # ── Panel 3: triangulated strata ─────────────────────────────────────
    ax = axes[2]
    _draw_zone(ax)
    excl = building_exclusions or []
    for i, shape in enumerate(dead_zones):
        color = colors[i % len(colors)]
        geoms = (list(shape.geoms)
                 if shape.geom_type in ("MultiPolygon", "GeometryCollection")
                 else [shape])
        for geom in geoms:
            if geom.geom_type != "Polygon" or geom.is_empty:
                continue
            boundary = list(geom.exterior.coords)[:-1]
            try:
                tv, _ = triangulate_zone(boundary, excl)
            except Exception:
                continue
            if len(tv) == 0:
                continue
            fc = (*color[:3], 0.4)
            ec = (*color[:3], 0.9)
            ax.add_collection(PolyCollection(tv, facecolor=fc, edgecolor=ec, linewidth=0.4))
    ax.autoscale_view()
    ax.set_title("3. Triangulated Strata")
    ax.set_xlabel("X (m)")

    plt.tight_layout()
    display(fig)
    plt.close(fig)


def _extract_per_rx_power(h_real, h_imag, tx_idx: int):
    """Return a TensorXf of shape (num_rx,) with power from tx_idx only.

    Stays in TensorXf throughout to preserve DrJit AD gradient tracking.
    Calling .array on a TensorXf returns drjit.cuda.Float (non-AD), which
    severs the gradient chain — so we never do that conversion here.

    h_real shape: (num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths) - DrJit TensorXf
    """
    hr = h_real[:, :, tx_idx:tx_idx + 1, :, :]
    hi = h_imag[:, :, tx_idx:tx_idx + 1, :, :]
    # Collapse num_paths, num_tx_ant, num_tx=1, num_rx_ant  →  shape (num_rx,)
    return dr.sum(dr.sum(dr.sum(dr.sum(cpx_abs_square((hr, hi)), axis=-1), axis=-1), axis=-1), axis=-1)


def _sir_loss_body(
    all_params, N, tx_configs, tx_states, scene, p_solver,
    noise_power, rx_objects,
    ref_powers_dbm, jam_configs, jam_scene, jam_rx_objects, jam_objects,
    num_inside=None,
    boundary_mask=None,
    gamma_db=0.0,
    sigma_db=10.0,
    lambda_in=0.0,
    lambda_out=10.0,
    lambda_sharp=0.0,
    lambda_pwr=7.0,
    epsilon=1e-30,
):
    """Containment SIR loss with hinge penalties and boundary sharpening.

    Loss components (all in dB-space):
      L_in    : squared hinge on (gamma - SINR) for inside cells
                -> penalizes service holes
      L_out   : squared hinge on (SINR - gamma) for outside cells
                -> penalizes detectable leakage
      L_sharp : Gaussian bump centered at gamma over boundary-shell cells
                -> penalizes fuzzy threshold crossings
      L_pwr   : L2 on jammer power scale factors
                -> discourages spending jammer power unnecessarily

    Parameters
    ----------
    boundary_mask : 1-D bool/int array of length total_rx, or None
        Marks receivers lying within delta of the zone boundary. If None,
        the sharpness term is skipped.
    gamma_db : float
        Detection / decode threshold in dB. Set from threat model.
    sigma_db : float
        Width of the boundary-fuzziness Gaussian, in dB.
    lambda_in, lambda_out, lambda_sharp, lambda_pwr : float
        Loss-term weights. Sweep lambda_out / lambda_in for the
        leakage / service-hole Pareto frontier.
    """
    deg2rad = Float(float(np.pi / 180.0))
    dr.disable_grad(deg2rad)

    _, offsets = _param_strides(tx_configs)

    # ------------------------------------------------------------------
    # Build power-scale factors for each gNB (DrJit-differentiable)
    # ------------------------------------------------------------------
    pow_scales = []
    for i, cfg in enumerate(tx_configs):
        if cfg.optimize_power:
            pow_i = all_params[offsets[i] + 4]
            dr.enable_grad(pow_i.array)
            scale_i = dr.power(
                Float(10.0),
                (pow_i - Float(float(ref_powers_dbm[i]))) / Float(10.0),
            )
            pow_scales.append(scale_i)
        else:
            c = Float(1.0)
            dr.disable_grad(c)
            pow_scales.append(c)

    # ------------------------------------------------------------------
    # Set TX orientations and positions (independent jitter per axis)
    # ------------------------------------------------------------------
    for i, cfg in enumerate(tx_configs):
        b    = offsets[i]
        az_i = all_params[b];     dr.enable_grad(az_i.array)
        el_i = all_params[b + 1]; dr.enable_grad(el_i.array)
        x_i  = all_params[b + 2]; dr.enable_grad(x_i.array)
        y_i  = all_params[b + 3]; dr.enable_grad(y_i.array)

        # Independent samples per DOF (previously a single scalar reused)
        jit_yaw   = Float(float(np.random.normal(0.0, 0.5 * np.pi / 180.0)))
        jit_pitch = Float(float(np.random.normal(0.0, 0.5 * np.pi / 180.0)))
        jit_x     = Float(float(np.random.normal(0.0, 0.1)))
        jit_y     = Float(float(np.random.normal(0.0, 0.1)))
        for j in (jit_yaw, jit_pitch, jit_x, jit_y):
            dr.disable_grad(j)

        yaw   = az_i * deg2rad + jit_yaw
        pitch = -(el_i * deg2rad) + jit_pitch
        roll  = Float(0.0); dr.disable_grad(roll)

        scene.get(cfg.name).orientation = [yaw, pitch, roll]
        scene.get(cfg.name).position = [
            x_i + jit_x,
            y_i + jit_y,
            Float(float(tx_states[i]["tx_height"])),
        ]

    # ------------------------------------------------------------------
    # gNB PathSolver call (directional pattern, scene)
    # ------------------------------------------------------------------
    paths = p_solver(
        scene,
        los=True,
        refraction=False,
        specular_reflection=True,
        diffuse_reflection=True,
    )
    h_real, h_imag = paths.a

    tx_power_vecs = []
    for j in range(N):
        p_raw = _extract_per_rx_power(h_real, h_imag, j)
        tx_power_vecs.append(p_raw * pow_scales[j])

    dr.eval(*tx_power_vecs)
    del paths, h_real, h_imag

    # ------------------------------------------------------------------
    # Compute total gNB parameter count correctly (per-config)
    # ------------------------------------------------------------------
    total_gnb_params = sum(
        5 if cfg.optimize_power else 4 for cfg in tx_configs
    )
    _, jam_offsets = _jam_param_strides(jam_configs)

    # ------------------------------------------------------------------
    # Set jammer positions and build power-scale factors
    # ------------------------------------------------------------------
    jam_pow_scales = []
    for j, jcfg in enumerate(jam_configs):
        b   = total_gnb_params + jam_offsets[j]
        xj  = all_params[b];     dr.enable_grad(xj.array)
        yj  = all_params[b + 1]; dr.enable_grad(yj.array)
        pj  = all_params[b + 2]; dr.enable_grad(pj.array)

        jit_x = Float(float(np.random.normal(0.0, 0.1)))
        jit_y = Float(float(np.random.normal(0.0, 0.1)))
        dr.disable_grad(jit_x); dr.disable_grad(jit_y)

        jam_scene.get(jcfg.name).position = [
            xj + jit_x,
            yj + jit_y,
            Float(10.0),
        ]
        scale_j = dr.power(
            Float(10.0),
            (pj - Float(float(jcfg.initial_power_dbm))) / Float(10.0),
        )
        jam_pow_scales.append(scale_j)

    # ------------------------------------------------------------------
    # Jammer PathSolver call (iso pattern, jam_scene)
    # ------------------------------------------------------------------
    jam_paths = p_solver(
        jam_scene,
        los=True,
        refraction=False,
        specular_reflection=True,
        diffuse_reflection=False,
    )
    jh_real, jh_imag = jam_paths.a

    J = len(jam_configs)
    jam_power_vecs = []
    for j in range(J):
        jp_raw = _extract_per_rx_power(jh_real, jh_imag, j)
        jam_power_vecs.append(jp_raw * jam_pow_scales[j])

    dr.eval(*jam_power_vecs)
    del jam_paths, jh_real, jh_imag

    # ------------------------------------------------------------------
    # Per-receiver SINR with best-BS selection
    # ------------------------------------------------------------------
    eps_f   = Float(float(epsilon));   dr.disable_grad(eps_f)
    noise_f = Float(float(noise_power)); dr.disable_grad(noise_f)

    total_jam_power = None
    for jp in jam_power_vecs:
        total_jam_power = jp if total_jam_power is None else (total_jam_power + jp)

    best_sinr = None
    for i in range(N):
        p_sig = tx_power_vecs[i]

        p_gnb_int = None
        for j in range(N):
            if j == i:
                continue
            p_gnb_int = tx_power_vecs[j] if p_gnb_int is None else (p_gnb_int + tx_power_vecs[j])

        denom = noise_f
        if p_gnb_int is not None:
            denom = denom + p_gnb_int
        if total_jam_power is not None:
            denom = denom + total_jam_power

        sinr_i = p_sig / denom
        best_sinr = sinr_i if best_sinr is None else dr.maximum(best_sinr, sinr_i)

    # ------------------------------------------------------------------
    # Convert SINR to dB once (linear -> dB), then build hinge losses
    # ------------------------------------------------------------------
    log10 = Float(float(np.log(10.0))); dr.disable_grad(log10)
    # Clamp before log: receivers with no signal paths have best_sinr≈0, giving
    # d(log)/d(x) = 1/eps_f ≈ 1e30 which explodes gradients and causes Adam NaN.
    # A floor of 1e-10 (-100 dB SINR) is physically below any useful threshold;
    # dr.maximum is differentiable (grad=1 above floor, 0 below), so the gradient
    # is cleanly zeroed for dead receivers rather than blowing up.
    sinr_floored = dr.maximum(best_sinr, Float(1e-10))
    sinr_db = Float(10.0) * dr.log(sinr_floored) / log10

    gamma_f = Float(float(gamma_db));  dr.disable_grad(gamma_f)
    sigma_f = Float(float(sigma_db));  dr.disable_grad(sigma_f)
    zero_f  = Float(0.0);              dr.disable_grad(zero_f)

    # Inside / outside split
    if num_inside is not None and num_inside < dr.width(best_sinr):
        n_in = int(num_inside)
        sinr_in  = sinr_db[:n_in]
        sinr_out = sinr_db[n_in:]

        # L_in: squared hinge on (gamma - SINR), only positive part
        deficit = dr.maximum(gamma_f - sinr_in, zero_f)
        loss_inside = dr.mean(deficit * deficit)

        # L_out: squared hinge on (SINR - gamma)
        excess = dr.maximum(sinr_out - gamma_f, zero_f)
        loss_outside = dr.mean(excess * excess)
    else:
        # No partition provided: treat all cells as "inside"
        deficit = dr.maximum(gamma_f - sinr_db, zero_f)
        loss_inside = dr.mean(deficit * deficit)
        loss_outside = Float(0.0); dr.disable_grad(loss_outside)

    # L_sharp: Gaussian bump centered at gamma, over boundary shell cells.
    # Penalizes cells lingering near threshold => forces a sharp transition.
    loss_sharp = Float(0.0); dr.disable_grad(loss_sharp)
    if boundary_mask is not None:
        idx = np.asarray(boundary_mask)
        if idx.dtype == bool:
            idx = np.where(idx)[0]
        if idx.size > 0:
            # Slice preserves the AD graph (same pattern as inside/outside).
            # Convert to a Python list of ints for DrJit indexing.
            idx_list = idx.tolist()
            # Build a stacked tensor of boundary SINRs.
            sinr_b = dr.gather(type(sinr_db), sinr_db, idx_list)
            delta = (sinr_b - gamma_f) / sigma_f
            loss_sharp = dr.mean(dr.exp(Float(-0.5) * delta * delta))

    # L_pwr: L2 on jammer linear power scales (cost of jammer activity)
    if J > 0:
        pwr_sq_sum = None
        for s in jam_pow_scales:
            term = s * s
            pwr_sq_sum = term if pwr_sq_sum is None else (pwr_sq_sum + term)
        loss_pwr = pwr_sq_sum / Float(float(J))
    else:
        loss_pwr = Float(0.0); dr.disable_grad(loss_pwr)

    # ------------------------------------------------------------------
    # Compose total loss
    # ------------------------------------------------------------------
    lam_in    = Float(float(lambda_in));    dr.disable_grad(lam_in)
    lam_out   = Float(float(lambda_out));   dr.disable_grad(lam_out)
    lam_sharp = Float(float(lambda_sharp)); dr.disable_grad(lam_sharp)
    lam_pwr   = Float(float(lambda_pwr));   dr.disable_grad(lam_pwr)

    total = (lam_in    * loss_inside +
             lam_out   * loss_outside +
             lam_sharp * loss_sharp +
             lam_pwr   * loss_pwr)

    # Optional debug (comment out for production training)
    if num_inside is not None and num_inside < dr.width(best_sinr):
        n_in = int(num_inside)
        print(f"  [dbg] mean SINR_dB inside : {dr.mean(sinr_db[:n_in])}")
        print(f"  [dbg] mean SINR_dB outside: {dr.mean(sinr_db[n_in:])}")
    print(f"  [dbg] L_in={dr.mean(loss_inside)}  L_out={dr.mean(loss_outside)}  "
          f"L_sharp={dr.mean(loss_sharp)}  L_pwr={dr.mean(loss_pwr)}")

    return total


def _make_compute_sir_loss(
    N, tx_configs, tx_states, scene, p_solver,
    noise_power, rx_objects, ref_powers_dbm,
    jam_configs, jam_scene, jam_rx_objects, jam_objects,
    num_inside=None,
):
    """Build and return the @dr.wrap-decorated SIR loss function.

    Uses exec() to produce a function with a *fixed* positional signature
    matching exactly the number of scalar parameters — required by @dr.wrap.
    The flat signature is: [gNB params...] + [jammer params...]
    where each jammer contributes [x, y, power_dbm].
    """
    _, gnb_offsets = _param_strides(tx_configs)
    _, jam_offsets = _jam_param_strides(jam_configs)
    total_jam_params = (jam_offsets[-1] + 3) if jam_configs else 0

    # gNB param names: p0, p1, ...
    arg_names = []
    for i, cfg in enumerate(tx_configs):
        b = gnb_offsets[i]
        arg_names += [f"p{b}", f"p{b+1}", f"p{b+2}", f"p{b+3}"]
        if cfg.optimize_power:
            arg_names.append(f"p{b+4}")

    # Jammer param names: j0, j1, ... (appended after gNB params)
    for k in range(total_jam_params):
        arg_names.append(f"j{k}")

    arg_str  = ", ".join(arg_names)
    list_str = "[" + ", ".join(arg_names) + "]"

    func_code = (
        f"def _inner({arg_str}):\n"
        f"    return _body({list_str}, _N, _cfgs, _states, _scene, _psolver,\n"
        f"                 _noise, _rxobj, _refpow,\n"
        f"                 _jam_cfgs, _jam_scene, _jam_rxobj, _jam_obj,\n"
        f"                 num_inside=_num_inside)\n"
    )

    globs = {
        "_body":       _sir_loss_body,
        "_N":          N,
        "_cfgs":       tx_configs,
        "_states":     tx_states,
        "_scene":      scene,
        "_psolver":    p_solver,
        "_noise":      noise_power,
        "_rxobj":      rx_objects,
        "_refpow":     ref_powers_dbm,
        "_jam_cfgs":   jam_configs,
        "_jam_scene":  jam_scene,
        "_jam_rxobj":  jam_rx_objects,
        "_jam_obj":    jam_objects,
        "_num_inside": num_inside,
    }
    exec(func_code, globs)
    inner_fn = globs["_inner"]
    return dr.wrap(source="torch", target="drjit")(inner_fn)


# ---------------------------------------------------------------------------
# Public API: optimize_multi_tx
# ---------------------------------------------------------------------------

def optimize_multi_tx(
    scene,
    tx_configs: list,
    map_config: dict,
    scene_xml_path: str,
    jammer_array: AntennaArray,
    jam_configs: list,
    learning_rate: float = 3.0,
    num_iterations: int = 50,
    num_sample_points: int = 100,
    noise_power: float = 1e-10,
    dead_tail_percentile: float = 1.0,
    max_dbscan_points: int = 100_000,
    lds: str = "Halton",
    sampler: str = "triangulated",
    sampling_strata: str = "proportional",
    verbose: bool = True,
    on_iteration_callback: Optional[callable] = None,
    debug_viz: bool = False,
) -> dict:
    """Jointly optimise N transmitters for SIR coverage.

    Parameters
    ----------
    scene : sionna.rt.Scene
        Sionna scene; all TxConfig.name transmitters must already be added.
    tx_configs : list[TxConfig]
        One entry per transmitter.  Order must match scene insertion order
        (determines paths.a axis-1 indexing).
    map_config : dict
        Same format as single-TX optimizer: {'center', 'size', 'cell_size'}.
    scene_xml_path : str
        Path to scene.xml (for building extraction).
    learning_rate : float
        Adam learning rate.
    num_iterations : int
        Optimisation iterations.
    noise_power : float
        Thermal noise floor (Watts) added to interference in SIR denominator.
    dead_tail_percentile : float
        Bottom X % of rays (by power) identified as dead-zone seed.
    max_dbscan_points : int
        Cap for HDBSCAN to keep runtime tractable.
    lds : str
        Low-discrepancy sequence: "Halton" | "Sobol" | "Latin" | "Uniform".
    sampler : str
        Receiver placement backend: "triangulated" | "rejection".
    sampling_strata : str
        "proportional" -> sample alive+dead zones proportionally each iteration.
        "dead_only"    -> existing behaviour (sample dead zones, fall back to full).
    verbose : bool
    on_iteration_callback : callable or None
        Optional callable invoked at the end of every iteration, after dead
        zones are accumulated and sample points are placed.  Signature::

            callback(iteration: int, tx_states: list, tx_configs: list)

        Each state dict will have ``current_tx_position`` set to the
        transmitter's current [x, y, z] coordinates for that iteration.
        Use ``visualize_multi_tx_strata`` from ``boresight_pathsolver`` as a
        ready-made callback.

    Returns
    -------
    dict with keys per TX name plus "joint":
        {
          tx_name: {
            "best_angles":      [az_deg, el_deg],
            "final_position":   [x, y, z],
            "initial_angles":   [az_deg, el_deg],
            "initial_position": [x, y, z],
            "best_power_dbm":   float,   # only if optimize_power=True
            "az_history":       list,
            "el_history":       list,
            "power_history":    list,    # only if optimize_power=True
          },
          "joint": {
            "loss_history":   list,
            "elapsed_time_s": float,
            "num_iterations": int,
            "noise_power":    float,
            "sampler":        str,
            "sampling_strata":str,
            "lds":            str,
          }
        }
    """
    N = len(tx_configs)
    assert N >= 1, "tx_configs must contain at least one TxConfig"

    # Set up randrange boundaries based on map size
    # Starting with randomly placed receivers
    lower_bound_x = map_config['center'][0] - map_config['size'][0] / 2
    print(lower_bound_x)
    upper_bound_x = map_config['center'][0] + map_config['size'][0] / 2
    lower_bound_y = map_config['center'][1] - map_config['size'][1] / 2
    upper_bound_y = map_config['center'][1] + map_config['size'][1] / 2

    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-TX SIR OPTIMIZATION  ({N} transmitters)")
        print(f"{'='*70}")

    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Build per-TX state dicts
    # ------------------------------------------------------------------
    qrand_shared = _make_qrand(lds)  # shared LDS instance
    tx_states = [_setup_tx_state(scene, cfg, scene_xml_path, qrand_shared)
                 for cfg in tx_configs]

    # ------------------------------------------------------------------
    # 2. All BSes share the same receiver pool (global shared zone).
    #    First num_sample_points receivers are inside the zone;
    #    next num_sample_points are in the outer ring (2x-scaled zone minus zone).
    # ------------------------------------------------------------------
    n_inside  = num_sample_points
    n_outside = num_sample_points
    total_rx  = n_inside + n_outside
    shared_slice = slice(0, total_rx)
    for state in tx_states:
        state["rx_slice"] = shared_slice
        state["rx_indices_drjit"] = None

    # ------------------------------------------------------------------
    # 3. Clear existing receivers; pre-create total_rx receivers at origin
    # ------------------------------------------------------------------
    for rx_name in list(scene.receivers.keys()):
        scene.remove(rx_name)

    rx_objects = {}
    for idx in range(total_rx):
        rx_name = f"opt_rx_{idx}"
        rx = Receiver(name=rx_name, position=[0.0, 0.0, 0.0])
        scene.add(rx)
        rx_objects[rx_name] = rx

    # ------------------------------------------------------------------
    # Build a separate jam_scene for the jammer solve.
    # Jammers use an isotropic antenna pattern and must be isolated from
    # the gNB scene so each solve uses a different tx_array.
    # ------------------------------------------------------------------
    jam_scene = load_scene(scene_xml_path)
    jam_scene.frequency = scene.frequency
    for mat_name, mat in scene.radio_materials.items():
        if mat_name in jam_scene.radio_materials:
            jam_scene.radio_materials[mat_name].scattering_coefficient = (
                mat.scattering_coefficient
            )
    jam_scene.tx_array = jammer_array
    jam_scene.rx_array = scene.rx_array

    # Add jammers as Transmitter objects driven by JammerConfig.
    # initial_position=None -> sample uniformly from map bounds.
    jam_objects = {}
    for jcfg in jam_configs:
        if jcfg.initial_position is not None:
            jx, jy = float(jcfg.initial_position[0]), float(jcfg.initial_position[1])
        else:
            jx = float(random.randrange(int(lower_bound_x), int(upper_bound_x), 1))
            jy = float(random.randrange(int(lower_bound_y), int(upper_bound_y), 1))
        pos = mi.Point3f([jx, jy, 45.0])
        jammer = Transmitter(name=jcfg.name, position=pos,
                             power_dbm=jcfg.initial_power_dbm)
        jam_scene.add(jammer)
        jam_objects[jcfg.name] = jammer

    # Mirror the receiver pool in jam_scene so the second solve has targets.
    jam_rx_objects = {}
    for idx in range(total_rx):
        rx_name = f"opt_rx_{idx}"
        jam_rx = Receiver(name=rx_name, position=[0.0, 0.0, 0.0])
        jam_scene.add(jam_rx)
        jam_rx_objects[rx_name] = jam_rx

    if verbose:
        print(f"Pre-created {total_rx} receivers ({n_inside} inside + {n_outside} outside) "
              f"across {N} base stations")

    # ------------------------------------------------------------------
    # Pre-sample fixed receiver positions once for the shared zone.
    # Receivers 0..n_inside-1   → inside zone
    # Receivers n_inside..total_rx-1 → outer ring (2x-scaled zone minus zone)
    # ------------------------------------------------------------------
    ground_z = float(map_config["center"][2]) if len(map_config["center"]) > 2 else 0.0
    pts = _sample_zone_points(tx_states[0], tx_configs[0], n_inside,
                              sampler, "full", ground_z)
    out_pts = _sample_outside_zone(tx_states[0], n_outside, ground_z)
    for state in tx_states:
        state["current_sample_points"] = pts
        state["outside_sample_points"] = out_pts
    for k, pos in enumerate(pts):
        rx_name = f"opt_rx_{k}"
        p3 = mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2]))
        rx_objects[rx_name].position = p3
        jam_rx_objects[rx_name].position = p3
    for k, pos in enumerate(out_pts):
        rx_name = f"opt_rx_{n_inside + k}"
        p3 = mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2]))
        rx_objects[rx_name].position = p3
        jam_rx_objects[rx_name].position = p3

    # ------------------------------------------------------------------
    # 4. PathSolver + @dr.wrap closure
    # ------------------------------------------------------------------
    p_solver = PathSolver()
    p_solver.loop_mode = "evaluated"

    ref_powers_dbm = [s["tx_ref_power_dbm"] for s in tx_states]

    compute_sir_loss = _make_compute_sir_loss(
        N, tx_configs, tx_states, scene, p_solver,
        noise_power, rx_objects, ref_powers_dbm,
        jam_configs, jam_scene, jam_rx_objects, jam_objects,
        num_inside=n_inside,
    )

    # ------------------------------------------------------------------
    # 5. Initialise PyTorch parameters
    # ------------------------------------------------------------------
    _, offsets = _param_strides(tx_configs)
    params = []
    for i, (cfg, state) in enumerate(zip(tx_configs, tx_states)):
        params.append(torch.tensor(state["initial_azimuth"],   device="cuda",
                                   dtype=torch.float32, requires_grad=True))
        params.append(torch.tensor(state["initial_elevation"], device="cuda",
                                   dtype=torch.float32, requires_grad=True))
        params.append(torch.tensor(state["tx_position"][0],   device="cuda",
                                   dtype=torch.float32, requires_grad=True))
        params.append(torch.tensor(state["tx_position"][1],   device="cuda",
                                   dtype=torch.float32, requires_grad=True))
        if cfg.optimize_power:
            params.append(torch.tensor(state["initial_power_dbm"], device="cuda",
                                       dtype=torch.float32, requires_grad=True))

    # Jammer params: [x, y, power_dbm] per jammer (always all three).
    jam_params = []
    for jcfg in jam_configs:
        jammer = jam_objects[jcfg.name]
        init_pos = jammer.position.numpy().flatten()
        jam_params.append(torch.tensor(float(init_pos[0]), device="cuda",
                                       dtype=torch.float32, requires_grad=True))
        jam_params.append(torch.tensor(float(init_pos[1]), device="cuda",
                                       dtype=torch.float32, requires_grad=True))
        jam_params.append(torch.tensor(jcfg.initial_power_dbm, device="cuda",
                                       dtype=torch.float32, requires_grad=True))

    optimizer = torch.optim.Adam(params + jam_params, lr=learning_rate, betas=(0.9, 0.999))
    #optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.25)

    # ------------------------------------------------------------------
    # 6. Tracking
    # ------------------------------------------------------------------
    loss_history = []
    rm_solver    = RadioMapSolver()

    # Buffers for final averaging (last 10 iterations)
    final_bufs = {i: {"az": [], "el": [], "pow": []} for i in range(N)}

    # ------------------------------------------------------------------
    # 7. Optimisation loop
    # ------------------------------------------------------------------
    for iteration in range(num_iterations):
        iter_start = time.time()

        # Current plain-float parameter values (detached from AD)
        pvals = [float(p.item()) for p in params]

        # Apply current parameter values to scene (once for all TXs)
        _, offsets = _param_strides(tx_configs)
        for k, (cfg_k, state_k) in enumerate(zip(tx_configs, tx_states)):
            b = offsets[k]
            az_k, el_k = pvals[b], pvals[b + 1]
            xp_k, yp_k = pvals[b + 2], pvals[b + 3]
            scene.get(cfg_k.name).orientation = [
                float(np.deg2rad(az_k)), -float(np.deg2rad(el_k)), 0.0
            ]
            scene.get(cfg_k.name).position = mi.Point3f(
                float(xp_k), float(yp_k), float(state_k["tx_height"])
            )
            if cfg_k.optimize_power:
                scene.get(cfg_k.name).power_dbm = [float(pvals[b + 4])]

        # Single RadioMap pass — rm.rss shape (N_tx, H, W) gives per-TX
        # cell-aggregated power, smoothing out multipath fades.
        if sampling_strata != "full":
            # Calculate Coverage
            rm = rm_solver(
                scene,
                max_depth=12,
                samples_per_tx=int(100e7),
                cell_size=[1.0, 1.0],
                center=map_config["center"],
                orientation=[0, 0, 0],
                size=map_config["size"],
                los=True,
                specular_reflection=True,
                diffuse_reflection=True,
                diffraction=True,
                edge_diffraction=True,
                refraction=False,
                stop_threshold=None,
            )
            # Cluster and build dead zones
            for i, (cfg, state) in enumerate(zip(tx_configs, tx_states)):
                _accumulate_dead_zones(
                    state, cfg, i, rm,
                    map_config, dead_tail_percentile, max_dbscan_points,
                    debug_viz=debug_viz, iteration=iteration,
                )
            
        # Differentiable forward pass
        loss = compute_sir_loss(*params, *jam_params)
        loss.backward()

        # Clip gradients before the Adam step to guard against log-gradient
        # explosions when receivers have near-zero signal (d(log)/dx = 1/x → ∞).
        torch.nn.utils.clip_grad_norm_(params + jam_params, max_norm=10.0)

        optimizer.step()
        optimizer.zero_grad()

        # Per-TX post-step constraints
        with torch.no_grad():
            for i, (cfg, state) in enumerate(zip(tx_configs, tx_states)):
                b = offsets[i]
                az_t = params[b];     el_t = params[b + 1]
                x_t  = params[b + 2]; y_t  = params[b + 3]

                # Azimuth: clamp and wrap to [0, 360)
                az_t.clamp_(0.0, 360.0)
                if az_t.item() >= 360.0:
                    az_t.fill_(az_t.item() % 360.0)

                # Position: project to building polygon if this TX is building-mounted.
                if cfg.on_building:
                    proj_x, proj_y = state["tx_placement"].project_to_polygon(
                        x_t.item(), y_t.item()
                    )
                    x_t.data.fill_(proj_x)
                    y_t.data.fill_(proj_y)

                # Power clamp
                if cfg.optimize_power:
                    pow_t = params[b + 4]
                    pow_t.clamp_(*cfg.power_dbm_bounds)

        # Track histories
        loss_val = float(loss.item())
        loss_history.append(loss_val)

        for i, cfg in enumerate(tx_configs):
            b = offsets[i]
            az_val = float(params[b].item())
            el_val = float(params[b + 1].item())
            tx_states[i]["az_history"].append(az_val)
            tx_states[i]["el_history"].append(el_val)
            if cfg.optimize_power:
                pw_val = float(params[b + 4].item())
                tx_states[i]["power_history"].append(pw_val)

        # Accumulate final-window values (last 10 iters)
        window_start = max(0, num_iterations - 10)
        if iteration >= window_start:
            for i, cfg in enumerate(tx_configs):
                b = offsets[i]
                final_bufs[i]["az"].append(float(params[b].item()))
                final_bufs[i]["el"].append(float(params[b + 1].item()))
                if cfg.optimize_power:
                    final_bufs[i]["pow"].append(float(params[b + 4].item()))

        # Expose current TX positions for visualisation callbacks
        for i, (cfg, state) in enumerate(zip(tx_configs, tx_states)):
            b = offsets[i]
            state["current_tx_position"] = [
                float(params[b + 2].item()),
                float(params[b + 3].item()),
                state["tx_height"],
            ]

        if on_iteration_callback is not None:
            _, joff = _jam_param_strides(jam_configs)
            jam_positions = [
                [float(jam_params[joff[j]].item()),
                 float(jam_params[joff[j] + 1].item())]
                for j in range(len(jam_configs))
            ]
            on_iteration_callback(iteration, tx_states, tx_configs,
                                  jam_positions=jam_positions)

        if verbose:
            dur = time.time() - iter_start
            print(f"  Iter {iteration+1:3d}/{num_iterations}  loss={loss_val:.4f}  "
                  f"({dur:.1f}s)")

        # Make sure there are no memory leaks...    
        del loss
        torch.cuda.empty_cache()
        dr.flush_malloc_cache()

    # ------------------------------------------------------------------
    # 8. Finalise: reset scene to plain-float state; remove temp receivers
    # ------------------------------------------------------------------
    for i, (cfg, state) in enumerate(zip(tx_configs, tx_states)):
        b      = offsets[i]
        best_az = float(np.mean(final_bufs[i]["az"])) if final_bufs[i]["az"] else float(params[b].item())
        best_el = float(np.mean(final_bufs[i]["el"])) if final_bufs[i]["el"] else float(params[b + 1].item())
        final_x = float(params[b + 2].item())
        final_y = float(params[b + 3].item())

        yaw_r, pitch_r = azimuth_elevation_to_yaw_pitch(best_az, best_el)
        scene.get(cfg.name).orientation = mi.Point3f(float(yaw_r), float(pitch_r), 0.0)
        scene.get(cfg.name).position    = mi.Point3f(float(final_x), float(final_y),
                                                       float(state["tx_height"]))
        if cfg.optimize_power:
            best_pow = float(np.mean(final_bufs[i]["pow"])) if final_bufs[i]["pow"] else float(params[b + 4].item())
            scene.get(cfg.name).power_dbm = [best_pow]
            state["best_power_dbm"] = best_pow
        state["best_angles"]    = [best_az, best_el]
        state["final_position"] = [final_x, final_y, state["tx_height"]]

    # Remove optimization receivers
    for rx_name in list(rx_objects.keys()):
        if rx_name in [obj.name for obj in scene.receivers.values()]:
            scene.remove(rx_name)

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # 9. Build result dict
    # ------------------------------------------------------------------
    result = {}
    for i, (cfg, state) in enumerate(zip(tx_configs, tx_states)):
        entry = {
            "best_angles":      state["best_angles"],
            "final_position":   state["final_position"],
            "initial_angles":   [state["initial_azimuth"], state["initial_elevation"]],
            "initial_position": state["tx_position"],
            "az_history":       state["az_history"],
            "el_history":       state["el_history"],
        }
        if cfg.optimize_power:
            entry["best_power_dbm"] = state["best_power_dbm"]
            entry["power_history"]  = state["power_history"]
        result[cfg.name] = entry

    result["joint"] = {
        "loss_history":    loss_history,
        "elapsed_time_s":  elapsed,
        "num_iterations":  num_iterations,
        "noise_power":     noise_power,
        "sampler":         sampler,
        "sampling_strata": sampling_strata,
        "lds":             lds,
    }

    # ------------------------------------------------------------------
    # 10. Persist final jammer state into result and jam_scene
    # ------------------------------------------------------------------
    if jam_configs:
        _, joff = _jam_param_strides(jam_configs)
        jammers_final = {}
        for j, jcfg in enumerate(jam_configs):
            b   = joff[j]
            xf  = float(jam_params[b].item())
            yf  = float(jam_params[b + 1].item())
            pf  = float(jam_params[b + 2].item())
            jam_scene.get(jcfg.name).position  = mi.Point3f(xf, yf, 45.0)
            jam_scene.get(jcfg.name).power_dbm = [pf]
            jammers_final[jcfg.name] = {
                "final_position":    [xf, yf, 45.0],
                "final_power_dbm":   pf,
                "initial_power_dbm": jcfg.initial_power_dbm,
            }
        result["joint"]["jammers"] = jammers_final

    if verbose:
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE  ({elapsed:.1f}s)")
        for cfg in tx_configs:
            r = result[cfg.name]
            print(f"  {cfg.name}: Az={r['best_angles'][0]:.1f}°, "
                  f"El={r['best_angles'][1]:.1f}°, "
                  f"pos=({r['final_position'][0]:.1f}, {r['final_position'][1]:.1f})")
        if jam_configs:
            for jcfg in jam_configs:
                jd = result["joint"]["jammers"][jcfg.name]
                print(f"  {jcfg.name}: pos=({jd['final_position'][0]:.1f}, "
                      f"{jd['final_position'][1]:.1f}), pwr={jd['final_power_dbm']:.1f} dBm")
        print(f"{'='*70}\n")

    return result, jam_scene if jam_configs else None


# ---------------------------------------------------------------------------
# Public API: compare_multi_tx_performance
# ---------------------------------------------------------------------------

def compare_multi_tx_performance(
    scene,
    tx_configs: list,
    multi_result: dict,
    map_config: dict,
    zone_masks: dict,
    noise_power: float = 1e-10,
    fig: bool = True,
    gamma_db: float = 0.0,
    jam_scene=None,
    jammer_configs: list = None,
) -> tuple:
    """Evaluate the optimised multi-TX configuration for coverage shaping.

    Reports how well SINR is contained within the target zone:
      rho_leak : fraction of OUTSIDE cells with SINR >= gamma  (lower is better)
      rho_hole : fraction of INSIDE  cells with SINR <  gamma  (lower is better)

    Signal = highest BS signal at each cell (jammers never count as signal).
    Interference = other BSs + friendly jammer power (if jam_scene provided).
    SINR formula mirrors the loss function exactly:
      SINR_i(r) = P_BS_i / (noise + Σ_{j≠i} P_BS_j + Σ_k P_jam_k)
      best_SINR  = max_i(SINR_i)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    N = len(tx_configs)
    gamma_db = float(gamma_db)

    def _watts_to_db(w):
        return 10.0 * np.log10(np.maximum(w, 1e-18))

    def _set_scene_optimized():
        for cfg in tx_configs:
            r = multi_result[cfg.name]
            angles  = r["best_angles"]
            pos     = r["final_position"]
            pow_dbm = r.get("best_power_dbm", None)
            yaw_r, pitch_r = azimuth_elevation_to_yaw_pitch(angles[0], angles[1])
            scene.get(cfg.name).orientation = mi.Point3f(float(yaw_r), float(pitch_r), 0.0)
            scene.get(cfg.name).position    = mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2]))
            if pow_dbm is not None:
                scene.get(cfg.name).power_dbm = [float(pow_dbm)]

    def _run_radiomap():
        solver = RadioMapSolver()
        return solver(
            scene,
            max_depth=8,
            samples_per_tx=int(1e9),
            cell_size=list(map_config["cell_size"]),
            center=map_config["center"],
            orientation=[0, 0, 0],
            size=map_config["size"],
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=False,
            stop_threshold=None,
        )

    def _bs_sinr_field(rss_list_2d, jam_map=None):
        """Best-SINR (max over BSs) over the full 2-D grid, in dB.

        Signal = P_BS_i; Interference = other BSs + optional jammer map.
        """
        best = None
        for i in range(N):
            p_sig = rss_list_2d[i]
            p_int = sum(rss_list_2d[j] for j in range(N) if j != i)
            denom = p_int + noise_power
            if jam_map is not None:
                denom = denom + jam_map
            sinr_i = p_sig / denom
            best = sinr_i if best is None else np.maximum(best, sinr_i)
        return _watts_to_db(best)

    def _summarize_sinr(values_db):
        if values_db.size == 0:
            return None
        return {
            "sinr_mean_db":   float(np.mean(values_db)),
            "sinr_median_db": float(np.median(values_db)),
            "sinr_p10_db":    float(np.percentile(values_db, 10)),
            "sinr_p90_db":    float(np.percentile(values_db, 90)),
            "sinr_values_db": values_db.tolist(),
        }

    # ------------------------------------------------------------------
    # Masks: inside = target zone, outside = complement of union
    # ------------------------------------------------------------------
    grid_shape   = next(iter(zone_masks.values())).shape
    inside_masks = {cfg.name: (zone_masks[cfg.name] == 1.0) for cfg in tx_configs}
    union_inside = np.zeros(grid_shape, dtype=bool)
    for cfg in tx_configs:
        union_inside |= inside_masks[cfg.name]
    outside_mask = ~union_inside

    # ------------------------------------------------------------------
    # Print parameter before/after table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("BS PARAMETERS  —  BEFORE / AFTER")
    print(f"{'='*70}")
    print(f"  {'Name':<14} {'Param':<10} {'Initial':>10}  {'Optimised':>10}  {'Delta':>10}")
    print(f"  {'-'*64}")
    for cfg in tx_configs:
        r  = multi_result[cfg.name]
        ia, ie = r["initial_angles"]
        oa, oe = r["best_angles"]
        ip = r["initial_position"]
        op = r["final_position"]
        rows = [
            ("Azimuth",   f"{ia:+8.2f}°",   f"{oa:+8.2f}°",   f"{oa-ia:+8.2f}°"),
            ("Elevation", f"{ie:+8.2f}°",   f"{oe:+8.2f}°",   f"{oe-ie:+8.2f}°"),
            ("X (m)",     f"{ip[0]:+8.1f}", f"{op[0]:+8.1f}", f"{op[0]-ip[0]:+8.1f}"),
            ("Y (m)",     f"{ip[1]:+8.1f}", f"{op[1]:+8.1f}", f"{op[1]-ip[1]:+8.1f}"),
        ]
        if "best_power_dbm" in r:
            init_pow = r.get("initial_power_dbm", float("nan"))
            opt_pow  = r["best_power_dbm"]
            rows.append(("Power dBm", f"{init_pow:+8.2f}", f"{opt_pow:+8.2f}",
                          f"{opt_pow - init_pow:+8.2f}"))
        for k, (param, iv, ov, dv) in enumerate(rows):
            name_col = cfg.name if k == 0 else ""
            print(f"  {name_col:<14} {param:<10} {iv:>10}  {ov:>10}  {dv:>10}")
        print(f"  {'-'*64}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Run RadioMapSolver for optimized BS configuration
    # ------------------------------------------------------------------
    print("Computing RadioMap for optimized BS configuration...")
    _set_scene_optimized()
    rm          = _run_radiomap()
    # NaN = no propagation path → treat as 0 received power
    rss_list_2d = [np.nan_to_num(rm.rss.numpy()[i], nan=0.0) for i in range(N)]

    # ------------------------------------------------------------------
    # Run RadioMapSolver for jammers (if provided)
    # ------------------------------------------------------------------
    jam_interference_map = None
    if jam_scene is not None and jammer_configs:
        jammers_data = multi_result["joint"].get("jammers", {})
        for jcfg in jammer_configs:
            if jcfg.name in jammers_data:
                jd  = jammers_data[jcfg.name]
                pos = jd["final_position"]
                jam_scene.get(jcfg.name).position  = mi.Point3f(*[float(v) for v in pos])
                jam_scene.get(jcfg.name).power_dbm = [float(jd["final_power_dbm"])]
        print("Computing RadioMap for jammer interference...")
        jam_solver = RadioMapSolver()
        jam_rm = jam_solver(
            jam_scene,
            max_depth=8,
            samples_per_tx=int(1e9),
            cell_size=list(map_config["cell_size"]),
            center=map_config["center"],
            orientation=[0, 0, 0],
            size=map_config["size"],
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            stop_threshold=None,
        )
        jam_rss = np.nan_to_num(jam_rm.rss.numpy(), nan=0.0)  # (J, H, W)
        jam_interference_map = np.sum(jam_rss, axis=0)         # (H, W)

    sinr_field = _bs_sinr_field(rss_list_2d, jam_map=jam_interference_map)

    # ------------------------------------------------------------------
    # Per-TX containment metrics
    # ------------------------------------------------------------------
    stats = {}
    for cfg in tx_configs:
        r        = multi_result[cfg.name]
        in_mask  = inside_masks[cfg.name]
        in_vals  = sinr_field[in_mask]
        out_vals = sinr_field[outside_mask]

        rho_leak = float(np.mean(out_vals >= gamma_db)) if out_vals.size else None
        rho_hole = float(np.mean(in_vals  <  gamma_db)) if in_vals.size  else None

        init_params = {"azimuth": r["initial_angles"][0], "elevation": r["initial_angles"][1],
                       "position": r["initial_position"]}
        opt_params  = {"azimuth": r["best_angles"][0],    "elevation": r["best_angles"][1],
                       "position": r["final_position"]}
        if "best_power_dbm" in r:
            init_params["power_dbm"] = r.get("initial_power_dbm", float("nan"))
            opt_params["power_dbm"]  = r["best_power_dbm"]

        stats[cfg.name] = {
            "initial_params":  init_params,
            "optimized_params": opt_params,
            "sinr_inside":     _summarize_sinr(in_vals),
            "sinr_outside":    _summarize_sinr(out_vals),
            "containment":     {"rho_leak": rho_leak, "rho_hole": rho_hole},
            "az_history":      r.get("az_history", []),
            "el_history":      r.get("el_history", []),
            "power_history":   r.get("power_history", []),
        }

    # ------------------------------------------------------------------
    # Print containment summary
    # ------------------------------------------------------------------
    sinr_mode = "BS+JAMMER" if jam_interference_map is not None else "BS-ONLY"
    print(f"\n{'='*70}")
    print(f"{sinr_mode} SINR CONTAINMENT  (gamma = {gamma_db:+.1f} dB)")
    print(f"{'='*70}")
    for cfg in tx_configs:
        s  = stats[cfg.name]
        c  = s["containment"]
        si = s["sinr_inside"]
        so = s["sinr_outside"]
        def _fmt(v): return f"{100*v:5.1f}%" if v is not None else "  n/a "
        print(f"  {cfg.name}:")
        print(f"    rho_leak  (outside >= gamma): {_fmt(c['rho_leak'])}")
        print(f"    rho_hole  (inside  <  gamma): {_fmt(c['rho_hole'])}")
        if si: print(f"    SINR inside  mean / p10: {si['sinr_mean_db']:+.1f} / {si['sinr_p10_db']:+.1f} dB")
        if so: print(f"    SINR outside mean / p10: {so['sinr_mean_db']:+.1f} / {so['sinr_p10_db']:+.1f} dB")
        print()
    print(f"{'='*70}\n")

    jnt = multi_result.get("joint", {})
    stats["joint"] = {
        "loss_history":         jnt.get("loss_history", []),
        "gamma_db":             gamma_db,
        "jammers":              jnt.get("jammers", {}),
        "jam_interference_used": jam_interference_map is not None,
    }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not fig:
        return None, stats

    # Layout: N rows (one per BS) + 1 summary row
    # Cols: Spatial SINR map | Inside vs Outside CDF | Trajectory
    n_rows = N + 1
    fig_obj, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for tx_idx, cfg in enumerate(tx_configs):
        s = stats[cfg.name]
        c = s["containment"]

        # Col 0: Spatial BS-only SINR map with target zone contour
        ax = axes[tx_idx, 0]
        vmin, vmax = gamma_db - 20, gamma_db + 20
        norm = TwoSlopeNorm(vmin=vmin, vcenter=gamma_db, vmax=vmax)
        im = ax.imshow(np.flipud(sinr_field), cmap="RdBu_r", norm=norm,
                       interpolation="nearest", aspect="equal")
        ax.contour(np.flipud(inside_masks[cfg.name].astype(float)),
                   levels=[0.5], colors="black", linewidths=1.5)
        # Overlay jammer positions
        if jam_scene is not None and jammer_configs:
            jammers_data = multi_result["joint"].get("jammers", {})
            cx_m, cy_m = map_config["center"][0], map_config["center"][1]
            sx_m, sy_m = map_config["size"][0],   map_config["size"][1]
            cw, ch     = map_config["cell_size"][0], map_config["cell_size"][1]
            H_px = int(round(sy_m / ch))
            for jcfg in jammer_configs:
                if jcfg.name in jammers_data:
                    jx, jy = jammers_data[jcfg.name]["final_position"][:2]
                    px = (jx - (cx_m - sx_m / 2)) / cw
                    py = H_px - (jy - (cy_m - sy_m / 2)) / ch
                    ax.scatter(px, py, marker="x", color="yellow", s=80,
                               linewidths=2, zorder=5, label=jcfg.name)
            ax.legend(fontsize=7, loc="upper right")
        jam_label = "+Jammers" if jam_interference_map is not None else "BS-only"
        ax.set_title(f"{cfg.name}  SINR ({jam_label}) dB", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="SINR (dB)")

        # Col 1: Inside vs Outside SINR CDF on the same axes
        ax = axes[tx_idx, 1]
        si = s["sinr_inside"]
        so = s["sinr_outside"]
        if si:
            arr = np.sort(np.asarray(si["sinr_values_db"], dtype=float))
            ax.plot(arr, np.arange(1, len(arr) + 1) / len(arr),
                    color="steelblue", linewidth=1.8, label=f"Inside  (N={len(arr)})")
        if so:
            arr = np.sort(np.asarray(so["sinr_values_db"], dtype=float))
            ax.plot(arr, np.arange(1, len(arr) + 1) / len(arr),
                    color="coral", linewidth=1.8, label=f"Outside (N={len(arr)})")
        ax.axvline(gamma_db, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"γ = {gamma_db:+.1f} dB")
        parts = []
        if c["rho_leak"] is not None: parts.append(f"ρ_leak={100*c['rho_leak']:.1f}%")
        if c["rho_hole"] is not None: parts.append(f"ρ_hole={100*c['rho_hole']:.1f}%")
        ax.set_title(f"{cfg.name}  SINR CDF — " + "  ".join(parts), fontsize=10)
        ax.set_xlabel("BS-only SINR (dB)"); ax.set_ylabel("CDF")
        ax.set_xlim(gamma_db - 30, gamma_db + 30)
        ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)

        # Col 2: Parameter trajectory (azimuth / elevation / power vs iteration)
        ax = axes[tx_idx, 2]
        az_h = s["az_history"]
        el_h = s["el_history"]
        pw_h = s["power_history"]
        if az_h or el_h:
            from matplotlib.lines import Line2D
            handles_t = []
            if az_h:
                ax.plot(np.arange(len(az_h)), az_h, color="steelblue", linewidth=1.5)
                ax.set_ylabel("Azimuth (°)", color="steelblue")
                ax.tick_params(axis="y", labelcolor="steelblue")
                handles_t.append(Line2D([0], [0], color="steelblue", label="Azimuth"))
            if el_h:
                ax2_t = ax.twinx()
                ax2_t.plot(np.arange(len(el_h)), el_h, color="darkorange", linewidth=1.5)
                ax2_t.set_ylabel("Elevation (°)", color="darkorange")
                ax2_t.tick_params(axis="y", labelcolor="darkorange")
                handles_t.append(Line2D([0], [0], color="darkorange", label="Elevation"))
                if pw_h:
                    ax2_t.plot(np.arange(len(pw_h)), pw_h, color="green",
                               linewidth=1.2, linestyle="--")
                    handles_t.append(Line2D([0], [0], color="green",
                                            linestyle="--", label="Power (dBm)"))
            ax.set_xlabel("Iteration")
            ax.set_title(f"{cfg.name}  Parameter Trajectory", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(handles=handles_t, fontsize=8, loc="upper right")
        else:
            ax.axis("off")

    # ------------------------------------------------------------------
    # Summary row: loss curve | containment bar | (off)
    # ------------------------------------------------------------------
    ax_loss = axes[N, 0]
    loss_hist = stats["joint"]["loss_history"]
    if loss_hist:
        ax_loss.plot(loss_hist, color="crimson", linewidth=1.8)
        ax_loss.set_title("Joint Loss History", fontsize=10)
        ax_loss.set_xlabel("Iteration"); ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.3)
    else:
        ax_loss.axis("off")

    ax_cont = axes[N, 1]
    names         = [cfg.name for cfg in tx_configs]
    rho_leak_vals = [100 * (stats[cfg.name]["containment"]["rho_leak"] or 0) for cfg in tx_configs]
    rho_hole_vals = [100 * (stats[cfg.name]["containment"]["rho_hole"] or 0) for cfg in tx_configs]
    x = np.arange(len(names))
    w = 0.35
    ax_cont.bar(x - w / 2, rho_leak_vals, w, color="coral",     alpha=0.85, label="ρ_leak (outside ≥ γ)")
    ax_cont.bar(x + w / 2, rho_hole_vals, w, color="steelblue", alpha=0.85, label="ρ_hole (inside < γ)")
    ax_cont.set_xticks(x); ax_cont.set_xticklabels(names, fontsize=9)
    ax_cont.set_ylabel("Fraction (%)"); ax_cont.set_ylim(0, 100)
    sinr_label = "BS+Jammer SINR" if jam_interference_map is not None else "BS-only SINR"
    ax_cont.set_title(f"{sinr_label} Containment  (γ = {gamma_db:+.1f} dB)", fontsize=10)
    ax_cont.legend(fontsize=8); ax_cont.grid(True, alpha=0.3, axis="y")

    axes[N, 2].axis("off")

    plt.suptitle(
        f"Friendly-Jammer Coverage Shaping — {sinr_label} vs Target Zone   "
        f"(γ = {gamma_db:+.1f} dB)",
        fontsize=13, weight="bold")
    plt.tight_layout()
    return fig_obj, stats

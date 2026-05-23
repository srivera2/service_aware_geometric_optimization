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
JammerConfig                   : dataclass describing one friendly jammer
seed_bs_positions()            : LOS-aware greedy farthest-point BS placement inside zone
seed_jammer_positions()        : concave-edge jammer placement outside zone
setup_bs_transmitters()        : seed + place + configure n BSs in one call
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
    initial_power_dbm : float
        Starting transmit power [dBm].
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
    initial_power_dbm: float = 40.0
    power_dbm_bounds: tuple = (40.0, 50.0)
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
    initial_azimuth_deg : float or None
        Override initial azimuth; None -> auto-compute from jammer -> zone centroid.
    initial_elevation_deg : float or None
        Override initial elevation; None -> auto-compute.
    """
    name: str
    initial_position: Optional[list] = None   # [x, y]; None = random
    initial_power_dbm: float = 23.0
    power_dbm_bounds: tuple = (0.0, 40.0)
    elevation_bounds: tuple = (0.0, -90.0)
    initial_azimuth_deg: Optional[float] = None
    initial_elevation_deg: Optional[float] = None


# ---------------------------------------------------------------------------
# Placement seeding utilities
# ---------------------------------------------------------------------------

def _los_blocked(bx, by, bz, tx, ty, tz, bldg_polys_2d, bldg_heights):
    """Return True if the 3-D segment (bx,by,bz)→(tx,ty,tz) is blocked.

    For each building whose 2-D footprint intersects the segment we compute
    the height of the LOS line at the nearest intersection centroid.  If any
    building roof exceeds that height the path is considered blocked.
    """
    from shapely.geometry import LineString as _LS2
    seg = _LS2([(bx, by), (tx, ty)])
    h_dist = np.hypot(tx - bx, ty - by)
    if h_dist < 1e-6:
        return False
    for poly, z_h in zip(bldg_polys_2d, bldg_heights):
        if not seg.intersects(poly):
            continue
        ic = seg.intersection(poly).centroid
        d = np.hypot(ic.x - bx, ic.y - by)
        h_at = bz + (tz - bz) * (d / h_dist)
        if z_h > h_at:
            return True
    return False


def _project_to_zone_edge(
    pos_xy: list,
    centroid,
    zone_poly,
    building_polygons=None,
    step_back_m: float = 3.0,
) -> list:
    """Project pos_xy to the farthest zone boundary along centroid→pos direction.

    Walks the ray from the zone centroid through pos_xy until it exits the zone
    polygon, then steps back step_back_m so the result is clearly interior.
    If the stepped-back point lands inside a building, it retreats in further
    increments until clear (or gives up after 20 tries).
    """
    from shapely.geometry import LineString as _LS, Point as _Pt

    cx, cy = float(centroid.x), float(centroid.y)
    dx = pos_xy[0] - cx
    dy = pos_xy[1] - cy
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length < 1e-6:
        return pos_xy

    dx /= length
    dy /= length

    far = 20_000.0
    ray = _LS([(cx, cy), (cx + dx * far, cy + dy * far)])
    inter = ray.intersection(zone_poly)

    def _coords(geom):
        if geom.is_empty:
            return []
        t = geom.geom_type
        if t == "Point":
            return [(geom.x, geom.y)]
        if t == "MultiPoint":
            return [(g.x, g.y) for g in geom.geoms]
        if t == "LineString":
            return list(geom.coords)
        if t in ("MultiLineString", "GeometryCollection"):
            out = []
            for g in geom.geoms:
                out.extend(_coords(g))
            return out
        return []

    coords = _coords(inter)
    if not coords:
        return pos_xy

    # Farthest point along the ray from the centroid
    best_t, best_pt = -1.0, pos_xy
    for (x, y) in coords:
        t = (x - cx) * dx + (y - cy) * dy
        if t > best_t:
            best_t = t
            best_pt = [x, y]

    # Step back from edge so the point is clearly inside the zone
    px = best_pt[0] - dx * step_back_m
    py = best_pt[1] - dy * step_back_m

    # Retreat further if we landed inside a building
    if building_polygons:
        pt = _Pt(px, py)
        for _ in range(20):
            if not any(bp.contains(pt) for bp in building_polygons):
                break
            px -= dx * step_back_m
            py -= dy * step_back_m
            pt = _Pt(px, py)

    return [px, py]


def seed_bs_positions(
    zone_params: dict,
    n_bs: int,
    building_polygons=None,
    building_info: dict = None,
    bs_height: float = 25.0,
    target_z: float = 1.5,
    seed: int = 42,
    project_to_edge: bool = False,
    min_building_clearance: float = 40.0,
    min_bs_separation: float = 300.0,
) -> list:
    """Place n_bs base stations inside the zone, maximally spread and LOS-clear.

    Candidates are sampled inside the zone polygon (building footprints
    excluded), then split into two pools:

    * **LOS pool** — positions with an unobstructed 3-D line of sight from
      ``(x, y, bs_height)`` to the zone centroid at ``target_z``.
    * **Fallback pool** — all remaining interior positions.

    Greedy farthest-point selection runs on the LOS pool first.  If the LOS
    pool is exhausted before ``n_bs`` stations are placed, the algorithm
    continues into the fallback pool (with a printed warning).

    Parameters
    ----------
    zone_params : dict
        ``{'vertices': [(x,y), ...]}`` or ``{'center', 'width', 'height'}``.
    n_bs : int
        Number of base stations to place.
    building_polygons : list[ShapelyPolygon] or None
        2-D building footprints used to exclude positions from the interior
        candidate grid.  Pass the same list used in ``_setup_tx_state``.
    building_info : dict or None
        Dict returned by ``extract_building_info``; used for the 3-D LOS
        check.  If ``None``, the LOS filter is skipped.
    bs_height : float
        BS elevation (metres above scene origin).  Default 25 m.
    target_z : float
        Height of the LOS target point at the zone centroid.  Default 1.5 m.
    seed : int
        RNG seed for reproducible candidate sampling.
    project_to_edge : bool
        If True, each selected position is projected outward from the zone
        centroid to the farthest zone boundary in that direction, pushing BSs
        into concave protrusions of the zone.  Default False.
    min_building_clearance : float
        Minimum distance (metres) between any candidate position and the
        nearest building footprint.  Building polygons are buffered by this
        amount before being subtracted from the candidate area.  Default 30 m.
    min_bs_separation : float
        Minimum distance (metres) between any two selected BS positions.
        Enforced greedily during cluster selection; if no candidate in a
        cluster satisfies the constraint a warning is printed and the nearest
        candidate is used anyway.  Default 50 m.

    Returns
    -------
    list of [x, y] positions (floats, metres).
    """
    from shapely import contains_xy as _cxy
    from shapely.geometry import Point as _Pt

    rng = np.random.default_rng(seed)

    if "vertices" in zone_params:
        zone_poly = ShapelyPolygon(zone_params["vertices"])
    else:
        cx0, cy0 = zone_params["center"][0], zone_params["center"][1]
        w, h = zone_params["width"], zone_params["height"]
        zone_poly = ShapelyPolygon([
            (cx0 - w / 2, cy0 - h / 2), (cx0 + w / 2, cy0 - h / 2),
            (cx0 + w / 2, cy0 + h / 2), (cx0 - w / 2, cy0 + h / 2),
        ])

    valid_poly = zone_poly
    if building_polygons:
        bldg_union = shapely.ops.unary_union(building_polygons)
        clearance = min_building_clearance if min_building_clearance > 0 else 0
        if clearance > 0:
            bldg_union = bldg_union.buffer(clearance)
        valid_poly = zone_poly.difference(bldg_union)

    minx, miny, maxx, maxy = valid_poly.bounds
    n_cand = max(4000, n_bs * 400)
    xs = rng.uniform(minx, maxx, n_cand * 5)
    ys = rng.uniform(miny, maxy, n_cand * 5)
    inside = _cxy(valid_poly, xs, ys)
    candidates = np.column_stack([xs[inside], ys[inside]])[:n_cand]

    if len(candidates) < n_bs:
        raise ValueError(
            f"seed_bs_positions: only {len(candidates)} valid interior candidates "
            f"for {n_bs} BSs — zone may be too small, heavily built-up, or the "
            f"{min_building_clearance} m building clearance leaves too little space "
            f"(try reducing min_building_clearance)."
        )

    # ── LOS filter ────────────────────────────────────────────────────────────
    # Used only as a preference signal within each k-means cluster below,
    # NOT as a hard gate — avoids starving non-convex zone lobes of candidates.
    if building_info:
        cent = zone_poly.centroid
        cx_los, cy_los = float(cent.x), float(cent.y)
        bldg_polys_2d = []
        bldg_heights  = []
        for info in building_info.values():
            verts = info.get("vertices", [])
            z_h   = info.get("z_height", 0.0)
            if len(verts) >= 3:
                bldg_polys_2d.append(ShapelyPolygon([(v[0], v[1]) for v in verts]))
                bldg_heights.append(float(z_h))

        los_ok = np.array([
            not _los_blocked(
                float(candidates[i, 0]), float(candidates[i, 1]), bs_height,
                cx_los, cy_los, target_z,
                bldg_polys_2d, bldg_heights,
            )
            for i in range(len(candidates))
        ])
    else:
        los_ok = np.ones(len(candidates), dtype=bool)

    # ── K-means++ spread: one BS per zone region ───────────────────────────────
    # Partition ALL interior candidates (no LOS gate) into n_bs clusters so
    # every part of the zone is represented.  Within each cluster, prefer the
    # LOS-clear candidate nearest the cluster centre; fall back to any candidate.
    from sklearn.cluster import KMeans as _KMeans

    km = _KMeans(n_clusters=n_bs, init="k-means++", n_init=10, random_state=seed)
    km.fit(candidates.astype(float))
    labels  = km.labels_           # cluster index per candidate
    centers = km.cluster_centers_  # n_bs × 2 ideal positions

    used   = set()
    result = []
    for k in range(n_bs):
        cluster_idx = np.where(labels == k)[0]
        if len(cluster_idx) == 0:
            continue

        los_in_cluster = cluster_idx[los_ok[cluster_idx]]
        pool_idx = los_in_cluster if len(los_in_cluster) > 0 else cluster_idx

        # Filter by minimum separation from already-placed BSs.
        if result and min_bs_separation > 0:
            placed = np.array(result)
            pool_cands = candidates[pool_idx]
            sep_ok = np.all(
                np.linalg.norm(pool_cands[:, None, :] - placed[None, :, :], axis=2)
                >= min_bs_separation,
                axis=1,
            )
            if sep_ok.any():
                pool_idx = pool_idx[sep_ok]
            else:
                print(
                    f"seed_bs_positions: cluster {k} has no candidate "
                    f">= {min_bs_separation} m from existing BSs; "
                    f"relaxing separation constraint for this station."
                )

        dists = np.linalg.norm(candidates[pool_idx] - centers[k], axis=1)
        best  = pool_idx[int(np.argmin(dists))]
        result.append(candidates[best])
        used.add(int(best))

    # Safety: fill any empty-cluster gaps from remaining unused candidates.
    # Two passes: first honoring min_bs_separation, then relaxing it.
    for strict in (True, False):
        if len(result) >= n_bs:
            break
        for i in range(len(candidates)):
            if len(result) >= n_bs:
                break
            if i in used:
                continue
            if strict and result and min_bs_separation > 0:
                placed = np.array(result)
                if np.linalg.norm(candidates[i] - placed, axis=1).min() < min_bs_separation:
                    continue
            result.append(candidates[i])
            used.add(i)

    positions = [[float(p[0]), float(p[1])] for p in result[:n_bs]]

    if project_to_edge:
        centroid = zone_poly.centroid
        positions = [
            _project_to_zone_edge(p, centroid, zone_poly, building_polygons)
            for p in positions
        ]

    return positions


def setup_bs_transmitters(
    scene,
    zone_params: dict,
    n_bs: int,
    scene_xml_path: str,
    bs_height: float = 25.0,
    target_z: float = 1.5,
    name_prefix: str = "bs",
    seed: int = 42,
    project_to_edge: bool = False,
    initial_power_dbm: float = 23.0,
    power_dbm_bounds: tuple = (0.0, 50.0),
) -> tuple:
    """Create and place n_bs base stations in the scene, ready for optimisation.

    Combines position seeding, scene registration, and ``TxConfig`` creation
    into a single call so adding more BSs only requires changing ``n_bs``.

    Steps
    -----
    1. Load building geometry from *scene_xml_path* for footprint exclusion and
       3-D LOS filtering.
    2. Call :func:`seed_bs_positions` (LOS-aware greedy farthest-point spread).
    3. For each BS: add a :class:`~sionna.rt.Transmitter` to the scene if it
       does not already exist, or update its position if it does.
    4. Return a ``TxConfig`` list and the seeded ``[x, y, z]`` positions.

    Parameters
    ----------
    scene : sionna.rt.Scene
    zone_params : dict
        ``{'vertices': [(x,y), ...]}`` or ``{'center', 'width', 'height'}``.
    n_bs : int
        Number of base stations to create.
    scene_xml_path : str
        Path to the scene XML; used to extract building info.
    bs_height : float
        Transmitter elevation (metres).  Default 25 m.
    target_z : float
        LOS target height at zone centroid (metres).  Default 1.5 m.
    name_prefix : str
        Transmitter names will be ``"{name_prefix}_{i}"`` for i = 0 … n_bs-1.
    seed : int
        Reproducibility seed passed to ``seed_bs_positions``.
    project_to_edge : bool
        If True, each BS is projected outward from the zone centroid to the
        farthest zone boundary in its direction, pushing BSs into concave
        protrusions.  Default False.
    initial_power_dbm : float
        Starting transmit power [dBm].
    power_dbm_bounds : tuple[float, float]
        (min, max) dBm clamp for power optimisation.

    Returns
    -------
    tx_configs : list[TxConfig]
    bs_positions_xyz : list[[x, y, z]]
        Seeded positions including elevation, for use as ``origin_point`` in
        :func:`~boresight_pathsolver.create_zone_mask`.
    """
    from scene_parser import extract_building_info

    building_info = extract_building_info(scene_xml_path)

    # 2-D building polygons for footprint exclusion (no height needed here)
    building_polygons = []
    for info in building_info.values():
        verts = info.get("vertices", [])
        if len(verts) >= 3:
            try:
                p = ShapelyPolygon([(v[0], v[1]) for v in verts])
                if p.is_valid:
                    building_polygons.append(p)
            except Exception:
                pass

    positions_xy = seed_bs_positions(
        zone_params=zone_params,
        n_bs=n_bs,
        building_polygons=building_polygons,
        building_info=building_info,
        bs_height=bs_height,
        target_z=target_z,
        seed=seed,
        project_to_edge=project_to_edge,
    )

    bs_positions_xyz = [[float(x), float(y), bs_height] for x, y in positions_xy]
    tx_configs = []

    for i, (x, y, z) in enumerate(bs_positions_xyz):
        name = f"{name_prefix}_{i}"
        existing = scene.get(name)
        if existing is None:
            tx = Transmitter(name=name, position=[x, y, z])
            scene.add(tx)
        else:
            existing.position = mi.Point3f(float(x), float(y), float(z))

        tx_configs.append(TxConfig(
            name=name,
            on_building=False,
            building_id=0,
            zone_params=zone_params,
            initial_power_dbm=initial_power_dbm,
            power_dbm_bounds=power_dbm_bounds,
        ))
        print(f"  {name}: ({x:.1f}, {y:.1f}, {z:.1f})")

    return tx_configs, bs_positions_xyz


def seed_jammer_positions(
    zone_params: dict,
    n_jammers: Optional[int] = None,
    bs_positions: list = None,
    standoff_distance: float = 100.0,
    min_bs_distance: float = 80.0,
    concave_order: int = 8,
    max_gap_deg: float = 90.0,
    interpolation_factor: int = 1,
    seed: int = 42,
) -> list:
    """Place jammers outside the zone at every boundary feature.

    Produces one jammer per *concave corner* (local radius minimum — inward
    dip of the zone boundary) **and** one jammer per *convex arc peak* (the
    vertex of maximum radius on each arc between consecutive concave corners).
    Together these seed points cover the full perimeter: inward pockets AND
    outward lobes all get a jammer to suppress leakage from every direction.

    Each jammer is placed radially outward from the zone centroid:
    ``position = boundary_vertex + outward_normal × standoff_distance``
    Jammers are then pushed further outward in 10 m steps until they are at
    least ``min_bs_distance`` metres from every BS.

    For box zones jammers are placed evenly around the exterior perimeter.

    Parameters
    ----------
    zone_params : dict
        ``{'vertices': [(x,y), ...]}`` or ``{'center', 'width', 'height'}``.
        Also accepts ``{'center', 'radius'}`` (circle),
        ``{'center', 'side'}`` (square), and
        ``{'center', 'width', 'height', 'angle_deg'}`` (rotated rectangle).
    n_jammers : int or None
        Maximum number of jammers to place.  ``None`` (default) → place one
        per detected boundary feature (all concave corners + all convex arc
        peaks), giving full perimeter coverage.
    bs_positions : list of [x, y] or [x, y, z]
        Friendly base-station positions; only x/y are used.
    standoff_distance : float
        Metres past the zone boundary vertex where the jammer is initially
        seeded.  A larger standoff gives the optimizer room to raise power
        without immediately leaking into the protected zone.  Default 100 m.
    min_bs_distance : float
        Minimum distance (metres) from any BS.  Jammers that start too close
        are pushed further outward in 10 m steps (max 30 steps).
    concave_order : int
        Half-window for local-minimum detection on the radius profile.
    max_gap_deg : float
        Maximum allowed angular gap (degrees, measured from the centroid)
        between consecutive jammers.  After feature-based seeding, any gap
        larger than this threshold gets a fill jammer inserted at the gap
        midpoint.  Default 90°.
    interpolation_factor : int
        Multiplier applied to the jammer count after all other placement
        logic.  ``1`` (default) leaves the set unchanged.  ``2`` inserts one
        contour-following jammer between every consecutive pair, doubling the
        count.  ``k`` inserts ``k-1`` evenly-spaced jammers per gap, so the
        total approaches ``k × original``.  Inserted jammers are ray-cast to
        the zone boundary and placed at ``standoff_distance`` outward, so they
        follow the zone contour rather than straight-line interpolating.
    seed : int
        Unused (placement is deterministic); kept for API consistency.

    Returns
    -------
    list of [x, y] positions (floats, metres).
    """
    from shapely.geometry import Point as _Pt, LineString as _LS

    # ── Normalise non-polygon zone types to vertex list ──────────────────────
    zone_params = dict(zone_params)  # shallow copy — don't mutate caller's dict
    if "radius" in zone_params:
        _cx, _cy = float(zone_params["center"][0]), float(zone_params["center"][1])
        _r = float(zone_params["radius"])
        _thetas = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        zone_params = {"vertices": [(_cx + _r * np.cos(t), _cy + _r * np.sin(t))
                                    for t in _thetas]}
    elif "side" in zone_params:
        _cx, _cy = float(zone_params["center"][0]), float(zone_params["center"][1])
        _s = float(zone_params["side"]) / 2
        zone_params = {"vertices": [(_cx - _s, _cy - _s), (_cx + _s, _cy - _s),
                                    (_cx + _s, _cy + _s), (_cx - _s, _cy + _s)]}
    elif "angle_deg" in zone_params:
        _cx, _cy = float(zone_params["center"][0]), float(zone_params["center"][1])
        _hw = float(zone_params["width"]) / 2
        _hh = float(zone_params["height"]) / 2
        _th = float(np.deg2rad(zone_params["angle_deg"]))
        _c, _s_th = np.cos(_th), np.sin(_th)
        _corners = [(-_hw, -_hh), (_hw, -_hh), (_hw, _hh), (-_hw, _hh)]
        zone_params = {"vertices": [(_cx + x * _c - y * _s_th,
                                     _cy + x * _s_th + y * _c)
                                    for x, y in _corners]}

    bs_xy = np.array([[float(p[0]), float(p[1])] for p in (bs_positions or [])],
                     dtype=float).reshape(-1, 2)

    def _push_to_min_bs_dist(jx, jy, nx, ny):
        for _ in range(30):
            if len(bs_xy) == 0:
                break
            if np.linalg.norm(bs_xy - np.array([jx, jy]), axis=1).min() >= min_bs_distance:
                break
            jx += nx * 10.0
            jy += ny * 10.0
        return jx, jy

    def _place_outward(vertex_idx, vertices, zone_poly, cx, cy, r):
        vx, vy = float(vertices[vertex_idx][0]), float(vertices[vertex_idx][1])
        dx, dy = vx - cx, vy - cy
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            return None
        nx, ny = dx / dist, dy / dist
        jx = vx + nx * standoff_distance
        jy = vy + ny * standoff_distance
        if zone_poly.contains(_Pt(jx, jy)):
            jx += nx * standoff_distance
            jy += ny * standoff_distance
        return jx, jy, nx, ny

    if "vertices" in zone_params:
        vertices = list(zone_params["vertices"])
        zone_poly = ShapelyPolygon(vertices)
        cx, cy = zone_poly.centroid.x, zone_poly.centroid.y

        n_v = len(vertices)
        r = np.array([np.hypot(v[0] - cx, v[1] - cy) for v in vertices])

        # ── Concave corners (local minima in radius) ──────────────────────────
        concave_idx = [
            i for i in range(n_v)
            if all(r[i] < r[(i + k) % n_v] for k in range(1, concave_order + 1))
            and all(r[i] < r[(i - k) % n_v] for k in range(1, concave_order + 1))
        ]

        def _prominence(i):
            nbrs = [r[(i + k) % n_v]
                    for k in range(-concave_order, concave_order + 1) if k != 0]
            return float(np.mean(nbrs)) - float(r[i])

        concave_sorted = sorted(concave_idx, key=_prominence, reverse=True)
        concave_candidates = []
        for i in concave_sorted:
            pt = _place_outward(i, vertices, zone_poly, cx, cy, r)
            if pt is not None:
                concave_candidates.append(pt)

        # ── Convex arc peaks (local maxima between consecutive concave corners) ─
        # Walk the boundary in position order and find the highest-radius vertex
        # on each arc between consecutive concave corners.
        convex_candidates = []
        if concave_idx:
            concave_by_pos = sorted(concave_idx)
            n_c = len(concave_by_pos)
            for arc_i in range(n_c):
                c_start = concave_by_pos[arc_i]
                c_end   = concave_by_pos[(arc_i + 1) % n_c]
                # Vertices strictly between the two concave corners (wraps if needed)
                if c_start < c_end:
                    arc_inner = list(range(c_start + 1, c_end))
                else:
                    arc_inner = list(range(c_start + 1, n_v)) + list(range(0, c_end))
                if not arc_inner:
                    continue
                # Highest-radius vertex on this arc is the convex lobe peak
                peak_idx = max(arc_inner, key=lambda i: r[i])
                pt = _place_outward(peak_idx, vertices, zone_poly, cx, cy, r)
                if pt is not None:
                    convex_candidates.append(pt)

            # Sort convex peaks by their peak radius (highest outward lobe first)
            convex_candidates.sort(key=lambda t: -np.hypot(t[0] - cx, t[1] - cy))
        else:
            # No concavities at all: use the single highest-radius vertex as the
            # only convex peak and pad with equally-spaced exterior points.
            peak_idx = int(np.argmax(r))
            pt = _place_outward(peak_idx, vertices, zone_poly, cx, cy, r)
            if pt is not None:
                convex_candidates.append(pt)

        # Concave jammers first (by prominence), then convex arc peaks (by r).
        all_candidates = concave_candidates + convex_candidates

        # If there are still not enough, pad with equally-spaced exterior points.
        cap = n_jammers if n_jammers is not None else len(all_candidates)
        if len(all_candidates) < cap:
            used_angles = {round(np.arctan2(t[1] - cy, t[0] - cx), 2)
                           for t in all_candidates}
            for angle in np.linspace(0, 2 * np.pi, cap * 4, endpoint=False):
                if len(all_candidates) >= cap:
                    break
                if round(angle, 2) in used_angles:
                    continue
                ray = _LS([(cx, cy),
                            (cx + 2000 * np.cos(angle), cy + 2000 * np.sin(angle))])
                inter = ray.intersection(zone_poly.boundary)
                if inter.is_empty:
                    continue
                bpt = (max(inter.geoms, key=lambda g: g.distance(_Pt(cx, cy)))
                       if hasattr(inter, "geoms") else inter)
                bx, by = float(bpt.x), float(bpt.y)
                nx, ny = np.cos(angle), np.sin(angle)
                all_candidates.append((bx + nx * standoff_distance,
                                       by + ny * standoff_distance, nx, ny))

    else:
        # Box zone: distribute evenly around exterior perimeter
        cx, cy = zone_params["center"][0], zone_params["center"][1]
        w, h   = zone_params["width"], zone_params["height"]
        zone_poly = ShapelyPolygon([
            (cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2),
        ])
        cap = n_jammers if n_jammers is not None else 4
        angles = np.linspace(0, 2 * np.pi, cap, endpoint=False)
        all_candidates = []
        for angle in angles:
            ray = _LS([(cx, cy), (cx + 2000 * np.cos(angle), cy + 2000 * np.sin(angle))])
            inter = ray.intersection(zone_poly.boundary)
            if inter.is_empty:
                continue
            pt = (max(inter.geoms, key=lambda g: g.distance(_Pt(cx, cy)))
                  if hasattr(inter, "geoms") else inter)
            bx, by = float(pt.x), float(pt.y)
            nx, ny = np.cos(angle), np.sin(angle)
            all_candidates.append((bx + nx * standoff_distance,
                                   by + ny * standoff_distance, nx, ny))

    # ── Shared ray-cast helper (used by gap fill and interpolation) ──────────
    _TWO_PI = 2 * np.pi

    def _fill_ray(angle):
        nx, ny = np.cos(angle), np.sin(angle)
        ray = _LS([(cx, cy), (cx + 2000 * nx, cy + 2000 * ny)])
        inter = ray.intersection(zone_poly.boundary)
        if inter.is_empty:
            return None
        bpt = (max(inter.geoms, key=lambda g: g.distance(_Pt(cx, cy)))
               if hasattr(inter, "geoms") else inter)
        bx, by = float(bpt.x), float(bpt.y)
        return (bx + nx * standoff_distance,
                by + ny * standoff_distance, nx, ny)

    # ── Angular gap fill ─────────────────────────────────────────────────────
    # Iteratively subdivide any angular sector (from centroid) larger than
    # max_gap_deg until all gaps are within the threshold.  Using midpoint
    # bisection handles both large initial gaps (e.g. a single starting
    # candidate on a smooth circle) and the n_cands==1 wrap-around edge case.
    if all_candidates and max_gap_deg > 0:
        max_gap_rad = np.deg2rad(max_gap_deg)

        fill_candidates = []
        live_angles = sorted(
            np.arctan2(t[1] - cy, t[0] - cx) for t in all_candidates
        )

        changed = True
        max_iters = 32  # safety cap (2^32 fills would be absurd)
        while changed and max_iters > 0:
            max_iters -= 1
            changed = False
            n_a = len(live_angles)
            new_angles = []
            for i in range(n_a):
                a0 = live_angles[i]
                a1 = live_angles[(i + 1) % n_a]
                # Correct wrap-around: when n_a==1 the gap is the full circle
                gap = (a1 - a0) % _TWO_PI or _TWO_PI
                if gap > max_gap_rad:
                    mid = a0 + gap / 2
                    pt = _fill_ray(mid)
                    if pt is not None:
                        fill_candidates.append(pt)
                        new_angles.append(mid)
                        changed = True
            live_angles = sorted(live_angles + new_angles)

        all_candidates = all_candidates + fill_candidates

    # ── Contour interpolation ────────────────────────────────────────────────
    # Insert (interpolation_factor - 1) evenly-spaced jammers between every
    # consecutive pair of existing jammers (in angular order around the
    # centroid).  Each inserted point is ray-cast to the zone boundary so it
    # follows the zone contour rather than straight-line interpolating.
    if interpolation_factor > 1 and all_candidates:
        n_inserts = interpolation_factor - 1
        sorted_cands = sorted(all_candidates,
                              key=lambda t: np.arctan2(t[1] - cy, t[0] - cx))
        n_sc = len(sorted_cands)
        interp_candidates = []
        for i in range(n_sc):
            a0 = np.arctan2(sorted_cands[i][1] - cy, sorted_cands[i][0] - cx)
            a1 = np.arctan2(sorted_cands[(i + 1) % n_sc][1] - cy,
                            sorted_cands[(i + 1) % n_sc][0] - cx)
            gap = (a1 - a0) % _TWO_PI or _TWO_PI
            for k in range(1, n_inserts + 1):
                mid = a0 + gap * k / (n_inserts + 1)
                pt = _fill_ray(mid)
                if pt is not None:
                    interp_candidates.append(pt)
        all_candidates = sorted_cands + interp_candidates

    cap = n_jammers if n_jammers is not None else len(all_candidates)
    result = []
    for jx, jy, nx, ny in all_candidates[:cap]:
        jx, jy = _push_to_min_bs_dist(jx, jy, nx, ny)
        result.append([float(jx), float(jy)])

    return result


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

    initial_power_dbm = cfg.initial_power_dbm

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
    strides = [6] * len(tx_configs)
    offsets = [sum(strides[:k]) for k in range(len(strides))]
    return strides, offsets


def _jam_param_strides(jam_configs):
    """Return (strides, offsets) for the jammer flat parameter list.

    Each jammer contributes 7 params: [az, el, x, y, z, power_dbm, gate_logit].
    gate_logit is an unconstrained scalar; sigmoid(gate_logit) gates the jammer's field
    contribution so the optimizer can drive it to ~0 to effectively turn the jammer off.
    """
    jam_configs = jam_configs or []
    strides = [7] * len(jam_configs)
    offsets = [sum(strides[:k]) for k in range(len(strides))]
    return strides, offsets


def _push_outside_buildings(x, y, building_polygons):
    """If (x, y) is inside any building polygon, snap it to the nearest exterior point."""
    pt = shapely.geometry.Point(x, y)
    for bp in building_polygons:
        if bp.contains(pt):
            nearest = shapely.ops.nearest_points(pt, bp.exterior)[1]
            return nearest.x, nearest.y
    return x, y


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


def _sample_outside_zone(state: dict, n_inside: int, ground_z: float,
                         outer_half_size: float = 500.0) -> np.ndarray:
    """Sample points outside the zone, area-proportional to n_inside.

    The sample count is scaled by (outer_ring.area / zone.area) so that the
    spatial density of outside samples matches the inside density, giving each
    square metre equal weight in the loss function.

    The outer region is a square of ±outer_half_size metres centred on the zone
    centroid with the zone polygon and building footprints subtracted.  Using a
    fixed square (rather than a scaled copy of the zone) gives a consistent,
    zone-shape-invariant outer sampling region across all simulation geometries.
    """
    from shapely import contains_xy as _cxy
    import shapely.ops

    box_poly = state["box_polygon"]
    centroid = box_poly.centroid
    cx, cy = centroid.x, centroid.y
    h = outer_half_size
    outer_poly = ShapelyPolygon([
        (cx - h, cy - h), (cx + h, cy - h),
        (cx + h, cy + h), (cx - h, cy + h),
    ])
    outer_ring = outer_poly.difference(box_poly)

    cached_bldgs = state.get("cached_building_polygons", [])
    if cached_bldgs:
        bldg_union = shapely.ops.unary_union(cached_bldgs)
        outer_ring = outer_ring.difference(bldg_union)

    zone_area = box_poly.area
    ring_area  = outer_ring.area
    area_ratio = ring_area / zone_area if zone_area > 0 else 1.0
    n = max(1, round(n_inside * area_ratio))

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
    gamma_db=0.0,
    lambda_in=0.0,
    lambda_out=10.0,
    lambda_uniform=0.0,
    lambda_min_j=1.0,
    min_sinr_db=10.0,
    soft_mean_weight=0.25,
    soft_mean_in_weight=0.0,
    epsilon=1e-30,
    lambda_spread=0.0,
    spread_min_dist=100.0,
):
    """Containment SIR loss with hinge penalties.

    Loss components (all in dB-space):
      L_in      : squared hinge on (gamma - SINR) for inside cells
                  -> penalizes service holes
      L_out     : squared hinge on (SINR - gamma) for outside cells
                  -> penalizes detectable leakage
      L_uniform : variance of inside-cell SINR + squared hinge below min_sinr_db
                  -> penalizes hotspot-driven coverage (non-uniform distribution)
      L_min_j   : mean gate value across jammers
                  -> sparsity pressure; drives unused jammers off

    Parameters
    ----------
    gamma_db : float
        Detection / decode threshold in dB. Set from threat model.
    lambda_in, lambda_out : float
        Loss-term weights. Sweep lambda_out / lambda_in for the
        leakage / service-hole Pareto frontier.
    lambda_uniform : float
        Weight for the uniformity penalty (L_uniform). Default 0 (disabled).
    min_sinr_db : float
        Minimum acceptable SINR inside the zone (dB). Cells below this
        threshold contribute a squared hinge to L_uniform, on top of the
        variance term. Default 10 dB.
    """
    deg2rad = Float(float(np.pi / 180.0))
    dr.disable_grad(deg2rad)

    _, offsets = _param_strides(tx_configs)

    # ------------------------------------------------------------------
    # Build power-scale factors for each gNB (DrJit-differentiable)
    # ------------------------------------------------------------------
    pow_scales = []
    for i, cfg in enumerate(tx_configs):
        pow_i = all_params[offsets[i] + 5]
        dr.enable_grad(pow_i.array)
        scale_i = dr.power(
            Float(10.0),
            (pow_i - Float(float(ref_powers_dbm[i]))) / Float(10.0),
        )
        pow_scales.append(scale_i)

    # ------------------------------------------------------------------
    # Set TX orientations and positions (independent jitter per axis)
    # ------------------------------------------------------------------
    for i, cfg in enumerate(tx_configs):
        b    = offsets[i]
        az_i = all_params[b];     dr.enable_grad(az_i.array)
        el_i = all_params[b + 1]; dr.enable_grad(el_i.array)
        x_i  = all_params[b + 2]; dr.enable_grad(x_i.array)
        y_i  = all_params[b + 3]; dr.enable_grad(y_i.array)
        z_i  = all_params[b + 4]; dr.enable_grad(z_i.array)

        # Independent samples per DOF (previously a single scalar reused)
        # jit_yaw   = Float(float(np.random.normal(0.0, 0.5 * np.pi / 180.0)))
        # jit_pitch = Float(float(np.random.normal(0.0, 0.5 * np.pi / 180.0)))
        # jit_x     = Float(float(np.random.normal(0.0, 0.1)))
        # jit_y     = Float(float(np.random.normal(0.0, 0.1)))
        # jit_z     = Float(float(np.random.normal(0.0, 0.1)))

        jit_yaw   = Float(float(0.0))
        jit_pitch = Float(float(0.0))
        jit_x     = Float(float(0.0))
        jit_y     = Float(float(0.0))
        jit_z     = Float(float(0.0))

        for j in (jit_yaw, jit_pitch, jit_x, jit_y, jit_z):
            dr.disable_grad(j)

        yaw   = az_i * deg2rad + jit_yaw
        pitch = -(el_i * deg2rad) + jit_pitch
        roll  = Float(0.0); dr.disable_grad(roll)

        scene.get(cfg.name).orientation = [yaw, pitch, roll]
        scene.get(cfg.name).position = [
            x_i + jit_x,
            y_i + jit_y,
            z_i + jit_z,
        ]

    # ------------------------------------------------------------------
    # gNB PathSolver call (directional pattern, scene)
    # ------------------------------------------------------------------
    paths = p_solver(
        scene,
        los=True,
        refraction=True,
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
    total_gnb_params = 6 * len(tx_configs)
    _, jam_offsets = _jam_param_strides(jam_configs)

    # ------------------------------------------------------------------
    # Set jammer positions and build power-scale factors
    # ------------------------------------------------------------------
    jam_pow_scales = []
    jam_gates     = []   # sigmoid(gate_logit) per jammer — 0 = off, 1 = on
    jam_power_vecs = []
    jam_xy_positions = []  # (xj, yj) DrJIT Floats for spread penalty
    J = 0
    if jam_configs:
        for j, jcfg in enumerate(jam_configs):
            b    = total_gnb_params + jam_offsets[j]
            azj  = all_params[b];     dr.enable_grad(azj.array)
            elj  = all_params[b + 1]; dr.enable_grad(elj.array)
            xj   = all_params[b + 2]; dr.enable_grad(xj.array)
            yj   = all_params[b + 3]; dr.enable_grad(yj.array)
            zj   = all_params[b + 4]; dr.enable_grad(zj.array)
            pj   = all_params[b + 5]; dr.enable_grad(pj.array)
            glj  = all_params[b + 6]; dr.enable_grad(glj.array)
            gate_j = dr.rcp(Float(1.0) + dr.exp(-glj))
            jam_gates.append(gate_j)
            jam_xy_positions.append((xj, yj))

            jit_yaw   = Float(float(np.random.normal(0.0, 0.5 * np.pi / 180.0)))
            jit_pitch = Float(float(np.random.normal(0.0, 0.5 * np.pi / 180.0)))
            jit_x = Float(float(np.random.normal(0.0, 2.0)))
            jit_y = Float(float(np.random.normal(0.0, 2.0)))
            jit_z = Float(float(np.random.normal(0.0, 0.1)))
            for _jv in (jit_yaw, jit_pitch, jit_x, jit_y, jit_z):
                dr.disable_grad(_jv)

            yaw_j   = azj * deg2rad + jit_yaw
            pitch_j = -(elj * deg2rad) + jit_pitch
            roll_j  = Float(0.0); dr.disable_grad(roll_j)

            jam_scene.get(jcfg.name).orientation = [yaw_j, pitch_j, roll_j]
            jam_scene.get(jcfg.name).position = [
                xj + jit_x,
                yj + jit_y,
                zj + jit_z,
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
        for j in range(J):
            jp_raw = _extract_per_rx_power(jh_real, jh_imag, j)
            jam_power_vecs.append(jp_raw * jam_pow_scales[j] * jam_gates[j])

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
    zero_f  = Float(0.0);              dr.disable_grad(zero_f)

    # Inside / outside split
    if num_inside is not None and num_inside < dr.width(best_sinr):
        n_in = int(num_inside)
        sinr_in  = sinr_db[:n_in]
        sinr_out = sinr_db[n_in:]

        # L_in: squared hinge + always-active soft mean term.
        # The hinge alone is silent while inside cells stay above gamma, so
        # jammer activation faces no resistance until inside coverage collapses.
        # The soft mean term provides a constant gradient that opposes any factor
        # (including friendly jammers) from lowering inside SINR, competing
        # symmetrically against the soft_mean_weight term in L_out.
        gamma_in = gamma_f
        dr.disable_grad(gamma_in)
        deficit = dr.maximum(gamma_in - sinr_in, zero_f)
        loss_inside = (dr.mean(deficit * deficit)
                       - Float(float(soft_mean_in_weight)) * dr.mean(sinr_in))

        # L_out: squared hinge + soft mean term.
        # The hinge alone is silent when BS optimisation already pushed most
        # outside cells below gamma, leaving no gradient to activate jammers.
        # The soft mean term provides an always-active signal to push outside
        # SINR downward regardless of whether cells are above the threshold.
        excess = dr.maximum(sinr_out - gamma_f, zero_f)
        loss_outside = dr.mean(excess * excess) + Float(float(soft_mean_weight)) * dr.mean(sinr_out)
    else:
        # No partition provided: treat all cells as "inside"
        deficit = dr.maximum(gamma_f - sinr_db, zero_f)
        loss_inside = dr.mean(deficit * deficit)
        loss_outside = Float(0.0); dr.disable_grad(loss_outside)

    # L_uniform: variance of inside-cell SINR + squared hinge below min_sinr_db.
    # Var = E[X²] - E[X]² avoids a scalar-broadcast subtraction across the array.
    # Both terms fire only when inside receivers exist; disabled when lambda_uniform=0.
    loss_uniform = Float(0.0); dr.disable_grad(loss_uniform)
    if lambda_uniform != 0.0 and num_inside is not None and num_inside < dr.width(best_sinr):
        n_in = int(num_inside)
        sinr_in_u = sinr_db[:n_in]
        mean_in    = dr.mean(sinr_in_u)
        variance   = dr.mean(sinr_in_u * sinr_in_u) - mean_in * mean_in
        floor_f    = Float(float(min_sinr_db)); dr.disable_grad(floor_f)
        floor_def  = dr.maximum(floor_f - sinr_in_u, zero_f)
        loss_floor = dr.mean(floor_def * floor_def)
        loss_uniform = variance + loss_floor

    # L_min_j: mean gate value across jammers. Gate = sigmoid(gate_logit) ∈ (0,1).
    # Penalizing the mean gate pushes unused jammers' logits negative → gate → 0 → off.
    loss_min_j = Float(0.0); dr.disable_grad(loss_min_j)
    if J > 0:
        for gate_j in jam_gates:
            loss_min_j = loss_min_j + gate_j
        loss_min_j = loss_min_j / Float(float(J))

    # L_spread: penalize pairwise proximity of jammer XY positions.
    # Uses a soft inverse: spread_ref² / (dist² + spread_ref²), which equals 1.0
    # at zero separation and 0.5 at spread_min_dist metres. Minimising this term
    # pushes jammers apart, giving each one a distinct sector to cover.
    if J > 1 and lambda_spread != 0.0:
        n_pairs = J * (J - 1) // 2
        spread_ref_sq = Float(float(spread_min_dist * spread_min_dist))
        spread_sum = Float(0.0); dr.disable_grad(spread_sum)
        for ji in range(J):
            for jk in range(ji + 1, J):
                dx = jam_xy_positions[ji][0] - jam_xy_positions[jk][0]
                dy = jam_xy_positions[ji][1] - jam_xy_positions[jk][1]
                dist_sq = dx * dx + dy * dy
                spread_sum = spread_sum + spread_ref_sq / (dist_sq + spread_ref_sq)
        loss_spread = spread_sum / Float(float(n_pairs))
    else:
        loss_spread = Float(0.0); dr.disable_grad(loss_spread)

    # ------------------------------------------------------------------
    # Compose total loss
    # ------------------------------------------------------------------
    lam_in      = Float(float(lambda_in));      dr.disable_grad(lam_in)
    lam_out     = Float(float(lambda_out));     dr.disable_grad(lam_out)
    lam_uniform = Float(float(lambda_uniform)); dr.disable_grad(lam_uniform)
    lam_loss_min_j = Float(float(lambda_min_j)); dr.disable_grad(lam_loss_min_j)
    lam_spread  = Float(float(lambda_spread));  dr.disable_grad(lam_spread)

    total = (lam_in         * loss_inside +
             lam_out        * loss_outside +
             lam_uniform    * loss_uniform +
             lam_loss_min_j * loss_min_j +
             lam_spread     * loss_spread)

    # Optional debug (comment out for production training)
    if num_inside is not None and num_inside < dr.width(best_sinr):
        n_in = int(num_inside)
        print(f"  [dbg] mean SINR_dB inside : {dr.mean(sinr_db[:n_in])}")
        print(f"  [dbg] mean SINR_dB outside: {dr.mean(sinr_db[n_in:])}")
    print(f"  [dbg] L_in={dr.mean(loss_inside)}  L_out={dr.mean(loss_outside)}  "
          f"L_uniform={dr.mean(loss_uniform)}  L_min_j={dr.mean(loss_min_j)}  "
          f"L_spread={dr.mean(loss_spread)}")

    return total


def _make_compute_sir_loss(
    N, tx_configs, tx_states, scene, p_solver,
    noise_power, rx_objects, ref_powers_dbm,
    jam_configs, jam_scene, jam_rx_objects, jam_objects,
    num_inside=None,
    gamma_db=0.0,
    lambda_in=1.0,
    lambda_out=10.0,
    lambda_uniform=0.0,
    lambda_min_j=1.0,
    min_sinr_db=10.0,
    soft_mean_weight=0.25,
    soft_mean_in_weight=0.0,
    epsilon=1e-30,
    lambda_spread=0.0,
    spread_min_dist=100.0,
):
    """Build and return the @dr.wrap-decorated SIR loss function.

    Uses exec() to produce a function with a *fixed* positional signature
    matching exactly the number of scalar parameters — required by @dr.wrap.
    The flat signature is: [gNB params...] + [jammer params...]
    where each gNB contributes [az, el, x, y, z, power_dbm] and
    each jammer contributes [x, y, z, power_dbm, gate_logit].
    """
    _, gnb_offsets = _param_strides(tx_configs)
    jam_strides, jam_offsets = _jam_param_strides(jam_configs)
    total_jam_params = (jam_offsets[-1] + jam_strides[-1]) if jam_configs else 0

    # gNB param names: p0, p1, ... layout: [az, el, x, y, z, power_dbm]
    arg_names = []
    for i, cfg in enumerate(tx_configs):
        b = gnb_offsets[i]
        arg_names += [f"p{b}", f"p{b+1}", f"p{b+2}", f"p{b+3}", f"p{b+4}", f"p{b+5}"]

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
        f"                 num_inside=_num_inside,\n"
        f"                 gamma_db=_gamma_db,\n"
        f"                 lambda_in=_lambda_in,\n"
        f"                 lambda_out=_lambda_out,\n"
        f"                 lambda_uniform=_lambda_uniform,\n"
        f"                 lambda_min_j=_lambda_min_j,\n"
        f"                 min_sinr_db=_min_sinr_db,\n"
        f"                 soft_mean_weight=_soft_mean_weight,\n"
        f"                 soft_mean_in_weight=_soft_mean_in_weight,\n"
        f"                 epsilon=_epsilon,\n"
        f"                 lambda_spread=_lambda_spread,\n"
        f"                 spread_min_dist=_spread_min_dist)\n"
    )

    globs = {
        "_body":           _sir_loss_body,
        "_N":              N,
        "_cfgs":           tx_configs,
        "_states":         tx_states,
        "_scene":          scene,
        "_psolver":        p_solver,
        "_noise":          noise_power,
        "_rxobj":          rx_objects,
        "_refpow":         ref_powers_dbm,
        "_jam_cfgs":       jam_configs,
        "_jam_scene":      jam_scene,
        "_jam_rxobj":      jam_rx_objects,
        "_jam_obj":        jam_objects,
        "_num_inside":     num_inside,
        "_gamma_db":       gamma_db,
        "_lambda_in":      lambda_in,
        "_lambda_out":     lambda_out,
        "_lambda_uniform":   lambda_uniform,
        "_lambda_min_j":     lambda_min_j,
        "_min_sinr_db":      min_sinr_db,
        "_soft_mean_weight":    soft_mean_weight,
        "_soft_mean_in_weight": soft_mean_in_weight,
        "_epsilon":             epsilon,
        "_lambda_spread":    lambda_spread,
        "_spread_min_dist":  spread_min_dist,
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
    jammer_array: Optional[AntennaArray] = None,
    jam_configs: Optional[list] = None,
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
    gamma_db: float = 0.0,
    lambda_in: float = 1.0,
    lambda_out: float = 10.0,
    lambda_uniform: float = 0.0,
    lambda_min_j: float = 1.0,
    min_sinr_db: float = 10.0,
    soft_mean_weight: float = 0.25,
    soft_mean_in_weight: float = 0.0,
    freeze_bs: bool = False,
    outside_half_size: float = 500.0,
    lambda_spread: float = 0.0,
    spread_min_dist: float = 100.0,
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
    gamma_db : float
        Detection threshold in dB. Cells above/below this drive the hinge losses.
    lambda_in, lambda_out : float
        Weights for the coverage/leakage hinge terms. Sweep their ratio for the
        service-hole / leakage Pareto frontier.
    lambda_uniform : float
        Weight for the uniformity penalty. When > 0, penalizes variance of
        inside-cell SINR (hotspot suppression) plus a squared hinge on cells
        below ``min_sinr_db``. Start around 0.5–2.0 relative to lambda_in.
    lambda_min_j : float
        Weight for the jammer gate sparsity penalty. Default 1.0.
    min_sinr_db : float
        Minimum acceptable SINR inside the zone in dB. Cells below this
        contribute a squared hinge to L_uniform. Default 10 dB.
    outside_half_size : float
        Half-side of the square outer sampling region in metres. The outside
        sample count scales with this area, so larger values produce more
        outside receivers. Default 500 m.

    Returns
    -------
    dict with keys per TX name plus "joint":
        {
          tx_name: {
            "best_angles":      [az_deg, el_deg],
            "final_position":   [x, y, z],
            "initial_angles":   [az_deg, el_deg],
            "initial_position": [x, y, z],
            "best_power_dbm":   float,
            "az_history":       list,
            "el_history":       list,
            "power_history":    list,
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
    # 2. Pre-sample fixed receiver positions once for the shared zone.
    #    Outside count is area-proportional to n_inside so each m² contributes
    #    equally to the loss (see _sample_outside_zone).
    #    Receivers 0..n_inside-1        → inside zone
    #    Receivers n_inside..total_rx-1 → outer ring
    # ------------------------------------------------------------------
    n_inside = num_sample_points
    ground_z = float(map_config["center"][2]) if len(map_config["center"]) > 2 else 0.0
    pts = _sample_zone_points(tx_states[0], tx_configs[0], n_inside,
                              sampler, "full", ground_z)
    out_pts = _sample_outside_zone(tx_states[0], n_inside, ground_z,
                                   outer_half_size=outside_half_size)
    n_outside = len(out_pts)
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
    jam_scene = None
    jam_objects = {}
    jam_rx_objects = {}
    jam_params = []

    if jammer_array is not None and jam_configs is not None:
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
            pos = mi.Point3f([jx, jy, 25.0])
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
    for state in tx_states:
        state["current_sample_points"] = pts
        state["outside_sample_points"] = out_pts
    for k, pos in enumerate(pts):
        rx_name = f"opt_rx_{k}"
        p3 = mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2]))
        rx_objects[rx_name].position = p3
        if jam_rx_objects:
            jam_rx_objects[rx_name].position = p3
    for k, pos in enumerate(out_pts):
        rx_name = f"opt_rx_{n_inside + k}"
        p3 = mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2]))
        rx_objects[rx_name].position = p3
        if jam_rx_objects:
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
        gamma_db=gamma_db,
        lambda_in=lambda_in,
        lambda_out=lambda_out,
        lambda_uniform=lambda_uniform,
        lambda_min_j=lambda_min_j,
        min_sinr_db=min_sinr_db,
        soft_mean_weight=soft_mean_weight,
        soft_mean_in_weight=soft_mean_in_weight,
        lambda_spread=lambda_spread,
        spread_min_dist=spread_min_dist,
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
        params.append(torch.tensor(float(np.clip(state["tx_position"][2], 40.0, 60.0)),
                                   device="cuda", dtype=torch.float32, requires_grad=True))
        params.append(torch.tensor(state["initial_power_dbm"], device="cuda",
                                   dtype=torch.float32, requires_grad=True))

    # Jammer params: [az, el, x, y, z, power_dbm, gate_logit] per jammer.
    zone_centroid = tx_states[0]["box_polygon"].centroid
    look_at_xyz   = [zone_centroid.x, zone_centroid.y, 1.5]
    for jcfg in (jam_configs or []):
        jammer = jam_objects[jcfg.name]
        init_pos = jammer.position.numpy().flatten()
        init_pos_3d = [float(init_pos[0]), float(init_pos[1]), 50.0]
        if jcfg.initial_azimuth_deg is not None and jcfg.initial_elevation_deg is not None:
            init_az = jcfg.initial_azimuth_deg
            init_el = jcfg.initial_elevation_deg
        else:
            init_az, init_el = compute_initial_angles_from_position(
                init_pos_3d, look_at_xyz, verbose=False
            )
        jam_params.append(torch.tensor(init_az, device="cuda",
                                       dtype=torch.float32, requires_grad=True))
        jam_params.append(torch.tensor(init_el, device="cuda",
                                       dtype=torch.float32, requires_grad=True))
        jam_params.append(torch.tensor(float(init_pos[0]), device="cuda",
                                       dtype=torch.float32, requires_grad=True))
        jam_params.append(torch.tensor(float(init_pos[1]), device="cuda",
                                       dtype=torch.float32, requires_grad=True))
        jam_params.append(torch.tensor(50.0, device="cuda",
                                       dtype=torch.float32, requires_grad=True))
        jam_params.append(torch.tensor(jcfg.initial_power_dbm, device="cuda",
                                       dtype=torch.float32, requires_grad=True))
        # Gate logit: sigmoid(4.0) ≈ 0.98 — jammer starts on; optimizer drives it
        # negative to turn the jammer off when it isn't needed.
        jam_params.append(torch.tensor(4.0, device="cuda",
                                       dtype=torch.float32, requires_grad=True))

    opt_params = jam_params if freeze_bs else params + jam_params
    optimizer = torch.optim.Adam(opt_params, lr=learning_rate, betas=(0.9, 0.999))
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
            xp_k, yp_k, zp_k = pvals[b + 2], pvals[b + 3], pvals[b + 4]
            scene.get(cfg_k.name).orientation = [
                float(np.deg2rad(az_k)), -float(np.deg2rad(el_k)), 0.0
            ]
            scene.get(cfg_k.name).position = mi.Point3f(
                float(xp_k), float(yp_k), float(zp_k)
            )
            scene.get(cfg_k.name).power_dbm = [float(pvals[b + 5])]

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

                # Elevation: clamp to [-90, 0] (downward-facing only)
                el_t.clamp_(-90.0, 0.0)

                # Position: project to building polygon if this TX is building-mounted.
                if cfg.on_building:
                    proj_x, proj_y = state["tx_placement"].project_to_polygon(
                        x_t.item(), y_t.item()
                    )
                    x_t.data.fill_(proj_x)
                    y_t.data.fill_(proj_y)
                else:
                   # Keep free-roaming TXs outside building footprints.
                    bldgs = state["cached_building_polygons"]
                    if bldgs:
                        px, py = _push_outside_buildings(x_t.item(), y_t.item(), bldgs)
                        x_t.data.fill_(px)
                        y_t.data.fill_(py)

                # Z clamp [40, 50]
                params[b + 4].clamp_(40.0, 50.0)
                # Power clamp
                params[b + 5].clamp_(*cfg.power_dbm_bounds)

            # Clamp jammer positions outside building footprints and power to bounds.
            if jam_configs:
                _, joff = _jam_param_strides(jam_configs)
                bldgs = tx_states[0]["cached_building_polygons"] if tx_states else []
                for j, jcfg in enumerate(jam_configs):
                    if bldgs:
                        jx = jam_params[joff[j] + 2].item()
                        jy = jam_params[joff[j] + 3].item()
                        jx, jy = _push_outside_buildings(jx, jy, bldgs)
                        jam_params[joff[j] + 2].data.fill_(jx)
                        jam_params[joff[j] + 3].data.fill_(jy)
                    # Z clamp [30, 60]
                    jam_params[joff[j] + 4].clamp_(5.0, 50.0)
                    jam_params[joff[j] + 5].clamp_(*jcfg.power_dbm_bounds)
                    jam_params[joff[j] + 1].clamp_(*jcfg.elevation_bounds)

        # Track histories
        loss_val = float(loss.item())
        loss_history.append(loss_val)

        for i, cfg in enumerate(tx_configs):
            b = offsets[i]
            az_val = float(params[b].item())
            el_val = float(params[b + 1].item())
            tx_states[i]["az_history"].append(az_val)
            tx_states[i]["el_history"].append(el_val)
            pw_val = float(params[b + 5].item())
            tx_states[i]["power_history"].append(pw_val)

        # Accumulate final-window values (last 10 iters)
        window_start = max(0, num_iterations - 10)
        if iteration >= window_start:
            for i, cfg in enumerate(tx_configs):
                b = offsets[i]
                final_bufs[i]["az"].append(float(params[b].item()))
                final_bufs[i]["el"].append(float(params[b + 1].item()))
                final_bufs[i]["pow"].append(float(params[b + 5].item()))

        # Expose current TX positions for visualisation callbacks
        for i, (cfg, state) in enumerate(zip(tx_configs, tx_states)):
            b = offsets[i]
            state["current_tx_position"] = [
                float(params[b + 2].item()),
                float(params[b + 3].item()),
                float(params[b + 4].item()),
            ]

        if on_iteration_callback is not None:
            if jam_configs:
                _, joff = _jam_param_strides(jam_configs)
                jam_positions = [
                    [float(jam_params[joff[j] + 2].item()),
                     float(jam_params[joff[j] + 3].item())]
                    for j in range(len(jam_configs))
                ]
            else:
                jam_positions = []
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
        final_z = float(params[b + 4].item())

        yaw_r, pitch_r = azimuth_elevation_to_yaw_pitch(best_az, best_el)
        scene.get(cfg.name).orientation = mi.Point3f(float(yaw_r), float(pitch_r), 0.0)
        scene.get(cfg.name).position    = mi.Point3f(float(final_x), float(final_y), float(final_z))
        best_pow = float(np.mean(final_bufs[i]["pow"])) if final_bufs[i]["pow"] else float(params[b + 5].item())
        scene.get(cfg.name).power_dbm = [best_pow]
        state["best_power_dbm"] = best_pow
        state["best_angles"]    = [best_az, best_el]
        state["final_position"] = [final_x, final_y, final_z]

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
            b    = joff[j]
            azf  = float(jam_params[b].item())
            elf  = float(jam_params[b + 1].item())
            xf   = float(jam_params[b + 2].item())
            yf   = float(jam_params[b + 3].item())
            zf   = float(jam_params[b + 4].item())
            pf   = float(jam_params[b + 5].item())
            glf  = float(jam_params[b + 6].item())
            gate_f = 1.0 / (1.0 + np.exp(-glf))
            jam_scene.get(jcfg.name).orientation = [
                float(np.deg2rad(azf)), -float(np.deg2rad(elf)), 0.0
            ]
            jam_scene.get(jcfg.name).position  = mi.Point3f(xf, yf, zf)
            jam_scene.get(jcfg.name).power_dbm = [pf]
            jammers_final[jcfg.name] = {
                "final_azimuth_deg":   azf,
                "final_elevation_deg": elf,
                "final_position":      [xf, yf, zf],
                "final_power_dbm":     pf,
                "initial_power_dbm":   jcfg.initial_power_dbm,
                "final_gate":          gate_f,
                "active":              gate_f >= 0.5,
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
                status = "ON " if jd.get("active", True) else "OFF"
                gate   = jd.get("final_gate", float("nan"))
                print(f"  {jcfg.name} [{status} gate={gate:.3f}]: "
                      f"Az={jd['final_azimuth_deg']:.1f}°, El={jd['final_elevation_deg']:.1f}°, "
                      f"pos=({jd['final_position'][0]:.1f}, {jd['final_position'][1]:.1f}, {jd['final_position'][2]:.1f}), "
                      f"pwr={jd['final_power_dbm']:.1f} dBm")
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
    sinr_bs_only_ref: "np.ndarray | None" = None,
    building_polygons: list = None,
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
        n_bs = max(1, len(tx_configs))
        return solver(
            scene,
            max_depth=8,
            samples_per_tx=max(1, int(1e9) // n_bs),
            cell_size=list(map_config["cell_size"]),
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

    # Build a mask of grid cells that fall inside building footprints so they
    # can be excluded from outdoor statistics.
    _building_mask = np.zeros(grid_shape, dtype=bool)
    if building_polygons:
        from shapely.geometry import Point
        from shapely.prepared import prep as shp_prep
        cx_m, cy_m = map_config["center"][0], map_config["center"][1]
        sx_m, sy_m = map_config["size"][0],   map_config["size"][1]
        cw,   ch   = map_config["cell_size"][0], map_config["cell_size"][1]
        H, W = grid_shape
        xs = np.linspace(cx_m - sx_m / 2 + cw / 2, cx_m + sx_m / 2 - cw / 2, W)
        ys = np.linspace(cy_m - sy_m / 2 + ch / 2, cy_m + sy_m / 2 - ch / 2, H)
        Xg, Yg = np.meshgrid(xs, ys)
        pts  = np.column_stack((Xg.ravel(), Yg.ravel()))
        flat = np.zeros(H * W, dtype=bool)
        for bp in building_polygons:
            minx, miny, maxx, maxy = bp.bounds
            in_bbox = (
                (pts[:, 0] >= minx) & (pts[:, 0] <= maxx) &
                (pts[:, 1] >= miny) & (pts[:, 1] <= maxy)
            )
            idxs = np.where(in_bbox)[0]
            if len(idxs) > 0:
                pbp = shp_prep(bp)
                for idx in idxs:
                    if pbp.contains(Point(pts[idx, 0], pts[idx, 1])):
                        flat[idx] = True
        _building_mask = flat.reshape(H, W)

    # Outdoor-only masks: exclude cells inside building footprints.
    outdoor_outside = outside_mask & ~_building_mask
    outdoor_inside  = union_inside  & ~_building_mask

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
        active_indices = []
        for j, jcfg in enumerate(jammer_configs):
            if jcfg.name not in jammers_data:
                continue
            jd  = jammers_data[jcfg.name]
            pos = jd["final_position"]
            if "final_azimuth_deg" in jd and "final_elevation_deg" in jd:
                jam_scene.get(jcfg.name).orientation = [
                    float(np.deg2rad(jd["final_azimuth_deg"])),
                    -float(np.deg2rad(jd["final_elevation_deg"])),
                    0.0,
                ]
            jam_scene.get(jcfg.name).position = mi.Point3f(*[float(v) for v in pos])
            if jd.get("active", True):
                jam_scene.get(jcfg.name).power_dbm = [float(jd["final_power_dbm"])]
                active_indices.append(j)
            else:
                jam_scene.get(jcfg.name).power_dbm = [-200.0]  # effectively zero

        if active_indices:
            print(f"Computing RadioMap for {len(active_indices)}/{len(jammer_configs)} active jammers...")
            jam_solver = RadioMapSolver()
            n_jam = max(1, len(active_indices))
            jam_rm = jam_solver(
                jam_scene,
                max_depth=8,
                samples_per_tx=max(1, int(1e9) // n_jam),
                cell_size=list(map_config["cell_size"]),
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
            jam_rss = np.nan_to_num(jam_rm.rss.numpy(), nan=0.0)  # (J_total, H, W)
            jam_interference_map = np.sum(jam_rss[active_indices], axis=0)  # (H, W)
        else:
            print("All jammers inactive — skipping jammer RadioMap.")

    has_jammers = jam_interference_map is not None
    sinr_bs_only = _bs_sinr_field(rss_list_2d, jam_map=None)
    sinr_bs_jam  = _bs_sinr_field(rss_list_2d, jam_map=jam_interference_map) if has_jammers else None

    if has_jammers:
        mean_jam_W  = float(np.mean(jam_interference_map))
        mean_delta  = float(np.mean(sinr_bs_only - sinr_bs_jam))
        print(f"[verify] jam_interference_map: mean={mean_jam_W:.3e} W  "
              f"| mean SINR shift (BS-only − BS+Jam) = {mean_delta:+.3f} dB")

    # ------------------------------------------------------------------
    # Per-TX containment metrics — always report BS-only; add jam when present
    # ------------------------------------------------------------------
    stats = {}
    def _fmt(v): return f"{100*v:5.1f}%" if v is not None else "  n/a "

    for cfg in tx_configs:
        r       = multi_result[cfg.name]
        in_mask = inside_masks[cfg.name]

        in_bs   = sinr_bs_only[in_mask & ~_building_mask]
        out_bs  = sinr_bs_only[outdoor_outside]
        rho_leak_bs = float(np.mean(out_bs >= gamma_db)) if out_bs.size else None
        rho_hole_bs = float(np.mean(in_bs  <  gamma_db)) if in_bs.size  else None

        init_params = {"azimuth": r["initial_angles"][0], "elevation": r["initial_angles"][1],
                       "position": r["initial_position"]}
        opt_params  = {"azimuth": r["best_angles"][0],    "elevation": r["best_angles"][1],
                       "position": r["final_position"]}
        if "best_power_dbm" in r:
            init_params["power_dbm"] = r.get("initial_power_dbm", float("nan"))
            opt_params["power_dbm"]  = r["best_power_dbm"]

        entry = {
            "initial_params":   init_params,
            "optimized_params": opt_params,
            "sinr_inside":      _summarize_sinr(in_bs),
            "sinr_outside":     _summarize_sinr(out_bs),
            "containment":      {"rho_leak": rho_leak_bs, "rho_hole": rho_hole_bs},
            "az_history":       r.get("az_history", []),
            "el_history":       r.get("el_history", []),
            "power_history":    r.get("power_history", []),
        }

        if has_jammers:
            in_jam  = sinr_bs_jam[in_mask & ~_building_mask]
            out_jam = sinr_bs_jam[outdoor_outside]
            rho_leak_jam = float(np.mean(out_jam >= gamma_db)) if out_jam.size else None
            rho_hole_jam = float(np.mean(in_jam  <  gamma_db)) if in_jam.size  else None
            entry["sinr_inside_jam"]  = _summarize_sinr(in_jam)
            entry["sinr_outside_jam"] = _summarize_sinr(out_jam)
            entry["containment_jam"]  = {"rho_leak": rho_leak_jam, "rho_hole": rho_hole_jam}

        stats[cfg.name] = entry

    # ------------------------------------------------------------------
    # Print containment summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"BS-ONLY SINR CONTAINMENT  (gamma = {gamma_db:+.1f} dB)")
    print(f"{'='*70}")
    for cfg in tx_configs:
        s  = stats[cfg.name]
        c  = s["containment"]
        si = s["sinr_inside"]
        so = s["sinr_outside"]
        print(f"  {cfg.name}:")
        print(f"    rho_leak  (outside >= gamma): {_fmt(c['rho_leak'])}")
        print(f"    rho_hole  (inside  <  gamma): {_fmt(c['rho_hole'])}")
        if si: print(f"    SINR inside  mean / p10: {si['sinr_mean_db']:+.1f} / {si['sinr_p10_db']:+.1f} dB")
        if so: print(f"    SINR outside mean / p10: {so['sinr_mean_db']:+.1f} / {so['sinr_p10_db']:+.1f} dB")
        print()

    if has_jammers:
        print(f"{'='*70}")
        print(f"BS+JAMMER SINR CONTAINMENT  (gamma = {gamma_db:+.1f} dB)")
        print(f"{'='*70}")
        for cfg in tx_configs:
            s  = stats[cfg.name]
            cj = s["containment_jam"]
            sij = s["sinr_inside_jam"]
            soj = s["sinr_outside_jam"]
            print(f"  {cfg.name}:")
            print(f"    rho_leak  (outside >= gamma): {_fmt(cj['rho_leak'])}")
            print(f"    rho_hole  (inside  <  gamma): {_fmt(cj['rho_hole'])}")
            if sij: print(f"    SINR inside  mean / p10: {sij['sinr_mean_db']:+.1f} / {sij['sinr_p10_db']:+.1f} dB")
            if soj: print(f"    SINR outside mean / p10: {soj['sinr_mean_db']:+.1f} / {soj['sinr_p10_db']:+.1f} dB")
            print()

    print(f"{'='*70}\n")

    jnt = multi_result.get("joint", {})
    stats["joint"] = {
        "loss_history":          jnt.get("loss_history", []),
        "gamma_db":              gamma_db,
        "jammers":               jnt.get("jammers", {}),
        "jam_interference_used": has_jammers,
        "sinr_bs_only_map":      sinr_bs_only,
    }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not fig:
        return None, stats

    _step = 8  # downsample before imshow/contour — increase to speed up rendering
    vmin, vmax = gamma_db - 20, gamma_db + 20
    norm = TwoSlopeNorm(vmin=vmin, vcenter=gamma_db, vmax=vmax)
    _lw_cycle = [2.0, 1.4, 1.0, 0.7]

    def _draw_sinr_map(ax, sinr_field, title, show_jammers=False):
        cx_m, cy_m = map_config["center"][0], map_config["center"][1]
        sx_m, sy_m = map_config["size"][0],   map_config["size"][1]
        cw, ch     = map_config["cell_size"][0], map_config["cell_size"][1]
        x_min, x_max = cx_m - sx_m / 2, cx_m + sx_m / 2
        y_min, y_max = cy_m - sy_m / 2, cy_m + sy_m / 2

        # rss and zone_mask both store row 0 = south; origin='lower' + extent
        # puts the image in world coordinates without any manual coordinate math.
        im = ax.imshow(sinr_field[::_step, ::_step], cmap="RdBu_r",
                       norm=norm, interpolation="nearest",
                       aspect="equal", rasterized=True,
                       origin="lower",
                       extent=[x_min, x_max, y_min, y_max])

        # Zone boundary contour — pass explicit world-coordinate X/Y so the
        # contour aligns with the extent-based imshow regardless of map center.
        for tx_idx, cfg in enumerate(tx_configs):
            lw = _lw_cycle[tx_idx % len(_lw_cycle)]
            mask_ds = inside_masks[cfg.name][::_step, ::_step].astype(float)
            H_ds, W_ds = mask_ds.shape
            x_coords = np.linspace(x_min + _step * cw / 2, x_max - _step * cw / 2, W_ds)
            y_coords = np.linspace(y_min + _step * ch / 2, y_max - _step * ch / 2, H_ds)
            ax.contour(x_coords, y_coords, mask_ds, levels=[0.5],
                       colors=["black"], linewidths=lw, linestyles="-")
            ax.plot([], [], color="black", linestyle="-", linewidth=lw, label=cfg.name)

        # Building footprint outlines — drawn in world coordinates from the
        # Shapely polygons, so no rasterization step is needed.
        if building_polygons:
            _bldg_added = False
            for bp in building_polygons:
                bx_poly, by_poly = bp.exterior.xy
                kw = dict(color="dimgray", linewidth=0.7, linestyle="-",
                          alpha=0.75, zorder=3)
                if not _bldg_added:
                    ax.plot(bx_poly, by_poly, label="Buildings", **kw)
                    _bldg_added = True
                else:
                    ax.plot(bx_poly, by_poly, **kw)
                for interior in bp.interiors:
                    ix, iy = interior.xy
                    ax.plot(ix, iy, **kw)

        # BS and jammer markers plotted directly in world coordinates.
        _bs_colors = ["lime", "cyan", "orange", "hotpink"]
        for tx_idx, cfg in enumerate(tx_configs):
            r = multi_result.get(cfg.name, {})
            if "final_position" in r:
                bx, by = r["final_position"][:2]
                color = _bs_colors[tx_idx % len(_bs_colors)]
                ax.scatter(bx, by, marker="*", color=color, s=160,
                           edgecolors="black", linewidths=0.6, zorder=6,
                           label=f"{cfg.name} (BS)")
        if show_jammers and jam_scene is not None and jammer_configs:
            jammers_data = multi_result["joint"].get("jammers", {})
            for jcfg in jammer_configs:
                if jcfg.name in jammers_data:
                    jd = jammers_data[jcfg.name]
                    jx, jy = jd["final_position"][:2]
                    is_active = jd.get("active", True)
                    color  = "yellow" if is_active else "grey"
                    label  = jcfg.name if is_active else f"{jcfg.name} (off)"
                    alpha  = 1.0 if is_active else 0.4
                    ax.scatter(jx, jy, marker="x", color=color, s=80,
                               linewidths=2, zorder=5, label=label, alpha=alpha)

        # Auto-zoom: tight view around the union zone with a margin sized to
        # keep all jammers (placed ~60-150 m outside the boundary) in frame.
        rows, cols = np.where(union_inside)
        if len(rows) > 0:
            x_z_min = x_min + cols.min() * cw
            x_z_max = x_min + (cols.max() + 1) * cw
            y_z_min = y_min + rows.min() * ch
            y_z_max = y_min + (rows.max() + 1) * ch
            margin = max(x_z_max - x_z_min, y_z_max - y_z_min) * 0.5
            ax.set_xlim(max(x_min, x_z_min - margin), min(x_max, x_z_max + margin))
            ax.set_ylim(max(y_min, y_z_min - margin), min(y_max, y_z_max + margin))

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="SINR (dB)")
        ax.legend(fontsize=8, loc="upper right")
        return im

    # ------------------------------------------------------------------
    # Figure 1: SINR map(s) — single column (BS-only) or two columns when
    #           jammers are present so both conditions are visible at once
    # ------------------------------------------------------------------
    n_map_cols = 2 if has_jammers else 1
    fig_map, map_axes = plt.subplots(1, n_map_cols, figsize=(7 * n_map_cols, 6), squeeze=False)

    _left_sinr = sinr_bs_only_ref if (sinr_bs_only_ref is not None and has_jammers) else sinr_bs_only
    _left_title = (f"BS-only SINR (dB)  —  γ = {gamma_db:+.1f} dB"
                   if sinr_bs_only_ref is None or not has_jammers
                   else f"BS-only result  —  γ = {gamma_db:+.1f} dB")
    _draw_sinr_map(map_axes[0, 0], _left_sinr, _left_title, show_jammers=False)
    if has_jammers:
        _draw_sinr_map(map_axes[0, 1], sinr_bs_jam,
                       f"BS+Jammer SINR (dB)  —  γ = {gamma_db:+.1f} dB",
                       show_jammers=True)

    fig_map.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2: single CDF — best-SINR inside union-of-zones vs outside
    # ------------------------------------------------------------------
    in_bs  = sinr_bs_only[outdoor_inside]
    out_bs = sinr_bs_only[outdoor_outside]
    ls_bs  = "--" if has_jammers else "-"

    fig_cdf, ax_cdf = plt.subplots(1, 1, figsize=(6, 4))

    def _plot_cdf(ax, vals, color, ls, label):
        arr = np.sort(vals)
        med = float(np.median(arr))
        ax.plot(arr, np.arange(1, len(arr) + 1) / len(arr),
                color=color, linewidth=1.8, linestyle=ls, label=label)
        ax.scatter([med], [0.5], color=color, s=50, zorder=5, clip_on=False)
        ax.annotate(f"{med:+.1f}", xy=(med, 0.5), xytext=(med, 0.55),
                    color=color, fontsize=7, ha="center", va="bottom")
        return med

    # Horizontal reference at the 50th percentile (median)
    ax_cdf.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.45)

    _plot_cdf(ax_cdf, in_bs,  "blue", ls_bs,
              f"BS-only Inside  (N={in_bs.size})"  if has_jammers else f"Inside  (N={in_bs.size})")
    _plot_cdf(ax_cdf, out_bs, "red",     ls_bs,
              f"BS-only Outside (N={out_bs.size})" if has_jammers else f"Outside (N={out_bs.size})")

    if has_jammers:
        in_jam  = sinr_bs_jam[outdoor_inside]
        out_jam = sinr_bs_jam[outdoor_outside]
        _plot_cdf(ax_cdf, in_jam,  "blue", "-", f"BS+Jam Inside  (N={in_jam.size})")
        _plot_cdf(ax_cdf, out_jam, "red",     "-", f"BS+Jam Outside (N={out_jam.size})")

    ax_cdf.axvline(gamma_db, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"γ = {gamma_db:+.1f} dB")
    cdf_title = "BS-only vs BS+Jammer SINR CDF" if has_jammers else "BS-only SINR CDF"
    ax_cdf.set_title(f"{cdf_title}  (γ = {gamma_db:+.1f} dB)", fontsize=11)
    ax_cdf.set_xlabel("Best-cell SINR (dB)"); ax_cdf.set_ylabel("CDF")
    ax_cdf.set_xlim(gamma_db - 30, gamma_db + 30)
    ax_cdf.legend(fontsize=8, loc="lower right"); ax_cdf.grid(True, alpha=0.3)
    fig_cdf.tight_layout()
    return fig_map, fig_cdf, stats

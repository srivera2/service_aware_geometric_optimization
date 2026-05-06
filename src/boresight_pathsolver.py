# ============================================
# BORESIGHT OPTIMIZATION USING PATHSOLVER + AD
# ============================================
# This approach uses PathSolver (which supports AD) instead of RadioMapSolver
# We sample grid points and use PathSolver to compute paths to those points
# This is similar to rm_diff.ipynb but optimizes boresight instead of TX position

import matplotlib.pyplot as plt
import torch
import numpy as np
import drjit as dr
from drjit.auto import Float, Array3f, UInt
import mitsuba as mi
from sionna.rt import PathSolver, RadioMapSolver, Receiver, cpx_abs_square
from sionna.rt.path_solvers.paths import Paths
import time
import shapely
from shapely import contains_xy, concave_hull
from shapely.geometry import Point, Polygon
from tx_placement import TxPlacement
from angle_utils import (
    azimuth_elevation_to_yaw_pitch,
    yaw_pitch_to_azimuth_elevation,
    compute_initial_angles_from_position,
    normalize_azimuth,
    clamp_elevation,
)
import scipy
from scipy.spatial import ConvexHull
from triangulate import (
    get_zone_polygon_with_exclusions,
    triangulate_zone,
    sample_triangulated_zone,
    sample_dead_zones,
)
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
import math as m
import alphashape
import os


def create_zone_mask(
    map_config,
    zone_type="box",
    origin_point=None,
    zone_params=None,
    target_height=1.5,
    scene_xml_path=None,
    exclude_buildings=True,
):

    if zone_params is None:
        raise ValueError("zone_params must be provided")

    # Configures the zone based on rectangular area described by zone_params
    if zone_type == "box":
        # 1. Set up outer area (map size)
        width_m, height_m = map_config["size"]
        cell_w, cell_h = map_config["cell_size"]
        center_x, center_y, _ = map_config["center"]

        n_x = int(width_m / cell_w)
        n_y = int(height_m / cell_h)

        # Create coordinate grids
        # Use endpoint=False to ensure proper cell spacing
        # With endpoint=False, linspace creates n_x cells of exactly cell_w width
        x = (
            np.linspace(
                center_x - width_m / 2, center_x + width_m / 2, n_x, endpoint=False
            )
            + cell_w / 2
        )
        y = (
            np.linspace(
                center_y - height_m / 2, center_y + height_m / 2, n_y, endpoint=False
            )
            + cell_h / 2
        )
        X, Y = np.meshgrid(x, y)

        # Initialize mask (all zeros = outside zone)
        mask = np.zeros((n_y, n_x), dtype=np.float32)

        # Default look_at position (will be overridden based on zone geometry)
        look_at_pos = np.array([center_x, center_y, target_height], dtype=np.float32)
        # Get box parameters
        box_center = zone_params.get("center", (center_x, center_y))
        bx, by = box_center
        bw = zone_params.get("width", 20)
        bh = zone_params.get("height", 20)

        # Validate box parameters
        if bw <= 0 or bh <= 0:
            raise ValueError(
                f"Box width and height must be positive, got width={bw}, height={bh}"
            )

        # Create box mask
        in_x = np.abs(X - bx) <= (bw / 2)
        in_y = np.abs(Y - by) <= (bh / 2)
        in_box = in_x & in_y
        mask[in_box] = 1.0

        # Look-at is simply the center of the box
        look_at_pos = np.array([bx, by, target_height], dtype=np.float32)

    elif zone_type == "polygon":
        print("Creating polygon")
        # Set up outer area (map size)
        width_m, height_m = map_config["size"]
        cell_w, cell_h = map_config["cell_size"]
        center_x, center_y, _ = map_config["center"]

        n_x = int(width_m / cell_w)
        n_y = int(height_m / cell_h)

        # Create coordinate grids
        x = (
            np.linspace(
                center_x - width_m / 2, center_x + width_m / 2, n_x, endpoint=False
            )
            + cell_w / 2
        )
        y = (
            np.linspace(
                center_y - height_m / 2, center_y + height_m / 2, n_y, endpoint=False
            )
            + cell_h / 2
        )
        X, Y = np.meshgrid(x, y)

        # Initialize mask (all zeros = outside zone)
        mask = np.zeros((n_y, n_x), dtype=np.float32)

        # Get the vertices of the polygon from zone_params
        vertices = zone_params.get("vertices", [(0, 0), (10, 0), (10, -10), (0, -10)])

        from shapely.geometry import Polygon
        from shapely import contains_xy

        # Create polygon from vertices
        zone = Polygon(vertices)

        # Vectorized containment check: flatten grid points and check all at once
        mask_flat = contains_xy(zone, X.flatten(), Y.flatten())
        mask = mask_flat.reshape(n_y, n_x).astype(np.float32)

        # Set the look at position at the centroid of the polygon
        centroid = zone.centroid
        look_at_pos = np.array(
            [centroid.x, centroid.y, target_height], dtype=np.float32
        )

        # Store bounding box for LDS sampling
        minx, miny, maxx, maxy = zone.bounds
        zone_params["center"] = [(minx + maxx) / 2, (miny + maxy) / 2]
        zone_params["width"] = maxx - minx
        zone_params["height"] = maxy - miny

    else:
        print("Polygon configuration is not clear. Double check your configuration")
        exit

    # 3. Exclude building footprints (independent of the zone type)
    num_excluded_buildings = 0
    if exclude_buildings:
        if scene_xml_path is None:
            raise ValueError("scene_xml_path is required when exclude_buildings=True")

        # Import here to avoid circular dependency
        from scene_parser import extract_building_info
        from shapely.geometry import Polygon, Point
        from shapely.prepared import prep

        # Get building information (reuse existing parser!)
        building_info = extract_building_info(scene_xml_path, verbose=False)

        # Get coordinates of all cells currently in the zone
        zone_indices = np.nonzero(mask)
        zone_y_idx, zone_x_idx = zone_indices

        if len(zone_y_idx) > 0:
            # Extract coordinates of zone cells
            zone_x_coords = X[zone_y_idx, zone_x_idx]
            zone_y_coords = Y[zone_y_idx, zone_x_idx]

            # Create array of (x, y) points for vectorized operations
            zone_points_array = np.column_stack((zone_x_coords, zone_y_coords))

            # For each building, create polygon and exclude cells inside it
            for info in building_info.values():
                # Get building polygon vertices (only X, Y coordinates)
                vertices_3d = info["vertices"]
                vertices_2d = [(v[0], v[1]) for v in vertices_3d]

                # Create Shapely polygon for this building
                try:
                    building_polygon = Polygon(vertices_2d)

                    # Get bounding box for quick rejection test
                    minx, miny, maxx, maxy = building_polygon.bounds

                    # Quick rejection: filter points that are definitely outside bounding box
                    in_bbox = (
                        (zone_points_array[:, 0] >= minx)
                        & (zone_points_array[:, 0] <= maxx)
                        & (zone_points_array[:, 1] >= miny)
                        & (zone_points_array[:, 1] <= maxy)
                    )

                    # Only check points that might be inside the building
                    candidate_indices = np.where(in_bbox)[0]

                    if len(candidate_indices) > 0:
                        # Prepare the polygon for faster repeated containment checks
                        prepared_polygon = prep(building_polygon)

                        # Check candidates against actual polygon boundary
                        for idx in candidate_indices:
                            px, py = zone_points_array[idx]
                            point = Point(px, py)
                            if prepared_polygon.contains(point):
                                # Exclude this cell
                                mask[zone_y_idx[idx], zone_x_idx[idx]] = 0.0

                    num_excluded_buildings += 1
                except Exception:
                    # Skip buildings with invalid geometry
                    pass

    # 4. Validate and compute zone statistics
    num_cells_in_zone = int(np.sum(mask))
    total_cells = n_x * n_y

    if num_cells_in_zone == 0:
        raise ValueError(
            f"Coverage zone is empty (no grid cells inside). Check zone parameters:\n"
            f"  zone_type: {zone_type}\n"
            f"  zone_params: {zone_params}\n"
            f"  origin_point: {origin_point}\n"
            f"  map_config: {map_config}\n"
            f"  Buildings excluded: {exclude_buildings} ({num_excluded_buildings} buildings processed)"
        )

    # Compute actual centroid (center of mass of the mask)
    # This may differ from the geometric look_at position
    # Use np.nonzero for faster indexing (avoids boolean mask overhead)
    zone_y_idx, zone_x_idx = np.nonzero(mask)
    centroid_x = np.mean(X[zone_y_idx, zone_x_idx])
    centroid_y = np.mean(Y[zone_y_idx, zone_x_idx])

    zone_stats = {
        "num_cells": num_cells_in_zone,
        "coverage_fraction": num_cells_in_zone / total_cells,
        "centroid_xy": [float(centroid_x), float(centroid_y)],
        "look_at_xyz": look_at_pos.tolist(),
        "buildings_excluded": exclude_buildings,
        "num_excluded_buildings": num_excluded_buildings,
        "zone_params": zone_params,
    }

    return mask, look_at_pos, zone_stats


def sample_grid_points(
    polygon,
    num_points,
    qrand,
    alphashapes=None,
    building_polygons=None,
    ground_z=0.0,
    dead_polygons=None,
    dead_fraction=0.8,
):
    from shapely import contains_xy
    from shapely.ops import unary_union

    if dead_polygons is not None:
        dead_union = unary_union(dead_polygons)
        alive_diff = polygon.difference(dead_union)
        if alphashapes is not None:
            alpha_union = unary_union(alphashapes)
            alive_diff = alive_diff.difference(alpha_union)
        dead_num = int(dead_fraction * num_points)
        alive_num = num_points - dead_num
        alive_pts, _, __, ___, ____ = sample_grid_points(
            alive_diff,
            alive_num,
            qrand,
            building_polygons=building_polygons,
            ground_z=ground_z,
        )

        # Use QMC sampler directly inside each dead polygon, allocating points
        # proportional to polygon area so the low-discrepancy sequence is mapped
        # tightly to each polygon's own bounding box rather than a large union bbox.
        total_area = sum(p.area for p in dead_polygons)
        all_dead_pts = []
        for poly in dead_polygons:
            n_poly = max(1, round(dead_num * poly.area / total_area))
            minx, miny, maxx, maxy = poly.bounds
            w, h = maxx - minx, maxy - miny
            sampled = []
            for iteration in range(1, 101):
                needed = n_poly - len(sampled)
                if needed == 0:
                    break
                batch = np.clip(
                    np.array(qrand.random(min(needed * max(2, iteration), 10000))), 0.0, 1.0
                )
                x = minx + batch[:, 0] * w
                y = miny + batch[:, 1] * h
                in_poly = contains_xy(poly, x, y)
                pts = np.column_stack([x[in_poly], y[in_poly]])
                if building_polygons and len(pts):
                    mask = np.ones(len(pts), dtype=bool)
                    for bp in building_polygons:
                        mask &= ~contains_xy(bp, pts[:, 0], pts[:, 1])
                    pts = pts[mask]
                sampled.extend(pts[:needed].tolist())
            all_dead_pts.extend(sampled[:n_poly])

        if all_dead_pts:
            dead_pts2d = np.array(all_dead_pts[:dead_num])
            dead_pts = np.hstack([dead_pts2d, np.full((len(dead_pts2d), 1), ground_z)])
        else:
            dead_pts = np.zeros((0, 3))

        return (
            np.vstack([dead_pts, alive_pts]),
            dead_union,
            alive_diff,
            dead_pts,
            alive_pts,
        )
    else:
        dead_union = None
        alive_diff = None
        alive_pts = None
        dead_pts = None

    minx, miny, maxx, maxy = polygon.bounds
    w, h = maxx - minx, maxy - miny

    sampled = []
    for iteration in range(1, 101):
        needed = num_points - len(sampled)
        if needed == 0:
            break
        batch = np.clip(
            np.array(qrand.random(min(needed * max(2, iteration), 10000))), 0.0, 1.0
        )
        x = minx + batch[:, 0] * w
        y = miny + batch[:, 1] * h
        pts = np.column_stack([x, y])

        pts = pts[contains_xy(polygon, x, y)]

        if building_polygons and len(pts):
            mask = np.ones(len(pts), dtype=bool)
            for bp in building_polygons:
                mask &= ~contains_xy(bp, pts[:, 0], pts[:, 1])
            pts = pts[mask]

        sampled.extend(pts[:needed].tolist())

    if len(sampled) < num_points:
        raise ValueError(
            f"Could only sample {len(sampled)}/{num_points} points after 100 iterations."
        )

    pts2d = np.array(sampled)
    return (
        np.hstack([pts2d, np.full((len(pts2d), 1), ground_z)]),
        dead_union,
        alive_diff,
        dead_pts,
        alive_pts,
    )


def filter_and_append(rx_data, dead_zone, tail_percentile=20.0):
    """
    Dynamically isolates the worst X% of receivers in the current batch.
    """
    # rx_data has columns [x, y, power]
    # Drop cells with NaN/inf power (e.g. unlit cells in the RadioMap); a NaN
    # threshold from np.percentile would make every comparison False and return
    # an empty dead_points even when valid data exists.
    valid_mask = np.isfinite(rx_data[:, 2])
    rx_data = rx_data[valid_mask]
    if len(rx_data) == 0:
        return dead_zone

    power_array = rx_data[:, 2]

    # Dynamically find the threshold for the bottom tail_percentile% of this batch
    dynamic_threshold = np.percentile(power_array, tail_percentile)

    # Filter rows where power is below the dynamic threshold
    dead_mask = power_array <= dynamic_threshold
    dead_points = rx_data[dead_mask]

    if dead_zone.size == 0:
        dead_zone = dead_points
    else:
        dead_zone = np.append(dead_zone, dead_points, axis=0)

    return dead_zone


def visualize_receiver_placement(
    sample_points,
    map_config,
    zone_mask=None,
    current_tx_position=None,
    scene_xml_path=None,
    title="Receiver Sampling Visualization",
    figsize=(14, 10),
    zone_polygon=None,
    dead_buffers=None,
    alphashapes=None,
    box_polygon=None,
    building_id=None,
):
    """
    Visualize where receivers are placed relative to zone mask and buildings

    This visualization helps verify that:
    1. Receivers are uniformly distributed across both target and interference zones
    2. No receivers are placed inside buildings
    3. 50/50 split is maintained between zones

    Parameters:
    -----------
    sample_points : np.ndarray
        Nx3 array of (x, y, z) receiver positions
    zone_mask : np.ndarray
        Binary mask defining target zone (1.0) and interference zone (0.0)
    map_config : dict
        Map configuration with 'center', 'size', 'cell_size'
    tx_position : tuple or None
        (x, y, z) position of transmitter to mark on plot
    scene_xml_path : str or None
        Path to scene XML for building overlays
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon

    width_m, height_m = map_config["size"]
    center_x, center_y, _ = map_config["center"]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot zone mask as background
    extent = [
        center_x - width_m / 2,
        center_x + width_m / 2,
        center_y - height_m / 2,
        center_y + height_m / 2,
    ]

    # Show zone mask as background if provided
    if zone_mask is not None:
        im = ax.imshow(
            zone_mask,
            origin="lower",
            cmap="RdYlGn",
            alpha=0.3,
            extent=extent,
            vmin=0,
            vmax=1,
        )
        cbar = plt.colorbar(im, ax=ax, label="Zone Mask")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Interference", "Target"])

    # --- Zone and building overlay ---
    # Strategy: layered fills inside the coverage box.
    #   1. Fill box green  (alive zone background)
    #   2. Fill dead zones red  (on top of green)
    #   3. Fill all buildings gray  (on top of both)
    #   4. Draw coverage box outline in blue dashed
    # This avoids all polygon-difference topology issues and handles holes correctly.

    def _fill_geom(geom, **kwargs):
        """Fill a Polygon or MultiPolygon (exterior only — later layers cover holes)."""
        if geom is None or geom.is_empty:
            return
        if geom.geom_type == "Polygon":
            ax.fill(*geom.exterior.xy, **kwargs)
        elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
            for part in geom.geoms:
                if part.geom_type == "Polygon":
                    ax.fill(*part.exterior.xy, **kwargs)

    # 1. Alive zone: fill coverage box green
    draw_box = box_polygon if box_polygon is not None else zone_polygon
    if draw_box is not None:
        _fill_geom(
            draw_box, alpha=0.15, fc="green", ec="none", zorder=1, label="Alive Zone"
        )

    # 2. Buffer zones: fill purple
    if dead_buffers:
        for i, dp in enumerate(dead_buffers):
            _fill_geom(
                dp,
                alpha=0.35,
                fc="red",
                ec="black",
                linewidth=1.0,
                zorder=2,
                label="Buffer Zone" if i == 0 else None,
            )
    # Dead zones
    if alphashapes:
        for i, dp in enumerate(alphashapes):
            _fill_geom(
                dp,
                alpha=0.9,
                fc="purple",
                ec="black",
                linewidth=1.0,
                zorder=3,
                label="Dead Zone" if i == 0 else None,
            )

    # 3. All buildings: gray fill; TX building gets a highlighted edge
    if scene_xml_path is not None:
        try:
            from scene_parser import extract_building_info

            building_info = extract_building_info(scene_xml_path, verbose=False)
            tx_building_drawn = False
            for bid, info in building_info.items():
                vertices_2d = [(v[0], v[1]) for v in info["vertices"]]
                try:
                    bpoly = Polygon(vertices_2d)
                    is_tx_building = str(bid) == str(building_id)
                    ax.fill(
                        *bpoly.exterior.xy,
                        facecolor="#c0392b" if is_tx_building else "dimgray",
                        edgecolor="black",
                        alpha=0.75 if is_tx_building else 0.55,
                        linewidth=2.0 if is_tx_building else 0.8,
                        zorder=3,
                        label=(
                            "TX Building"
                            if is_tx_building and not tx_building_drawn
                            else None
                        ),
                    )
                    if is_tx_building:
                        tx_building_drawn = True
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: Could not overlay buildings: {e}")

    # 4. Coverage box outline
    if draw_box is not None:
        if draw_box.geom_type == "Polygon":
            ax.plot(
                *draw_box.exterior.xy,
                color="blue",
                linewidth=2,
                linestyle="--",
                label="Coverage Zone",
                zorder=4,
            )
        elif draw_box.geom_type == "MultiPolygon":
            for i, part in enumerate(draw_box.geoms):
                ax.plot(
                    *part.exterior.xy,
                    color="blue",
                    linewidth=2,
                    linestyle="--",
                    label="Coverage Zone" if i == 0 else None,
                    zorder=4,
                )

    # Plot all sample points
    ax.scatter(
        sample_points[:, 0],
        sample_points[:, 1],
        c="white",
        s=20,
        alpha=0.8,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
        label=f"Receivers ({len(sample_points)})",
    )

    # Plot transmitter position
    if current_tx_position is not None:
        ax.plot(
            current_tx_position[0],
            current_tx_position[1],
            "b*",
            markersize=20,
            label="Transmitter",
            markeredgecolor="navy",
            markeredgewidth=1.5,
            zorder=6,
        )

    ax.legend(loc="upper right", fontsize=10)

    # Labels and title
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_aspect("equal")

    # Add text box with sampling statistics
    stats_text = f"Total Receivers: {len(sample_points)}"
    if dead_buffers:
        stats_text += f"\nDead Zones: {len(dead_buffers)}"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout()

    return fig


def visualize_multi_tx_strata(
    tx_states,
    tx_configs,
    map_config,
    iteration=None,
    title=None,
    figsize=None,
    jam_positions=None,
):
    """
    Single-plot visualization of the shared coverage zone showing base station
    positions, jammer positions, sample points, and buildings.

    Parameters
    ----------
    tx_states : list[dict]
    tx_configs : list[TxConfig]
    map_config : dict
    iteration : int or None
    title : str or None
    figsize : tuple or None
    jam_positions : list[[x, y]] or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (9, 9)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    tx_colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#c0392b",
                 "#16a085", "#d35400", "#2c3e50"]

    def _fill_geom(geom, **kwargs):
        if geom is None or geom.is_empty:
            return
        if geom.geom_type == "Polygon":
            ax.fill(*geom.exterior.xy, **kwargs)
        elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
            for part in geom.geoms:
                if part.geom_type == "Polygon":
                    ax.fill(*part.exterior.xy, **kwargs)

    from shapely.affinity import scale as _shapely_scale

    # 1. Zone background, outer-ring, and outlines — use first state
    first_state = tx_states[0]
    draw_box = first_state.get("box_polygon") or first_state.get("zone_polygon")
    zone_poly = first_state.get("zone_polygon")

    box_poly = first_state.get("box_polygon")
    if box_poly is not None:
        # Outer ring: scale raw box 2x, subtract original box, then punch out buildings.
        # Mirrors _sample_outside_zone exactly — never scale the building-subtracted polygon.
        centroid = box_poly.centroid
        outer_poly = _shapely_scale(box_poly, xfact=2.0, yfact=2.0, origin=centroid)
        outer_ring = outer_poly.difference(box_poly)
        cached_bldgs = first_state.get("cached_building_polygons", [])
        if cached_bldgs:
            import shapely.ops as _shops
            outer_ring = outer_ring.difference(_shops.unary_union(cached_bldgs))
        _fill_geom(outer_ring, alpha=0.10, fc="#e74c3c", ec="none", zorder=0)
        outer_geoms = [outer_poly] if outer_poly.geom_type == "Polygon" else list(outer_poly.geoms)
        for j, part in enumerate(outer_geoms):
            ax.plot(*part.exterior.xy, color="#c0392b", linewidth=1.5,
                    linestyle=":", zorder=5,
                    label="Outer Ring (2×)" if j == 0 else None)

    if draw_box is not None:
        _fill_geom(draw_box, alpha=0.15, fc="green", ec="none", zorder=1)
        geoms = [draw_box] if draw_box.geom_type == "Polygon" else list(draw_box.geoms)
        for j, part in enumerate(geoms):
            ax.plot(*part.exterior.xy, color="steelblue", linewidth=2,
                    linestyle="--", zorder=5,
                    label="Coverage Zone" if j == 0 else None)

    # 2. Buildings — drawn once from first state's cache
    for bpoly in first_state.get("cached_building_polygons", []):
        try:
            ax.fill(*bpoly.exterior.xy,
                    facecolor="dimgray", edgecolor="black",
                    alpha=0.45, linewidth=0.7, zorder=4)
        except Exception:
            pass

    # 3a. Inside sample points
    sample_pts = None
    for state in tx_states:
        pts = state.get("current_sample_points")
        if pts is not None and len(pts) > 0:
            sample_pts = pts
            break
    if sample_pts is not None:
        ax.scatter(sample_pts[:, 0], sample_pts[:, 1],
                   c="white", s=18, alpha=0.85, marker="o",
                   edgecolors="black", linewidths=0.4,
                   zorder=6, label=f"RX inside ({len(sample_pts)})")

    # 3b. Outside sample points
    outside_pts = None
    for state in tx_states:
        pts = state.get("outside_sample_points")
        if pts is not None and len(pts) > 0:
            outside_pts = pts
            break
    if outside_pts is not None:
        ax.scatter(outside_pts[:, 0], outside_pts[:, 1],
                   c="#e74c3c", s=18, alpha=0.75, marker="x",
                   linewidths=0.6,
                   zorder=6, label=f"RX outside ({len(outside_pts)})")

    # 4. Base station positions
    for k, (state, cfg) in enumerate(zip(tx_states, tx_configs)):
        tx_pos = state.get("current_tx_position", state.get("tx_position"))
        if tx_pos is not None:
            color = tx_colors[k % len(tx_colors)]
            ax.plot(tx_pos[0], tx_pos[1], "*", markersize=18,
                    color=color, markeredgecolor="black",
                    markeredgewidth=1.0, zorder=9, label=cfg.name)

    # 5. Jammer positions
    if jam_positions:
        jx = [p[0] for p in jam_positions]
        jy = [p[1] for p in jam_positions]
        ax.scatter(jx, jy, marker="^", s=130, c="orange",
                   edgecolors="darkorange", linewidths=1.5,
                   zorder=10, label=f"Jammers ({len(jam_positions)})")

    iter_str = f"  —  Iteration {iteration}" if iteration is not None else ""
    ax.set_title(title if title is not None else f"Coverage Zone{iter_str}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig

def compare_boresight_performance(
    scene,
    tx_name,
    map_config,
    zone_mask,
    naive_angles,
    optimized_angles,
    naive_transmitter_pos=None,
    optimized_transmitter_pos=None,
    title="Boresight Optimization Comparison",
    fig=False,
):
    """
    Compare naive baseline vs optimized boresight performance using RadioMapSolver.

    Generates radio maps for both configurations and compares power distribution
    in the coverage zone using histograms and CDFs.

    Parameters:
    -----------
    scene : sionna.rt.Scene
        The scene with transmitter
    tx_name : str
        Name of transmitter
    map_config : dict
        Map configuration (center, size, cell_size)
    zone_mask : np.ndarray
        Binary mask defining coverage zone
    naive_angles : list or array
        [azimuth_deg, elevation_deg] naive baseline antenna angles
    optimized_angles : list or array
        [azimuth_deg, elevation_deg] optimized antenna angles
    naive_transmitter_pos : list or array, optional
        [x, y, z] naive baseline transmitter position
    optimized_transmitter_pos : list or array, optional
        [x, y, z] optimized transmitter position
    title : str
        Title for the comparison plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with comparison plots
    stats : dict
        Statistics comparing both configurations
    """
    from sionna.rt import RadioMapSolver
    import matplotlib.pyplot as plt

    tx = scene.get(tx_name)
    rm_solver = RadioMapSolver()

    results = {}

    for config_name, angles, tx_pos in [
        ("Naive Baseline", naive_angles, naive_transmitter_pos),
        ("Optimized", optimized_angles, optimized_transmitter_pos),
    ]:
        print(f"Computing RadioMap for {config_name}...")

        # Set antenna orientation using angles
        azimuth_deg, elevation_deg = angles[0], angles[1]
        yaw_rad, pitch_rad = azimuth_elevation_to_yaw_pitch(azimuth_deg, elevation_deg)
        tx.orientation = mi.Point3f(float(yaw_rad), float(pitch_rad), 0.0)

        # Set transmitter position if provided
        if tx_pos is not None:
            tx.position = mi.Point3f(tx_pos)
            # print(f"  Position: x={tx_pos[0]:.2f}, y={tx_pos[1]:.2f}, z={tx_pos[2]:.2f}")

        print(f"  Angles: Azimuth={azimuth_deg:.1f}°, Elevation={elevation_deg:.1f}°")

        # Generate RadioMap
        rm = rm_solver(
            scene,
            max_depth=8,
            samples_per_tx=int(10e8),
            cell_size=[0.5, 0.5],
            center=map_config["center"],
            orientation=[0, 0, 0],
            size=map_config["size"],
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=False,
            stop_threshold=None,
        )

        # Extract signal strength
        rss_watts = rm.rss.numpy()[0, :, :]

        # Commenting this out to test if the linear average is improved
        # signal_strength_dBm = 10.0 * np.log10(rss_watts + 1e-30) + 30.0

        # Extract power values in zone only
        zone_power = rss_watts[zone_mask == 1.0]

        # Penalizing dead zones as a saturated value
        live_zone_power = zone_power

        # All points are dead zones - use raw values
        # This was biasing the results!!! Keep this as-is
        mean_val = np.mean(zone_power)
        median_val = np.median(zone_power)
        std_val = np.std(zone_power)
        min_val = np.min(zone_power)
        max_val = np.max(zone_power)
        p10_val = np.percentile(zone_power, 10)
        p90_val = np.percentile(zone_power, 90)
        coverage_fraction = 0.0

        # Store results
        results[config_name] = {
            "power_values": zone_power,  # Keep all values for plotting
            "live_power_values": live_zone_power,  # Only live points
            "radiomap": rss_watts,
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "p10": p10_val,
            "p90": p90_val,
            "coverage_fraction": coverage_fraction,  # Fraction of zone with actual coverage
            "num_live_points": len(live_zone_power),
            "num_total_points": len(zone_power),
        }

    # Calculate improvement (for best and naive)
    improvement_mean = results["Optimized"]["mean"] - results["Naive Baseline"]["mean"]
    improvement_median = (
        results["Optimized"]["median"] - results["Naive Baseline"]["median"]
    )

    if fig is True:

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Helper function to convert Watts to dBm
        # Setting the noise floor to 150 dB
        def watts_to_dbm(watts):
            return 10.0 * np.log10(watts + 1e-18) + 30.0

        # Convert all power values to dBm for better visualization
        all_power_watts = np.concatenate(
            [
                results["Naive Baseline"]["power_values"],
                results["Optimized"]["power_values"],
            ]
        )
        all_power_dbm = watts_to_dbm(all_power_watts)
        naive_power_dbm = watts_to_dbm(results["Naive Baseline"]["power_values"])
        optimized_power_dbm = watts_to_dbm(results["Optimized"]["power_values"])
        naive_mean_dbm = watts_to_dbm(results["Naive Baseline"]["mean"])
        optimized_mean_dbm = watts_to_dbm(results["Optimized"]["mean"])

        data_min_dbm = np.min(all_power_dbm)
        data_max_dbm = np.max(all_power_dbm)

        # Plot 1: Histograms (PDF)
        ax = axes[0, 0]
        # Use linear binning in dBm space (dB is already logarithmic)
        if data_max_dbm > data_min_dbm:
            bins = np.linspace(data_min_dbm, data_max_dbm, 150)
        else:
            # Handle edge case where all values are the same
            bins = np.linspace(data_min_dbm - 1, data_max_dbm + 1, 150)

        ax.hist(
            naive_power_dbm,
            bins=bins,
            alpha=0.6,
            label="Naive Baseline",
            color="orange",
            density=True,
        )
        ax.hist(
            optimized_power_dbm,
            bins=bins,
            alpha=0.6,
            label="Optimized",
            color="green",
            density=True,
        )
        ax.axvline(
            naive_mean_dbm,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Naive Mean: {naive_mean_dbm:.2f} dBm",
        )
        ax.axvline(
            optimized_mean_dbm,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Optimized Mean: {optimized_mean_dbm:.2f} dBm",
        )
        ax.set_xlabel("Signal Strength (dBm)")
        ax.set_ylabel("Probability Density")
        ax.set_title("Power Distribution in Coverage Zone (PDF)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(data_min_dbm, data_max_dbm)

        # Plot 2: CDFs
        ax = axes[0, 1]
        for config_name in ["Naive Baseline", "Optimized"]:
            power_watts = results[config_name]["power_values"]
            power_dbm = watts_to_dbm(power_watts)
            sorted_power = np.sort(power_dbm)
            cdf = np.arange(1, len(sorted_power) + 1) / len(sorted_power)
            color = "orange" if config_name == "Naive Baseline" else "green"
            ax.plot(sorted_power, cdf, label=config_name, color=color, linewidth=2)

            # Mark median
            median_watts = results[config_name]["median"]
            median_dbm = watts_to_dbm(median_watts)
            ax.axvline(
                median_dbm,
                color=color,
                linestyle="--",
                alpha=0.5,
                label=f"{config_name} Median: {median_dbm:.2f} dBm",
            )

        ax.set_xlabel("Signal Strength (dBm)")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("Cumulative Distribution Function (CDF)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(data_min_dbm, data_max_dbm)

        # Plot 3: Box plot comparison
        ax = axes[1, 0]
        data_to_plot = [
            naive_power_dbm,
            optimized_power_dbm,
        ]
        bp = ax.boxplot(
            data_to_plot,
            labels=["Naive Baseline", "Optimized"],
            patch_artist=True,
            showmeans=True,
        )
        bp["boxes"][0].set_facecolor("orange")
        bp["boxes"][1].set_facecolor("green")
        ax.set_ylabel("Signal Strength (dBm)")
        ax.set_title("Power Distribution Comparison (Box Plot)")
        ax.grid(True, alpha=0.3, axis="y")

        # Add improvement annotation
        # Calculate improvement in dB
        improvement_db_mean = optimized_mean_dbm - naive_mean_dbm
        naive_median_dbm = watts_to_dbm(results["Naive Baseline"]["median"])
        optimized_median_dbm = watts_to_dbm(results["Optimized"]["median"])
        improvement_db_median = optimized_median_dbm - naive_median_dbm

        ax.text(
            1.5,
            optimized_mean_dbm + 2,  # 2 dB above optimized mean
            f"Improvement:\nMean: {improvement_db_mean:+.2f} dB\nMedian: {improvement_db_median:+.2f} dB",
            bbox=dict(
                boxstyle="round",
                facecolor="lightgreen" if improvement_db_mean > 0 else "lightcoral",
                alpha=0.8,
            ),
            fontsize=10,
            ha="center",
        )

        # Plot 4: Statistics table
        ax = axes[1, 1]
        ax.axis("off")

        stats_data = [
            ["Metric", "Naive Baseline", "Optimized", "Improvement"],
            [
                "Mean (dBm)",
                f"{watts_to_dbm(results['Naive Baseline']['mean']):.2f}",
                f"{watts_to_dbm(results['Optimized']['mean']):.2f}",
                f"{watts_to_dbm(results['Optimized']['mean']) - watts_to_dbm(results['Naive Baseline']['mean']):+.2f} dB",
            ],
            [
                "Median (dBm)",
                f"{watts_to_dbm(results['Naive Baseline']['median']):.2f}",
                f"{watts_to_dbm(results['Optimized']['median']):.2f}",
                f"{watts_to_dbm(results['Optimized']['median']) - watts_to_dbm(results['Naive Baseline']['median']):+.2f} dB",
            ],
            [
                "Std Dev (dB)",
                f"{watts_to_dbm(results['Naive Baseline']['std']):.2f}",
                f"{watts_to_dbm(results['Optimized']['std']):.2f}",
                f"{watts_to_dbm(results['Optimized']['std']) - watts_to_dbm(results['Naive Baseline']['std']):+.2f} dB",
            ],
            [
                "Min (dBm)",
                f"{watts_to_dbm(results['Naive Baseline']['min']):.2f}",
                f"{watts_to_dbm(results['Optimized']['min']):.2f}",
                f"{watts_to_dbm(results['Optimized']['min']) - watts_to_dbm(results['Naive Baseline']['min']):+.2f} dB",
            ],
            [
                "Max (dBm)",
                f"{watts_to_dbm(results['Naive Baseline']['max']):.2f}",
                f"{watts_to_dbm(results['Optimized']['max']):.2f}",
                f"{watts_to_dbm(results['Optimized']['max']) - watts_to_dbm(results['Naive Baseline']['max']):+.2f} dB",
            ],
            [
                "10th %ile (dBm)",
                f"{watts_to_dbm(results['Naive Baseline']['p10']):.2f}",
                f"{watts_to_dbm(results['Optimized']['p10']):.2f}",
                f"{watts_to_dbm(results['Optimized']['p10']) - watts_to_dbm(results['Naive Baseline']['p10']):+.2f} dB",
            ],
            [
                "90th %ile (dBm)",
                f"{watts_to_dbm(results['Naive Baseline']['p90']):.2f}",
                f"{watts_to_dbm(results['Optimized']['p90']):.2f}",
                f"{watts_to_dbm(results['Optimized']['p90']) - watts_to_dbm(results['Naive Baseline']['p90']):+.2f} dB",
            ],
        ]

        table = ax.table(
            cellText=stats_data,
            cellLoc="center",
            loc="center",
            colWidths=[0.25, 0.25, 0.25, 0.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Color header row
        for i in range(4):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Color improvement column green if positive
        for i in range(1, len(stats_data)):
            improvement_str = stats_data[i][3]
            # Parse percentage string (e.g., "+10.5%" or "-5.2%") or dB string (e.g., "+0.88 dB")
            if improvement_str != "N/A":
                # Handle both percentage and dB formats
                if improvement_str.endswith("%"):
                    improvement_val = float(improvement_str.rstrip("%"))
                elif improvement_str.endswith(" dB"):
                    improvement_val = float(improvement_str.rstrip(" dB"))
                else:
                    continue  # Skip if format is unexpected

                if improvement_val > 0:
                    table[(i, 3)].set_facecolor("#90EE90")
                elif improvement_val < 0:
                    table[(i, 3)].set_facecolor("#FFB6C6")

        ax.set_title(
            "Performance Statistics Comparison", fontsize=12, weight="bold", pad=20
        )

        plt.suptitle(title, fontsize=14, weight="bold")
        plt.tight_layout()

        # Prepare stats dictionary for return
        stats = {
            "naive": results["Naive Baseline"],
            "optimized": results["Optimized"],
            "improvement_mean_watts": improvement_mean,
            "improvement_median_watts": improvement_median,
            "improvement_percent": (
                improvement_mean / abs(results["Naive Baseline"]["mean"])
            )
            * 100,
        }

        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"Naive Baseline Mean:  {results['Naive Baseline']['mean']:.2e} W")
        print(f"Optimized Mean:       {results['Optimized']['mean']:.2e} W")
        print(
            f"Improvement:          {improvement_mean:+.2e} W ({stats['improvement_percent']:+.1f}%)"
        )
        print(f"{'='*70}\n")

        return fig, stats

    else:
        return None, stats


def optimize_boresight_pathsolver(
    scene,
    tx_name,
    map_config,
    scene_xml_path,
    zone_params=None,
    num_sample_points=100,
    building_id=10,
    learning_rate=1.0,
    num_iterations=20,
    verbose=True,
    lds="Sobol",  # Random seed for reproducible sampling
    tx_placement_mode="skip",  # "center", "fixed", "line", "skip" (skip = don't move TX)
    Tx_start_pos=[0.0, 0.0],
    filepath=os.getcwd() + "/figures/",
    dead_tail_percentile=5.0,   # Bottom X% of rays (by power) treated as dead zone
    max_dbscan_points=50_000,   # Cap before DBSCAN to keep runtime tractable
):
    """
    Optimize boresight using PathSolver with automatic differentiation

    This follows the rm_diff.ipynb pattern:
    - Use @dr.wrap to enable PyTorch-DrJit AD
    - Use PathSolver (not RadioMapSolver) for gradient computation
    - Sample grid points as receivers
    """

    # Get TX position for angle calculations
    tx = scene.get(tx_name)

    # Get TX position (new)
    tx = scene.get(tx_name)
    tx_x = tx.position[0]
    tx_y = tx.position[1]
    tx_z = tx.position[2]
    tx_position = [tx_x, tx_y, tx_z]

    # Save initial position (detached from AD graph)
    initial_tx_position = [dr.detach(tx_position[i]) for i in range(3)]

    # Get transmit power for proper power calculation
    tx_power_dbm = float(tx.power_dbm[0])
    print(f"Transmit Power in dBm: {tx_power_dbm}")

    tx_height = float(dr.detach(tx_z)[0])

    # Initialize TxPlacement for accessing building info and edge projection
    # We need this regardless of tx_placement_mode for the edge constraint
    tx_placement = TxPlacement(
        scene, tx_name, scene_xml_path, building_id, create_if_missing=False
    )

    # Handle TX placement based on mode
    if tx_placement_mode == "skip":
        # Don't move the TX - use its current position
        # This is useful when TX was already placed correctly before calling optimization
        x_start_position = float(dr.detach(tx.position[0])[0])
        y_start_position = float(dr.detach(tx.position[1])[0])
        if verbose:
            print(f"TX placement mode: skip (using current position)")
            print(
                # f"  Current TX position: ({x_start_position:.2f}, {y_start_position:.2f}, {tx_height:.2f})"
            )
    else:
        # Sets the initial location based on mode
        if tx_placement_mode == "center":
            tx_placement.set_rooftop_center()
            x_start_position = tx_placement.building["center"][0]
            y_start_position = tx_placement.building["center"][1]
        elif tx_placement_mode == "fixed":
            x_start_position = Tx_start_pos[0]
            y_start_position = Tx_start_pos[1]
            z_pos = tx_placement.building["z_height"]
            tx.position = mi.Point3f(
                float(x_start_position), float(y_start_position), float(z_pos)
            )
        else:
            raise ValueError(
                f"Unknown tx_placement_mode: {tx_placement_mode}. Must be 'skip', 'center', 'fixed', or 'line'"
            )

    # Select LDS from available list
    # Changing the dimension to 3 to support CDT + Turk's
    if lds == "Sobol":
        qrand = scipy.stats.qmc.Sobol(d=3, scramble=True, seed=None)
    elif lds == "Halton":
        qrand = scipy.stats.qmc.Halton(d=3, scramble=True, seed=None)
    elif lds == "Latin":
        qrand = scipy.stats.qmc.LatinHypercube(d=3, scramble=True, seed=None)
    elif lds == "Uniform":
        qrand = None
    else:
        print("This LDS is not supported. Using Sobol as default")
        qrand = scipy.stats.qmc.Sobol(d=3, scramble=True, seed=None)

    # Auto-detect zone type from zone_params
    if "vertices" in zone_params:
        zone_type = "polygon"
    elif "center" in zone_params and "width" in zone_params and "height" in zone_params:
        zone_type = "box"
    else:
        raise ValueError(
            "zone_params must contain either 'vertices' (for polygon) or "
            "'center', 'width', 'height' (for box)"
        )

    # Solve for polygon zone and remove building exclusions.
    target_zone, building_exclusions, _ = get_zone_polygon_with_exclusions(
        zone_type=zone_type,
        zone_params=zone_params,
        scene_xml_path=scene_xml_path,
        exclude_buildings=True,
    )

    # Build a clean outer boundary polygon (no holes) for dead-zone clipping and visualization.
    from shapely.geometry import Polygon as ShapelyPolygon

    if zone_type == "box":
        bx, by = zone_params["center"][0], zone_params["center"][1]
        bw, bh = zone_params["width"], zone_params["height"]
        box_polygon = ShapelyPolygon(
            [
                (bx - bw / 2, by - bh / 2),
                (bx + bw / 2, by - bh / 2),
                (bx + bw / 2, by + bh / 2),
                (bx - bw / 2, by + bh / 2),
            ]
        )
    else:
        box_polygon = ShapelyPolygon(zone_params["vertices"])

    # PRE-CACHE building polygons for rejection sampling (avoids re-parsing XML every iteration)
    # Convert building_exclusions (list of coordinate lists) to Shapely Polygons once
    cached_building_polygons = []
    for building_coords in building_exclusions:
        try:
            building_poly = ShapelyPolygon(building_coords)
            if building_poly.is_valid:
                cached_building_polygons.append(building_poly)
        except:
            pass
    if verbose:
        print(
            f"Cached {len(cached_building_polygons)} building polygons for rejection sampling"
        )

    # Zone polygon for sampling: box minus in-zone building footprints.
    # Using difference avoids the polygon-with-holes invalidity that plagued the old approach.
    if cached_building_polygons:
        zone_polygon = box_polygon.difference(
            shapely.ops.unary_union(cached_building_polygons)
        )
    else:
        zone_polygon = box_polygon

    # Pre-triangulate the full zone once for use as fallback when no dead zones exist.
    # NOTE: qrand is currently d=3 but sample_triangulated_zone only uses dims 0 and 1.
    # Consider switching to d=2 for marginally better discrepancy.
    tri_verts_full, _ = triangulate_zone(target_zone, building_exclusions)

    # CRITICAL: Remove ALL existing receivers from the scene first
    # This ensures paths.a indexing matches our optimization receivers exactly
    existing_receivers = list(scene.receivers.keys())
    if verbose and existing_receivers:
        print(
            f"Removing {len(existing_receivers)} existing receiver(s): {existing_receivers}"
        )
    for rx_name in existing_receivers:
        scene.remove(rx_name)

    # PRE-CREATE all receivers ONCE (outside the optimization loop)
    # This avoids the expensive remove/add cycle on every iteration
    rx_objects = {}  # Maps rx_name -> Receiver object for position updates
    rx_names = []
    if verbose:
        print(f"Pre-creating {num_sample_points} receivers...")
    for idx in range(num_sample_points):
        rx_name = f"opt_rx_{idx}"
        rx_names.append(rx_name)
        # Initialize at origin - positions will be updated in compute_loss
        rx = Receiver(name=rx_name, position=[0.0, 0.0, 0.0])
        scene.add(rx)
        rx_objects[rx_name] = rx
    if verbose:
        print(f"  Created {len(rx_objects)} receivers (will reposition each iteration)")

    # Create PathSolver w/ Gradients Enabled
    p_solver = PathSolver()
    p_solver.loop_mode = "evaluated"  # Required for gradient computation

    # Storage for sample points (accessible from outside the wrapped function)
    sample_points_storage = {}

    # This function is used to accumulate "dead" samples to isolate the shape of the dead zones
    def accumulate_samples(dead_zone, qrand_op):
        # Sample grid points via rejection sampling (full zone w/o any masking)
        new_sample_points, _, __, ___, ____ = sample_grid_points(
            zone_polygon,
            num_sample_points,
            qrand_op,
            building_polygons=cached_building_polygons,
            ground_z=map_config["center"][2],
        )

        # Store for visualization outside the loss function
        sample_points_storage["current"] = new_sample_points

        # REPOSITION existing receivers (much faster than remove/add)
        for idx, position in enumerate(new_sample_points):
            rx_name = f"opt_rx_{idx}"
            rx_objects[rx_name].position = mi.Point3f(
                float(position[0]), float(position[1]), float(position[2])
            )

        # Run path solver with updated receivers and antenna orientation
        paths = p_solver(
            scene,
            los=True,
            refraction=False,
            specular_reflection=True,
            diffuse_reflection=True,
        )

        # Extract channel coefficients and convert to numpy
        h_real, h_imag = paths.a
        h_real_np = np.array(h_real)
        h_imag_np = np.array(h_imag)

        # Compute per-receiver power: |h|^2 summed over all dims except receiver
        # h shape: (num_rx, num_tx, max_paths, num_rx_ant, num_tx_ant)
        power_all = h_real_np**2 + h_imag_np**2
        power_per_rx = power_all.sum(axis=tuple(range(1, power_all.ndim)))  # (num_rx,)

        # Build combined array: [x, y, power] per receiver
        rx_data = np.column_stack(
            [
                new_sample_points[:, 0],  # x
                new_sample_points[:, 1],  # y
                power_per_rx,  # power
            ]
        )  # shape: (num_rx, 3)

        dead_zone = filter_and_append(rx_data, dead_zone, 5.0)

        return dead_zone

    # Define differentiable loss function using @dr.wrap
    loss_type = "else"

    @dr.wrap(source="torch", target="drjit")
    def compute_loss(azimuth_deg, elevation_deg, x_pos, y_pos, loss_type=loss_type):
        """
        Compute loss with AD enabled through field_calculator only

        This uses pre-traced ray geometry and recomputes only field coefficients
        when antenna orientation changes, providing significant performance improvement.

        Parameters:
        -----------
        azimuth_deg, elevation_deg : DrJit Float scalars (converted from PyTorch tensors)
            Antenna pointing angles in degrees
            After @dr.wrap conversion, these are DrJit Float scalars with gradient tracking enabled
        p_solver : PathSolver
            PathSolver instance used to access _field_calculator
        """

        # Orientation gradients
        dr.enable_grad(azimuth_deg.array)
        dr.enable_grad(elevation_deg.array)
        # Position gradients
        dr.enable_grad(x_pos.array)
        dr.enable_grad(y_pos.array)

        # Convert degrees to radians for angle conversion
        # Use DrJit's pi constant for differentiability
        deg_to_rad = dr.auto.ad.Float(np.pi / 180.0)
        dr.disable_grad(deg_to_rad)  # Conversion factor is constant

        # Remember to bring this back to .array
        azimuth_rad = azimuth_deg * deg_to_rad
        elevation_rad = elevation_deg * deg_to_rad

        # Convert azimuth/elevation to yaw/pitch (roll = 0)
        # yaw = azimuth (rotation around Z-axis)
        # pitch = -elevation (negative because positive pitch tilts up in Mitsuba)
        yaw_rad = azimuth_rad
        pitch_rad = -elevation_rad
        roll_rad = dr.auto.ad.Float(0.0)
        dr.disable_grad(roll_rad)  # Roll is always 0 for antenna pointing

        # Adding jitter to smooth out the "Needle" effect and avoid overfitting to sample points
        # Use numpy for random jitter since this happens outside the differentiable path
        jitter_std_deg = 0.5  # Small jitter in degrees
        jitter_std_rad = jitter_std_deg * (np.pi / 180.0)

        jitter_std_m = 0.1
        pos_jitter = dr.auto.ad.Float(np.random.normal(-jitter_std_m, jitter_std_m))
        dr.disable_grad(pos_jitter)

        # Generate random jitter using numpy (converted to DrJit Float)
        yaw_jitter = dr.auto.ad.Float(np.random.normal(0.0, jitter_std_rad))
        pitch_jitter = dr.auto.ad.Float(np.random.normal(0.0, jitter_std_rad))
        dr.disable_grad(yaw_jitter)  # Jitter is not differentiable
        dr.disable_grad(pitch_jitter)

        # Apply jitter to orientation
        yaw_rad_jittered = yaw_rad + yaw_jitter
        pitch_rad_jittered = pitch_rad + pitch_jitter
        x_pos_jittered = x_pos + pos_jitter
        y_pos_jittered = y_pos + pos_jitter

        # Set antenna orientation directly using yaw, pitch, roll
        scene.get(tx_name).orientation = [
            yaw_rad_jittered,
            pitch_rad_jittered,
            roll_rad,
        ]
        print(f"TX orientation: {scene.get(tx_name).orientation}")

        # Minimal arithmetic - just identity to register in gradient graph
        # Adding in some jitter to avoid getting stuck in corners
        x_pos_val = x_pos_jittered * dr.auto.ad.Float(1.0)
        y_pos_val = y_pos_jittered * dr.auto.ad.Float(1.0)

        scene.get(tx_name).position = [x_pos_val, y_pos_val, tx_position[2]]
        print(f"Tx position: {scene.get(tx_name).position}")

        # Sample dead zone strata proportionally by area.
        # Falls back to full zone when no dead zones exist yet.
        ground_z = map_config["center"][2]
        new_sample_points = sample_dead_zones(dead_zones, num_sample_points)
        if new_sample_points is None:
            new_sample_points = sample_triangulated_zone(
                tri_verts_full, num_sample_points, qrand, ground_z=ground_z
            )
        else:
            new_sample_points[:, 2] = ground_z

        fig = visualize_receiver_placement(
        new_sample_points,
        map_config,
        current_tx_position=[dr.detach(x_pos), dr.detach(y_pos), tx_position[2]],
        box_polygon=box_polygon,
        dead_buffers=dead_buffs if dead_buffs else None,
        alphashapes=dead_zones,
        scene_xml_path=scene_xml_path,
        building_id=building_id,
        )
        plt.show()
        fig.savefig(filepath + f"_{iteration}.png", dpi=350)
        plt.close(fig)

        # Store for visualization/debugging
        sample_points_storage["current"] = new_sample_points

        # Reposition existing receivers using the new spread of receivers
        for idx, position in enumerate(new_sample_points):
            rx_name = f"opt_rx_{idx}"
            rx_objects[rx_name].position = mi.Point3f(
                float(position[0]), float(position[1]), float(position[2])
            )

        # Run path solver with updated receivers and antenna orientation
        paths = p_solver(
            scene,
            los=True,
            refraction=False,
            specular_reflection=True,
            diffuse_reflection=True,
        )

        # Extract channel coefficients
        h_real, h_imag = paths.a

        # Compute incoherent sum (Raw Channel Gain |h|^2)
        power_relative = dr.sum(
            dr.sum(cpx_abs_square((h_real, h_imag)), axis=-1), axis=-1
        )
        power_relative = dr.sum(power_relative, axis=-1)

        if loss_type == "LSE":
            # LSE (Soft Min)
            dead_zone_threshold = 1e-14
            valid_mask = power_relative > dead_zone_threshold
            alpha = 5.0
            epsilon = 1e-30
            log_P = dr.log(power_relative + epsilon)
            exponents = -alpha * log_P
            safe_exponents = dr.select(valid_mask, exponents, -1e9)
            max_exponent = dr.max(safe_exponents)
            shifted_exponents = safe_exponents - max_exponent
            terms = dr.exp(shifted_exponents)
            masked_terms = dr.select(valid_mask, terms, 0.0)
            sum_terms = dr.sum(masked_terms)
            lse = max_exponent + dr.log(sum_terms + epsilon)
            loss = (1.0 / alpha) * lse

        else:
            # Sum Log(Power) -> Automatically heavily penalizes low values
            # Epsilon prevents log(0) from returning -inf for blocked rays
            epsilon = 1e-30
            log_P = dr.log(power_relative + epsilon)
            # dr.mean automatically handles the division by the total static number of receivers
            avg_utility = dr.mean(log_P)

            loss = -avg_utility

        return loss, paths

    # Calculate the initial azimuth and elevation angles based on the position of the transmitter + center of the zone
    # Add z-coordinate (target_height) to zone center which only has [x, y]
    target_z = map_config.get("target_height", 1.5)
    look_at_xyz = list(zone_params["center"])[:2] + [target_z]
    initial_azimuth, initial_elevation = compute_initial_angles_from_position(
        [x_start_position, y_start_position, tx_height],
        look_at_xyz,
        verbose=False,
    )

    # Save the initial angles for analysis with multiple scenarios
    initial_angles = [initial_azimuth, initial_elevation]

    # PyTorch parameters: azimuth and elevation angles (in degrees)
    # Using 0-D tensors (scalars) - this is the ONLY pattern that works with @dr.wrap
    # 1-D tensors lose gradients during indexing
    azimuth = torch.tensor(
        initial_azimuth, device="cuda", dtype=torch.float32, requires_grad=True
    )
    elevation = torch.tensor(
        initial_elevation, device="cuda", dtype=torch.float32, requires_grad=True
    )
    # Adding in transmitter positions x, y
    # Ignoring z for now
    x_pos = torch.tensor(
        tx_position[0], device="cuda", dtype=torch.float32, requires_grad=True
    )
    y_pos = torch.tensor(
        tx_position[1], device="cuda", dtype=torch.float32, requires_grad=True
    )

    print(f"\n{'='*70}")
    print("PYTORCH TENSOR INITIALIZATION")
    print(f"{'='*70}")
    print(f"PyTorch azimuth tensor initialized to: {azimuth.item():.2f}°")
    print(f"PyTorch elevation tensor initialized to: {elevation.item():.2f}°")
    print(f"PyTorch tx_x tensor initialized to: {x_pos.item():.2f}m")
    print(f"")
    print(f"STEP 1: Position parameters created but NOT being optimized yet")
    print(f"        Testing if passing through @dr.wrap breaks PathSolver")
    print(f"{'='*70}\n")

    # Optimizer: Adam shows the best performance out of the gradient descent family
    optimizer = torch.optim.Adam(
        [azimuth, elevation, x_pos, y_pos], lr=learning_rate, betas=(0.9, 0.999)
    )
    # optimizer = torch.optim.SGD([azimuth, elevation, x_pos, y_pos], lr=learning_rate)

    # Learning rate scheduler: required to jump out of local minima for difficult loss surfaces...
    use_scheduler = num_iterations >= 200

    if use_scheduler:
        # Using cosine anneling warm restarts to avoid being trapped in local minima
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=25, T_mult=2, eta_min=learning_rate * 0.01
        )

    # Tracking
    loss_history = []
    angle_history = [[initial_azimuth, initial_elevation]]
    gradient_history = []  # Track gradient norms for diagnostics

    best_azimuth_final = initial_azimuth
    best_elevation_final = initial_elevation

    # List to save the average of the last 10 values
    final_el_list = np.array([])
    final_az_list = np.array([])

    # Creating empty numpy array of dead zone (x, y columns)
    dead_points = np.zeros((0, 3))
    # List of shapely polygons representing dead zones
    dead_buffs = []
    dead_zones = []

    # Start the optimization
    start_time = time.time()

    # Used to check for convergence
    x_pos_prev = None
    y_pos_prev = None

    rm_solver = RadioMapSolver()
    # Hard coding the number of iterations
    loop_iterations = num_iterations
    _use_radio_map = True

    for iteration in range(loop_iterations):
        iter_time = time.time()
        if verbose and iteration == 0:
            print(f"\n{'='*70}")
            print(f"STARTING OPTIMIZATION - Iteration {iteration+1}/{num_iterations}")
            print(f"{'='*70}")
            print(f"  Starting Azimuth: {azimuth.item():.2f}°")
            print(f"  Starting Elevation: {elevation.item():.2f}°")
            print(f"  Starting Position [x]: {tx_position[0]} m")
            print(f"  Starting Position [y]: {tx_position[1]} m")

        # ---- PathSolver accumulation path (original behaviour) ---- #
        if not _use_radio_map:
            dead_points = accumulate_samples(dead_points, qrand)

            if (iteration + 1) % 20 != 0:
                continue

            # Build dead zone polygons from accumulated PathSolver samples
            clusters = DBSCAN(eps=10, min_samples=5).fit(dead_points[:, :2])
            labels = clusters.labels_
            unique_labels = set(labels) - {-1}

            for cluster_id in sorted(unique_labels):
                pts = dead_points[labels == cluster_id, :2]
                shape = alphashape.alphashape(pts, alpha=0.05)
                shape = shapely.make_valid(shape)
                dead_zones.append(shape)
                #clipped = shapely.buffer(shape, .5)
                #donut = shapely.difference(clipped, shape).intersection(zone_polygon)
                #if not donut.is_empty:
                #    dead_buffs.append(donut)
                dead_buffs = dead_zones

            loss, path_out = compute_loss(azimuth, elevation, x_pos, y_pos)
            # Reset for the next accumulation window
            dead_points = np.zeros((0, 3))
            dead_buffs = []
            dead_zones = []

        # ---- RadioMap path: use in place of the PathSolver method ---- #
        else:
            from shapely import contains_xy as _contains_xy
            # Reset dead zone state so each iteration uses only the current radio map
            dead_points = np.zeros((0, 3))
            dead_buffs = []
            dead_zones = []

            # Snapshot current parameter values as plain Python floats, completely
            # detached from the DrJit / PyTorch AD graph, so that rm_solver cannot
            # corrupt gradient state by touching the scene object.
            az_val  = float(azimuth.item())
            el_val  = float(elevation.item())
            xp_val  = float(x_pos.item())
            yp_val  = float(y_pos.item())

            az_rad = float(np.deg2rad(az_val))
            el_rad = float(np.deg2rad(el_val))

            # Write plain-float TX state so rm_solver sees no AD tensors on the scene.
            scene.get(tx_name).orientation = [az_rad, -el_rad, 0.0]
            scene.get(tx_name).position = mi.Point3f(xp_val, yp_val, tx_z)

            # Calculate the RadioMap during this iteration
            rm = rm_solver(
                scene,
                max_depth=8,
                samples_per_tx=int(1e6),
                cell_size=[0.5, 0.5],
                center=map_config["center"],
                orientation=[0, 0, 0],
                size=map_config["size"],
                los=True,
                specular_reflection=True,
                diffuse_reflection=True,
                refraction=False,
                stop_threshold=None,
                capture_rays=int(1e6)
            )

            # Save the buffer values
            rm_pos, rm_ef, _ = rm.captured_rays

            if rm_pos is None:
                raise ValueError(
                    "radio_map.captured_rays returned None. "
                    "Was capture_rays > 0 passed to RadioMapSolver?"
                )

            # Power = |a_cap|^2  (already Friis-corrected in add_paths)
            rm_power = (rm_ef[:, 0]**2 + rm_ef[:, 1]**2
                    + rm_ef[:, 2]**2 + rm_ef[:, 3]**2)

            # Crop to zone polygon
            in_zone = _contains_xy(zone_polygon, rm_pos[:, 0], rm_pos[:, 1])
            rm_pos_z  = rm_pos[in_zone]
            rm_power_z = rm_power[in_zone]
            print(f"  Rays in zone: {in_zone.sum():,} / {len(rm_pos):,}")

            # Build [x, y, power] and keep only the bottom dead_tail_percentile%
            rx_data = np.column_stack([rm_pos_z[:, 0], rm_pos_z[:, 1], rm_power_z])
            dead_points = filter_and_append(rx_data, dead_points, dead_tail_percentile)
            print(f"  Dead points (bottom {dead_tail_percentile}%): {len(dead_points):,}")

            # Subsample so DBSCAN stays tractable
            if len(dead_points) > max_dbscan_points:
                rng = np.random.default_rng(42)
                dead_points = dead_points[
                    rng.choice(len(dead_points), max_dbscan_points, replace=False)
                ]
                print(f"  Subsampled to {max_dbscan_points:,} for DBSCAN")

            # DBSCAN clustering
            #clusters = DBSCAN(eps=2.0, min_samples=20).fit(dead_points[:, :2])
            clusters = HDBSCAN(min_samples=1).fit(dead_points[:, :2])
            labels = clusters.labels_
            unique_labels = set(labels) - {-1}
            print(f"  Clusters found: {len(unique_labels)}")

            for cluster_id in sorted(unique_labels):
                pts = dead_points[labels == cluster_id, :2]
                shape = alphashape.alphashape(pts, alpha=0.05)
                shape = shapely.make_valid(shape)
                dead_zones.append(shape)
                #clipped = shapely.buffer(shape, 5)
                #donut = shapely.difference(clipped, shape).intersection(zone_polygon)
                #if not donut.is_empty:
                #    dead_buffs.append(donut)
                dead_buffs = dead_zones

            print(f"  Dead zones built: {len(dead_zones)}  |  Buffers: {len(dead_buffs)}")
            print(f"{'='*70}\n")
            loss, path_out = compute_loss(azimuth, elevation, x_pos, y_pos)

        # Calculate the gradients in terms of each parameter
        loss.backward()

        # Printing gradients to analyze behavior
        print(f"X position gradient: {x_pos.grad}")
        print(f"Y position gradient: {y_pos.grad}")

        # Track gradient norm for diagnostics
        grad_azimuth_val = azimuth.grad.item() if azimuth.grad is not None else 0.0
        grad_elevation_val = (
            elevation.grad.item() if elevation.grad is not None else 0.0
        )

        # Gradient norm (2 parameters now instead of 3)
        grad_norm = np.sqrt(grad_azimuth_val**2 + grad_elevation_val**2)
        gradient_history.append(grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate if scheduler is enabled (only on optimizer step iterations)
        if use_scheduler:
            scheduler.step()

        # Apply constraints on angles
        with torch.no_grad():
            # Azimuth: clamp to [0, 360) degrees
            azimuth.clamp_(min=0.0, max=360.0)
            # Wrap azimuth around if needed (360° → 0°)
            if azimuth.item() >= 360.0:
                azimuth.fill_(azimuth.item() % 360.0)

        # Apply constraints on Transmitter Position
        # Constrain to the roof polygon (interior is fine, snap to edge if outside)
        with torch.no_grad():
            proj_x, proj_y = tx_placement.project_to_polygon(x_pos.item(), y_pos.item())
            x_pos.data.fill_(proj_x)
            y_pos.data.fill_(proj_y)

        # Modified to reflect robust value over the lowest loss
        window_start = (num_iterations - 10) if _use_radio_map else 5 * (num_iterations - 10)
        if iteration >= window_start:
            # Save the values to a list
            final_az_list = np.append(final_az_list, azimuth.item())
            final_el_list = np.append(final_el_list, elevation.item())

        # Convergence check
        if x_pos_prev is not None:
            if (
                abs(dr.detach(x_pos.item()) - x_pos_prev) < 0.101
                and abs(dr.detach(y_pos.item()) - y_pos_prev) < 0.101
            ):
                print("Local Minimum Reached")

                # Reset and return the values
                final_x = float(x_pos.item())
                final_y = float(y_pos.item())
                scene.get(tx_name).position = mi.Point3f(final_x, final_y, tx_height)
                reset_yaw, reset_pitch = azimuth_elevation_to_yaw_pitch(
                    float(best_azimuth_final), float(best_elevation_final)
                )
                scene.get(tx_name).orientation = mi.Point3f(
                    float(reset_yaw), float(reset_pitch), 0.0
                )

                # Cleanup Rx
                # This is required to make sure the RadioMap calculation doesn't leave any extraneous recievers
                for rx_name in rx_names:
                    if rx_name in [obj.name for obj in scene.receivers.values()]:
                        scene.remove(rx_name)

                # Grab the final position
                tx = scene.get(tx_name)
                final_tx_position = [dr.detach(tx.position[i]) for i in range(3)]

                # Compute final coverage statistics
                coverage_stats = {
                    "loss_type": loss_type,
                    "best_azimuth_deg": best_azimuth_final,
                    "best_elevation_deg": best_elevation_final,
                    "tx_power_dbm": tx_power_dbm,
                }

                # Return angles and positions
                best_angles = [best_azimuth_final, best_elevation_final]
                return (
                    best_angles,
                    loss_history,
                    angle_history,
                    gradient_history,
                    coverage_stats,
                    initial_angles,
                    initial_tx_position,
                    final_tx_position,
                    path_out
                )

        # Saving this run as previous values
        x_pos_prev = dr.detach(x_pos.item())
        y_pos_prev = dr.detach(y_pos.item())
        iter_end = time.time()
        dur = abs(iter_time - iter_end)
        print(f"Time per Iteration: {dur}")

    # Save the average of the final 10 values
    best_azimuth_final = np.mean(final_az_list)
    best_elevation_final = np.mean(final_el_list)

    # Reset TX position and orientation to clean (non-AD) mi.Point3f values.
    # compute_loss sets scene variables to AD-tracked DrJIT tensors; leaving them
    # in that state causes dr.while_loop() failures in any subsequent RadioMapSolver call.
    final_x = float(x_pos.item())
    final_y = float(y_pos.item())
    scene.get(tx_name).position = mi.Point3f(final_x, final_y, tx_height)
    reset_yaw, reset_pitch = azimuth_elevation_to_yaw_pitch(
        float(best_azimuth_final), float(best_elevation_final)
    )
    scene.get(tx_name).orientation = mi.Point3f(
        float(reset_yaw), float(reset_pitch), 0.0
    )

    # Save the elapsed time for metrics
    elapsed_time = time.time() - start_time

    # Cleanup Rx
    # This is required to make sure the RadioMap calculation doesn't leave any extraneous recievers
    for rx_name in rx_names:
        if rx_name in [obj.name for obj in scene.receivers.values()]:
            scene.remove(rx_name)

    # Grab the final position
    tx = scene.get(tx_name)
    final_tx_position = [dr.detach(tx.position[i]) for i in range(3)]

    # Compute final coverage statistics
    coverage_stats = {
        "loss_type": loss_type,
        "best_azimuth_deg": best_azimuth_final,
        "best_elevation_deg": best_elevation_final,
        "tx_power_dbm": tx_power_dbm,
    }

    # Return angles and positions
    best_angles = [best_azimuth_final, best_elevation_final]
    return (
        best_angles,
        loss_history,
        angle_history,
        gradient_history,
        coverage_stats,
        initial_angles,
        initial_tx_position,
        final_tx_position,
        path_out
    )

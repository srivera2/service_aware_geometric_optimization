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
from sionna.rt import PathSolver, Receiver, cpx_abs_square
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
import triangulate
from triangulate import sample_triangulated_zone, get_zone_polygon_with_exclusions, triangulate_zone
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN


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
            np.linspace(center_x - width_m / 2, center_x + width_m / 2, n_x, endpoint=False)
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
        x = (np.linspace(center_x - width_m / 2, center_x + width_m / 2, n_x, endpoint=False) + cell_w / 2)
        y = (np.linspace(center_y - height_m / 2, center_y + height_m / 2, n_y, endpoint=False) + cell_h / 2)
        X, Y = np.meshgrid(x, y)

        # Initialize mask (all zeros = outside zone)
        mask = np.zeros((n_y, n_x), dtype=np.float32)

        # Get the vertices of the polygon from zone_params
        vertices = zone_params.get("vertices", [(0,0), (10,0), (10,-10), (0,-10)])

        from shapely.geometry import Polygon
        from shapely import contains_xy

        # Create polygon from vertices
        zone = Polygon(vertices)

        # Vectorized containment check: flatten grid points and check all at once
        mask_flat = contains_xy(zone, X.flatten(), Y.flatten())
        mask = mask_flat.reshape(n_y, n_x).astype(np.float32)

        # Set the look at position at the centroid of the polygon
        centroid = zone.centroid
        look_at_pos = np.array([centroid.x, centroid.y, target_height], dtype=np.float32)

        # Store bounding box for LDS sampling
        minx, miny, maxx, maxy = zone.bounds
        zone_params['center'] = [(minx + maxx) / 2, (miny + maxy) / 2]
        zone_params['width'] = maxx - minx
        zone_params['height'] = maxy - miny

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
    map_config,
    scene_xml_path=None,
    exclude_buildings=True,
    zone_mask=None,
    zone_params=None,
    qrand=None,
    # Making the number of points configurable from now on
    num_points=20,
    # Pre-loaded building polygons to avoid re-parsing XML every call
    cached_building_polygons=None,
):

    width_m, height_m = map_config["size"]
    cell_w, cell_h = map_config["cell_size"]

    n_x = int(width_m / cell_w)
    n_y = int(height_m / cell_h)

    center_x, center_y, ground_z = map_config["center"]
    
    from shapely.geometry import Polygon, Point
    from shapely import contains_xy
    from shapely.prepared import prep

    # Use cached building polygons if provided, otherwise load them
    # This avoids re-parsing the XML file on every call
    if cached_building_polygons is not None:
        building_polygons = cached_building_polygons
    elif exclude_buildings and scene_xml_path is not None:
        # Load building polygons (expensive - should be cached by caller for repeated calls)
        from scene_parser import extract_building_info
        building_polygons = []
        building_info = extract_building_info(scene_xml_path, verbose=False)
        for building_id, info in building_info.items():
            vertices_3d = info["vertices"]
            vertices_2d = [(v[0], v[1]) for v in vertices_3d]
            try:
                building_polygon = Polygon(vertices_2d)
                if building_polygon.is_valid:
                    building_polygons.append(building_polygon)
            except:
                pass
    else:
        building_polygons = []

    # Helper function to filter out points inside buildings (VECTORIZED)
    def filter_buildings(points_2d):
        """Filter out points inside any building. Returns boolean mask of valid points."""
        if not building_polygons or len(points_2d) == 0:
            return np.ones(len(points_2d), dtype=bool)

        # Use vectorized contains_xy - MUCH faster than Point-by-Point checks
        valid_mask = np.ones(len(points_2d), dtype=bool)
        x_coords = points_2d[:, 0]
        y_coords = points_2d[:, 1]
        for building_poly in building_polygons:
            # contains_xy checks all points against polygon at once (no Point objects)
            inside_building = contains_xy(building_poly, x_coords, y_coords)
            valid_mask &= ~inside_building
        return valid_mask
    
    print(zone_params)
    print(qrand)

    # Quasi-random sampling in continuous space within the zone
    if zone_params is not None and qrand is not None:
        # Extract zone parameters directly
        zone_center = zone_params.get("center")
        if zone_center is None:
            # Fallback: compute from map_config if center not in zone_params
            zone_center = [map_config["center"][0], map_config["center"][1]]
        zone_width = zone_params["width"]
        zone_height = zone_params["height"]

        # Check if this is a polygon zone (has vertices)
        if "vertices" in zone_params:
            # Polygon + building rejection sampling - guarantees exactly num_points
            zone_polygon = Polygon(zone_params["vertices"])
            min_bound_x = zone_center[0] - zone_width / 2
            min_bound_y = zone_center[1] - zone_height / 2

            sampled_points_list = []
            max_iterations = 100  # Safety limit to prevent infinite loops
            iteration = 0

            while len(sampled_points_list) < num_points and iteration < max_iterations:
                iteration += 1
                samples_needed = num_points - len(sampled_points_list)

                # Oversample to account for both polygon and building rejection
                # Start with 2x, increase if we're not getting enough valid points
                oversample_factor = max(2, int(np.ceil(samples_needed / max(1, len(sampled_points_list) / max(1, iteration)))))
                batch_size = min(samples_needed * oversample_factor, 10000)  # Cap batch size

                # Draw from bounding box using quasi-random sequence
                unit_points = np.array(qrand.random(batch_size))
                unit_points = np.clip(unit_points, 0.0, 1.0)

                # Scale to bounding box
                scaled_x = min_bound_x + unit_points[:, 0] * zone_width
                scaled_y = min_bound_y + unit_points[:, 1] * zone_height
                candidate_points = np.column_stack([scaled_x, scaled_y])

                # Stage 1: Keep only points inside polygon
                inside_polygon = contains_xy(zone_polygon, candidate_points[:, 0], candidate_points[:, 1])
                polygon_valid_points = candidate_points[inside_polygon]

                if len(polygon_valid_points) == 0:
                    continue

                # Stage 2: Filter out points inside buildings
                if building_polygons:
                    building_valid_mask = filter_buildings(polygon_valid_points)
                    valid_points = polygon_valid_points[building_valid_mask]
                else:
                    valid_points = polygon_valid_points

                # Add valid points (up to what we need)
                points_to_add = min(len(valid_points), num_points - len(sampled_points_list))
                if points_to_add > 0:
                    sampled_points_list.extend(valid_points[:points_to_add].tolist())

            if len(sampled_points_list) < num_points:
                raise ValueError(
                    f"Could not sample {num_points} points after {max_iterations} iterations. "
                    f"Only found {len(sampled_points_list)} valid points. "
                    "Zone may be too small or mostly covered by buildings."
                )

            sampled_points = np.array(sampled_points_list)

        else:
            # Box + building rejection sampling - guarantees exactly num_points
            min_bound_x = zone_center[0] - zone_width / 2
            min_bound_y = zone_center[1] - zone_height / 2

            sampled_points_list = []
            max_iterations = 100
            iteration = 0

            while len(sampled_points_list) < num_points and iteration < max_iterations:
                iteration += 1
                samples_needed = num_points - len(sampled_points_list)
                oversample_factor = max(2, iteration)  # Increase oversampling if struggling
                batch_size = min(samples_needed * oversample_factor, 10000)

                unit_points = np.array(qrand.random(batch_size))
                unit_points = np.clip(unit_points, 0.0, 1.0)

                scaled_x = min_bound_x + unit_points[:, 0] * zone_width
                scaled_y = min_bound_y + unit_points[:, 1] * zone_height
                candidate_points = np.column_stack([scaled_x, scaled_y])

                # Filter out points inside buildings
                if building_polygons:
                    building_valid_mask = filter_buildings(candidate_points)
                    valid_points = candidate_points[building_valid_mask]
                else:
                    valid_points = candidate_points

                points_to_add = min(len(valid_points), num_points - len(sampled_points_list))
                if points_to_add > 0:
                    sampled_points_list.extend(valid_points[:points_to_add].tolist())

            if len(sampled_points_list) < num_points:
                raise ValueError(
                    f"Could not sample {num_points} points after {max_iterations} iterations. "
                    f"Only found {len(sampled_points_list)} valid points."
                )

            sampled_points = np.array(sampled_points_list)

    # Uniform grid sampling: sample entire grid then filter by mask
    else:
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
        all_points = np.column_stack([X.flatten(), Y.flatten()])

        # Filter by zone mask
        if zone_mask is None:
            candidate_points = all_points
        else:
            mask_flat = zone_mask.flatten()
            candidate_points = all_points[mask_flat == 1.0]

        # Filter out building locations
        if building_polygons:
            building_valid_mask = filter_buildings(candidate_points)
            sampled_points = candidate_points[building_valid_mask]

            if len(sampled_points) == 0:
                raise ValueError(
                    f"All {len(candidate_points)} grid points are inside buildings! "
                    "Try adjusting the zone mask or cell size."
                )
        else:
            sampled_points = candidate_points

    # Add z-coordinate
    z_coords = np.full((len(sampled_points), 1), ground_z)
    sampled_points_3d = np.hstack([sampled_points, z_coords])

    return sampled_points_3d

def filter_and_append(rx_data, dead_zone, threshold):
    # rx_data has columns [x, y, power] — filter rows where power < threshold
    dead_mask = rx_data[:, 2] < threshold
    dead_points = rx_data[dead_mask]  # keeps all 3 columns: [x, y, power]
    # Append dead points along rows (axis=0 preserves the (N, 3) shape)
    if dead_zone.size == 0:
        dead_zone = dead_points
    else:
        dead_zone = np.append(dead_zone, dead_points, axis=0)

    return dead_zone

def visualize_receiver_placement(
    sample_points,
    zone_mask,
    map_config,
    tx_position=None,
    scene_xml_path=None,
    title="Receiver Sampling Visualization",
    figsize=(14, 10),
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
    import matplotlib.patches as mpatches
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

    # Show zone mask with alpha for visibility
    im = ax.imshow(
        zone_mask,
        origin="lower",
        cmap="RdYlGn",
        alpha=0.3,
        extent=extent,
        vmin=0,
        vmax=1,
    )

    # Separate receivers by zone
    cell_w, cell_h = map_config["cell_size"]
    n_x = int(width_m / cell_w)
    n_y = int(height_m / cell_h)

    target_receivers = []
    interference_receivers = []

    for idx, point in enumerate(sample_points):
        x, y = point[0], point[1]
        # Convert to grid indices
        # Grid cells are centered, so we use floor division to find the cell
        i = int((x - (center_x - width_m / 2)) / cell_w)
        j = int((y - (center_y - height_m / 2)) / cell_h)

        # Debug: check if clipping is changing the indices
        i_orig, j_orig = i, j
        i = np.clip(i, 0, n_x - 1)
        j = np.clip(j, 0, n_y - 1)

        if zone_mask[j, i] > 0.5:  # Target zone
            target_receivers.append([x, y])
        else:  # Interference zone
            interference_receivers.append([x, y])
            # Debug: Print details about misclassified points
            if (
                len(interference_receivers) <= 2
            ):  # Only print first 2 misclassified points
                print(
                    f"  [DEBUG] Point at ({x:.2f}, {y:.2f}) classified as interference:"
                )
                print(f"    Grid indices: i={i_orig}->{i}, j={j_orig}->{j}")
                print(f"    Mask value: {zone_mask[j, i]:.2f}")

    target_receivers = np.array(target_receivers)
    interference_receivers = np.array(interference_receivers)

    # Plot receivers
    if len(target_receivers) > 0:
        ax.scatter(
            target_receivers[:, 0],
            target_receivers[:, 1],
            c="green",
            s=30,
            alpha=0.7,
            marker="o",
            edgecolors="darkgreen",
            linewidths=0.5,
            label=f"Target Zone RX ({len(target_receivers)})",
        )

    if len(interference_receivers) > 0:
        ax.scatter(
            interference_receivers[:, 0],
            interference_receivers[:, 1],
            c="red",
            s=30,
            alpha=0.7,
            marker="s",
            edgecolors="darkred",
            linewidths=0.5,
            label=f"Interference Zone RX ({len(interference_receivers)})",
        )

    # Plot transmitter if provided
    if tx_position is not None:
        ax.plot(
            tx_position[0],
            tx_position[1],
            "b*",
            markersize=20,
            label="Transmitter",
            markeredgecolor="navy",
            markeredgewidth=1.5,
        )

    # Overlay building footprints if scene XML provided
    if scene_xml_path is not None:
        try:
            from scene_parser import extract_building_info

            building_info = extract_building_info(scene_xml_path, verbose=False)

            for building_id, info in building_info.items():
                vertices_3d = info["vertices"]
                vertices_2d = [(v[0], v[1]) for v in vertices_3d]

                try:
                    building_polygon = Polygon(vertices_2d)
                    x_coords, y_coords = building_polygon.exterior.xy
                    ax.fill(
                        x_coords,
                        y_coords,
                        facecolor="gray",
                        edgecolor="black",
                        alpha=0.5,
                        linewidth=1,
                    )
                except:
                    pass

            # Add building legend entry
            building_patch = mpatches.Patch(
                facecolor="gray", edgecolor="black", alpha=0.5, label="Buildings"
            )
            handles, labels = ax.get_legend_handles_labels()
            handles.append(building_patch)
            ax.legend(handles=handles, loc="upper right", fontsize=10)
        except Exception as e:
            print(f"Warning: Could not overlay buildings: {e}")
            ax.legend(loc="upper right", fontsize=10)
    else:
        ax.legend(loc="upper right", fontsize=10)

    # Add colorbar for zone mask
    cbar = plt.colorbar(im, ax=ax, label="Zone Mask")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Interference", "Target"])

    # Labels and title
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_aspect("equal")

    # Add text box with sampling statistics
    total_rx = len(sample_points)
    num_target = len(target_receivers)
    num_interference = len(interference_receivers)
    pct_target = 100.0 * num_target / total_rx if total_rx > 0 else 0
    pct_interference = 100.0 * num_interference / total_rx if total_rx > 0 else 0

    stats_text = (
        f"Total Receivers: {total_rx}\n"
        f"Target Zone: {num_target} ({pct_target:.1f}%)\n"
        f"Interference Zone: {num_interference} ({pct_interference:.1f}%)"
    )

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
            #print(f"  Position: x={tx_pos[0]:.2f}, y={tx_pos[1]:.2f}, z={tx_pos[2]:.2f}")

        print(f"  Angles: Azimuth={azimuth_deg:.1f}°, Elevation={elevation_deg:.1f}°")

        # Generate RadioMap
        rm = rm_solver(
            scene,
            max_depth=5,
            samples_per_tx=int(6e8),
            cell_size=map_config["cell_size"],
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

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Helper function to convert Watts to dBm
    def watts_to_dbm(watts):
        return 10.0 * np.log10(watts + 1e-30) + 30.0

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


def optimize_boresight_pathsolver(
    scene,
    tx_name,
    map_config,
    scene_xml_path,
    zone_mask=None,
    zone_params=None,
    num_sample_points=100,
    building_id=10,
    learning_rate=1.0,
    num_iterations=20,
    loss_type="coverage_maximize",
    verbose=True,
    seed=None,
    lds="Sobol",  # Random seed for reproducible sampling
    tx_placement_mode="skip",  # "center", "fixed", "line", "skip" (skip = don't move TX)
    # If true, the center position of the roof polygon is used
    # Else, use the start position
    sampler = 'Rejection',
    Tx_start_pos=[0.0, 0.0],
    save_radiomap_frames=False,
    frame_save_interval=1,
    output_dir="/home/tingjunlab/Development/optimize_tx/figures",
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
    #tx_x = float(dr.detach(tx.position[0])[0])
    #tx_y = float(dr.detach(tx.position[1])[0])
    #tx_z = float(dr.detach(tx.position[2])[0])
    #tx_position = [tx_x, tx_y, tx_z]

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

    # Determine if initial_boresight is angles or position
    # If it's a position [x, y, z], convert to angles
    # If grid search is used, angles will be overridden anyway

    if verbose:
        print(f"\n{'='*70}")
        print("Binary Mask Zone Coverage Optimization")
        print(f"{'='*70}")
        print(f"Transmit power: {tx_power_dbm:.2f} dBm")
        print(f"Learning rate: {learning_rate}")
        print(f"Iterations: {num_iterations}")
        print(f"Sample points: {num_sample_points}")
        print(f"Loss type: {loss_type}")
        print(f"Map config: {map_config}")
        print(f"{'='*70}\n")

    # TX height already extracted above (tx_z is already detached)
    tx_height = tx_z

    # Initialize TxPlacement for accessing building info and edge projection
    # We need this regardless of tx_placement_mode for the edge constraint
    tx_placement = TxPlacement(scene, tx_name, scene_xml_path, building_id, create_if_missing=False)

    # Handle TX placement based on mode
    if tx_placement_mode == "skip":
        # Don't move the TX - use its current position
        # This is useful when TX was already placed correctly before calling optimization
        x_start_position = float(dr.detach(tx.position[0])[0])
        y_start_position = float(dr.detach(tx.position[1])[0])
        if verbose:
            print(f"TX placement mode: skip (using current position)")
            print(
                #f"  Current TX position: ({x_start_position:.2f}, {y_start_position:.2f}, {tx_height:.2f})"
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
            tx.position = mi.Point3f(float(x_start_position), float(y_start_position), float(z_pos))
        else:
            raise ValueError(
                f"Unknown tx_placement_mode: {tx_placement_mode}. Must be 'skip', 'center', 'fixed', or 'line'"
            )

    if verbose:
        #print(f"TX height: {tx_height:.1f}m")
        print(
            #f"Boresight Z constraint: must be < {tx_height:.1f}m (no pointing upward)\n"
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
    if 'vertices' in zone_params:
        zone_type = 'polygon'
    elif 'center' in zone_params and 'width' in zone_params and 'height' in zone_params:
        zone_type = 'box'
    else:
        raise ValueError(
            "zone_params must contain either 'vertices' (for polygon) or "
            "'center', 'width', 'height' (for box)"
        )

    if verbose:
        print(f"Auto-detected zone type: {zone_type}")

    # Solve for polygon zone and remove building exclusions.
    target_zone, building_exclusions, _ = get_zone_polygon_with_exclusions(
        zone_type=zone_type,
        zone_params=zone_params,
        scene_xml_path=scene_xml_path,
        exclude_buildings=True
    )

    # Triangulate using CDT (only need to do this once)
    triangles, _ = triangulate_zone(
        target_zone,
        building_exclusions,
        buffer_distance=-.01,
        verbose=True
    )

    # PRE-CACHE building polygons for rejection sampling (avoids re-parsing XML every iteration)
    # Convert building_exclusions (list of coordinate lists) to Shapely Polygons once
    cached_building_polygons = []
    for building_coords in building_exclusions:
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            building_poly = ShapelyPolygon(building_coords)
            if building_poly.is_valid:
                cached_building_polygons.append(building_poly)
        except:
            pass
    if verbose:
        print(f"Cached {len(cached_building_polygons)} building polygons for rejection sampling")

    # Print to see the number of sample points after sampling
    if verbose:
        print(f"Actual sample points after building exclusion: {num_sample_points}")

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

    # Create PathSolver
    p_solver = PathSolver()
    p_solver.loop_mode = "evaluated"  # Required for gradient computation

    # Storage for sample points (accessible from outside the wrapped function)
    sample_points_storage = {}

    def accumulate_samples(dead_zone, qrand_op):
        # Sample grid points via rejection sampling (global)
        new_sample_points = sample_grid_points(
            map_config,
            scene_xml_path=scene_xml_path,
            exclude_buildings=True,
            zone_mask=zone_mask,
            zone_params=zone_params,
            qrand=qrand_op,
            num_points=num_sample_points,
            cached_building_polygons=cached_building_polygons,  # Use pre-cached polygons
        )

        # Store for visualization outside the loss function
        sample_points_storage['current'] = new_sample_points

        # REPOSITION existing receivers (much faster than remove/add)
        # Receivers were pre-created before the optimization loop
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
        rx_data = np.column_stack([
            new_sample_points[:, 0],  # x
            new_sample_points[:, 1],  # y
            power_per_rx,             # power
        ])  # shape: (num_rx, 3)

        dead_zone = filter_and_append(rx_data, dead_zone, 10e-13)

        return dead_zone
    
    # Define differentiable loss function using @dr.wrap
    @dr.wrap(source="torch", target="drjit")
    def compute_loss(azimuth_deg, elevation_deg, x_pos, y_pos, qrand_op, num_sample_points, dead_zone, loss_type='CVaR'):
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
        jitter_std_deg = .5  # Small jitter in degrees
        jitter_std_rad = jitter_std_deg * (np.pi / 180.0)

        # Generate random jitter using numpy (converted to DrJit Float)
        yaw_jitter = dr.auto.ad.Float(np.random.normal(0.0, jitter_std_rad))
        pitch_jitter = dr.auto.ad.Float(np.random.normal(0.0, jitter_std_rad))
        dr.disable_grad(yaw_jitter)  # Jitter is not differentiable
        dr.disable_grad(pitch_jitter)

        # Apply jitter to orientation
        yaw_rad_jittered = yaw_rad + yaw_jitter
        pitch_rad_jittered = pitch_rad + pitch_jitter

        # Set antenna orientation directly using yaw, pitch, roll
        scene.get(tx_name).orientation = [yaw_rad_jittered, pitch_rad_jittered, roll_rad]
        print(f"TX orientation: {scene.get(tx_name).orientation}")

        # Minimal arithmetic - just identity to register in gradient graph
        x_pos_val = x_pos * dr.auto.ad.Float(1.0)
        y_pos_val = y_pos * dr.auto.ad.Float(1.0)

        scene.get(tx_name).position = [x_pos_val, y_pos_val, tx_position[2]]
        print(f"Tx position: {scene.get(tx_name).position}")

        # Sample grid points via rejection sampling (global)
        new_sample_points = sample_grid_points(
            map_config,
            scene_xml_path=scene_xml_path,
            exclude_buildings=True,
            zone_mask=zone_mask,
            zone_params=zone_params,
            qrand=qrand_op,
            num_points=num_sample_points,
            cached_building_polygons=cached_building_polygons,  # Use pre-cached polygons
        )

        # Store for visualization outside the loss function
        sample_points_storage['current'] = new_sample_points

        # REPOSITION existing receivers (much faster than remove/add)
        # Receivers were pre-created before the optimization loop
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

        # Check if PathSolver found any paths
        # Use dr.width() to check actual data entries, not just array structure
        try:
            h_real_width = dr.width(h_real) if hasattr(h_real, '__len__') else 0
        except:
            h_real_width = 0

        if len(h_real) == 0 or h_real_width == 0:
            # Initialize all counters individually if needed
            if not hasattr(compute_loss, "_iter_count"):
                compute_loss._iter_count = 0
            if not hasattr(compute_loss, "_failed_first_iter"):
                compute_loss._failed_first_iter = False
            if not hasattr(compute_loss, "_empty_path_count"):
                compute_loss._empty_path_count = 0
            if not hasattr(compute_loss, "_too_many_failures"):
                compute_loss._too_many_failures = False

            # Increment empty path counter
            compute_loss._empty_path_count += 1

            print(f"\n  [WARNING] PathSolver found 0 paths! (Empty count: {compute_loss._empty_path_count}/5)")
            print(f"            Azimuth={float(azimuth_deg.item()):.1f}°, Elevation={float(elevation_deg.item()):.1f}°")

            # Check if we've exceeded the threshold
            if compute_loss._empty_path_count > 5:
                compute_loss._too_many_failures = True
                print(f"            CRITICAL: More than 5 empty PathSolver results detected!")
                print(f"            Marking optimization for abandonment.\n")
                penalty = dr.auto.ad.Float(-1e10) - elevation_deg * 0.1 - azimuth_deg * 0.01
                return penalty

            if compute_loss._iter_count == 0:
                # Mark first iteration as failed
                compute_loss._failed_first_iter = True
                print(f"            First iteration failed - bad zone/TX setup.")
                print(f"            Signaling caller to skip this configuration.\n")
                # Return penalty with synthetic gradient
                # Gradient pushes elevation up (away from ground)
                penalty = dr.auto.ad.Float(-1e10) - elevation_deg * 0.1
                return penalty

            # Later iterations - create synthetic gradient to guide optimizer away
            print(f"            Injecting synthetic gradient to guide optimizer.\n")
            # Penalty loss with gradient that encourages increasing elevation (pointing up)
            penalty = dr.auto.ad.Float(-1e10) - elevation_deg * 0.1 - azimuth_deg * 0.01
            return penalty

        # Increment iteration counter
        if not hasattr(compute_loss, "_iter_count"):
            compute_loss._iter_count = 0
        compute_loss._iter_count += 1

        # DEBUG: Check if any paths were found (only print on first call)
        # We use a simple flag via a mutable default to track first call
        if not hasattr(compute_loss, "_first_call_done"):
            compute_loss._first_call_done = False

        if verbose and not compute_loss._first_call_done:
            compute_loss._first_call_done = True
            # Compute total power across all receivers as a quick check
            total_power = dr.sum(dr.sum(cpx_abs_square((h_real, h_imag))))
            print(f"  [DEBUG] PathSolver found {len(h_real)} paths for {len(new_sample_points)} receivers")
            print(f"  [DEBUG] Total path power (linear): {total_power}")
            if total_power < 1e-25:
                print(
                    f"  [WARNING] Very low or zero path power - PathSolver may not be finding paths!"
                )

        # Additional safety check before tensor reduction
        # Verify h_real and h_imag have actual data to prevent TensorXf creation with 0 entries
        try:
            # Test if we can compute power on a small subset first
            test_power = cpx_abs_square((h_real, h_imag))
            if dr.width(test_power) == 0:
                # Increment empty path counter (initialize if needed)
                if not hasattr(compute_loss, "_empty_path_count"):
                    compute_loss._empty_path_count = 0
                if not hasattr(compute_loss, "_too_many_failures"):
                    compute_loss._too_many_failures = False
                compute_loss._empty_path_count += 1

                print(f"\n  [WARNING] cpx_abs_square returned empty tensor! (Empty count: {compute_loss._empty_path_count}/5)")
                print(f"            h_real width: {dr.width(h_real)}, h_imag width: {dr.width(h_imag)}")

                # Check if we've exceeded the threshold
                if compute_loss._empty_path_count > 5:
                    compute_loss._too_many_failures = True
                    print(f"            CRITICAL: More than 5 empty PathSolver results detected!")
                    print(f"            Marking optimization for abandonment.\n")

                print(f"            Returning penalty loss.")
                penalty = dr.auto.ad.Float(-1e10) - elevation_deg * 0.1 - azimuth_deg * 0.01
                return penalty
        except Exception as e:
            # Increment empty path counter
            if not hasattr(compute_loss, "_empty_path_count"):
                compute_loss._empty_path_count = 0
            if not hasattr(compute_loss, "_too_many_failures"):
                compute_loss._too_many_failures = False
            compute_loss._empty_path_count += 1

            print(f"\n  [ERROR] Failed to compute cpx_abs_square: {e} (Empty count: {compute_loss._empty_path_count}/5)")

            # Check if we've exceeded the threshold
            if compute_loss._empty_path_count > 5:
                compute_loss._too_many_failures = True
                print(f"           CRITICAL: More than 5 empty PathSolver results detected!")
                print(f"           Marking optimization for abandonment.\n")

            print(f"         Returning penalty loss.")
            penalty = dr.auto.ad.Float(-1e10) - elevation_deg * 0.1 - azimuth_deg * 0.01
            return penalty

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
        
        elif loss_type == "CVaR":
            # 1. Convert to dB and Floor Clamp
            # -120dB floor provides a finite gradient for dead zones
            val_db = 10.0 * dr.log(power_relative + 1e-35) / dr.log(10.0)

            # 2. Determine the Target via NumPy (The Waterline)
            # We detach to ensure the goalpost doesn't "cheat" by moving itself
            val_db_np = np.array(dr.detach(val_db))
            target_scalar = np.median(val_db_np)
            target = type(val_db)(target_scalar)

            # 3. Deficit-Weighted (Quadratic) Hinge Loss
            deficit = target - val_db
            hinge = dr.maximum(deficit, 0.0)
            loss_contribution = hinge * hinge

            # 4. Normalize by the Tail Count
            # This is the "Conditional" part of CVaR. 
            # We only average over the users who are actually failing.
            tail_mask = deficit > 0.0
            tail_count = dr.sum(dr.select(tail_mask, 1.0, 0.0))
            
            # Divide total loss by the number of contributors
            loss = dr.sum(loss_contribution) / (tail_count + 1e-5)

        elif loss_type == "threshold":
            # Target: -90 dBm (The goal line we want users to cross)
            dead_threshold = 1e-14
            target_db = -70.0

            # Filter to calculate gradient only using alive transmitters
            is_alive = power_relative > dead_threshold
            epsilon = 1e-30
            safe_power = dr.select(is_alive, power_relative, 1.0)
            vals_db = 10.0 * dr.log(safe_power + epsilon) / dr.log(10.0)
            
            # Hinge error
            error = target_db - vals_db
            
            # Params: Mask, True Value, False Value
            active_loss = dr.select(is_alive, dr.maximum(error, 0.0), 0.0)
            
            # Normalize
            pushing_mask = is_alive & (error > 0)
            pushing_count = dr.sum(dr.select(pushing_mask, 1.0, 0.0))
            
            # Minimize the total contributing error
            loss = dr.sum(active_loss) / (pushing_count + 1e-5)

        else:
            # Sum Log(Power) -> Penalizes low values
            dead_zone_threshold = 1e-14
            valid_mask = power_relative > dead_zone_threshold
            epsilon = 1e-30
            log_P = dr.log(power_relative + epsilon)
            masked_log_P = dr.select(valid_mask, log_P, 0.0)
            log_utility = dr.sum(masked_log_P)
            count = dr.sum(dr.select(valid_mask, 1.0, 0.0))
            avg_utility = log_utility / (count + 1e-5)
            loss = -avg_utility

        # Use DBSCAN to cluster dead zones by spatial position (x, y columns only)
        clusters = HDBSCAN(
            min_cluster_size=5,       
            min_samples=25,           
            cluster_selection_epsilon=1.0,
            allow_single_cluster=False,
            copy=True
        ).fit(dead_zone[:, :2])
        
        labels = clusters.labels_
        # Save number of clusters and noise points
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        return loss, clusters

    # Calculate the initial azimuth and elevation angles based on the position of the transmitter + center of the zone
    # Add z-coordinate (target_height) to zone center which only has [x, y]
    target_z = map_config.get('target_height', 1.5)
    look_at_xyz = list(zone_params['center'])[:2] + [target_z]
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

    # Optimizer: Adam shows the best performance
    # STEP 1: Only optimize angles (2 parameters)
    # TODO STEP 3: Add tx_x, tx_y to optimizer after Step 1 passes
    optimizer = torch.optim.Adam([azimuth, elevation, x_pos, y_pos], lr=learning_rate, betas=(0.9, 0.999))

    # Learning rate scheduler: required to jump out of local minima for difficult loss surfaces...
    use_scheduler = num_iterations >= 50

    if use_scheduler:
        # Cosine annealing with warm restarts: periodically resets LR to escape local minima.
        # T_0 = restart period (iterations), T_mult = multiply period after each restart.
        # With T_0=25, T_mult=2: restarts at iter 25, 75 (25+50), giving 3 exploration phases.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=25, T_mult=2, eta_min=learning_rate * 0.01
        )
        if verbose:
            print(f"Learning rate scheduler enabled (cosine annealing with warm restarts)")
            print(f"  Initial LR: {learning_rate:.4f}")
            print(f"  Restart period: T_0=25, T_mult=2 (restarts at iter 25, 75)")
            print(f"  Min LR: {learning_rate * 0.01:.6f}")

    # Create output directory for frame saving
    if save_radiomap_frames:
        import os

        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Saving frames to: {output_dir}")

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
    dead_areas = np.zeros((0, 3))

    # Start the optimization
    start_time = time.time()
    for iteration in range(num_iterations):
        if verbose and iteration == 0:
            print(f"\n{'='*70}")
            print(f"STARTING OPTIMIZATION - Iteration {iteration+1}/{num_iterations}")
            print(f"{'='*70}")
            print(f"  Starting Azimuth: {azimuth.item():.2f}°")
            print(f"  Starting Elevation: {elevation.item():.2f}°")
            print(f"  (These should match the naive baseline angles shown above)")
            print(f"{'='*70}\n")

        # DEBUG: Verify tensor setup before compute_loss
        if iteration == 0:
            print(f"[TENSOR DEBUG] azimuth requires_grad: {azimuth.requires_grad}")
            print(f"[TENSOR DEBUG] elevation requires_grad: {elevation.requires_grad}")
            print(f"[TENSOR DEBUG] x_pos requires_grad: {x_pos.requires_grad}")
            print(f"[TENSOR DEBUG] x_pos value: {x_pos.item()}")

        # Accumulate dead zone samples every iteration
        dead_areas = accumulate_samples(dead_areas, qrand)

        # Every 5th accumulated batch, compute loss and reset
        if (iteration + 1) % 10 == 0:
            dead_zone_pos = dead_areas[:,:2]
            loss, clusters = compute_loss(azimuth, elevation, x_pos, y_pos, qrand, num_sample_points, dead_areas, sampler)
            # Reset the dead zone for the next accumulation window
            dead_areas = np.zeros((0, 3))

        loss, clusters = compute_loss(azimuth, elevation, x_pos, y_pos, qrand, num_sample_points, dead_areas, sampler)
        # Take convex hull of each cluster (2D: x, y only)
        labels = clusters.labels_
        unique_labels = set(labels) - {-1}
        dead_xy = dead_areas[:, :2]

        if len(unique_labels) > 0:
            fig, ax = plt.subplots()
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

            for i, cluster_id in enumerate(sorted(unique_labels)):
                mask = labels == cluster_id
                pts = dead_xy[mask]
                ax.scatter(pts[:, 0], pts[:, 1], color=colors[i],
                           label=f'Cluster {cluster_id}', alpha=0.5, s=10)

                if len(pts) >= 3:
                    try:
                        hull = ConvexHull(pts)
                        hull_verts = np.append(hull.vertices, hull.vertices[0])
                        ax.plot(pts[hull_verts, 0], pts[hull_verts, 1],
                                color=colors[i], linewidth=1.5)
                        ax.fill(pts[hull_verts, 0], pts[hull_verts, 1],
                                color=colors[i], alpha=0.15)
                    except Exception as e:
                        print(f"  ConvexHull failed for cluster {cluster_id}: {e}")

            noise_mask = labels == -1
            if np.any(noise_mask):
                ax.scatter(dead_xy[noise_mask, 0], dead_xy[noise_mask, 1],
                           color='gray', label='Noise', alpha=0.3, s=5, marker='x')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            ax.legend()
            ax.set_title(f'Dead Zone Clusters (iter {iteration+1})')
            plt.show()


        # Check if first iteration failed (no paths found)
        if iteration == 0 and hasattr(compute_loss, "_failed_first_iter") and compute_loss._failed_first_iter:
            print("\n" + "="*70)
            print("OPTIMIZATION ABORTED: First iteration found 0 paths")
            print("="*70)
            print("This zone/TX configuration cannot find any propagation paths.")
            print("Possible causes:")
            print("  - TX is too low (pointing into ground)")
            print("  - Zone is entirely blocked by buildings")
            print("  - Scene geometry issues")
            print("\nReturning None to signal failure to caller.")
            print("="*70 + "\n")
            initial_pos = [float(tx_position[0]) if hasattr(tx_position[0], 'item') else float(tx_position[0]),
                           float(tx_position[1]) if hasattr(tx_position[1], 'item') else float(tx_position[1]),
                           float(tx_position[2]) if hasattr(tx_position[2], 'item') else float(tx_position[2])]
            return initial_angles, None, None, None, None, initial_angles, initial_pos, initial_pos

        # Check if too many failures occurred during optimization
        if hasattr(compute_loss, "_too_many_failures") and compute_loss._too_many_failures:
            print("\n" + "="*70)
            print("OPTIMIZATION ABORTED: More than 5 empty PathSolver results")
            print("="*70)
            print(f"Empty path count: {compute_loss._empty_path_count}")
            print(f"Current iteration: {iteration+1}/{num_iterations}")
            print("\nThis configuration is consistently failing to find propagation paths.")
            print("Possible causes:")
            print("  - Optimizer is pointing antenna into ground or sky")
            print("  - Zone geometry is incompatible with TX placement")
            print("  - Extreme antenna angles causing no valid paths")
            print("\nReturning None to signal failure to caller.")
            print("="*70 + "\n")
            initial_pos = [float(tx_position[0]) if hasattr(tx_position[0], 'item') else float(tx_position[0]),
                           float(tx_position[1]) if hasattr(tx_position[1], 'item') else float(tx_position[1]),
                           float(tx_position[2]) if hasattr(tx_position[2], 'item') else float(tx_position[2])]
            return initial_angles, None, None, None, None, initial_angles, initial_pos, initial_pos
        
        loss.backward()

        # DEBUG: Check gradient computation
        print(f"[GRAD DEBUG] azimuth.grad: {azimuth.grad}")
        print(f"[GRAD DEBUG] elevation.grad: {elevation.grad}")
        print(f"[GRAD DEBUG] x_pos.grad: {x_pos.grad}")
        if x_pos.grad is not None:
            print(f"[GRAD DEBUG] x_pos.grad value: {x_pos.grad.item()}")
        else:
            print("[GRAD DEBUG] x_pos.grad is None - gradient not computed!")

        # Visualize the Sobol sampling pattern (only save every N iterations to reduce file count)
        if save_radiomap_frames and (iteration % frame_save_interval == 0):
            current_points = sample_points_storage.get('current', None)
            if current_points is not None:
                fig = visualize_receiver_placement(
                    current_points,
                    zone_mask,
                    map_config,
                    tx_position=tx_position,
                    scene_xml_path=scene_xml_path,
                    title=f"LDS - Iteration {iteration}",
                    figsize=(14, 10),
                )
                plt.savefig(
                    f"{output_dir}/sampling_iteration_{iteration:04d}.png",
                    dpi=100,
                    bbox_inches="tight",
                )
                plt.close(fig)

        # Track gradient norm for diagnostics
        grad_azimuth_val = azimuth.grad.item() if azimuth.grad is not None else 0.0
        grad_elevation_val = (
            elevation.grad.item() if elevation.grad is not None else 0.0
        )

        # Gradient norm (2 parameters now instead of 3)
        grad_norm = np.sqrt(grad_azimuth_val**2 + grad_elevation_val**2)
        gradient_history.append(grad_norm)

        if verbose:
            print(
                f"  Gradients: dAz={grad_azimuth_val:+.3e}°, dEl={grad_elevation_val:+.3e}°"
            )
            # Convert loss back to mean power in dBm
            # Since loss = -mean(dBm), we just negate it to get mean(dBm)
            mean_power_in_zone_dbm = -loss.item()
            print(
                f"  Loss: {loss.item():.4f}, Mean Power in Zone: {mean_power_in_zone_dbm:.2f} dBm"
            )

        accumulation_steps = 10
        if (iteration + 1) % accumulation_steps == 0:
            # Average the accumulated gradients instead of using the sum
            for param in [azimuth, elevation, x_pos, y_pos]:
                if param.grad is not None:
                    param.grad /= accumulation_steps
            print("Stepping optimizer")
            optimizer.step()
            optimizer.zero_grad()

        # Update learning rate if scheduler is enabled (only on optimizer step iterations)
        if use_scheduler and (iteration + 1) % accumulation_steps == 0:
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
            proj_x, proj_y = tx_placement.project_to_polygon(
                x_pos.item(),
                y_pos.item()
            )
            x_pos.data.fill_(proj_x)
            y_pos.data.fill_(proj_y)

        # Track
        #loss_history.append(loss.item())
        #angle_history.append([float(dr.detach(azimuth)), float(dr.detach(elevation))])

        # Modified to reflect robust value over the lowest loss
        if iteration > (num_iterations - 10):
            # Save the values to a list
            final_az_list = np.append(final_az_list, azimuth.item())
            final_el_list = np.append(final_el_list, elevation.item())

    # Save the average of the final 10 values
    best_azimuth_final = np.mean(final_az_list)
    best_elevation_final = np.mean(final_el_list)

    # Save the elapsed time for metrics
    elapsed_time = time.time() - start_time

    # Cleanup receivers
    # This is required to make sure the RadioMap calculation doesn't have any extraneous recievers
    for rx_name in rx_names:
        if rx_name in [obj.name for obj in scene.receivers.values()]:
            scene.remove(rx_name)

    # Set up final scene
    tx = scene.get(tx_name)

    # Set final antenna orientation using best angles (non-AD) - use pure Python floats
    # Failing to do this can cause issues with the RadioMap solver...
    final_yaw_rad, final_pitch_rad = azimuth_elevation_to_yaw_pitch(
        best_azimuth_final, best_elevation_final
    )
    tx.orientation = mi.Point3f(float(final_yaw_rad), float(final_pitch_rad), 0.0)

    # Extract final position from scene after optimization
    final_tx_position = [dr.detach(tx.position[i]) for i in range(3)]

    # Compute final coverage statistics
    coverage_stats = {
        "loss_type": loss_type,
        "best_azimuth_deg": best_azimuth_final,
        "best_elevation_deg": best_elevation_final,
        "tx_power_dbm": tx_power_dbm,
    }

    if verbose:
        print(f"\n{'='*70}")
        print("Optimization Complete!")
        print(f"{'='*70}")
        print(
            f"Best angles: Azimuth={best_azimuth_final:.1f}°, Elevation={best_elevation_final:.1f}°"
        )
        #if final_tx_position[0] != initial_tx_position[0]:
        #    print(f"TX Position change (Δx): {final_tx_position[0] - initial_tx_position[0]:.2f}m")
        print(f"Total time: {elapsed_time:.1f}s")
        print(f"Time per iteration: {elapsed_time/num_iterations:.2f}s")
        print(f"{'='*70}\n")

    # Return angles and positions
    best_angles = [best_azimuth_final, best_elevation_final]
    return best_angles, loss_history, angle_history, gradient_history, coverage_stats, initial_angles, initial_tx_position, final_tx_position

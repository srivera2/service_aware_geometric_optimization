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
from shapely import contains_xy
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


def create_optimization_gif(
    frame_dir,
    output_path="optimization.gif",
    duration=200,
    loop=0,
    sector_angles=None,
    tx_position=None,
    map_config=None,
):
    """
    Create GIF from saved optimization frames with optional sector coverage overlay

    Parameters:
    -----------
    frame_dir : str
        Directory containing frame_*.png files
    output_path : str
        Path to save the output GIF
    duration : int
        Duration of each frame in milliseconds (default: 200ms)
    loop : int
        Number of times to loop (0 = infinite)
    sector_angles : list of dict or None
        List of sector definitions to overlay on frames. Each dict should contain:
        - 'angle_start': Start angle in degrees (0° = East, counter-clockwise)
        - 'angle_end': End angle in degrees
        - 'color': (Optional) Color for the sector overlay (default: 'red')
        - 'alpha': (Optional) Transparency for the sector (default: 0.2)
        Example: [{'angle_start': 0, 'angle_end': 120, 'color': 'red', 'alpha': 0.2}]
    tx_position : tuple or None
        (x, y, z) position of transmitter for sector overlay origin
    map_config : dict or None
        Map configuration with 'center', 'size'. Required if sector_angles is provided.

    Returns:
    --------
    str : Path to created GIF
    """
    import os
    import glob

    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Wedge
        import io
    except ImportError:
        print("PIL or matplotlib not found, trying imageio...")
        import imageio

        # Get all frame files in order
        frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))

        if not frame_files:
            print(f"No frames found in {frame_dir}")
            return None

        print(f"Creating GIF from {len(frame_files)} frames...")

        # Read images and create GIF
        images = [imageio.imread(f) for f in frame_files]
        imageio.mimsave(output_path, images, duration=duration / 1000.0, loop=loop)

        print(f"GIF saved to: {output_path}")
        return output_path

    # Get all frame files in order
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))

    if not frame_files:
        print(f"No frames found in {frame_dir}")
        return None

    print(f"Creating GIF from {len(frame_files)} frames...")

    # Process frames with sector overlay if requested
    if sector_angles is not None and tx_position is not None and map_config is not None:
        print(f"Adding sector overlays to frames...")
        processed_images = []

        for frame_file in frame_files:
            # Load the frame
            img = Image.open(frame_file)

            # Create figure from image
            fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)
            ax.imshow(img)
            ax.axis("off")

            # Calculate sector overlay position
            # Map tx_position to image coordinates
            center_x, center_y, _ = map_config["center"]
            width_m, height_m = map_config["size"]

            # Image extent (data coordinates)
            extent = [
                center_x - width_m / 2,
                center_x + width_m / 2,
                center_y - height_m / 2,
                center_y + height_m / 2,
            ]

            # TX position in data coordinates
            # Convert to float to handle numpy arrays or mitsuba objects
            tx_x = float(tx_position[0])
            tx_y = float(tx_position[1])

            # Calculate radius for sector visualization (should cover the map)
            max_radius = float(np.sqrt(width_m**2 + height_m**2))

            # Draw each sector
            for sector in sector_angles:
                angle_start = sector["angle_start"]
                angle_end = sector["angle_end"]
                color = sector.get("color", "red")
                alpha = sector.get("alpha", 0.2)

                # Convert angles: matplotlib uses degrees counter-clockwise from East
                # Our convention: 0° = East, counter-clockwise
                # matplotlib Wedge: theta1 is start, theta2 is end (counter-clockwise from East)

                # Handle wraparound case (e.g., sector from 350° to 10°)
                if angle_end < angle_start:
                    # Draw two wedges: one from angle_start to 360, another from 0 to angle_end
                    wedge1 = Wedge(
                        (tx_x, tx_y),
                        max_radius,
                        angle_start,
                        360,
                        facecolor=color,
                        edgecolor=color,
                        alpha=alpha,
                        linewidth=2,
                    )
                    wedge2 = Wedge(
                        (tx_x, tx_y),
                        max_radius,
                        0,
                        angle_end,
                        facecolor=color,
                        edgecolor=color,
                        alpha=alpha,
                        linewidth=2,
                    )
                    ax.add_patch(wedge1)
                    ax.add_patch(wedge2)

                    # Draw boundary lines
                    for angle_deg in [angle_start, angle_end]:
                        angle_rad = np.radians(angle_deg)
                        dx = max_radius * np.cos(angle_rad)
                        dy = max_radius * np.sin(angle_rad)
                        ax.plot(
                            [tx_x, tx_x + dx],
                            [tx_y, tx_y + dy],
                            color=color,
                            linewidth=2,
                            alpha=0.8,
                        )
                else:
                    # Normal sector
                    wedge = Wedge(
                        (tx_x, tx_y),
                        max_radius,
                        angle_start,
                        angle_end,
                        facecolor=color,
                        edgecolor=color,
                        alpha=alpha,
                        linewidth=2,
                    )
                    ax.add_patch(wedge)

                    # Draw boundary lines to make sector more visible
                    for angle_deg in [angle_start, angle_end]:
                        angle_rad = np.radians(angle_deg)
                        dx = max_radius * np.cos(angle_rad)
                        dy = max_radius * np.sin(angle_rad)
                        ax.plot(
                            [tx_x, tx_x + dx],
                            [tx_y, tx_y + dy],
                            color=color,
                            linewidth=2,
                            alpha=0.8,
                        )

            # Set the correct limits to match the original image
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            processed_img = Image.open(buf)
            processed_images.append(processed_img.copy())
            buf.close()
            plt.close(fig)

        images = processed_images
    else:
        # Load images without modification
        images = [Image.open(f) for f in frame_files]

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=False,
    )

    print(f"GIF saved to: {output_path}")
    print(f"  Total frames: {len(images)}")
    print(f"  Frame duration: {duration}ms")
    print(f"  Total duration: {len(images) * duration / 1000:.1f}s")

    return output_path


def estimate_achievable_power(
    tx_position,
    map_config,
    antenna_gain_dBi=8.0,
    tx_power_dBm=30.0,
    frequency_GHz=3.5,
    path_loss_exponent=2.5,
    beamwidth_3dB=65.0,
):
    """
    Estimate achievable power levels at different locations on the map

    This helps auto-scale target power levels to be realistic based on:
    - Distance from TX to coverage area
    - Antenna characteristics (gain, beamwidth)
    - Path loss model

    Parameters:
    -----------
    tx_position : tuple
        (x, y, z) position of transmitter
    map_config : dict
        Map configuration with 'center', 'size'
    antenna_gain_dBi : float
        Antenna gain in main lobe (dBi). TR38.901 typical: 8 dBi
    tx_power_dBm : float
        Transmit power in dBm
    frequency_GHz : float
        Carrier frequency in GHz
    path_loss_exponent : float
        Path loss exponent (2.0 = free space, 2.5-4 = urban)
    beamwidth_3dB : float
        3dB beamwidth in degrees. TR38.901 typical: 65°

    Returns:
    --------
    dict with keys:
        'peak_power_dbm': Maximum achievable power (center of main lobe, closest point)
        'min_power_dbm': Minimum achievable power (edge of map, sidelobe)
        'mainlobe_power_dbm': Typical power in main lobe at map edges
        'sidelobe_power_dbm': Typical power in sidelobes
    """
    tx_x, tx_y, tx_z = tx_position
    center_x, center_y, center_z = map_config["center"]
    width_m, height_m = map_config["size"]

    # Convert to scalars to avoid numpy array propagation
    tx_x = float(tx_x)
    tx_y = float(tx_y)
    tx_z = float(tx_z)
    center_x = float(center_x)
    center_y = float(center_y)
    center_z = float(center_z)
    width_m = float(width_m)
    height_m = float(height_m)

    # Compute distances to key map locations
    # Center of map (ground level)
    dist_to_center = np.sqrt(
        (center_x - tx_x) ** 2 + (center_y - tx_y) ** 2 + (center_z - tx_z) ** 2
    )

    # Corner of map (furthest point)
    corner_x = center_x + width_m / 2
    corner_y = center_y + height_m / 2
    dist_to_corner = np.sqrt(
        (corner_x - tx_x) ** 2 + (corner_y - tx_y) ** 2 + (center_z - tx_z) ** 2
    )

    # Free space path loss calculation
    freq_MHz = frequency_GHz * 1000
    fspl_1m = 20 * np.log10(freq_MHz) + 32.44

    # Path loss at different distances
    pl_center = fspl_1m + 10 * path_loss_exponent * np.log10(max(dist_to_center, 1.0))
    pl_corner = fspl_1m + 10 * path_loss_exponent * np.log10(max(dist_to_corner, 1.0))

    # Antenna pattern approximation (simplified)
    # Main lobe: full gain
    # At 3dB beamwidth edge: gain - 3dB
    # Sidelobes: typically 15-20 dB below main lobe
    sidelobe_attenuation = 18.0  # dB below main lobe (typical for directive antenna)

    # EIRP (Effective Isotropic Radiated Power)
    eirp_mainlobe = tx_power_dBm + antenna_gain_dBi
    eirp_sidelobe = tx_power_dBm + antenna_gain_dBi - sidelobe_attenuation

    # Received power = EIRP - Path Loss
    # We compute power at different antenna directions and distances
    # to provide reasonable targets for different coverage goals

    # Peak: center of map, main lobe (best case)
    peak_power_dbm = eirp_mainlobe - pl_center

    # Main lobe at edge of map (far but well-covered)
    mainlobe_edge_power_dbm = eirp_mainlobe - pl_corner

    # Sidelobe at edge (weakest achievable - use for "low" coverage)
    sidelobe_edge_power_dbm = eirp_sidelobe - pl_corner

    # Minimum achievable (sidelobe at furthest point)
    min_power_dbm = eirp_sidelobe - pl_corner

    return {
        "peak_power_dbm": peak_power_dbm,  # ~-52 dBm (TX at 45.5, -33.7, 34m, map centered at 45.5, -33.7, 0)
        "min_power_dbm": min_power_dbm,  # ~-96 dBm (sidelobe, corner)
        "mainlobe_power_dbm": mainlobe_edge_power_dbm,  # ~-78 dBm (main lobe, corner)
        "sidelobe_power_dbm": sidelobe_edge_power_dbm,  # ~-96 dBm (sidelobe, corner)
        "center_distance_m": dist_to_center,
        "corner_distance_m": dist_to_corner,
        "path_loss_center_dB": pl_center,
        "path_loss_corner_dB": pl_corner,
    }


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
        # 1. Set up outer area (map size)
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
    zone_stats=None,
    qrand=None,
    # Making the number of points configurable from now on
    num_points=20,
):

    width_m, height_m = map_config["size"]
    cell_w, cell_h = map_config["cell_size"]

    n_x = int(width_m / cell_w)
    n_y = int(height_m / cell_h)

    center_x, center_y, ground_z = map_config["center"]
    
    from shapely.geometry import Polygon
    from shapely import contains_xy

    # Quasi-random sampling in continuous space within the zone
    if zone_stats is not None and qrand is not None:
        zone_params = zone_stats.get("zone_params")
        zone_center = zone_params.get("center", zone_stats["look_at_xyz"][:2])
        zone_width = zone_params["width"]
        zone_height = zone_params["height"]

        # Check if this is a polygon zone (has vertices)
        if "vertices" in zone_params:
            # Polygon sampling with rejection method
            zone_polygon = Polygon(zone_params["vertices"])

            # Oversample to account for rejection (2x should be enough for most polygons)
            oversample_factor = 2
            samples_needed = num_points
            sampled_points_list = []

            while len(sampled_points_list) < num_points:
                # Draw from bounding box
                unit_points = np.array(qrand.random(samples_needed))
                unit_points = np.clip(unit_points, 0.0, 1.0)

                # Scale to bounding box
                min_bound_x = zone_center[0] - zone_width / 2
                min_bound_y = zone_center[1] - zone_height / 2
                scaled_x = min_bound_x + unit_points[:, 0] * zone_width
                scaled_y = min_bound_y + unit_points[:, 1] * zone_height

                # Keep only points inside polygon
                inside_mask = contains_xy(zone_polygon, scaled_x, scaled_y)
                valid_points = np.column_stack([scaled_x[inside_mask], scaled_y[inside_mask]])
                sampled_points_list.append(valid_points)

                # Update how many more we need
                samples_needed = num_points - sum(len(pts) for pts in sampled_points_list)

            # Combine and trim to exact count
            sampled_points = np.vstack(sampled_points_list)[:num_points]
        else:
            # Box sampling (original logic)
            unit_points = np.array(qrand.random(num_points))
            unit_points = np.clip(unit_points, 0.0, 1.0)

            min_bound_x = zone_center[0] - zone_width / 2
            min_bound_y = zone_center[1] - zone_height / 2

            scaled_x = unit_points[:, 0] * zone_width
            scaled_y = unit_points[:, 1] * zone_height

            trans_x = min_bound_x + scaled_x
            trans_y = min_bound_y + scaled_y

            sampled_points = np.column_stack([trans_x, trans_y])

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
            sampled_points = all_points
        else:
            mask_flat = zone_mask.flatten()
            sampled_points = all_points[mask_flat == 1.0]

    # Filter out building locations if requested
    if exclude_buildings and scene_xml_path is not None:
        from scene_parser import extract_building_info
        from shapely.geometry import Polygon, Point

        # Get building information
        building_info = extract_building_info(scene_xml_path, verbose=False)

        # Create list of building polygons
        building_polygons = []
        for building_id, info in building_info.items():
            vertices_3d = info["vertices"]
            vertices_2d = [(v[0], v[1]) for v in vertices_3d]
            try:
                building_polygon = Polygon(vertices_2d)
                if building_polygon.is_valid:
                    building_polygons.append(building_polygon)
            except:
                pass

        # Filter out points inside buildings
        if building_polygons:
            filtered_points = []
            num_excluded = 0
            for point_2d in sampled_points:
                point = Point(point_2d[0], point_2d[1])
                # Check if point is inside any building
                inside_building = False
                for poly in building_polygons:
                    if poly.contains(point):
                        inside_building = True
                        num_excluded += 1
                        break
                if not inside_building:
                    filtered_points.append(point_2d)

            if len(filtered_points) == 0:
                raise ValueError(
                    f"All {len(sampled_points)} sampled points are inside buildings! "
                    "Try increasing num_samples or adjusting the zone mask."
                )

            sampled_points = np.array(filtered_points)

    # Add z-coordinate
    z_coords = np.full((len(sampled_points), 1), ground_z)
    sampled_points_3d = np.hstack([sampled_points, z_coords])

    return sampled_points_3d


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

    for config_name, angles in [
        ("Naive Baseline", naive_angles),
        ("Optimized", optimized_angles),
    ]:
        print(f"Computing RadioMap for {config_name}...")

        # Set antenna orientation using angles
        azimuth_deg, elevation_deg = angles[0], angles[1]
        yaw_rad, pitch_rad = azimuth_elevation_to_yaw_pitch(azimuth_deg, elevation_deg)
        tx.orientation = mi.Point3f(float(yaw_rad), float(pitch_rad), 0.0)

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
            diffuse_reflection=False,
            diffraction=True,
            edge_diffraction=True,
            refraction=False,
            stop_threshold=None,
        )

        # Extract signal strength
        rss_watts = rm.rss.numpy()[0, :, :]
        # Commenting this out to test if the linear average is improved
        # signal_strength_dBm = 10.0 * np.log10(rss_watts + 1e-30) + 30.0

        # Extract power values in zone only
        zone_power = rss_watts[zone_mask == 1.0]

        # Filter out dead zones (values below -200 dBm are likely numerical artifacts)
        # Dead zones occur when PathSolver finds no propagation paths
        # DEAD_ZONE_THRESHOLD = -200.0  # dBm
        DEAD_ZONE_THRESHOLD = 0  # 0 watts
        live_zone_power = zone_power[zone_power > DEAD_ZONE_THRESHOLD]

        # Compute statistics on live points only (exclude dead zones)
        # This gives more meaningful metrics for coverage quality
        if len(live_zone_power) > 0:
            mean_val = np.mean(live_zone_power)
            median_val = np.median(live_zone_power)
            std_val = np.std(live_zone_power)
            min_val = np.min(live_zone_power)
            max_val = np.max(live_zone_power)
            p10_val = np.percentile(live_zone_power, 10)
            p90_val = np.percentile(live_zone_power, 90)
            coverage_fraction = len(live_zone_power) / len(zone_power)
        else:
            # All points are dead zones - use raw values
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

    # Determine data range for better visualization
    all_power = np.concatenate(
        [
            results["Naive Baseline"]["power_values"],
            results["Optimized"]["power_values"],
        ]
    )

    # Filter out dead zones for range calculation
    # For Watts: dead zones are near 0, valid power is > 1e-30
    live_power = all_power[all_power > 1e-30]
    if len(live_power) > 0:
        # Use full range (min to max) for completely inclusive binning
        data_min = np.min(live_power)
        data_max = np.max(live_power)
    else:
        data_min, data_max = 1e-15, 1e-10  # Default range in Watts

    # Plot 1: Histograms (PDF)
    ax = axes[0, 0]
    # Use logarithmic binning for better spread across orders of magnitude
    # This creates bins that are evenly spaced in log-space
    if data_min > 0 and data_max > data_min:
        # Extend range slightly beyond data to avoid edge effects
        bin_min = data_min * 0.5
        bin_max = data_max * 2.0
        bins = np.logspace(np.log10(bin_min), np.log10(bin_max), 150)
    else:
        # Fallback to linear bins if log doesn't work
        bins = np.linspace(data_min, data_max, 150)
    ax.hist(
        results["Naive Baseline"]["power_values"],
        bins=bins,
        alpha=0.6,
        label="Naive Baseline",
        color="orange",
        density=True,
    )
    ax.hist(
        results["Optimized"]["power_values"],
        bins=bins,
        alpha=0.6,
        label="Optimized",
        color="green",
        density=True,
    )
    ax.axvline(
        results["Naive Baseline"]["mean"],
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Naive Mean: {results['Naive Baseline']['mean']:.2e} W",
    )
    ax.axvline(
        results["Optimized"]["mean"],
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Optimized Mean: {results['Optimized']['mean']:.2e} W",
    )
    ax.set_xlabel("Signal Strength (Watts)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Power Distribution in Coverage Zone (PDF)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(data_min * 0.5, data_max * 2.0)  # Wider margin for better visibility
    ax.set_xscale("log")  # Use log scale for x-axis to show wide range

    # Plot 2: CDFs
    ax = axes[0, 1]
    for config_name in ["Naive Baseline", "Optimized"]:
        power = results[config_name]["power_values"]
        sorted_power = np.sort(power)
        cdf = np.arange(1, len(sorted_power) + 1) / len(sorted_power)
        color = "orange" if config_name == "Naive Baseline" else "green"
        ax.plot(sorted_power, cdf, label=config_name, color=color, linewidth=2)

        # Mark median
        median = results[config_name]["median"]
        ax.axvline(
            median,
            color=color,
            linestyle="--",
            alpha=0.5,
            label=f"{config_name} Median: {median:.2e} W",
        )

    ax.set_xlabel("Signal Strength (Watts)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution Function (CDF)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Use adaptive x-axis range
    ax.set_xlim(data_min * 0.5, data_max * 2.0)  # Wider margin for better visibility
    ax.set_xscale("log")  # Use log scale for x-axis to show wide range

    # Plot 3: Box plot comparison
    ax = axes[1, 0]
    data_to_plot = [
        results["Naive Baseline"]["power_values"],
        results["Optimized"]["power_values"],
    ]
    bp = ax.boxplot(
        data_to_plot,
        labels=["Naive Baseline", "Optimized"],
        patch_artist=True,
        showmeans=True,
    )
    bp["boxes"][0].set_facecolor("orange")
    bp["boxes"][1].set_facecolor("green")
    ax.set_ylabel("Signal Strength (Watts)")
    ax.set_title("Power Distribution Comparison (Box Plot)")
    ax.grid(True, alpha=0.3, axis="y")

    # Add improvement annotation
    # Calculate improvement percentage for Watts
    improvement_pct_mean = (
        (improvement_mean / results["Naive Baseline"]["mean"]) * 100
        if results["Naive Baseline"]["mean"] != 0
        else 0
    )
    improvement_pct_median = (
        (improvement_median / results["Naive Baseline"]["median"]) * 100
        if results["Naive Baseline"]["median"] != 0
        else 0
    )

    ax.text(
        1.5,
        results["Optimized"]["mean"] * 1.1,  # 10% above optimized mean
        f"Improvement:\nMean: {improvement_pct_mean:+.1f}%\nMedian: {improvement_pct_median:+.1f}%",
        bbox=dict(
            boxstyle="round",
            facecolor="lightgreen" if improvement_pct_mean > 0 else "lightcoral",
            alpha=0.8,
        ),
        fontsize=10,
        ha="center",
    )

    # Plot 4: Statistics table
    ax = axes[1, 1]
    ax.axis("off")

    # Helper function to convert Watts to dBm
    def watts_to_dbm(watts):
        return 10.0 * np.log10(watts + 1e-30) + 30.0

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
    zone_stats=None,
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
    tx_x = float(dr.detach(tx.position[0])[0])
    tx_y = float(dr.detach(tx.position[1])[0])
    tx_z = float(dr.detach(tx.position[2])[0])
    tx_position = [tx_x, tx_y, tx_z]

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

    # Handle TX placement based on mode
    if tx_placement_mode == "skip":
        # Don't move the TX - use its current position
        # This is useful when TX was already placed correctly before calling optimization
        x_start_position = float(dr.detach(tx.position[0])[0])
        y_start_position = float(dr.detach(tx.position[1])[0])
        if verbose:
            print(f"TX placement mode: skip (using current position)")
            print(
                f"  Current TX position: ({x_start_position:.2f}, {y_start_position:.2f}, {tx_height:.2f})"
            )
    else:
        # Initialize TxPlacement for modes that need to move the TX
        tx_placement = TxPlacement(scene, tx_name, scene_xml_path, building_id)

        # Sets the initial location based on mode
        if tx_placement_mode == "center":
            tx_placement.set_rooftop_center()
            x_start_position = tx_placement.building["center"][0]
            y_start_position = tx_placement.building["center"][1]
        elif tx_placement_mode == "fixed":
            x_start_position = Tx_start_pos[0]
            y_start_position = Tx_start_pos[1]
            z_pos = tx_placement.building["z_height"]
            tx.position = mi.Point3f(x_start_position, y_start_position, z_pos)
        else:
            raise ValueError(
                f"Unknown tx_placement_mode: {tx_placement_mode}. Must be 'skip', 'center', 'fixed', or 'line'"
            )

    if verbose:
        print(f"TX height: {tx_height:.1f}m")
        print(
            f"Boresight Z constraint: must be < {tx_height:.1f}m (no pointing upward)\n"
        )

    # Select LDS from available list
    if lds == "Sobol":
        qrand = scipy.stats.qmc.Sobol(d=2, scramble=True, seed=None)
    elif lds == "Halton":
        qrand = scipy.stats.qmc.Halton(d=2, scramble=True, seed=None)
    elif lds == "Latin":
        qrand = scipy.stats.qmc.LatinHypercube(d=2, scramble=True, seed=None)
    elif lds == "Uniform":
        qrand = None
    else:
        print("This LDS is not supported. Using Sobol as default")
        qrand = scipy.stats.qmc.Sobol(d=2, scramble=True, seed=None)

    # Sample Grid Points using Quasi-Monte Carlo (Sobol sequence)
    sample_points = sample_grid_points(
        map_config,
        scene_xml_path=scene_xml_path,
        exclude_buildings=True,
        zone_mask=zone_mask,
        zone_stats=zone_stats,
        qrand=qrand,
        num_points=num_sample_points
    )

    # Print to see the number of sample points after sampling
    if verbose:
        print(f"Actual sample points after building exclusion: {num_sample_points}")

    # Visualize the receiver placement to ensure everything looks correct
    fig = visualize_receiver_placement(
        sample_points,
        zone_mask,
        map_config,
        tx_position=tx_position,
        scene_xml_path="../scene/scene.xml",
        title="Receiver Sampling Visualization",
        figsize=(14, 10),
    )

    # Display the visualization
    import matplotlib.pyplot as plt
    plt.show()

    # Extract zone mask values at sampled points
    width_m, height_m = map_config["size"]
    cell_w, cell_h = map_config["cell_size"]
    n_x = int(width_m / cell_w)
    n_y = int(height_m / cell_h)
    center_x, center_y, _ = map_config["center"]

    # Map sample points to grid indices to get mask values (1.0 = in zone, 0.0 = out of zone)
    mask_values = []
    for point in sample_points:
        x, y = point[0], point[1]
        # Convert to grid indices
        i = int((x - (center_x - width_m / 2)) / cell_w)
        j = int((y - (center_y - height_m / 2)) / cell_h)
        i = np.clip(i, 0, n_x - 1)
        j = np.clip(j, 0, n_y - 1)
        mask_values.append(zone_mask[j, i])

    mask_values = np.array(mask_values, dtype=np.float32)
    num_in_zone = int(np.sum(mask_values))
    num_out_zone = len(mask_values) - num_in_zone

    if verbose:
        print(f"Sample points: {num_in_zone} inside zone, {num_out_zone} outside zone")
        print(
            f"Zone coverage: {100.0 * num_in_zone / len(mask_values):.1f}% of samples"
        )

    # CRITICAL: Remove ALL existing receivers from the scene first
    # This ensures paths.a indexing matches our optimization receivers exactly
    existing_receivers = list(scene.receivers.keys())
    if verbose and existing_receivers:
        print(
            f"Removing {len(existing_receivers)} existing receiver(s): {existing_receivers}"
        )
    for rx_name in existing_receivers:
        scene.remove(rx_name)

    # Add receivers to scene
    rx_names = []
    for idx, position in enumerate(sample_points):
        rx_name = f"opt_rx_{idx}"
        rx_names.append(rx_name)
        # Convert to float32 to avoid type errors with Mitsuba
        position_f32 = [float(position[0]), float(position[1]), float(position[2])]
        rx = Receiver(name=rx_name, position=position_f32)
        scene.add(rx)

    # Create PathSolver
    p_solver = PathSolver()
    p_solver.loop_mode = "evaluated"  # Required for gradient computation

    # Define differentiable loss function using @dr.wrap
    @dr.wrap(source="torch", target="drjit")
    def compute_loss(azimuth_deg, elevation_deg, qrand_op, num_sample_points):
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
        # CRITICAL: Enable gradients for each input parameter
        # The @dr.wrap decorator converts PyTorch 0-D tensors → DrJit Float scalars
        # We need to access .array to get the actual DrJit scalars
        dr.enable_grad(azimuth_deg.array)
        dr.enable_grad(elevation_deg.array)

        # Convert degrees to radians for angle conversion
        # Use DrJit's pi constant for differentiability
        deg_to_rad = dr.auto.ad.Float(np.pi / 180.0)
        dr.disable_grad(deg_to_rad)  # Conversion factor is constant

        # Remember to bring this back to .array
        azimuth_rad = azimuth_deg * deg_to_rad
        elevation_rad = elevation_deg.array * deg_to_rad

        # Convert azimuth/elevation to yaw/pitch (roll = 0)
        # yaw = azimuth (rotation around Z-axis)
        # pitch = -elevation (negative because positive pitch tilts up in Mitsuba)
        yaw_rad = azimuth_rad
        pitch_rad = -elevation_rad
        roll_rad = dr.auto.ad.Float(0.0)
        dr.disable_grad(roll_rad)  # Roll is always 0 for antenna pointing

        # Adding jitter to smooth out the "Needle" effect and avoid overfitting to sample points
        # Use numpy for random jitter since this happens outside the differentiable path
        jitter_std_deg = 0.3  # Small jitter in degrees
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
        tx = scene.get(tx_name)
        tx.orientation = mi.Point3f(yaw_rad_jittered, pitch_rad_jittered, roll_rad)

        # Generate new sample points using Sobol sequence
        new_sample_points = sample_grid_points(
            map_config,
            scene_xml_path=scene_xml_path,
            exclude_buildings=True,
            zone_mask=zone_mask,
            zone_stats=zone_stats,
            qrand=qrand_op,
            num_points=num_sample_points
        )

        # Store for visualization outside the loss function
        compute_loss.current_sample_points = new_sample_points

        # Remove existing receivers from scene
        for rx_name in rx_names:
            scene.remove(rx_name)

        # Add new receivers at sampled positions
        rx_names.clear()
        for idx, position in enumerate(new_sample_points):
            rx_name = f"opt_rx_{idx}"
            rx_names.append(rx_name)
            position_f32 = [float(position[0]), float(position[1]), float(position[2])]
            rx = Receiver(name=rx_name, position=position_f32)
            scene.add(rx)

        # Run path solver with updated receivers and antenna orientation
        paths = p_solver(
            scene,
            los=True,
            refraction=False,
            specular_reflection=True,
            diffuse_reflection=False,
            diffraction=True,
            edge_diffraction=True,
        )

        # Extract channel coefficients
        h_real, h_imag = paths.a

        # GRADIENT DIAGNOSTICS: Check if gradients are tracking through the pipeline
        if verbose and not hasattr(compute_loss, "_gradient_check_printed"):
            compute_loss._gradient_check_printed = True
            print(
                f"  [GRADIENT CHECK] yaw_rad grad enabled: {dr.grad_enabled(yaw_rad)}"
            )
            print(
                f"  [GRADIENT CHECK] pitch_rad grad enabled: {dr.grad_enabled(pitch_rad)}"
            )
            print(
                f"  [GRADIENT CHECK] elevation_deg grad enabled: {dr.grad_enabled(elevation_deg.array)}"
            )
            print(f"  [GRADIENT CHECK] h_real grad enabled: {dr.grad_enabled(h_real)}")
            print(f"  [GRADIENT CHECK] h_imag grad enabled: {dr.grad_enabled(h_imag)}")
            print(f"  [DIAGNOSTIC] h_real shape: {dr.shape(h_real)}")

        # DEBUG: Check if any paths were found (only print on first call)
        # We use a simple flag via a mutable default to track first call
        if not hasattr(compute_loss, "_first_call_done"):
            compute_loss._first_call_done = False

        if verbose and not compute_loss._first_call_done:
            compute_loss._first_call_done = True
            # Compute total power across all receivers as a quick check
            total_power = dr.sum(dr.sum(cpx_abs_square((h_real, h_imag))))
            print(f"  [DEBUG] Total path power (linear): {total_power}")
            if total_power < 1e-25:
                print(
                    f"  [WARNING] Very low or zero path power - PathSolver may not be finding paths!"
                )

        # Extract path coefficients
        h_real, h_imag = paths.a

        # Compute incoherent sum (Raw Channel Gain |h|^2)
        power_relative = dr.sum(
            dr.sum(cpx_abs_square((h_real, h_imag)), axis=-1), axis=-1
        )
        power_relative = dr.sum(power_relative, axis=-1)

        # Apply mask to solve for average
        mask_target = dr.auto.ad.Float(mask_values)
        dr.disable_grad(mask_target)
        valid_points = dr.maximum(dr.sum(mask_target), 1.0)

        # Isolate power in the target zone
        target_power = power_relative * mask_target
        valid_points = dr.maximum(dr.sum(mask_target), 1.0)
        epsilon = 1e-16

        # "Mean of Logs"
        # Punishes shadows. Drives the "Median" and "10th Percentile" up.
        loss_coverage = dr.sum(dr.log(target_power + epsilon)) / valid_points

        # "Log of Means"
        # Attempts to improve the overall power in the reigon
        total_watts = dr.sum(target_power)
        loss_peak = dr.log((total_watts / valid_points) + epsilon)

        # Greediness Factor
        # Set to split the goal evenly
        #alpha = 1.0
        alpha = 0.0

        # Alpha should be biased closed to 1.0 since there is a magnitude difference between the objectives
        loss = -(alpha * loss_peak + (1.0 - alpha) * loss_coverage)

        return loss

    # Calculate the initial azimuth and elevation angles based on the position of the transmitter + center of the zone
    initial_azimuth, initial_elevation = compute_initial_angles_from_position(
        [x_start_position, y_start_position, tx_height],
        zone_stats["look_at_xyz"],
        verbose=False,
    )
    print(f"initial azimuth (after function): {initial_azimuth}")
    print(f"initial elevation (after function): {initial_elevation}")

    # PyTorch parameters: azimuth and elevation angles (in degrees)
    # Using 0-D tensors (scalars) - this is the ONLY pattern that works with @dr.wrap
    # 1-D tensors lose gradients during indexing
    azimuth = torch.tensor(
        initial_azimuth, device="cuda", dtype=torch.float32, requires_grad=True
    )
    elevation = torch.tensor(
        initial_elevation, device="cuda", dtype=torch.float32, requires_grad=True
    )

    print(f"\n{'='*70}")
    print("PYTORCH TENSOR INITIALIZATION")
    print(f"{'='*70}")
    print(f"PyTorch azimuth tensor initialized to: {azimuth.item():.2f}°")
    print(f"PyTorch elevation tensor initialized to: {elevation.item():.2f}°")
    print(f"These values will be used as starting point for optimization")
    print(f"{'='*70}\n")

    # Optimizer: Adam shows the best performance
    # Optimize azimuth and elevation angles directly (2 parameters instead of 3)
    # Adding a heavier beta term for momentum through noisy measurements
    optimizer = torch.optim.Adam([azimuth, elevation], lr=learning_rate, betas=(0.98, 0.999))

    # Learning rate scheduler: required to jump out of local minima for difficult loss surfaces...
    use_scheduler = num_iterations >= 50

    if use_scheduler:
        # Use exponential decay for smoother convergence after grid search
        # This is more stable than ReduceLROnPlateau for noisy loss landscapes
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95  # Decay by 5% each iteration (smooth reduction)
        )
        if verbose:
            print(f"Learning rate scheduler enabled (exponential decay, gamma=0.95)")
            print(f"  Initial LR: {learning_rate:.4f}")
            print(f"  LR after 50 iters: {learning_rate * (0.95**50):.6f}")
            print(f"  LR after 100 iters: {learning_rate * (0.95**100):.6f}")

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

        # Using an independent (random) set of Sobol points using different seeds
        # qrand_run = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=iteration)
        # Could build in different sampling sequences for customizable performance
        loss = compute_loss(azimuth, elevation, qrand, num_sample_points)
        optimizer.zero_grad()
        loss.backward()

        # Visualize the Sobol sampling pattern (only save every N iterations to reduce file count)
        if save_radiomap_frames and (iteration % frame_save_interval == 0):
            current_points = getattr(
                compute_loss, "current_sample_points", sample_points
            )
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
                f"  Gradients: dAz={grad_azimuth_val:+.3e}°, dEl={grad_elevation_val:+.3e}° (norm={grad_norm:.3e})"
            )
            # Convert loss back to mean power in dBm
            # Since loss = -mean(dBm), we just negate it to get mean(dBm)
            mean_power_in_zone_dbm = -loss.item()
            print(
                f"  Loss: {loss.item():.4f}, Mean Power in Zone: {mean_power_in_zone_dbm:.2f} dBm"
            )

        torch.nn.utils.clip_grad_norm_(elevation.grad, max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(elevation.grad, max_norm=0.5)

        # Update
        optimizer.step()

        # Update learning rate if scheduler is enabled
        if use_scheduler:
            scheduler.step()

        # Apply constraints on angles
        with torch.no_grad():
            # Azimuth: clamp to [0, 360) degrees
            azimuth.clamp_(min=0.0, max=360.0)
            # Wrap azimuth around if needed (360° → 0°)
            if azimuth.item() >= 360.0:
                azimuth.fill_(azimuth.item() % 360.0)

        # Track
        loss_history.append(loss.item())
        angle_history.append([azimuth.item(), elevation.item()])

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

    # Compute final coverage statistics
    coverage_stats = {
        "num_samples_in_zone": num_in_zone,
        "num_samples_total": len(mask_values),
        "zone_coverage_fraction": num_in_zone / len(mask_values),
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
        print(
            f"Zone samples: {num_in_zone}/{len(mask_values)} ({100.0*num_in_zone/len(mask_values):.1f}%)"
        )
        print(f"Total time: {elapsed_time:.1f}s")
        print(f"Time per iteration: {elapsed_time/num_iterations:.2f}s")
        print(f"{'='*70}\n")

    # Return angles instead of Cartesian coordinates
    best_angles = [best_azimuth_final, best_elevation_final]
    return best_angles, loss_history, angle_history, gradient_history, coverage_stats

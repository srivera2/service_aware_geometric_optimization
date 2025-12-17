# ============================================
# BORESIGHT OPTIMIZATION USING PATHSOLVER + AD
# ============================================
# This approach uses PathSolver (which supports AD) instead of RadioMapSolver
# We sample grid points and use PathSolver to compute paths to those points
# This is similar to rm_diff.ipynb but optimizes boresight instead of TX position

import torch
import numpy as np
import drjit as dr
from drjit.auto import Float, Array3f, UInt
import mitsuba as mi
from sionna.rt import PathSolver, Receiver, cpx_abs_square
import time
from shapely.geometry import Point, Polygon
from tx_placement import TxPlacement
from angle_utils import (
    azimuth_elevation_to_yaw_pitch,
    yaw_pitch_to_azimuth_elevation,
    compute_initial_angles_from_position,
    normalize_azimuth,
    clamp_elevation,
)


def create_optimization_gif(frame_dir, output_path="optimization.gif", duration=200, loop=0,
                           sector_angles=None, tx_position=None, map_config=None):
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
        imageio.mimsave(output_path, images, duration=duration/1000.0, loop=loop)

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
            fig, ax = plt.subplots(figsize=(img.width/100, img.height/100), dpi=100)
            ax.imshow(img)
            ax.axis('off')

            # Calculate sector overlay position
            # Map tx_position to image coordinates
            center_x, center_y, _ = map_config['center']
            width_m, height_m = map_config['size']

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
                angle_start = sector['angle_start']
                angle_end = sector['angle_end']
                color = sector.get('color', 'red')
                alpha = sector.get('alpha', 0.2)

                # Convert angles: matplotlib uses degrees counter-clockwise from East
                # Our convention: 0° = East, counter-clockwise
                # matplotlib Wedge: theta1 is start, theta2 is end (counter-clockwise from East)

                # Handle wraparound case (e.g., sector from 350° to 10°)
                if angle_end < angle_start:
                    # Draw two wedges: one from angle_start to 360, another from 0 to angle_end
                    wedge1 = Wedge((tx_x, tx_y), max_radius, angle_start, 360,
                                  facecolor=color, edgecolor=color, alpha=alpha, linewidth=2)
                    wedge2 = Wedge((tx_x, tx_y), max_radius, 0, angle_end,
                                  facecolor=color, edgecolor=color, alpha=alpha, linewidth=2)
                    ax.add_patch(wedge1)
                    ax.add_patch(wedge2)

                    # Draw boundary lines
                    for angle_deg in [angle_start, angle_end]:
                        angle_rad = np.radians(angle_deg)
                        dx = max_radius * np.cos(angle_rad)
                        dy = max_radius * np.sin(angle_rad)
                        ax.plot([tx_x, tx_x + dx], [tx_y, tx_y + dy],
                               color=color, linewidth=2, alpha=0.8)
                else:
                    # Normal sector
                    wedge = Wedge((tx_x, tx_y), max_radius, angle_start, angle_end,
                                 facecolor=color, edgecolor=color, alpha=alpha, linewidth=2)
                    ax.add_patch(wedge)

                    # Draw boundary lines to make sector more visible
                    for angle_deg in [angle_start, angle_end]:
                        angle_rad = np.radians(angle_deg)
                        dx = max_radius * np.cos(angle_rad)
                        dy = max_radius * np.sin(angle_rad)
                        ax.plot([tx_x, tx_x + dx], [tx_y, tx_y + dy],
                               color=color, linewidth=2, alpha=0.8)

            # Set the correct limits to match the original image
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
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
        optimize=False
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
    zone_type="angular_sector",
    origin_point=None,
    zone_params=None,
    target_height=1.5,
    scene_xml_path=None,
    exclude_buildings=True,
):
    """
    Create a binary coverage zone mask for simplified optimization.

    This function replaces create_target_radiomap() by using a binary mask approach:
    - Mask value = 1.0 inside the coverage zone (optimize for high power here)
    - Mask value = 0.0 outside the zone (don't care about power here)

    This simplifies the optimization problem - instead of matching specific power
    levels at each location, we just maximize coverage uniformly within the zone.
    The goal is to lift the median coverage level in the selected area.

    Parameters:
    -----------
    map_config : dict
        Map configuration containing:
        - 'size': [width, height] in meters
        - 'cell_size': (cell_w, cell_h) grid resolution
        - 'center': [x, y, z] center of the map

    zone_type : str
        Type of coverage zone:
        - 'angular_sector': Wedge-shaped sector from TX (azimuth-based)
        - 'box': Rectangular region

    origin_point : tuple or list
        (x, y, z) position of the transmitter in meters.
        Required for 'angular_sector' (defines the sector origin).
        Optional for 'box' (if provided, used for validation).

    zone_params : dict
        Zone-specific parameters:

        For 'angular_sector':
            - 'angle_start': Start angle in degrees (0° = East, counter-clockwise)
            - 'angle_end': End angle in degrees
            - 'radius': Maximum distance from TX in meters

        For 'box':
            - 'center': (x, y) center of box in meters
            - 'width': Box width in meters
            - 'height': Box height in meters

    target_height : float
        Z-coordinate for the returned look_at position (typically UE height: 1.5m).
        This defines the vertical aim point for a naive baseline boresight.

    scene_xml_path : str, optional
        Path to scene XML file. Required if exclude_buildings=True.

    exclude_buildings : bool
        If True, exclude building footprints from the coverage zone mask.
        This prevents the optimizer from targeting areas inside buildings.

    Returns:
    --------
    mask : np.ndarray
        2D binary mask, shape (n_y, n_x), dtype float32
        - 1.0 = inside coverage zone (target for optimization)
        - 0.0 = outside coverage zone (ignored)

    look_at_pos : np.ndarray
        [x, y, z] coordinates of the zone's geometric center.
        Use this for a naive baseline: tx.look_at(look_at_pos)

    zone_stats : dict
        Statistics about the zone:
        - 'num_cells': Number of grid cells in the zone (mask sum)
        - 'coverage_fraction': Fraction of map covered by zone
        - 'centroid_xy': [x, y] geometric center of the zone
        - 'look_at_xyz': [x, y, z] same as look_at_pos

    Raises:
    -------
    ValueError
        - If zone_type is invalid
        - If required parameters are missing
        - If zone is empty (no cells inside)

    Example:
    --------
    >>> # Angular sector: 45° wide cone pointing northeast, 100m range
    >>> mask, look_at, stats = create_zone_mask(
    ...     map_config={'center': [0, 0, 0], 'size': [200, 200], 'cell_size': (5, 5)},
    ...     zone_type='angular_sector',
    ...     origin_point=[0, 0, 20],  # TX at 20m height
    ...     zone_params={'angle_start': 22.5, 'angle_end': 67.5, 'radius': 100},
    ...     target_height=1.5
    ... )
    >>> print(f"Coverage zone has {stats['num_cells']} cells")

    >>> # Rectangular box: 50m x 30m area
    >>> mask, look_at, stats = create_zone_mask(
    ...     map_config={'center': [0, 0, 0], 'size': [200, 200], 'cell_size': (5, 5)},
    ...     zone_type='box',
    ...     zone_params={'center': (50, 30), 'width': 50, 'height': 30},
    ...     target_height=1.5
    ... )
    """

    # Validate inputs
    if zone_type not in ["angular_sector", "box"]:
        raise ValueError(
            f"Invalid zone_type '{zone_type}'. Must be 'angular_sector' or 'box'."
        )

    if zone_params is None:
        raise ValueError("zone_params must be provided")

    # 1. Setup Grid
    width_m, height_m = map_config["size"]
    cell_w, cell_h = map_config["cell_size"]
    center_x, center_y, _ = map_config["center"]

    n_x = int(width_m / cell_w)
    n_y = int(height_m / cell_h)

    # Create coordinate grids
    x = np.linspace(center_x - width_m / 2, center_x + width_m / 2, n_x)
    y = np.linspace(center_y - height_m / 2, center_y + height_m / 2, n_y)
    X, Y = np.meshgrid(x, y)

    # Initialize mask (all zeros = outside zone)
    mask = np.zeros((n_y, n_x), dtype=np.float32)

    # Default look_at position (will be overridden based on zone geometry)
    look_at_pos = np.array([center_x, center_y, target_height], dtype=np.float32)

    # 2. Generate Mask & Calculate Look-At Position
    if zone_type == "angular_sector":
        # Validate required parameters
        if origin_point is None:
            raise ValueError(
                "zone_type='angular_sector' requires 'origin_point' (TX location)."
            )

        tx_x, tx_y, _ = origin_point
        start_deg = zone_params.get("angle_start", 0)
        end_deg = zone_params.get("angle_end", 360)
        radius = zone_params.get("radius", 100.0)

        # Validate sector parameters
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")

        # Calculate angles from TX to each grid point
        # Angle convention: 0° = East (+X), 90° = North (+Y), counter-clockwise
        theta = np.degrees(np.arctan2(Y - tx_y, X - tx_x))
        theta = np.where(theta < 0, theta + 360, theta)  # Normalize to [0, 360)

        # Calculate distances from TX
        dist = np.sqrt((X - tx_x)**2 + (Y - tx_y)**2)

        # Handle wrap-around sectors (e.g., 350° to 10° wraps through 0°)
        if start_deg > end_deg:
            # Sector crosses 0° boundary
            in_wedge = (theta >= start_deg) | (theta <= end_deg)
            # Calculate bisector angle (wraps correctly)
            mid_angle_deg = (start_deg + end_deg + 360) / 2
            if mid_angle_deg >= 360:
                mid_angle_deg -= 360
        else:
            # Normal sector (no wrap-around)
            in_wedge = (theta >= start_deg) & (theta <= end_deg)
            mid_angle_deg = (start_deg + end_deg) / 2

        # Apply distance constraint
        in_sector = in_wedge & (dist <= radius)
        mask[in_sector] = 1.0

        # Calculate Look-At Position
        # Use 75% of radius along bisector for better beam centering
        # (empirically works better than edge or full center)
        target_dist = radius * 0.75

        mid_rad = np.radians(mid_angle_deg)
        target_x = tx_x + target_dist * np.cos(mid_rad)
        target_y = tx_y + target_dist * np.sin(mid_rad)

        look_at_pos = np.array([target_x, target_y, target_height], dtype=np.float32)

    elif zone_type == "box":
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
        mask[in_x & in_y] = 1.0

        # Look-at is simply the center of the box
        look_at_pos = np.array([bx, by, target_height], dtype=np.float32)

    # 3. Exclude building footprints (optional)
    num_excluded_buildings = 0
    if exclude_buildings:
        if scene_xml_path is None:
            raise ValueError("scene_xml_path is required when exclude_buildings=True")

        # Import here to avoid circular dependency
        from scene_parser import extract_building_info
        from shapely.geometry import Polygon, Point

        # Get building information (reuse existing parser!)
        building_info = extract_building_info(scene_xml_path, verbose=False)

        # For each building, create polygon and exclude cells inside it
        for building_id, info in building_info.items():
            # Get building polygon vertices (only X, Y coordinates)
            vertices_3d = info['vertices']
            vertices_2d = [(v[0], v[1]) for v in vertices_3d]

            # Create Shapely polygon for this building
            try:
                building_polygon = Polygon(vertices_2d)

                # Check each grid point that's currently in the zone
                for i in range(n_y):
                    for j in range(n_x):
                        if mask[i, j] > 0:  # Only check cells already in the zone
                            point = Point(X[i, j], Y[i, j])
                            if building_polygon.contains(point):
                                mask[i, j] = 0.0  # Exclude this cell

                num_excluded_buildings += 1
            except Exception as e:
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
    zone_x = X[mask == 1.0]
    zone_y = Y[mask == 1.0]
    centroid_x = np.mean(zone_x)
    centroid_y = np.mean(zone_y)

    zone_stats = {
        'num_cells': num_cells_in_zone,
        'coverage_fraction': num_cells_in_zone / total_cells,
        'centroid_xy': [float(centroid_x), float(centroid_y)],
        'look_at_xyz': look_at_pos.tolist(),
        'buildings_excluded': exclude_buildings,
        'num_excluded_buildings': num_excluded_buildings,
    }

    return mask, look_at_pos, zone_stats


def sample_grid_points(map_config, num_samples=100, seed=None, zone_mask=None):
    """
    Sample points from the radio map grid

    Parameters:
    -----------
    map_config : dict
        Map configuration
    num_samples : int
        Number of points to sample
    seed : int or None
        Random seed for reproducibility. If None, sampling is random.
    zone_mask : np.ndarray, optional
        Binary mask (shape: n_y x n_x) defining coverage zone.
        If provided, samples ONLY from cells where mask == 1.0

    Returns:
    --------
    sampled_points_3d : numpy array
        Nx3 array of (x, y, z) coordinates
    """
    width_m, height_m = map_config["size"]
    cell_w, cell_h = map_config["cell_size"]

    n_x = int(width_m / cell_w)
    n_y = int(height_m / cell_h)

    center_x, center_y, ground_z = map_config["center"]

    # Sample uniformly from grid
    x = np.linspace(center_x - width_m / 2, center_x + width_m / 2, n_x)
    y = np.linspace(center_y - height_m / 2, center_y + height_m / 2, n_y)

    X, Y = np.meshgrid(x, y)

    # Filter by zone mask if provided
    if zone_mask is not None:
        # Get coordinates only where mask == 1.0
        in_zone_mask = zone_mask == 1.0
        zone_x = X[in_zone_mask]
        zone_y = Y[in_zone_mask]
        all_points = np.column_stack([zone_x, zone_y])

        if len(all_points) == 0:
            raise ValueError("Zone mask is empty - no valid sampling points!")

        if len(all_points) < num_samples:
            print(f"Warning: Zone has only {len(all_points)} cells, but {num_samples} samples requested.")
            print(f"         Using all {len(all_points)} available points in zone.")
            sampled_points = all_points
        else:
            # Randomly select num_samples points from zone
            rng = np.random.RandomState(seed) if seed is not None else np.random
            indices = rng.choice(len(all_points), num_samples, replace=False)
            sampled_points = all_points[indices]
    else:
        # Original behavior: sample from entire grid
        all_points = np.column_stack([X.flatten(), Y.flatten()])

        # Randomly select num_samples points with optional seed
        if num_samples < len(all_points):
            rng = np.random.RandomState(seed) if seed is not None else np.random
            indices = rng.choice(len(all_points), num_samples, replace=False)
            sampled_points = all_points[indices]
        else:
            sampled_points = all_points

    # Add z-coordinate
    z_coords = np.full((len(sampled_points), 1), ground_z)
    sampled_points_3d = np.hstack([sampled_points, z_coords])

    return sampled_points_3d


def compare_boresight_performance(
    scene,
    tx_name,
    map_config,
    zone_mask,
    naive_angles,
    optimized_angles,
    stablized_angles,
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

    for config_name, angles in [("Naive Baseline", naive_angles),
                                 ("Optimized", optimized_angles),
                                 ("Stablized Angles", stablized_angles)]:
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
            samples_per_tx=int(1e8),
            cell_size=map_config['cell_size'],
            center=map_config['center'],
            orientation=[0, 0, 0],
            size=map_config['size'],
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=False,
        )

        # Extract signal strength
        rss_watts = rm.rss.numpy()[0, :, :]
        signal_strength_dBm = 10.0 * np.log10(rss_watts + 1e-30) + 30.0

        # Extract power values in zone only
        zone_power = signal_strength_dBm[zone_mask == 1.0]

        # Filter out dead zones (values below -200 dBm are likely numerical artifacts)
        # Dead zones occur when PathSolver finds no propagation paths
        DEAD_ZONE_THRESHOLD = -200.0  # dBm
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
            'power_values': zone_power,  # Keep all values for plotting
            'live_power_values': live_zone_power,  # Only live points
            'radiomap': signal_strength_dBm,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'p10': p10_val,
            'p90': p90_val,
            'coverage_fraction': coverage_fraction,  # Fraction of zone with actual coverage
            'num_live_points': len(live_zone_power),
            'num_total_points': len(zone_power),
        }

    # Calculate improvement (for best and naive)
    improvement_mean = results['Optimized']['mean'] - results['Naive Baseline']['mean']
    improvement_median = results['Optimized']['median'] - results['Naive Baseline']['median']

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Determine data range for better visualization
    all_power = np.concatenate([results['Naive Baseline']['power_values'],
                                 results['Optimized']['power_values'], 
                                 results['Stablized Angles']['power_values']])
    
    # Filter out dead zones for range calculation
    live_power = all_power[all_power > -269]
    if len(live_power) > 0:
        data_min = max(np.percentile(live_power, 1), -200)  # 1st percentile or -200 dBm
        data_max = min(np.percentile(live_power, 99), -40)   # 99th percentile or -40 dBm
    else:
        data_min, data_max = -120, -60

    # Plot 1: Histograms (PDF)
    ax = axes[0, 0]
    # Use adaptive binning based on data range
    bins = np.linspace(data_min, data_max, 60)
    ax.hist(results['Naive Baseline']['power_values'], bins=bins, alpha=0.6,
            label='Naive Baseline', color='orange', density=True)
    ax.hist(results['Optimized']['power_values'], bins=bins, alpha=0.6,
            label='Optimized', color='green', density=True)
    ax.hist(results['Stablized Angles']['power_values'], bins=bins, alpha=0.6,
            label='Stablized', color='white', density=True)
    ax.axvline(results['Naive Baseline']['mean'], color='orange', linestyle='--',
               linewidth=2, label=f"Naive Mean: {results['Naive Baseline']['mean']:.1f} dBm")
    ax.axvline(results['Optimized']['mean'], color='green', linestyle='--',
               linewidth=2, label=f"Optimized Mean: {results['Optimized']['mean']:.1f} dBm")
    ax.axvline(results['Stablized Angles']['mean'], color='white', linestyle='--',
               linewidth=2, label=f"Stablized Mean: {results['Stablized Angles']['mean']:.1f} dBm")
    ax.set_xlabel('Signal Strength (dBm)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Power Distribution in Coverage Zone (PDF)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(data_min - 5, data_max + 5)

    # Plot 2: CDFs
    ax = axes[0, 1]
    for config_name in ['Naive Baseline', 'Optimized']:
        power = results[config_name]['power_values']
        sorted_power = np.sort(power)
        cdf = np.arange(1, len(sorted_power) + 1) / len(sorted_power)
        color = 'orange' if config_name == 'Naive Baseline' else 'green'
        ax.plot(sorted_power, cdf, label=config_name, color=color, linewidth=2)

        # Mark median
        median = results[config_name]['median']
        ax.axvline(median, color=color, linestyle='--', alpha=0.5,
                   label=f"{config_name} Median: {median:.1f} dBm")

    ax.set_xlabel('Signal Strength (dBm)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function (CDF)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Use adaptive x-axis range
    ax.set_xlim(data_min - 5, data_max + 5)

    # Plot 3: Box plot comparison
    ax = axes[1, 0]
    data_to_plot = [results['Naive Baseline']['power_values'],
                    results['Optimized']['power_values']]
    bp = ax.boxplot(data_to_plot, labels=['Naive Baseline', 'Optimized'],
                    patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('orange')
    bp['boxes'][1].set_facecolor('green')
    ax.set_ylabel('Signal Strength (dBm)')
    ax.set_title('Power Distribution Comparison (Box Plot)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add improvement annotation
    ax.text(1.5, results['Optimized']['mean'] + 2,
            f"Improvement:\nMean: +{improvement_mean:.1f} dBm\nMedian: +{improvement_median:.1f} dBm",
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            fontsize=10, ha='center')

    # Plot 4: Statistics table
    ax = axes[1, 1]
    ax.axis('off')

    # Helper function to format improvement values
    def format_improvement(value):
        if value >= 0:
            return f"+{value:.2f}"
        else:
            return f"{value:.2f}"

    stats_data = [
        ['Metric', 'Naive Baseline', 'Optimized', 'Improvement'],
        ['Mean (dBm)', f"{results['Naive Baseline']['mean']:.2f}",
         f"{results['Optimized']['mean']:.2f}", format_improvement(improvement_mean)],
        ['Median (dBm)', f"{results['Naive Baseline']['median']:.2f}",
         f"{results['Optimized']['median']:.2f}", format_improvement(improvement_median)],
        ['Std Dev (dBm)', f"{results['Naive Baseline']['std']:.2f}",
         f"{results['Optimized']['std']:.2f}",
         format_improvement(results['Optimized']['std'] - results['Naive Baseline']['std'])],
        ['Min (dBm)', f"{results['Naive Baseline']['min']:.2f}",
         f"{results['Optimized']['min']:.2f}",
         format_improvement(results['Optimized']['min'] - results['Naive Baseline']['min'])],
        ['Max (dBm)', f"{results['Naive Baseline']['max']:.2f}",
         f"{results['Optimized']['max']:.2f}",
         format_improvement(results['Optimized']['max'] - results['Naive Baseline']['max'])],
        ['10th %ile (dBm)', f"{results['Naive Baseline']['p10']:.2f}",
         f"{results['Optimized']['p10']:.2f}",
         format_improvement(results['Optimized']['p10'] - results['Naive Baseline']['p10'])],
        ['90th %ile (dBm)', f"{results['Naive Baseline']['p90']:.2f}",
         f"{results['Optimized']['p90']:.2f}",
         format_improvement(results['Optimized']['p90'] - results['Naive Baseline']['p90'])],
    ]

    table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color improvement column green if positive
    for i in range(1, len(stats_data)):
        improvement_val = float(stats_data[i][3])
        if improvement_val > 0:
            table[(i, 3)].set_facecolor('#90EE90')
        elif improvement_val < 0:
            table[(i, 3)].set_facecolor('#FFB6C6')

    ax.set_title('Performance Statistics Comparison', fontsize=12, weight='bold', pad=20)

    plt.suptitle(title, fontsize=14, weight='bold')
    plt.tight_layout()

    # Prepare stats dictionary for return
    stats = {
        'naive': results['Naive Baseline'],
        'optimized': results['Optimized'],
        'improvement_mean_dBm': improvement_mean,
        'improvement_median_dBm': improvement_median,
        'improvement_percent': (improvement_mean / abs(results['Naive Baseline']['mean'])) * 100
    }

    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Naive Baseline Mean:  {results['Naive Baseline']['mean']:.2f} dBm")
    print(f"Optimized Mean:       {results['Optimized']['mean']:.2f} dBm")
    print(f"Improvement:          +{improvement_mean:.2f} dBm ({stats['improvement_percent']:.1f}%)")
    print(f"{'='*70}\n")

    return fig, stats


def grid_search_initial_boresight(
    scene,
    tx_name,
    map_config,
    zone_mask,
    num_angular_samples=12,
    num_elevation_samples=5,
    min_elevation_deg=0,
    max_elevation_deg=-45,
    radius_meters=100.0,
    num_sample_points=50,
    seed=42,
    verbose=True,
):
    """
    Perform coarse grid search to find good initial boresight direction.

    This function evaluates multiple boresight directions on a spherical grid
    and returns the one with best coverage in the target zone. Use this to
    initialize gradient-based optimization.

    Parameters:
    -----------
    scene : sionna.rt.Scene
        The scene
    tx_name : str
        Transmitter name
    map_config : dict
        Map configuration
    zone_mask : np.ndarray
        Binary zone mask
    num_angular_samples : int
        Number of azimuth angles to test (default: 12 = every 30°)
    num_elevation_samples : int
        Number of elevation angles to test (default: 5)
    min_elevation_deg : float
        Minimum elevation angle in degrees (0° = horizontal)
    max_elevation_deg : float
        Maximum elevation angle in degrees (negative = pointing down)
    radius_meters : float
        Distance from TX to boresight target point
    num_sample_points : int
        Number of sample points for evaluation (fewer = faster)
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns:
    --------
    best_azimuth : float
        Best azimuth angle in degrees
    best_elevation : float
        Best elevation angle in degrees
    best_mean_power : float
        Mean power (dBm) achieved with best boresight
    grid_results : dict
        Results for all tested directions
    """
    from sionna.rt import PathSolver, cpx_abs_square

    tx = scene.get(tx_name)
    # Extract TX position using the same pattern as elsewhere in the file
    tx_x = float(dr.detach(tx.position[0])[0])
    tx_y = float(dr.detach(tx.position[1])[0])
    tx_z = float(dr.detach(tx.position[2])[0])

    if verbose:
        print(f"\n{'='*70}")
        print("Grid Search for Initial Boresight")
        print(f"{'='*70}")
        print(f"TX position: ({tx_x:.1f}, {tx_y:.1f}, {tx_z:.1f})")
        print(f"Angular samples: {num_angular_samples}")
        print(f"Elevation samples: {num_elevation_samples}")
        print(f"Total directions to test: {num_angular_samples * num_elevation_samples}")
        print(f"{'='*70}\n")

    # Sample receivers in zone
    sample_points = sample_grid_points(map_config, num_sample_points, seed=seed, zone_mask=zone_mask)

    # Get mask values for samples
    width_m, height_m = map_config["size"]
    cell_w, cell_h = map_config["cell_size"]
    n_x = int(width_m / cell_w)
    n_y = int(height_m / cell_h)
    center_x, center_y, _ = map_config["center"]

    mask_values = []
    for point in sample_points:
        x, y = point[0], point[1]
        i = int((x - (center_x - width_m / 2)) / cell_w)
        j = int((y - (center_y - height_m / 2)) / cell_h)
        i = np.clip(i, 0, n_x - 1)
        j = np.clip(j, 0, n_y - 1)
        mask_values.append(zone_mask[j, i])
    mask_values = np.array(mask_values, dtype=np.float32)

    # Add receivers (reuse for all grid points)
    rx_names = []
    for idx, position in enumerate(sample_points):
        rx_name = f"grid_rx_{idx}"
        rx_names.append(rx_name)
        if rx_name in [obj.name for obj in scene.receivers.values()]:
            scene.remove(rx_name)
        position_f32 = [float(position[0]), float(position[1]), float(position[2])]
        rx = Receiver(name=rx_name, position=position_f32)
        scene.add(rx)

    p_solver = PathSolver()

    # Generate grid of directions
    azimuth_angles = np.linspace(0, 360, num_angular_samples, endpoint=False)
    elevation_angles = np.linspace(min_elevation_deg, max_elevation_deg, num_elevation_samples)

    best_mean_power = -np.inf
    best_azimuth = None
    best_elevation = None
    grid_results = []

    total_tests = len(azimuth_angles) * len(elevation_angles)
    test_idx = 0

    for azimuth_deg in azimuth_angles:
        for elevation_deg in elevation_angles:
            test_idx += 1

            # Convert azimuth/elevation to yaw/pitch
            yaw_rad, pitch_rad = azimuth_elevation_to_yaw_pitch(azimuth_deg, elevation_deg)

            # Set antenna orientation directly (roll = 0 for antenna pointing)
            tx.orientation = mi.Point3f(float(yaw_rad), float(pitch_rad), 0.0)

            # Compute paths
            paths = p_solver(
                scene,
                los=True,
                refraction=False,
                specular_reflection=True,
                diffuse_reflection=False,
                diffraction=True,
                edge_diffraction=True,  # Enable for better obstacle handling in heavily obstructed scenarios
            )

            # Compute mean power in zone
            h_real, h_imag = paths.a
            sum_power_db = 0.0
            count_in_zone = 0

            for rx_idx in range(num_sample_points):
                if mask_values[rx_idx] > 0.5:  # In zone
                    h_real_rx = h_real[rx_idx : rx_idx + 1, ...]
                    h_imag_rx = h_imag[rx_idx : rx_idx + 1, ...]

                    path_gain_linear = dr.sum(dr.sum(cpx_abs_square((h_real_rx, h_imag_rx))))
                    path_gain_db = 10.0 * dr.log(dr.maximum(path_gain_linear, 1e-30)) / dr.log(10.0)

                    # Extract scalar value from DrJit tensor
                    # Convert to numpy array first, then extract scalar
                    path_gain_db_val = float(np.array(path_gain_db))

                    sum_power_db += path_gain_db_val
                    count_in_zone += 1

            mean_power = sum_power_db / max(count_in_zone, 1)

            grid_results.append({
                'azimuth_deg': azimuth_deg,
                'elevation_deg': elevation_deg,
                'mean_power_db': mean_power,
            })

            if mean_power > best_mean_power:
                best_mean_power = mean_power
                best_azimuth = azimuth_deg
                best_elevation = elevation_deg

                if verbose:
                    print(f"[{test_idx}/{total_tests}] NEW BEST: Az={azimuth_deg:.0f}°, El={elevation_deg:.0f}° → {mean_power:.1f} dB")
            elif verbose and test_idx % 5 == 0:
                print(f"[{test_idx}/{total_tests}] Az={azimuth_deg:.0f}°, El={elevation_deg:.0f}° → {mean_power:.1f} dB")

    # Cleanup receivers
    for rx_name in rx_names:
        if rx_name in [obj.name for obj in scene.receivers.values()]:
            scene.remove(rx_name)

    if verbose:
        print(f"\n{'='*70}")
        print("Grid Search Complete!")
        print(f"{'='*70}")
        print(f"Best angles: Azimuth={best_azimuth:.1f}°, Elevation={best_elevation:.1f}°")
        print(f"Best mean power: {best_mean_power:.1f} dB")
        print(f"{'='*70}\n")

    return best_azimuth, best_elevation, best_mean_power, grid_results


def optimize_boresight_pathsolver(
    scene,
    tx_name,
    map_config,
    scene_xml_path,
    zone_mask = None,
    initial_boresight=[100.0, 100.0, 10.0],
    num_sample_points=100,
    building_id=10,
    learning_rate=1.0,
    num_iterations=20,
    loss_type="coverage_maximize",
    power_threshold_dbm=-90.0,
    use_grid_search_init=False,
    grid_search_params=None,
    verbose=True,
    seed=42,  # Random seed for reproducible sampling
    tx_placement_mode="skip", # "center", "fixed", "line", "skip" (skip = don't move TX)
    # If true, the center position of the roof polygon is used
    # Else, use the start position
    Tx_Center=True,
    Tx_start_pos=[0.0, 0.0],
    save_radiomap_frames=False,
    frame_save_interval=1,
    output_dir="./optimization_frames",
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

    # Determine if initial_boresight is angles or position
    # If it's a position [x, y, z], convert to angles
    # If grid search is used, angles will be overridden anyway
    if not use_grid_search_init:
        # Convert initial_boresight (position) to angles
        initial_azimuth, initial_elevation = compute_initial_angles_from_position(
            tx_position, initial_boresight
        )
    else:
        # Grid search will provide angles (set placeholder values for now)
        initial_azimuth = 0.0
        initial_elevation = -10.0

    if verbose:
        print(f"\n{'='*70}")
        print("Binary Mask Zone Coverage Optimization")
        print(f"{'='*70}")
        if not use_grid_search_init:
            print(f"Initial angles: Azimuth={initial_azimuth:.1f}°, Elevation={initial_elevation:.1f}°")
            print(f"  (converted from look_at position: {initial_boresight})")
        print(f"Use grid search init: {use_grid_search_init}")
        print(f"Learning rate: {learning_rate}")
        print(f"Iterations: {num_iterations}")
        print(f"Sample points: {num_sample_points}")
        print(f"Loss type: {loss_type}")
        if loss_type == "coverage_threshold":
            print(f"Power threshold: {power_threshold_dbm:.1f} dB")
        print(f"Map config: {map_config}")
        print(f"{'='*70}\n")

    # Run grid search for initial boresight if requested
    if use_grid_search_init:
        if verbose:
            print("\n" + "="*70)
            print("PHASE 1: Grid Search for Initial Boresight")
            print("="*70 + "\n")

        # Set default grid search parameters if not provided
        # These defaults provide a good balance between coverage and speed:
        # - 16 azimuth samples (every 22.5°) gives full 360° coverage
        # - 5 elevation samples from 0° to -45° covers typical downward tilt range
        # - 100m radius is reasonable for most urban scenarios
        # - 50 sample points makes grid search fast while still accurate
        if grid_search_params is None:
            grid_search_params = {
                'num_angular_samples': 16,      # Test every 22.5° (full circle)
                'num_elevation_samples': 5,     # Test 5 elevations from 0° to -45°
                'min_elevation_deg': 0,         # Horizontal (0° = level with horizon)
                'max_elevation_deg': -45,       # 45° downward tilt
                'radius_meters': 100.0,         # 100m distance to boresight target
                'num_sample_points': 50,        # 50 sample points for evaluation
            }
            if verbose:
                print("Using default grid search parameters:")
                print(f"  Angular samples: {grid_search_params['num_angular_samples']} (every {360/grid_search_params['num_angular_samples']:.1f}°)")
                print(f"  Elevation samples: {grid_search_params['num_elevation_samples']} ({grid_search_params['min_elevation_deg']}° to {grid_search_params['max_elevation_deg']}°)")
                print(f"  Radius: {grid_search_params['radius_meters']:.0f}m")
                print(f"  Sample points: {grid_search_params['num_sample_points']}")
                print(f"  Total directions to test: {grid_search_params['num_angular_samples'] * grid_search_params['num_elevation_samples']}\n")

        # Run grid search
        best_azimuth, best_elevation, best_grid_power, grid_results = grid_search_initial_boresight(
            scene=scene,
            tx_name=tx_name,
            map_config=map_config,
            zone_mask=zone_mask,
            num_angular_samples=grid_search_params.get('num_angular_samples', 16),
            num_elevation_samples=grid_search_params.get('num_elevation_samples', 5),
            min_elevation_deg=grid_search_params.get('min_elevation_deg', 0),
            max_elevation_deg=grid_search_params.get('max_elevation_deg', -45),
            radius_meters=grid_search_params.get('radius_meters', 100.0),
            num_sample_points=grid_search_params.get('num_sample_points', 50),
            seed=seed,
            verbose=verbose,
        )

        # Store initial angles from grid search (will be used later to initialize optimization)
        initial_azimuth = best_azimuth
        initial_elevation = best_elevation

        if verbose:
            print("\n" + "="*70)
            print("PHASE 2: Gradient-Based Refinement")
            print("="*70)
            print(f"Starting from grid search result: Azimuth={initial_azimuth:.1f}°, Elevation={initial_elevation:.1f}°")
            print(f"Grid search mean power: {best_grid_power:.1f} dB")
            print("="*70 + "\n")

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
            print(f"  Current TX position: ({x_start_position:.2f}, {y_start_position:.2f}, {tx_height:.2f})")
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
            raise ValueError(f"Unknown tx_placement_mode: {tx_placement_mode}. Must be 'skip', 'center', 'fixed', or 'line'")


    if verbose:
        print(f"TX height: {tx_height:.1f}m")
        print(
            f"Boresight Z constraint: must be < {tx_height:.1f}m (no pointing upward)\n"
        )

    if zone_mask is not None:
        # Sample grid points for receivers (this is independent of cell size)
        sample_points = sample_grid_points(map_config, num_sample_points, seed=seed, zone_mask=zone_mask)
    else:
        sample_points = sample_points(map_config, num_sample_points, seed=seed)

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
        print(f"Zone coverage: {100.0 * num_in_zone / len(mask_values):.1f}% of samples")

    # CRITICAL: Remove ALL existing receivers from the scene first
    # This ensures paths.a indexing matches our optimization receivers exactly
    existing_receivers = list(scene.receivers.keys())
    if verbose and existing_receivers:
        print(f"Removing {len(existing_receivers)} existing receiver(s): {existing_receivers}")
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
    def compute_loss(azimuth_deg, elevation_deg):
        """
        Compute loss with AD enabled through PathSolver

        Parameters:
        -----------
        azimuth_deg, elevation_deg : DrJit Float scalars (converted from PyTorch tensors)
            Antenna pointing angles in degrees
            After @dr.wrap conversion, these are DrJit Float scalars with gradient tracking enabled
        """

        # CRITICAL: Enable gradients for each input parameter
        # The @dr.wrap decorator converts PyTorch 0-D tensors → DrJit Float scalars
        # We need to access .array to get the actual DrJit scalars
        #dr.enable_grad(azimuth_deg.array)
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

        # Set antenna orientation directly using yaw, pitch, roll
        tx = scene.get(tx_name)
        tx.orientation = mi.Point3f(yaw_rad, pitch_rad, roll_rad)

        # Position
        # tx.position = mi.Point3f(tx_x.array, tx_y.array, tx_height)

        # Compute paths with AD (this is built into the Dr.Jit framework)
        paths = p_solver(
            scene,
            los=True,
            refraction=False,
            specular_reflection=True,
            diffuse_reflection=False,
            diffraction=True,
            edge_diffraction=True,  # Enable for better obstacle handling in heavily obstructed scenarios
        )

        # Extract channel coefficients
        h_real, h_imag = paths.a

        # DEBUG: Verify channel coefficient shapes match expectations
        # Expected shape: (num_receivers, num_tx_antennas, num_rx_antennas, num_paths)
        h_real_shape = dr.shape(h_real)
        if verbose and h_real_shape[0] != num_sample_points:
            print(f"  [WARNING] Channel coefficient shape mismatch!")
            print(f"    Expected {num_sample_points} receivers, got {h_real_shape[0]}")
            print(f"    Full h_real shape: {h_real_shape}")

        # DEBUG: Check if any paths were found (only print on first call)
        # We use a simple flag via a mutable default to track first call
        if not hasattr(compute_loss, '_first_call_done'):
            compute_loss._first_call_done = False

        if verbose and not compute_loss._first_call_done:
            compute_loss._first_call_done = True
            # Compute total power across all receivers as a quick check
            total_power = dr.sum(dr.sum(cpx_abs_square((h_real, h_imag))))
            print(f"  [DEBUG] Total path power (linear): {total_power}")
            if total_power < 1e-25:
                print(f"  [WARNING] Very low or zero path power - PathSolver may not be finding paths!")

        # Collect path gains for all receivers
        path_gains_db_list = []  # Store path gains in dB

        for rx_idx in range(num_sample_points):
            # Extract channel coefficients for this receiver
            h_real_rx = h_real[rx_idx : rx_idx + 1, ...]
            h_imag_rx = h_imag[rx_idx : rx_idx + 1, ...]

            # Compute path gain in LINEAR scale
            path_gain_linear = dr.sum(dr.sum(cpx_abs_square((h_real_rx, h_imag_rx))))

            # Convert to dB (clamped to avoid log(0))
            path_gain_db = (
                10.0 * dr.log(dr.maximum(path_gain_linear, 1e-30)) / dr.log(10.0)
            )

            path_gains_db_list.append(path_gain_db)

        # Compute loss based on type
        if loss_type == "coverage_maximize":
            # COVERAGE MAXIMIZE: Maximize mean power in zone (in dB)
            # Loss = -mean(power_in_zone_dB) + variance_penalty_weight * variance(power_in_zone_dB)

            sum_power_in_zone = dr.auto.ad.Float(0.0)
            count_in_zone = dr.auto.ad.Float(0.0)

            # First pass: compute mean in dB
            for rx_idx in range(num_sample_points):
                in_zone = float(mask_values[rx_idx])  # 1.0 or 0.0
                in_zone_dr = dr.auto.ad.Float(in_zone)
                dr.disable_grad(in_zone_dr)  # Mask is constant

                power_db = path_gains_db_list[rx_idx]

                # Accumulate power for receivers in zone
                sum_power_in_zone += in_zone_dr * power_db
                count_in_zone += in_zone_dr

            # Mean power in zone (dB)
            mean_power = sum_power_in_zone / dr.maximum(count_in_zone, dr.auto.ad.Float(1.0))

            normalized_loss = -mean_power

        elif loss_type == "coverage_threshold":
            # COVERAGE THRESHOLD: Maximize fraction of zone above threshold
            # IMPROVED VERSION with enhancements:
            # 1. Smooth sigmoid for counting points above threshold
            # 2. Margin bonus for exceeding threshold (prevents gradient saturation)
            # 3. Soft-minimum penalty to eliminate dead zones/outliers below threshold
            #
            # The soft-minimum penalty is KEY for ensuring ALL points reach the threshold.
            # It creates strong gradients toward improving the weakest points.

            count_above_threshold = dr.auto.ad.Float(0.0)
            count_in_zone = dr.auto.ad.Float(0.0)
            threshold = dr.auto.ad.Float(power_threshold_dbm)

            # Track soft-minimum for worst-case penalty
            # Soft-min gives strong gradients toward improving weakest points
            # Lower alpha = sharper focus on worst points (more aggressive)
            soft_min_alpha = dr.auto.ad.Float(5.0)  # 5 dB smoothing for soft-min
            soft_min_sum = dr.auto.ad.Float(0.0)

            for rx_idx in range(num_sample_points):
                in_zone = float(mask_values[rx_idx])
                in_zone_dr = dr.auto.ad.Float(in_zone)
                dr.disable_grad(in_zone_dr)

                power_db = path_gains_db_list[rx_idx]

                # 1. Smooth sigmoid for counting points above threshold
                distance_from_threshold = power_db - threshold
                smoothness = dr.auto.ad.Float(8.0)  # 8 dB smoothing for sigmoid

                above_threshold = dr.auto.ad.Float(1.0) / (
                    dr.auto.ad.Float(1.0) + dr.exp(-distance_from_threshold / smoothness)
                )

                # 2. Margin bonus for exceeding threshold
                # This gives continuous gradients even when well above threshold
                margin = dr.auto.ad.Float(5.0)  # 5 dB safety margin above threshold
                excess_power = dr.maximum(distance_from_threshold - margin, dr.auto.ad.Float(0.0))
                margin_bonus = excess_power / dr.auto.ad.Float(100.0)  # Small bonus to avoid dominating

                count_above_threshold += in_zone_dr * (above_threshold + margin_bonus)
                count_in_zone += in_zone_dr

                # 3. Accumulate soft-minimum (focuses on worst points)
                # soft_min ≈ -α * log(sum(exp(-x/α)))
                # This heavily penalizes low outliers (dead zones)
                soft_min_sum += in_zone_dr * dr.exp(-power_db / soft_min_alpha)

            # Primary objective: maximize fraction above threshold
            fraction = count_above_threshold / dr.maximum(count_in_zone, dr.auto.ad.Float(1.0))

            # Soft-minimum of power in zone (approximates worst-case point)
            # Lower soft_min = worse dead zones
            soft_min_power = -soft_min_alpha * dr.log(soft_min_sum / dr.maximum(count_in_zone, dr.auto.ad.Float(1.0)))

            # Worst-case penalty: penalize if worst point is below threshold
            # This creates strong gradients to lift dead zones above threshold
            # Gap = how far below threshold the worst point is
            worst_case_gap = threshold - soft_min_power  # Positive if worst point is below threshold
            worst_case_penalty = dr.maximum(worst_case_gap, dr.auto.ad.Float(0.0)) / dr.auto.ad.Float(10.0)

            # Combined loss:
            # 1. Maximize fraction above threshold (primary objective)
            # 2. Lift worst points above threshold (eliminate dead zones)
            worst_case_weight = dr.auto.ad.Float(0.5)  # Strong weight - aggressive dead zone elimination

            normalized_loss = -fraction + worst_case_weight * worst_case_penalty
            
        elif loss_type == "percentile_maximize":
            # PERCENTILE MAXIMIZE: Maximize the Nth percentile (e.g., median)
            # This is harder in differentiable form, so we approximate with soft-minimum
            # Loss = -soft_minimum(power_in_zone)

            # Soft minimum using log-sum-exp trick (differentiable)
            # soft_min(x) ≈ -α * log(sum(exp(-x/α)))
            # where α controls smoothness (smaller α → sharper min)

            alpha = dr.auto.ad.Float(10.0)  # Smoothness parameter
            sum_exp = dr.auto.ad.Float(0.0)
            count_in_zone = dr.auto.ad.Float(0.0)

            for rx_idx in range(num_sample_points):
                in_zone = float(mask_values[rx_idx])
                in_zone_dr = dr.auto.ad.Float(in_zone)
                dr.disable_grad(in_zone_dr)

                power_db = path_gains_db_list[rx_idx]

                # Accumulate for soft-min
                sum_exp += in_zone_dr * dr.exp(-power_db / alpha)
                count_in_zone += in_zone_dr

            # Soft minimum (represents worst-case coverage in zone)
            soft_min = -alpha * dr.log(sum_exp / dr.maximum(count_in_zone, dr.auto.ad.Float(1.0)))

            # Loss = -soft_min (we want to maximize worst-case coverage)
            normalized_loss = -soft_min

        else:
            raise ValueError(
                f"Unknown loss_type: '{loss_type}'. "
                f"Must be 'coverage_maximize', 'coverage_threshold', or 'percentile_maximize'"
            )

        return normalized_loss

    # PyTorch parameters: azimuth and elevation angles (in degrees)
    # Using 0-D tensors (scalars) - this is the ONLY pattern that works with @dr.wrap
    # 1-D tensors lose gradients during indexing
    #azimuth = torch.tensor(
    #    initial_azimuth, device="cuda", dtype=torch.float32, requires_grad=True
    #)
    elevation = torch.tensor(
        initial_elevation, device="cuda", dtype=torch.float32, requires_grad=True
    )

    # Optimizer: Adam shows the best performance
    # Optimize azimuth and elevation angles directly (2 parameters instead of 3)
    optimizer = torch.optim.Adam(
        [elevation], lr=learning_rate
    )

    # Learning rate scheduler: required to jump out of local minima for difficult loss surfaces...
    use_scheduler = num_iterations >= 50
    if use_scheduler:
        # Use exponential decay for smoother convergence after grid search
        # This is more stable than ReduceLROnPlateau for noisy loss landscapes
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95  # Decay by 5% each iteration (smooth reduction)
        )
        if verbose:
            print(f"Learning rate scheduler enabled (exponential decay, gamma=0.95)")
            print(f"  Initial LR: {learning_rate:.4f}")
            print(f"  LR after 50 iters: {learning_rate * (0.95**50):.6f}")
            print(f"  LR after 100 iters: {learning_rate * (0.95**100):.6f}")

    # Tracking
    loss_history = []
    angle_history = [
        [initial_azimuth, initial_elevation]
    ]
    gradient_history = []  # Track gradient norms for diagnostics

    best_loss = float("inf")
    best_azimuth_final = initial_azimuth
    best_elevation_final = initial_elevation

    # Setup frame saving if requested
    if save_radiomap_frames:
        import os
        os.makedirs(output_dir, exist_ok=True)
        from sionna.rt import RadioMapSolver
        rm_solver = RadioMapSolver()
        print(f"RadioMap frames will be saved to: {output_dir}")
        print(f"Saving every {frame_save_interval} iteration(s)")

    start_time = time.time()

    # Verify scene state before optimization
    if verbose:
        print(f"\nScene verification before optimization:")
        print(f"  Number of receivers: {len(scene.receivers)}")
        print(f"  Number of transmitters: {len(scene.transmitters)}")
        tx = scene.get(tx_name)
        print(f"  TX '{tx_name}' position: {tx.position}")
        print(f"  TX '{tx_name}' orientation: {tx.orientation}")
        print()

    # Run the optimization for the specified number of iterations
    for iteration in range(num_iterations):

        if verbose and iteration == 0:
            print(f"\nIteration {iteration+1}/{num_iterations}:")
            print(
                #f"  Angles: Azimuth={azimuth.item():.2f}°, Elevation={elevation.item():.2f}°"
            )

        # Forward pass
        loss = compute_loss(initial_azimuth, elevation)

        # Backward pass (using AD)
        # The gradients are backpropagated through the scene
        optimizer.zero_grad()
        loss.backward()

        # Track gradient norm for diagnostics
        #grad_azimuth_val = azimuth.grad.item() if azimuth.grad is not None else 0.0
        grad_elevation_val = elevation.grad.item() if elevation.grad is not None else 0.0

        # Gradient norm (2 parameters now instead of 3)
        grad_norm = np.sqrt(
            #grad_azimuth_val**2
            + grad_elevation_val**2
        )
        gradient_history.append(grad_norm)

        if verbose:
            print(
                #f"  Gradients: dAz={grad_azimuth_val:+.3e}°, dEl={grad_elevation_val:+.3e}° (norm={grad_norm:.3e})"
            )
            # RMSE is not applicable for a maximization objective, print mean power instead
            mean_power_in_zone = -loss.item()
            print(f"  Loss: {loss.item():.4f}, Mean Power in Zone: {mean_power_in_zone:.2f} dB")

        # Update
        optimizer.step()

        # Save RadioMap frame if requested
        if save_radiomap_frames and (iteration % frame_save_interval == 0 or iteration == num_iterations - 1):
            # Apply current angles to scene (temporarily)
            tx = scene.get(tx_name)
            #current_azimuth = azimuth.item()
            current_elevation = elevation.item()
            #yaw_rad, pitch_rad = azimuth_elevation_to_yaw_pitch(current_azimuth, current_elevation)
            tx.orientation = mi.Point3f(float(yaw_rad), float(pitch_rad), 0.0)

            # Generate RadioMap
            rm = rm_solver(
                scene,
                max_depth=5,
                samples_per_tx=int(1e7),  # Lower for speed, increase for quality
                cell_size=map_config['cell_size'],
                center=map_config['center'],
                orientation=[0, 0, 0],
                size=map_config['size'],
                los=True,
                specular_reflection=True,
                diffuse_reflection=True,
                refraction=False,
            )

            # Extract and save visualization
            import matplotlib.pyplot as plt
            rss_watts = rm.rss.numpy()[0, :, :]
            signal_strength_dBm = 10.0 * np.log10(rss_watts + 1e-30) + 30.0

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(signal_strength_dBm, origin='lower', cmap='viridis',
                          vmin=-120, vmax=-60, extent=[
                              map_config['center'][0] - map_config['size'][0]/2,
                              map_config['center'][0] + map_config['size'][0]/2,
                              map_config['center'][1] - map_config['size'][1]/2,
                              map_config['center'][1] + map_config['size'][1]/2,
                          ])
            plt.colorbar(im, ax=ax, label='Signal Strength (dB)')
            ax.set_title(f'Iteration {iteration} | Loss: {loss.item():.2f}')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')

            # Save frame
            frame_path = os.path.join(output_dir, f'frame_{iteration:04d}.png')
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            if verbose:
                print(f"  Saved frame: {frame_path}")

        # Update learning rate if scheduler is enabled
        if use_scheduler:
            scheduler.step()

        # Apply constraints on angles
        with torch.no_grad():
            # Azimuth: clamp to [0, 360) degrees
            #azimuth.clamp_(min=0.0, max=360.0)
            # Wrap azimuth around if needed (360° → 0°)
            #if azimuth.item() >= 360.0:
            #    azimuth.fill_(azimuth.item() % 360.0)

            # Elevation: clamp to prevent pointing upward (only downward tilt)
            # 0° = horizontal, negative = downward tilt
            elevation.clamp_(min=-90.0, max=0.0)

        # Track
        loss_history.append(loss.item())
        angle_history.append(
            [initial_azimuth, elevation.item()]
        )

        # Updates the ideal parameters if the loss is the lowest to date
        if loss.item() < best_loss:
            best_loss = loss.item()
            #best_azimuth_final = azimuth.item()
            best_elevation_final = elevation.item()

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
    tx.orientation = mi.Point3f(
        float(final_yaw_rad), float(final_pitch_rad), 0.0
    )

    # Compute final coverage statistics
    coverage_stats = {
        'num_samples_in_zone': num_in_zone,
        'num_samples_total': len(mask_values),
        'zone_coverage_fraction': num_in_zone / len(mask_values),
        'final_loss': best_loss,
        'loss_type': loss_type,
        'best_azimuth_deg': best_azimuth_final,
        'best_elevation_deg': best_elevation_final,
    }

    if verbose:
        print(f"\n{'='*70}")
        print("Optimization Complete!")
        print(f"{'='*70}")
        print(
            f"Best angles: Azimuth={best_azimuth_final:.1f}°, Elevation={best_elevation_final:.1f}°"
        )
        print(f"Best loss: {best_loss:.4f}")
        if loss_type == "coverage_maximize":
            print(f"  (Maximizing mean power in zone)")
            print(f"  Negative loss = {-best_loss:.2f} dB (approximate mean power)")
        elif loss_type == "coverage_threshold":
            print(f"  (Maximizing fraction above {power_threshold_dbm} dB)")
            print(f"  Estimated coverage: {-best_loss*100:.1f}% of zone")
        elif loss_type == "percentile_maximize":
            print(f"  (Maximizing soft-minimum power in zone)")
        print(f"Zone samples: {num_in_zone}/{len(mask_values)} ({100.0*num_in_zone/len(mask_values):.1f}%)")
        print(f"Total time: {elapsed_time:.1f}s")
        print(f"Time per iteration: {elapsed_time/num_iterations:.2f}s")
        print(f"{'='*70}\n")

    # Return angles instead of Cartesian coordinates
    best_angles = [best_azimuth_final, best_elevation_final]
    return best_angles, loss_history, angle_history, gradient_history, coverage_stats


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    print("Boresight optimization using PathSolver + AD")
    print("\nExample usage:")
    print(
        """
    from mcp_optimization_boresight_pathsolver import (
        optimize_boresight_pathsolver,
        create_target_radiomap
    )

    # Get TX position from scene
    tx = scene.get("gnb")
    tx_pos = tx.position  # (x, y, z)

    map_config = {
        'center': [0, 0, 1.5],
        'size': [400, 400],
        'cell_size': (20, 20),  # Coarser for PathSolver
        'ground_height': 1.5,
    }

    # Option 1: Path-loss based target (omnidirectional)
    target_map = create_target_radiomap(
        map_config,
        target_type='path_loss',
        tx_position=tx_pos,
        tx_power_dBm=30.0,  # 1 Watt
        frequency_GHz=3.5,  # 5G mid-band
        path_loss_exponent=2.5  # Urban environment
    )

    # Option 2: Path-loss with sector (directional antenna)
    target_map = create_target_radiomap(
        map_config,
        target_type='path_loss_sector',
        tx_position=tx_pos,
        tx_power_dBm=30.0,
        frequency_GHz=3.5,
        path_loss_exponent=2.5,
        sector_angle=45,  # Point northeast
        sector_width=120  # 120° beamwidth
    )

    # Option 3: Simple Gaussian (for testing)
    target_map = create_target_radiomap(map_config, target_type='circular')

    # Option 4: Angular sectors with custom power levels
    angular_sectors_config = [
        {'angle_start': 0, 'angle_end': 120, 'power_dbm': -80},      # East to SE: -80 dBm
        {'angle_start': 120, 'angle_end': 240, 'power_dbm': -100},   # SE to NW: -100 dBm
        {'angle_start': 240, 'angle_end': 360, 'power_dbm': -90}     # NW to East: -90 dBm
    ]
    target_map = create_target_radiomap(
        map_config,
        target_type='angular_sectors',
        tx_position=tx_pos,
        angular_sectors=angular_sectors_config
    )

    # Option 5: Angular sectors with AUTO-SCALED power levels (RECOMMENDED for large maps)
    # This automatically computes achievable power levels based on map size and antenna characteristics
    angular_sectors_auto = [
        {'angle_start': 0, 'angle_end': 120, 'relative_power': 'high'},     # Strong coverage sector
        {'angle_start': 120, 'angle_end': 240, 'relative_power': 'low'},   # Weak coverage sector
        {'angle_start': 240, 'angle_end': 360, 'relative_power': 'medium'} # Medium coverage sector
    ]
    target_map = create_target_radiomap(
        map_config,
        target_type='angular_sectors',
        tx_position=tx_pos,
        angular_sectors=angular_sectors_auto,
        auto_scale_power=True,  # Enable auto-scaling
        antenna_gain_dBi=8.0,   # TR38.901 default
        frequency_GHz=3.5,      # 5G mid-band
        path_loss_exponent=2.5  # Urban environment
    )

    best_boresight, loss_hist, bore_hist = optimize_boresight_pathsolver(
        scene=scene,
        tx_name="gnb",
        map_config=map_config,
        target_map=target_map,
        initial_boresight=[100.0, 100.0, 10.0],
        num_sample_points=100,  # Number of receiver points
        learning_rate=1.0,
        num_iterations=50,
        loss_type='mse',  # Use MSE for path-loss targets
        verbose=True
    )
    """
    )

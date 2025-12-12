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


def create_optimization_gif(frame_dir, output_path="optimization.gif", duration=200, loop=0):
    """
    Create GIF from saved optimization frames

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

    Returns:
    --------
    str : Path to created GIF
    """
    import os
    import glob
    try:
        from PIL import Image
    except ImportError:
        print("PIL not found, trying imageio...")
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

    # Load images
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


def create_target_radiomap(
    map_config,
    target_type="circular",
    sector_angle=0,
    sector_width=90,
    tx_position=None,
    tx_power_dBm=30.0,
    frequency_GHz=3.5,
    path_loss_exponent=2.0,
    angular_sectors=None,
    auto_scale_power=False,
    antenna_gain_dBi=8.0,
    beamwidth_3dB=65.0,
):
    """
    Create target radio map distribution

    Parameters:
    -----------
    map_config : dict
        Map configuration with 'size', 'cell_size', 'center'
    target_type : str
        Type of target: 'circular', 'sector', 'uniform', 'easy_test', 'path_loss', 'path_loss_sector', 'angular_sectors'
    sector_angle : float
        Center angle for sector (degrees)
    sector_width : float
        Width of sector (degrees)
    tx_position : tuple or None
        (x, y, z) position of transmitter for path-loss-based targets
    tx_power_dBm : float
        Transmit power in dBm (default: 30 dBm = 1 Watt)
    frequency_GHz : float
        Carrier frequency in GHz (default: 3.5 GHz for 5G)
    path_loss_exponent : float
        Path loss exponent (2.0 = free space, 3-4 = urban)
    angular_sectors : list of dict or None
        For 'angular_sectors' target type. List of sector definitions, each with:
        - 'angle_start': Start angle in degrees (0° = East, counter-clockwise)
        - 'angle_end': End angle in degrees
        - 'power_dbm': Target power level in dBm for this sector (ignored if auto_scale_power=True)
        - 'relative_power': (Optional) Relative power level. 'high', 'medium', 'low' (used with auto_scale_power)
        Example: [{'angle_start': 0, 'angle_end': 120, 'power_dbm': -80, 'relative_power': 'high'},
                  {'angle_start': 120, 'angle_end': 240, 'power_dbm': -100, 'relative_power': 'low'},
                  {'angle_start': 240, 'angle_end': 360, 'power_dbm': -90, 'relative_power': 'medium'}]
    auto_scale_power : bool
        If True, automatically compute achievable power levels based on map geometry and antenna.
        This overrides 'power_dbm' values in angular_sectors and uses 'relative_power' instead.
    antenna_gain_dBi : float
        Antenna gain in dBi (for auto_scale_power). TR38.901 typical: 8 dBi
    beamwidth_3dB : float
        3dB beamwidth in degrees (for auto_scale_power). TR38.901 typical: 65°
    """
    width_m, height_m = map_config["size"]
    cell_w, cell_h = map_config["cell_size"]

    n_x = int(width_m / cell_w)
    n_y = int(height_m / cell_h)

    center_x, center_y, center_z = map_config["center"]

    x = np.linspace(center_x - width_m / 2, center_x + width_m / 2, n_x)
    y = np.linspace(center_y - height_m / 2, center_y + height_m / 2, n_y)

    X, Y = np.meshgrid(x, y)

    if target_type == "circular":
        # 2D Gaussian distribution centered at coverage center
        # TRUE Gaussian in LINEAR scale: P = A * exp(-(dist^2) / (2*sigma^2))
        # Convert to dB: P_dB = 10*log10(A) + 10*log10(exp(-dist^2 / (2*sigma^2)))
        #              = peak_dB - (dist^2 / (2*sigma^2)) * 10*log10(e)
        #              = peak_dB - (dist^2 / (2*sigma^2)) * 4.343

        dist_squared = (X - center_x) ** 2 + (Y - center_y) ** 2

        # Parameters for Gaussian
        peak_dB = -30.0  # Peak power at center (achievable with your setup)
        sigma = (
            width_m / 8
        )  # Std dev = 10m for 40m wide map (covers ~95% within 2*sigma = 20m)

        # Convert to dB: This creates a TRUE Gaussian in linear power
        # At distance = sigma, power drops by -4.343 dB (this is correct for Gaussian)
        # At distance = sqrt(2)*sigma, power drops by -8.686 dB
        gaussian_dB = peak_dB - (dist_squared / (2 * sigma**2)) * 4.343

        # Use unclipped Gaussian for true distribution
        # NOTE: Clipping with np.maximum() would make this NOT a true Gaussian
        target_map = gaussian_dB

    elif target_type == "sector":
        angle = np.arctan2(Y - center_y, X - center_x) * 180 / np.pi
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        # Define sector using parameters
        in_sector = np.abs(angle - sector_angle) < (sector_width / 2)

        # Base: very weak everywhere
        target_map = np.full((n_y, n_x), -140.0)

        # In-sector targets (directive antenna should achieve this)
        target_map[in_sector & (dist < 150)] = (
            -95.0
        )  # Good coverage within 150m in-sector
        target_map[in_sector & (dist < 100)] = -90.0  # Better coverage within 100m
        target_map[in_sector & (dist < 50)] = -85.0  # Best coverage close-in

        # Out-of-sector: don't care (keep at -140 dB)
    elif target_type == "uniform":
        # More realistic uniform target for long distances
        target_map = np.full((n_y, n_x), -50.0)  # Changed from -70 dB to -100 dB

    elif target_type == "easy_test":
        # EXTREMELY EASY: Only care about a small region directly in one direction
        # This should be trivial for the optimizer to solve
        angle = np.arctan2(Y - center_y, X - center_x) * 180 / np.pi
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        # Very narrow sector (30 degrees) pointing in one direction
        target_angle = sector_angle  # Use parameter
        in_target_sector = np.abs(angle - target_angle) < 15  # ±15° = 30° total

        # Close range only (within 100m)
        close_range = dist < 100

        # Target: strong signal in this narrow sector, don't care elsewhere
        target_map = np.full((n_y, n_x), -150.0)  # Very weak baseline (ignored)
        target_map[in_target_sector & close_range & (dist < 50)] = -85.0  # Close
        target_map[in_target_sector & close_range & (dist >= 50)] = -95.0  # Medium

        # This is easy: optimizer just needs to point antenna at target_angle

    elif target_type == "path_loss":
        # Realistic path-loss model based on distance from TX position
        # Uses RELATIVE power (independent of absolute TX power in simulator)
        # Target represents the SHAPE of the coverage, not absolute levels

        if tx_position is None:
            raise ValueError("tx_position must be provided for 'path_loss' target type")

        tx_x, tx_y, tx_z = tx_position

        # Compute 3D distance from TX to each grid point
        # Grid points are at center_z (typically ground level)
        dist_3d = np.sqrt((X - tx_x) ** 2 + (Y - tx_y) ** 2 + (center_z - tx_z) ** 2)

        # Free Space Path Loss (FSPL) formula
        # FSPL(d) = 20*log10(d) + 20*log10(f) + 32.44 (for d in meters, f in MHz)
        freq_MHz = frequency_GHz * 1000
        fspl_1m = 20 * np.log10(freq_MHz) + 32.44

        # Path loss model: FSPL at 1m + path loss exponent for additional distance
        # PL(d) = FSPL(1m) + 10*n*log10(d) for d >= 1m
        path_loss = np.where(
            dist_3d >= 1.0,
            fspl_1m + 10 * path_loss_exponent * np.log10(dist_3d),
            fspl_1m,  # At distances < 1m, use FSPL at 1m
        )

        # OPTION 1: If you want absolute power targets (assumes tx_power_dBm matches Sionna)
        # target_map = tx_power_dBm - path_loss

        # OPTION 2: Relative power (independent of TX power - RECOMMENDED)
        # Use path loss relative to minimum distance
        # This creates a target based on SHAPE, not absolute levels
        min_path_loss = path_loss.min()
        target_map = -(
            path_loss - min_path_loss
        )  # 0 dB at closest point, negative elsewhere

        # If you want to set a specific peak power, add an offset:
        # target_map = tx_power_dBm - path_loss  # Uncomment for absolute power

        # NOTE: For MSE loss, absolute power matters. For normalized MSE or cross-entropy,
        # only the relative shape matters. Choose based on your loss function!

    elif target_type == "path_loss_sector":
        # Path-loss model with sector-based antenna gain pattern
        # This models a realistic directional antenna:
        # - Strong signal in desired sector (antenna main lobe)
        # - Weaker signal outside sector (antenna sidelobes/backlobe)

        if tx_position is None:
            raise ValueError(
                "tx_position must be provided for 'path_loss_sector' target type"
            )

        tx_x, tx_y, tx_z = tx_position

        # Compute 3D distance from TX
        dist_3d = np.sqrt((X - tx_x) ** 2 + (Y - tx_y) ** 2 + (center_z - tx_z) ** 2)

        # Compute angle from TX to each grid point
        angle = np.arctan2(Y - tx_y, X - tx_x) * 180 / np.pi

        # Determine if point is in main sector
        angle_diff = np.abs(angle - sector_angle)
        # Handle wraparound at ±180°
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)
        in_sector = angle_diff < (sector_width / 2)

        # Free Space Path Loss
        freq_MHz = frequency_GHz * 1000
        fspl_1m = 20 * np.log10(freq_MHz) + 32.44
        path_loss = np.where(
            dist_3d >= 1.0,
            fspl_1m + 10 * path_loss_exponent * np.log10(dist_3d),
            fspl_1m,
        )

        # Antenna gain pattern (simplified)
        # Main lobe: +15 dB gain (typical for directional antenna)
        # Sidelobes: -10 dB (25 dB front-to-back ratio)
        antenna_gain = np.where(in_sector, 15.0, -10.0)

        # OPTION 1: Absolute power (if tx_power_dBm matches Sionna)
        # target_map = tx_power_dBm + antenna_gain - path_loss

        # OPTION 2: Relative power (RECOMMENDED - independent of TX power)
        # Combine path loss with antenna pattern to get effective path loss
        effective_path_loss = path_loss - antenna_gain

        # Normalize to minimum (peak power = 0 dB relative)
        min_effective_loss = effective_path_loss.min()
        target_map = -(effective_path_loss - min_effective_loss)

        # If you want absolute power, uncomment:
        # target_map = tx_power_dBm + antenna_gain - path_loss

    elif target_type == "angular_sectors":
        # Angular sectors centered at transmitter position
        # Each sector has a defined power level based on angle from TX

        if tx_position is None:
            raise ValueError(
                "tx_position must be provided for 'angular_sectors' target type"
            )

        if angular_sectors is None or len(angular_sectors) == 0:
            raise ValueError(
                "angular_sectors must be provided as a list of sector definitions"
            )

        tx_x, tx_y, tx_z = tx_position

        # Compute angle from TX to each grid point
        # arctan2 returns angles in range [-180, 180], so we convert to [0, 360]
        angle = np.arctan2(Y - tx_y, X - tx_x) * 180 / np.pi
        angle = np.where(angle < 0, angle + 360, angle)  # Convert to [0, 360]

        # Auto-scale power levels if requested
        if auto_scale_power:
            # Estimate achievable power levels
            power_est = estimate_achievable_power(
                tx_position,
                map_config,
                antenna_gain_dBi=antenna_gain_dBi,
                tx_power_dBm=tx_power_dBm,
                frequency_GHz=frequency_GHz,
                path_loss_exponent=path_loss_exponent,
                beamwidth_3dB=beamwidth_3dB,
            )

            # Map relative power levels to achievable values
            power_mapping = {
                "high": power_est["peak_power_dbm"],  # Main lobe, close to TX
                "medium": power_est["mainlobe_power_dbm"],  # Main lobe, far from TX
                "low": power_est["sidelobe_power_dbm"],  # Sidelobe regions
                "verylow": power_est["min_power_dbm"],  # Sidelobe, far from TX
            }

            print(f"\n[Auto-Scale Power] Estimated achievable levels:")
            print(f"  High:     {power_mapping['high']:.1f} dBm (main lobe, near)")
            print(f"  Medium:   {power_mapping['medium']:.1f} dBm (main lobe, far)")
            print(f"  Low:      {power_mapping['low']:.1f} dBm (sidelobe, near)")
            print(f"  Very Low: {power_mapping['verylow']:.1f} dBm (sidelobe, far)")
            print(f"  Center distance: {power_est['center_distance_m']:.1f} m")
            print(f"  Corner distance: {power_est['corner_distance_m']:.1f} m\n")

        # Initialize with a default low power level
        target_map = np.full((n_y, n_x), -100.0)

        # Apply power levels for each sector
        for sector in angular_sectors:
            angle_start = sector["angle_start"]
            angle_end = sector["angle_end"]

            # Determine power level
            if auto_scale_power:
                # Use relative_power if provided, otherwise default to 'medium'
                rel_power = sector.get("relative_power", "medium")
                if rel_power not in power_mapping:
                    raise ValueError(
                        f"Invalid relative_power: {rel_power}. "
                        f"Must be one of: {list(power_mapping.keys())}"
                    )
                power_dbm = power_mapping[rel_power]
            else:
                # Use explicit power_dbm from sector definition
                power_dbm = sector["power_dbm"]

            # Handle wraparound case (e.g., sector from 350° to 10°)
            if angle_end < angle_start:
                # Sector wraps around 0°/360°
                in_sector = (angle >= angle_start) | (angle <= angle_end)
            else:
                # Normal sector
                in_sector = (angle >= angle_start) & (angle <= angle_end)

            # Set power level for this sector
            target_map[in_sector] = power_dbm

    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return target_map


def sample_grid_points(map_config, num_samples=100, seed=None):
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

    # Flatten and randomly sample
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


def optimize_boresight_pathsolver(
    scene,
    tx_name,
    map_config,
    scene_xml_path,
    target_map,
    initial_boresight=[100.0, 100.0, 10.0],
    num_sample_points=100,
    building_id=10,
    learning_rate=1.0,
    num_iterations=20,
    loss_type="mse",
    normalize_power=False,
    verbose=True,
    seed=42,  # Random seed for reproducible sampling
    tx_placement_mode="center", # "center", "fixed", "line"
    # If true, the center position of the roof polygon is used
    # Else, use the start position
    Tx_Center=True,
    # Sets the start position in the XY plane of the building polygon
    Tx_start_pos=[0.0, 0.0],
    # RadioMap visualization options
    save_radiomap_frames=False,  # Set to True to save RadioMap at each iteration
    frame_save_interval=1,  # Save every N iterations (1 = every iteration)
    output_dir="./optimization_frames",  # Directory to save frames
):
    """
    Optimize boresight using PathSolver with automatic differentiation

    This follows the rm_diff.ipynb pattern:
    - Use @dr.wrap to enable PyTorch-DrJit AD
    - Use PathSolver (not RadioMapSolver) for gradient computation
    - Sample grid points as receivers
    """

    if verbose:
        print(f"\n{'='*70}")
        print("Boresight Optimization using PathSolver + AD")
        print(f"{'='*70}")
        print(
            f"Initial boresight: ({initial_boresight[0]:.1f}, {initial_boresight[1]:.1f}, {initial_boresight[2]:.1f})"
        )
        print(f"Learning rate: {learning_rate}")
        print(f"Iterations: {num_iterations}")
        print(f"Sample points: {num_sample_points}")
        print(f"Loss type: {loss_type}")
        if loss_type == "mse":
            print(f"Power normalization: {'ON' if normalize_power else 'OFF'}")
            if normalize_power:
                print(f"  → Optimizing spatial pattern (normalized to [0,1])")
            else:
                print(f"  → Optimizing absolute power levels (dB)")
        print(f"Map config: {map_config}")
        print(f"{'='*70}\n")

    # Get TX height for boresight Z constraint (prevent pointing upward)
    tx = scene.get(tx_name)
    tx_height = float(dr.detach(tx.position[2])[0])

    # Initialize TxPlacement
    tx_placement = TxPlacement(scene, tx_name, scene_xml_path, building_id)

    # Sets the initial location
    if tx_placement_mode == "center":
        tx_placement.set_rooftop_center()
        x_start_position = tx_placement.building["center"][0]
        y_start_position = tx_placement.building["center"][1]
    elif tx_placement_mode == "fixed":
        x_start_position = Tx_start_pos[0]
        y_start_position = Tx_start_pos[1]
        z_pos = tx_placement.building["z_height"]
        tx.position = mi.Point3f(x_start_position, y_start_position, z_pos)
    else: # Default to center
        tx_placement.set_rooftop_center()
        x_start_position = tx_placement.building["center"][0]
        y_start_position = tx_placement.building["center"][1]


    if verbose:
        print(f"TX height: {tx_height:.1f}m")
        print(
            f"Boresight Z constraint: must be < {tx_height:.1f}m (no pointing upward)\n"
        )

    # Sample grid points for receivers (this is independent of cell size)
    sample_points = sample_grid_points(map_config, num_sample_points, seed=seed)

    # Extract target values at sampled points
    width_m, height_m = map_config["size"]
    cell_w, cell_h = map_config["cell_size"]
    n_x = int(width_m / cell_w)
    n_y = int(height_m / cell_h)
    center_x, center_y, _ = map_config["center"]

    # Map sample points to grid indices to get target values
    target_values = []
    for point in sample_points:
        x, y = point[0], point[1]
        # Convert to grid indices
        i = int((x - (center_x - width_m / 2)) / cell_w)
        j = int((y - (center_y - height_m / 2)) / cell_h)
        i = np.clip(i, 0, n_x - 1)
        j = np.clip(j, 0, n_y - 1)
        target_values.append(target_map[j, i])

    target_values = np.array(target_values, dtype=np.float32)

    if verbose:
        print(
            f"Target values range: [{target_values.min():.1f}, {target_values.max():.1f}] dB"
        )

    # Add receivers to scene
    rx_names = []
    for idx, position in enumerate(sample_points):
        rx_name = f"opt_rx_{idx}"
        rx_names.append(rx_name)
        # Remove if exists
        if rx_name in [obj.name for obj in scene.receivers.values()]:
            scene.remove(rx_name)
        # Convert to float32 to avoid type errors with Mitsuba
        position_f32 = [float(position[0]), float(position[1]), float(position[2])]
        rx = Receiver(name=rx_name, position=position_f32)
        scene.add(rx)

    # Create PathSolver
    p_solver = PathSolver()
    p_solver.loop_mode = "evaluated"  # Required for gradient computation

    # Define differentiable loss function using @dr.wrap
    @dr.wrap(source="torch", target="drjit")
    def compute_loss(boresight_x, boresight_y, boresight_z):
        """
        Compute loss with AD enabled through PathSolver

        Parameters:
        -----------
        boresight_x, boresight_y, boresight_z : DrJit Float scalars (converted from PyTorch tensors)
            After @dr.wrap conversion, these are DrJit Float scalars with gradient tracking enabled
        """

        # CRITICAL: Enable gradients for each input parameter
        # The @dr.wrap decorator converts PyTorch 0-D tensors → DrJit Float scalars
        # We need to access .array to get the actual DrJit scalars
        dr.enable_grad(boresight_x.array)
        dr.enable_grad(boresight_y.array)
        dr.enable_grad(boresight_z.array)

        # Create Mitsuba Point3f from the .array attributes
        # This extracts the actual DrJit scalars that mi.Point3f expects
        look_at_target = mi.Point3f(
            boresight_x.array,
            boresight_y.array,
            boresight_z.array,
        )

        # Set boresight direction
        tx = scene.get(tx_name)
        tx.look_at(look_at_target)

        # Position
        # tx.position = mi.Point3f(tx_x.array, tx_y.array, tx_height)

        # Compute paths with AD (this is built into the Dr.Jit framework)
        paths = p_solver(
            scene,
            los=True,
            refraction=False,
            specular_reflection=True,
            diffuse_reflection=True,
        )

        # Extract channel coefficients
        h_real, h_imag = paths.a

        # Store values for logging (optional)
        loss_target = []
        loss_his = []

        # Step 1: FIRST PASS - Compute softmax normalization denominators
        # We need sum(exp(x)) for both predicted and target distributions
        # This must be done before computing probabilities

        path_gains_list = []  # Store dr.auto.ad.Float values
        exp_gains_list = []  # Store exp(path_gain) to avoid recomputing
        valid_paths_list = []  # Store validity booleans

        # Predicted softmax denominator (differentiable through Dr.Jit)
        sum_exp_predicted = dr.auto.ad.Float(0.0)

        # Target softmax denominator (constant, no gradients needed)
        sum_exp_target = 0.0  # Regular Python float

        for rx_idx in range(num_sample_points):
            # Extract channel coefficients for this receiver
            h_real_rx = h_real[rx_idx : rx_idx + 1, ...]
            h_imag_rx = h_imag[rx_idx : rx_idx + 1, ...]

            # Compute path gain in LINEAR scale
            path_gain_linear = dr.sum(dr.sum(cpx_abs_square((h_real_rx, h_imag_rx))))

            # Check validity: significant signal exists
            # This removes extremely low power paths from the optimization -> avoids gradient explosion
            has_valid_path = path_gain_linear > 1e-20

            # Linear values are ~1e-14, which causes gradient issues
            # dB values are -140 to -90 (better scale)
            path_gain_db = (
                10.0 * dr.log(dr.maximum(path_gain_linear, 1e-30)) / dr.log(10.0)
            )

            # Scale dB for softmax: divide by 10 → values become -14 to -9
            path_gain_scaled = path_gain_db / dr.auto.ad.Float(10.0)

            # Compute exp for softmax on scaled dB
            exp_gain = dr.exp(path_gain_scaled)

            # Store SCALED values for second pass
            path_gains_list.append(path_gain_scaled)
            exp_gains_list.append(exp_gain)
            valid_paths_list.append(has_valid_path)

            # Accumulate softmax denominators
            sum_exp_predicted += exp_gain

            # Target: scale dB the same way
            target_db = float(target_values[rx_idx])
            target_scaled = target_db / 10.0  # -140 dB → -14
            sum_exp_target += np.exp(target_scaled)

        # Step 2: SECOND PASS - Compute loss based on loss_type
        if loss_type == "mse":
            # MSE Loss with optional power normalization

            if normalize_power:
                # Power-Normalized MSE: Focuses on spatial pattern, not absolute power
                # Normalize both predicted and target to [0, 1] range

                # First pass: Find min/max of predicted values
                min_pred = dr.auto.ad.Float(1e10)
                max_pred = dr.auto.ad.Float(-1e10)

                for rx_idx in range(num_sample_points):
                    path_gain_scaled = path_gains_list[rx_idx]
                    has_valid_path = valid_paths_list[rx_idx]
                    path_gain_db = path_gain_scaled * dr.auto.ad.Float(10.0)

                    # Update min/max only for valid paths
                    min_pred = dr.select(
                        has_valid_path, dr.minimum(min_pred, path_gain_db), min_pred
                    )
                    max_pred = dr.select(
                        has_valid_path, dr.maximum(max_pred, path_gain_db), max_pred
                    )

                # Compute target min/max (constant)
                target_min = float(target_values.min())
                target_max = float(target_values.max())
                target_range = target_max - target_min

                # Prevent division by zero
                pred_range = max_pred - min_pred
                epsilon = dr.auto.ad.Float(1e-6)

                # Second pass: Compute normalized MSE
                mse_loss = dr.auto.ad.Float(0.0)
                valid_count = dr.auto.ad.Float(0.0)

                for rx_idx in range(num_sample_points):
                    path_gain_scaled = path_gains_list[rx_idx]
                    has_valid_path = valid_paths_list[rx_idx]
                    path_gain_db = path_gain_scaled * dr.auto.ad.Float(10.0)

                    # Normalize predicted to [0, 1]
                    pred_normalized = (path_gain_db - min_pred) / (pred_range + epsilon)

                    # Normalize target to [0, 1]
                    target_db_value = float(target_values[rx_idx])
                    if target_range > 1e-6:
                        target_normalized = (
                            target_db_value - target_min
                        ) / target_range
                    else:
                        target_normalized = 0.5  # If uniform target, center at 0.5

                    target_norm_dr = dr.auto.ad.Float(target_normalized)
                    dr.disable_grad(target_norm_dr)

                    # Squared error on normalized values
                    error = pred_normalized - target_norm_dr
                    squared_error = error * error

                    # Apply validity mask
                    contribution = dr.select(
                        has_valid_path, squared_error, dr.auto.ad.Float(0.0)
                    )
                    mse_loss += contribution
                    valid_count += dr.select(
                        has_valid_path, dr.auto.ad.Float(1.0), dr.auto.ad.Float(0.0)
                    )

                normalized_loss = mse_loss / dr.maximum(
                    valid_count, dr.auto.ad.Float(1.0)
                )

            else:
                # Standard MSE: mean((predicted_dB - target_dB)^2)
                # This is spatially aware - directly compares power at each location

                mse_loss = dr.auto.ad.Float(0.0)
                valid_count = dr.auto.ad.Float(0.0)

                for rx_idx in range(num_sample_points):
                    # Retrieve path gain in dB (unscaled)
                    path_gain_scaled = path_gains_list[rx_idx]
                    has_valid_path = valid_paths_list[rx_idx]

                    # Convert back from scaled to actual dB
                    # We scaled by /10, so multiply by 10 to get back to dB
                    path_gain_db = path_gain_scaled * dr.auto.ad.Float(10.0)

                    # Target in dB (as constant)
                    target_db_value = float(target_values[rx_idx])
                    target_db = dr.auto.ad.Float(target_db_value)
                    dr.disable_grad(target_db)  # No gradients for target

                    # Squared error: (predicted - target)^2
                    error = path_gain_db - target_db
                    squared_error = error * error

                    # Apply validity mask
                    contribution = dr.select(
                        has_valid_path, squared_error, dr.auto.ad.Float(0.0)
                    )
                    mse_loss += contribution

                    # Track valid count
                    valid_count += dr.select(
                        has_valid_path, dr.auto.ad.Float(1.0), dr.auto.ad.Float(0.0)
                    )

                # Normalize by number of valid receivers
                normalized_loss = mse_loss / dr.maximum(
                    valid_count, dr.auto.ad.Float(1.0)
                )

        elif loss_type == "cross_entropy":
            # Cross-entropy: CE = -sum(p_target * log(p_predicted))
            # where p = softmax(x) = exp(x) / sum(exp(x))

            cross_entropy_loss = dr.auto.ad.Float(0.0)
            valid_count = dr.auto.ad.Float(0.0)

            # CRITICAL: epsilon must be dr.auto.ad.Float for proper AD
            epsilon = dr.auto.ad.Float(1e-10)

            for rx_idx in range(num_sample_points):
                # Retrieve stored values (all dr.auto.ad.Float types)
                exp_gain = exp_gains_list[rx_idx]
                has_valid_path = valid_paths_list[rx_idx]

                # Compute PREDICTED probability: softmax(path_gain)
                predicted_prob = exp_gain / sum_exp_predicted

                # Compute TARGET probability: softmax(target_scaled)
                target_db = float(target_values[rx_idx])
                target_scaled = target_db / 10.0  # Match the scaling from first pass
                target_prob = np.exp(target_scaled) / sum_exp_target

                # Cross-entropy contribution: -p_target * log(p_predicted + epsilon)
                target_prob_dr = dr.auto.ad.Float(target_prob)
                dr.disable_grad(target_prob_dr)  # Mark as constant (no gradients)
                # Adding epsilon (dr.auto.ad.Float) prevents log(0) = -inf
                log_pred = dr.log(predicted_prob + epsilon)

                # Now both operands are Dr.Jit types - AD graph preserved
                ce_term = -target_prob_dr * log_pred

                # Apply validity mask using dr.select (differentiable conditional)
                # Only include this receiver if it has a valid path
                contribution = dr.select(has_valid_path, ce_term, dr.auto.ad.Float(0.0))
                cross_entropy_loss += contribution

                # Track number of valid receivers
                valid_count += dr.select(
                    has_valid_path, dr.auto.ad.Float(1.0), dr.auto.ad.Float(0.0)
                )

            # Step 3: Normalize by number of valid receivers
            # Prevents loss magnitude from depending on number of valid paths
            normalized_loss = cross_entropy_loss / dr.maximum(
                valid_count, dr.auto.ad.Float(1.0)
            )

        # Changing the loss type to huber to see if I can make the loss function more robust against outliers
        elif loss_type == "huber":

            huber_loss = dr.auto.ad.Float(0.0)
            valid_count = dr.auto.ad.Float(0.0)
            delta = dr.auto.ad.Float(1.5)  # Huber loss threshold

            for rx_idx in range(num_sample_points):
                # Retrieve path gain in dB (unscaled)
                path_gain_scaled = path_gains_list[rx_idx]
                has_valid_path = valid_paths_list[rx_idx]

                # Convert back from scaled to actual dB
                # We scaled by /10, so multiply by 10 to get back to dB
                path_gain_db = path_gain_scaled * dr.auto.ad.Float(10.0)

                # Target in dB (as constant)
                target_db_value = float(target_values[rx_idx])
                target_db = dr.auto.ad.Float(target_db_value)
                dr.disable_grad(target_db)  # No gradients for target

                # Huber loss: differentiable using dr.select instead of Python if/else
                # Huber(a) = 0.5 * a^2                     if |a| <= delta
                #          = delta * (|a| - 0.5 * delta)   if |a| > delta
                error = path_gain_db - target_db
                abs_error = dr.abs(error)

                # Compute both branches
                quadratic = dr.auto.ad.Float(0.5) * error * error
                linear = delta * (abs_error - dr.auto.ad.Float(0.5) * delta)

                # Select based on threshold (differentiable)
                huber_error = dr.select(abs_error <= delta, quadratic, linear)

                # Apply validity mask (same as the other loss functions)
                contribution = dr.select(
                    has_valid_path, huber_error, dr.auto.ad.Float(0.0)
                )
                huber_loss += contribution

                # Track valid count
                valid_count += dr.select(
                    has_valid_path, dr.auto.ad.Float(1.0), dr.auto.ad.Float(0.0)
                )

            # Normalize by number of valid receivers
            normalized_loss = huber_loss / dr.maximum(
                valid_count, dr.auto.ad.Float(1.0)
            )

        else:
            raise ValueError(
                f"Unknown loss_type: {loss_type}. Must be 'mse' or 'cross_entropy'"
            )

        # DEBUG Statements
        if loss_type == "cross_entropy":
            print(f"  [DEBUG] Raw CE loss: {cross_entropy_loss}")
        elif loss_type == "mse":
            print(f"  [DEBUG] Raw MSE loss: {mse_loss}")

        return normalized_loss

    # PyTorch parameters (each as a separate 0-D tensor for scalar conversion)
    # Using 0-D tensors (scalars) - this is the ONLY pattern that works with @dr.wrap
    # 1-D tensors lose gradients during indexing
    boresight_x = torch.tensor(
        initial_boresight[0], device="cuda", dtype=torch.float32, requires_grad=True
    )
    boresight_y = torch.tensor(
        initial_boresight[1], device="cuda", dtype=torch.float32, requires_grad=True
    )
    boresight_z = torch.tensor(
        initial_boresight[2], device="cuda", dtype=torch.float32, requires_grad=True
    )

    # Also attempting to optimize the transmitter's X and Y positions
    # tx_x = torch.tensor(
    #    [x_start_position], device="cuda", dtype=torch.float32, requires_grad=True
    # )
    # tx_y = torch.tensor(
    #    [y_start_position], device="cuda", dtype=torch.float32, requires_grad=True
    # )

    # Optimizer: Adam shows the best performance
    # Optimize all three boresight parameters together
    optimizer = torch.optim.Adam(
        [boresight_x, boresight_y, boresight_z], lr=learning_rate
    )

    # Learning rate scheduler: required to jump out of local minima for difficult loss surfaces...
    use_scheduler = num_iterations >= 50
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,  # Reduce LR by half
            patience=20,  # Wait 20 iterations without improvement (increased from 10)
            threshold=0.0005,  # Require 0.05% improvement (relaxed from 0.01%)
            min_lr=1e-3,  # Don't let LR get too small (increased from 1e-4)
        )
        if verbose:
            print(f"Learning rate scheduler enabled (patience=20, threshold=0.05%)")

    # Tracking
    loss_history = []
    boresight_history = [
        [initial_boresight[0], initial_boresight[1], initial_boresight[2]]
    ]
    # tx_history = [[x_start_position, y_start_position]]  # Track TX position
    gradient_history = []  # Track gradient norms for diagnostics

    best_loss = float("inf")
    best_boresight = initial_boresight.copy()

    # Setup frame saving if requested
    if save_radiomap_frames:
        import os
        os.makedirs(output_dir, exist_ok=True)
        from sionna.rt import RadioMapSolver
        rm_solver = RadioMapSolver()
        print(f"RadioMap frames will be saved to: {output_dir}")
        print(f"Saving every {frame_save_interval} iteration(s)")

    start_time = time.time()

    # Run the optimization for the specified number of iterations
    for iteration in range(num_iterations):

        # if verbose:
        #    print(f"\nIteration {iteration+1}/{num_iterations}:")
        #    print(
        #        f"  Boresight: ({boresight_x.item():.2f}, {boresight_y.item():.2f}, {boresight_z.item():.2f})"
        #    )

        # Forward pass
        loss = compute_loss(boresight_x, boresight_y, boresight_z)

        # Backward pass (using AD)
        # The gradients are backpropagated through the scene
        optimizer.zero_grad()
        loss.backward()

        # Track gradient norm for diagnostics
        grad_x_val = boresight_x.grad.item() if boresight_x.grad is not None else 0.0
        grad_y_val = boresight_y.grad.item() if boresight_y.grad is not None else 0.0
        grad_z_val = boresight_z.grad.item() if boresight_z.grad is not None else 0.0
        # grad_tx_x_val = tx_x.grad.item() if tx_x.grad is not None else 0.0
        # grad_tx_y_val = tx_y.grad.item() if tx_y.grad is not None else 0.0

        # Update the norm (including transmitter XY position)
        grad_norm = np.sqrt(
            grad_x_val**2
            + grad_y_val**2
            + grad_z_val**2
            # + grad_tx_x_val**2
            # + grad_tx_y_val**2
        )
        gradient_history.append(grad_norm)

        if verbose:
            print(
                f"  Gradients: dx={grad_x_val:+.3e}, dy={grad_y_val:+.3e}, dz={grad_z_val:+.3e} (norm={grad_norm:.3e})"
            )
            print(f"  Loss: {loss.item()}, RMSE: {np.sqrt(loss.item())} dB")

        # Update
        optimizer.step()

        # Save RadioMap frame if requested
        if save_radiomap_frames and (iteration % frame_save_interval == 0 or iteration == num_iterations - 1):
            # Apply current boresight to scene (temporarily)
            tx = scene.get(tx_name)
            current_boresight = [boresight_x.item(), boresight_y.item(), boresight_z.item()]
            tx.look_at(mi.Point3f(float(current_boresight[0]), float(current_boresight[1]), float(current_boresight[2])))

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
            scheduler.step(loss.item())

        # Apply constraints on boresight Z:
        # Must be below TX height (cannot point upward)
        with torch.no_grad():
            boresight_z.clamp_(min=-10.0, max=tx_height - 0.1)

            # Project TX position to stay within roof polygon
            # proj_x, proj_y = tx_placement.project_to_polygon(
            #    tx_x.item(),
            #    tx_y.item()
            # )
            # tx_x.fill_(proj_x)
            # tx_y.fill_(proj_y)

        # Track
        loss_history.append(loss.item())
        boresight_history.append(
            [boresight_x.item(), boresight_y.item(), boresight_z.item()]
        )
        # tx_history.append([tx_x.item(), tx_y.item()])

        # Updates the ideal parameters if the loss is the lowest to date
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_boresight = [boresight_x.item(), boresight_y.item(), boresight_z.item()]
            # best_tx_xy = [tx_x.item(), tx_y.item()]

    # Save the elapsed time for metrics
    elapsed_time = time.time() - start_time

    # Cleanup receivers
    # This is required to make sure the RadioMap calculation doesn't have any extraneous recievers
    for rx_name in rx_names:
        if rx_name in [obj.name for obj in scene.receivers.values()]:
            scene.remove(rx_name)

    # Set up final scene
    tx = scene.get(tx_name)

    # Set final boresight direction (non-AD) - use pure Python floats
    # Failing to do this can cause issues with the RadioMap solver...
    tx.look_at(
        mi.Point3f(
            float(best_boresight[0]), float(best_boresight[1]), float(best_boresight[2])
        )
    )

    # Sets the final Tx position
    # tx.position = mi.Point3f(float(best_tx_xy[0]), float(best_tx_xy[1]), float(building["z_height"]))

    if verbose:
        print(f"\n{'='*70}")
        print("Optimization Complete!")
        print(f"{'='*70}")
        print(
            f"Best boresight: ({best_boresight[0]:.1f}, {best_boresight[1]:.1f}, {best_boresight[2]:.1f})"
        )
        # print(
        #    f"Best tx_position: ({best_tx_xy[0]}, {best_tx_xy[1]}, {building['z_height']})"
        # )
        print(f"Best loss: {best_loss:.2f}")
        print(f"Best RMSE: {np.sqrt(best_loss):.2f} dB")
        print(f"Total time: {elapsed_time:.1f}s")
        print(f"Time per iteration: {elapsed_time/num_iterations:.2f}s")
        print(f"{'='*70}\n")

    return best_boresight, loss_history, boresight_history, gradient_history


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

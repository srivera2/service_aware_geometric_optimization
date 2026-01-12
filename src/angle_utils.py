"""
Angle Conversion Utilities for Antenna Orientation

This module provides utilities for converting between different angle representations:
- Azimuth/Elevation (standard antenna pointing angles)
- Yaw/Pitch/Roll (Mitsuba orientation representation)

Convention:
-----------
Azimuth (horizontal angle):
    - 0° = East (+X axis)
    - 90° = North (+Y axis)
    - Counter-clockwise rotation when viewed from above
    - Range: [0°, 360°]

Elevation (vertical angle):
    - 0° = Horizontal (parallel to ground plane)
    - Negative = Downward tilt (typical for base stations)
    - Positive = Upward tilt
    - Range: [-90°, +90°]

Mitsuba Orientation:
    - Yaw: Rotation around Z-axis (controls azimuth)
    - Pitch: Rotation around Y-axis (controls elevation)
    - Roll: Rotation around X-axis (always 0 for antenna pointing)
"""

import numpy as np


def azimuth_elevation_to_yaw_pitch(azimuth_deg, elevation_deg):
    """
    Convert azimuth and elevation angles to Mitsuba yaw and pitch (in radians).

    Roll is always 0 for antenna orientation and is handled separately.

    Parameters:
    -----------
    azimuth_deg : float
        Azimuth angle in degrees (0° = East, counter-clockwise)
    elevation_deg : float
        Elevation angle in degrees (0° = horizontal, negative = downward)

    Returns:
    --------
    yaw_rad : float
        Yaw angle in radians
    pitch_rad : float
        Pitch angle in radians

    Examples:
    ---------
    >>> # Point East with 10° downward tilt
    >>> yaw, pitch = azimuth_elevation_to_yaw_pitch(0, -10)
    >>> # Point North with 20° downward tilt
    >>> yaw, pitch = azimuth_elevation_to_yaw_pitch(90, -20)
    """
    # Convert degrees to radians
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)

    # Yaw: Direct mapping from azimuth
    yaw_rad = azimuth_rad

    # Pitch: Negative elevation (positive pitch tilts up in Mitsuba)
    pitch_rad = -elevation_rad

    return yaw_rad, pitch_rad


def yaw_pitch_to_azimuth_elevation(yaw_rad, pitch_rad):
    """
    Convert Mitsuba yaw and pitch back to azimuth and elevation angles.

    This is the inverse of azimuth_elevation_to_yaw_pitch().

    Parameters:
    -----------
    yaw_rad : float
        Yaw angle in radians
    pitch_rad : float
        Pitch angle in radians

    Returns:
    --------
    azimuth_deg : float
        Azimuth angle in degrees [0, 360)
    elevation_deg : float
        Elevation angle in degrees [-90, 90]

    Examples:
    ---------
    >>> yaw, pitch = azimuth_elevation_to_yaw_pitch(45, -15)
    >>> az, el = yaw_pitch_to_azimuth_elevation(yaw, pitch)
    >>> print(f"Azimuth: {az:.1f}°, Elevation: {el:.1f}°")
    Azimuth: 45.0°, Elevation: -15.0°
    """
    # Convert back to degrees
    azimuth_deg = np.degrees(yaw_rad)
    elevation_deg = -np.degrees(pitch_rad)  # Flip sign back

    # Normalize azimuth to [0, 360)
    azimuth_deg = azimuth_deg % 360.0

    # Clamp elevation to [-90, 90]
    elevation_deg = np.clip(elevation_deg, -90.0, 90.0)

    return azimuth_deg, elevation_deg


def compute_initial_angles_from_position(tx_position, look_at_position, verbose=False):
    """
    Compute azimuth and elevation angles from TX position to a look_at point.

    This is useful for converting legacy look_at positions to angle representation.

    Parameters:
    -----------
    tx_position : array-like
        [x, y, z] position of transmitter
    look_at_position : array-like
        [x, y, z] position of look_at target point
    verbose : bool, optional
        If True, print detailed computation steps for debugging

    Returns:
    --------
    azimuth_deg : float
        Azimuth angle in degrees
    elevation_deg : float
        Elevation angle in degrees

    Examples:
    ---------
    >>> # TX at origin, looking at point 100m East and 10m down
    >>> az, el = compute_initial_angles_from_position([0, 0, 20], [100, 0, 10])
    >>> print(f"Azimuth: {az:.1f}°, Elevation: {el:.1f}°")
    """
    tx_x, tx_y, tx_z = tx_position
    print(f"Tx Position: {tx_position}")
    target_x, target_y, target_z = look_at_position
    print(f"look_at_position: {look_at_position}")

    # Compute direction vector
    dx = target_x - tx_x
    dy = target_y - tx_y
    dz = target_z - tx_z

    # Horizontal distance
    horizontal_dist = np.sqrt(dx**2 + dy**2)

    # Azimuth: angle in XY plane (0° = East, counter-clockwise)
    azimuth_rad = np.arctan2(dy, dx)
    azimuth_deg = np.degrees(azimuth_rad)
    if azimuth_deg < 0:
        azimuth_deg += 360.0

    # Elevation: angle from horizontal plane
    elevation_rad = np.arctan2(dz, horizontal_dist)
    elevation_deg = np.degrees(elevation_rad)

    if verbose:
        print(f"  Direction vector: dx={dx:.2f}m, dy={dy:.2f}m, dz={dz:.2f}m")
        print(f"  Horizontal distance: {horizontal_dist:.2f}m")
        print(f"  Azimuth calculation: arctan2({dy:.2f}, {dx:.2f}) = {azimuth_deg:.2f}°")
        print(f"  Elevation calculation: arctan2({dz:.2f}, {horizontal_dist:.2f}) = {elevation_deg:.2f}°")

    return azimuth_deg, elevation_deg


def normalize_azimuth(azimuth_deg):
    """
    Normalize azimuth angle to [0, 360) range.

    Parameters:
    -----------
    azimuth_deg : float
        Azimuth angle in degrees

    Returns:
    --------
    normalized_azimuth : float
        Azimuth normalized to [0, 360) range
    """
    return azimuth_deg % 360.0


def clamp_elevation(elevation_deg, min_deg=-90.0, max_deg=90.0):
    """
    Clamp elevation angle to valid range.

    Parameters:
    -----------
    elevation_deg : float
        Elevation angle in degrees
    min_deg : float
        Minimum elevation (default: -90°)
    max_deg : float
        Maximum elevation (default: +90°)

    Returns:
    --------
    clamped_elevation : float
        Elevation clamped to [min_deg, max_deg] range
    """
    return np.clip(elevation_deg, min_deg, max_deg)


def poisson_disk_sampling_2d(
    points_pool,
    num_samples,
    min_distance=None,
    seed=None,
    max_attempts=30
):
    """
    Poisson disk sampling from a pool of candidate points

    Uses Bridson's algorithm adapted for discrete point sets.
    Ensures minimum distance between selected points for uniform coverage.
    This provides better spatial uniformity than random sampling (no clustering)
    while being more flexible than grid sampling (avoids buildings).

    Parameters:
    -----------
    points_pool : np.ndarray
        Nx2 array of candidate (x, y) points to sample from
    num_samples : int
        Target number of samples to select
    min_distance : float, optional
        Minimum distance between samples
        If None, auto-calculated based on area and num_samples
    seed : int, optional
        Random seed for reproducibility
    max_attempts : int
        Maximum attempts to find valid point per iteration

    Returns:
    --------
    selected_points : np.ndarray
        Mx2 array of selected points (M <= num_samples)
    """
    if len(points_pool) == 0:
        return np.array([]).reshape(0, 2)

    if len(points_pool) <= num_samples:
        return points_pool.copy()

    # Auto-calculate min_distance if not provided
    if min_distance is None:
        # Estimate based on area and desired density
        x_min, x_max = np.min(points_pool[:, 0]), np.max(points_pool[:, 0])
        y_min, y_max = np.min(points_pool[:, 1]), np.max(points_pool[:, 1])
        area = (x_max - x_min) * (y_max - y_min)
        # Target: fill area with circles of radius r
        # num_samples ≈ area / (π * r²)
        # r = sqrt(area / (π * num_samples))
        min_distance = np.sqrt(area / (np.pi * num_samples)) * 0.8  # 0.8 for some overlap tolerance

    rng = np.random.RandomState(seed) if seed is not None else np.random

    selected = []
    available_indices = set(range(len(points_pool)))

    # Start with random point
    first_idx = rng.choice(list(available_indices))
    selected.append(points_pool[first_idx])
    available_indices.remove(first_idx)

    # Active list for progressive sampling
    active_list = [0]  # Index into selected list

    while active_list and len(selected) < num_samples and available_indices:
        # Pick random point from active list
        active_idx = rng.choice(len(active_list))
        base_point_idx = active_list[active_idx]
        base_point = selected[base_point_idx]

        # Try to find a point near this one
        found = False
        for _ in range(max_attempts):
            if not available_indices:
                break

            # Pick random available point
            candidate_idx = rng.choice(list(available_indices))
            candidate = points_pool[candidate_idx]

            # Check distance to base point
            dist = np.linalg.norm(candidate - base_point)

            # Check if candidate is far enough from all selected points
            if dist >= min_distance:
                valid = True
                for sel_point in selected:
                    if np.linalg.norm(candidate - sel_point) < min_distance:
                        valid = False
                        break

                if valid:
                    selected.append(candidate)
                    available_indices.remove(candidate_idx)
                    active_list.append(len(selected) - 1)
                    found = True
                    break

        # If no valid point found near base_point, remove it from active list
        if not found:
            active_list.pop(active_idx)

    # If we didn't get enough points via Poisson disk, fill remainder with random points
    if len(selected) < num_samples and available_indices:
        remaining_needed = num_samples - len(selected)
        remaining_indices = rng.choice(
            list(available_indices),
            size=min(remaining_needed, len(available_indices)),
            replace=False
        )
        for idx in remaining_indices:
            selected.append(points_pool[idx])

    return np.array(selected)

"""
Zone Validation Module for TX Optimization

This module provides functions to validate whether a target zone is suitable
for optimization by checking signal coverage characteristics.
"""

import numpy as np
from sionna.rt import RadioMapSolver
import mitsuba as mi
from angle_utils import azimuth_elevation_to_yaw_pitch, compute_initial_angles_from_position


def validate_zone_coverage(
    scene,
    tx_name,
    zone_mask,
    zone_params,
    map_config,
    p10_min_dbm=-140.0,
    p10_max_dbm=-80.0,
    p90_min_dbm=-130.0,
    min_percentile_range_db=15.0,
    median_max_dbm=-80.0,
    verbose=True
):
    """
    Validate whether a target zone provides a good scenario for optimization.

    This function performs a quick RadioMapSolver simulation to check if the zone:
    1. Has 10th percentile coverage in valid range (not too weak, not too strong)
    2. Has 90th percentile coverage >= threshold (not too weak/dead)
    3. Has adequate dynamic range between 10th and 90th percentiles

    Parameters:
    -----------
    scene : sionna.rt.Scene
        The scene with transmitter already placed
    tx_name : str
        Name of the transmitter
    zone_mask : np.ndarray
        Binary mask defining the target zone
    zone_params : dict
        Zone parameters (must contain 'center', 'width', 'height')
    map_config : dict
        Map configuration with 'center', 'size', 'cell_size'
    p10_min_dbm : float
        Minimum acceptable 10th percentile power (default: -140 dBm)
        Zones with p10 < this are rejected as "too weak/dead"
    p10_max_dbm : float
        Maximum acceptable 10th percentile power (default: -80 dBm)
        Zones with p10 > this are rejected as "too strong"
    p90_min_dbm : float
        Minimum acceptable 90th percentile power (default: -130 dBm)
        Zones with p90 < this are rejected as "too weak/dead"
    min_percentile_range_db : float
        Minimum required range between p90 and p10 (default: 15 dB)
        Zones with (p90 - p10) < this are rejected as "too uniform"
    verbose : bool
        Print validation details

    Returns:
    --------
    is_valid : bool
        True if zone passes all validation checks
    validation_stats : dict
        Dictionary with validation metrics:
        - 'min_power_dbm': Minimum signal power in zone
        - 'max_power_dbm': Maximum signal power in zone
        - 'mean_power_dbm': Mean signal power in zone
        - 'median_power_dbm': Median signal power in zone
        - 'std_power_db': Standard deviation of signal in dB
        - 'signal_range_db': Max - Min signal variation
        - 'p10_power_dbm': 10th percentile signal power
        - 'p90_power_dbm': 90th percentile signal power
        - 'percentile_range_db': Range between p90 and p10
        - 'failed_checks': List of failed validation checks
        - 'reason': Human-readable rejection reason (if rejected)
    """

    if verbose:
        print(f"\n{'='*70}")
        print("ZONE VALIDATION")
        print(f"{'='*70}")

    # Get transmitter and compute naive baseline orientation
    tx = scene.get(tx_name)
    tx_position = tx.position.numpy().flatten().tolist()

    # Compute initial angles pointing at zone center
    target_z = map_config.get('target_height', 1.5)
    look_at_xyz = list(zone_params['center'])[:2] + [target_z]
    initial_azimuth, initial_elevation = compute_initial_angles_from_position(
        tx_position,
        look_at_xyz,
        verbose=False
    )

    # Set transmitter orientation to naive baseline
    yaw_rad, pitch_rad = azimuth_elevation_to_yaw_pitch(initial_azimuth, initial_elevation)
    tx.orientation = mi.Point3f(float(yaw_rad), float(pitch_rad), 0.0)

    if verbose:
        print(f"TX Position: ({tx_position[0]:.1f}, {tx_position[1]:.1f}, {tx_position[2]:.1f})")
        print(f"Zone Center: ({look_at_xyz[0]:.1f}, {look_at_xyz[1]:.1f}, {look_at_xyz[2]:.1f})")
        print(f"Naive Orientation: Az={initial_azimuth:.1f}°, El={initial_elevation:.1f}°")
        print(f"\nRunning quick coverage check...")

    # Run RadioMapSolver with reduced samples for speed
    rm_solver = RadioMapSolver()
    rm = rm_solver(
        scene,
        max_depth=5,
        samples_per_tx=int(2e8),  # Reduced samples for faster validation
        cell_size=map_config['cell_size'],
        center=map_config['center'],
        orientation=[0, 0, 0],
        size=map_config['size'],
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        diffraction=True,
        edge_diffraction=True,
        refraction=False,
        stop_threshold=None,
    )

    # Extract signal strength in zone
    rss_watts = rm.rss.numpy()[0, :, :]
    signal_strength_dbm = 10.0 * np.log10(rss_watts + 1e-30) + 30.0
    zone_power_dbm = signal_strength_dbm[zone_mask == 1.0]

    # Compute statistics
    min_power_dbm = np.min(zone_power_dbm)
    max_power_dbm = np.max(zone_power_dbm)
    mean_power_dbm = np.mean(zone_power_dbm)
    median_power_dbm = np.median(zone_power_dbm)
    std_power_db = np.std(zone_power_dbm)
    signal_range_db = max_power_dbm - min_power_dbm
    p10_power_dbm = np.percentile(zone_power_dbm, 10)
    p90_power_dbm = np.percentile(zone_power_dbm, 90)
    percentile_range_db = p90_power_dbm - p10_power_dbm

    # Validation checks
    failed_checks = []
    is_valid = True
    rejection_reason = None

    # Check 1: 10th percentile lower bound (ensure not too weak/dead)
    if p10_power_dbm < p10_min_dbm:
        is_valid = False
        failed_checks.append('p10_too_weak')
        rejection_reason = f"10th percentile too weak: {p10_power_dbm:.1f} dBm < {p10_min_dbm:.1f} dBm threshold"

    # Check 2: 10th percentile upper bound (ensure not too strong)
    if p10_power_dbm > p10_max_dbm:
        is_valid = False
        failed_checks.append('p10_too_strong')
        if rejection_reason is None:
            rejection_reason = f"10th percentile too strong: {p10_power_dbm:.1f} dBm > {p10_max_dbm:.1f} dBm threshold"

    # Check 3: 90th percentile (ensure not too weak/dead)
    if p90_power_dbm < p90_min_dbm:
        is_valid = False
        failed_checks.append('p90_too_weak')
        if rejection_reason is None:
            rejection_reason = f"90th percentile too weak: {p90_power_dbm:.1f} dBm < {p90_min_dbm:.1f} dBm threshold"

    # Check 4: Percentile range (dynamic range between p10 and p90)
    if percentile_range_db < min_percentile_range_db:
        is_valid = False
        failed_checks.append('insufficient_range')
        if rejection_reason is None:
            rejection_reason = f"Percentile range too small: {percentile_range_db:.1f} dB < {min_percentile_range_db:.1f} dB threshold"

    if median_max_dbm < median_power_dbm:
        is_valid = False
        failed_checks.append('median is too high')
        if rejection_reason is None:
            rejection_reason = f"Median is greater than {median_max_dbm} dBm"

    # Prepare validation stats
    validation_stats = {
        'min_power_dbm': min_power_dbm,
        'max_power_dbm': max_power_dbm,
        'mean_power_dbm': mean_power_dbm,
        'median_power_dbm': median_power_dbm,
        'std_power_db': std_power_db,
        'signal_range_db': signal_range_db,
        'p10_power_dbm': p10_power_dbm,
        'p90_power_dbm': p90_power_dbm,
        'percentile_range_db': percentile_range_db,
        'failed_checks': failed_checks,
        'reason': rejection_reason if not is_valid else "Zone passed all validation checks"
    }

    if verbose:
        print(f"\nCoverage Statistics:")
        print(f"  Min Power:      {min_power_dbm:>8.2f} dBm")
        print(f"  10th %ile:      {p10_power_dbm:>8.2f} dBm")
        print(f"  Median Power:   {median_power_dbm:>8.2f} dBm")
        print(f"  Mean Power:     {mean_power_dbm:>8.2f} dBm")
        print(f"  90th %ile:      {p90_power_dbm:>8.2f} dBm")
        print(f"  Max Power:      {max_power_dbm:>8.2f} dBm")
        print(f"  Std Dev:        {std_power_db:>8.2f} dB")
        print(f"  P10-P90 Range:  {percentile_range_db:>8.2f} dB")

        print(f"\nValidation Checks:")
        p10_check_pass = 'p10_too_weak' not in failed_checks and 'p10_too_strong' not in failed_checks
        print(f"  P10 Range:      {'✓ PASS' if p10_check_pass else '✗ FAIL'} (p10={p10_power_dbm:.1f} dBm, range=[{p10_min_dbm:.1f}, {p10_max_dbm:.1f}] dBm)")
        print(f"  P90 Threshold:  {'✓ PASS' if 'p90_too_weak' not in failed_checks else '✗ FAIL'} (p90={p90_power_dbm:.1f} dBm, min={p90_min_dbm:.1f} dBm)")
        print(f"  Percentile Range: {'✓ PASS' if 'insufficient_range' not in failed_checks else '✗ FAIL'} (range={percentile_range_db:.1f} dB, min={min_percentile_range_db:.1f} dB)")

        print(f"\n{'='*70}")
        if is_valid:
            print("✓ ZONE VALID: Suitable for optimization")
        else:
            print(f"✗ ZONE REJECTED: {rejection_reason}")
        print(f"{'='*70}\n")

    return is_valid, validation_stats


def find_valid_zone(
    scene,
    tx_name,
    tx_position,
    map_config,
    scene_xml_path,
    zone_params_template,
    min_distance=200.0,
    max_distance=400.0,
    max_attempts=20,
    validation_kwargs=None,
    verbose=True
):
    """
    Attempt to find a valid zone by randomly placing it and validating.

    This function repeatedly places zones at random locations around the TX
    until a valid zone is found or max_attempts is reached.

    Parameters:
    -----------
    scene : sionna.rt.Scene
        The scene with transmitter
    tx_name : str
        Name of transmitter
    tx_position : list or array
        [x, y, z] position of transmitter
    map_config : dict
        Map configuration
    scene_xml_path : str
        Path to scene XML file
    zone_params_template : dict
        Template zone parameters (e.g., {'width': 250, 'height': 250})
    min_distance : float
        Minimum distance from TX to zone center (meters)
    max_distance : float
        Maximum distance from TX to zone center (meters)
    max_attempts : int
        Maximum number of zone placement attempts
    validation_kwargs : dict or None
        Optional keyword arguments for validate_zone_coverage()
    verbose : bool
        Print progress

    Returns:
    --------
    zone_mask : np.ndarray or None
        Valid zone mask, or None if no valid zone found
    zone_params : dict or None
        Zone parameters (containing 'center', 'width', 'height'), or None if no valid zone found
    zone_center : list or None
        [x, y] coordinates of zone center, or None if no valid zone found
    validation_stats : dict or None
        Validation statistics for the found zone, or None if no valid zone found
    attempt_count : int
        Number of attempts taken
    """
    from boresight_pathsolver import create_zone_mask

    if validation_kwargs is None:
        validation_kwargs = {}

    if verbose:
        print(f"\n{'='*70}")
        print("SEARCHING FOR VALID ZONE")
        print(f"{'='*70}")
        print(f"Distance range: {min_distance:.0f}m - {max_distance:.0f}m from TX")
        print(f"Max attempts: {max_attempts}")
        print(f"{'='*70}\n")

    for attempt in range(max_attempts):
        if verbose:
            print(f"Attempt {attempt + 1}/{max_attempts}:")

        # Generate random zone placement
        random_angle = np.random.uniform(0, 2 * np.pi)
        random_distance = np.random.uniform(min_distance, max_distance)

        zone_center_x = tx_position[0] + random_distance * np.cos(random_angle)
        zone_center_y = tx_position[1] + random_distance * np.sin(random_angle)

        if verbose:
            print(f"  Zone placed at {random_distance:.1f}m, {np.degrees(random_angle):.1f}° from TX")

        # Create zone parameters with this center
        zone_params = zone_params_template.copy()
        zone_params['center'] = [zone_center_x, zone_center_y]

        # Create zone mask
        try:
            zone_mask, naive_look_at, centriod = create_zone_mask(
                map_config=map_config,
                zone_type='box',
                origin_point=tx_position,
                zone_params=zone_params,
                target_height=map_config.get('target_height', 1.5),
                scene_xml_path=scene_xml_path,
                exclude_buildings=True
            )
        except ValueError as e:
            if verbose:
                print(f"  ✗ Failed to create zone: {e}")
            continue

        # Validate zoneF
        is_valid, validation_stats = validate_zone_coverage(
            scene=scene,
            tx_name=tx_name,
            zone_mask=zone_mask,
            zone_params=zone_params,
            map_config=map_config,
            verbose=verbose,
            **validation_kwargs
        )

        if is_valid:
            if verbose:
                print(f"\n✓ Found valid zone after {attempt + 1} attempt(s)!")
            # Return zone_params directly (no longer wrapping in zone_stats)
            return zone_mask, zone_params, [zone_center_x, zone_center_y], validation_stats, attempt + 1

    if verbose:
        print(f"\n✗ Failed to find valid zone after {max_attempts} attempts")

    return None, None, None, None, max_attempts


def suggest_high_stakes_thresholds(verbose=True):
    """
    Suggest threshold values for creating "high stakes" optimization scenarios.

    High stakes scenarios are challenging but realistic:
    - Weak spots exist (10th percentile <= -80 dBm)
    - Strong spots exist (90th percentile >= -130 dBm)
    - High dynamic range between weak and strong areas (>= 15 dB)

    Returns:
    --------
    thresholds : dict
        Recommended threshold values for validation_kwargs
    """
    thresholds = {
        # P10 LOWER BOUND: Minimum acceptable 10th percentile
        'p10_min_dbm': -140.0,
        # Zones with p10 < -140 dBm are rejected (too weak/dead at weak end)
        # -140 dBm ensures even the weakest areas have some signal
        # Prevents completely dead zones
        # Adjust lower (e.g., -150 dBm) to be more permissive
        # Adjust higher (e.g., -130 dBm) to require stronger weak areas

        # P10 UPPER BOUND: Maximum acceptable 10th percentile
        'p10_max_dbm': -80.0,
        # Zones with p10 > -80 dBm are rejected (too strong overall)
        # -80 dBm ensures weakest 10% of zone has poor-to-moderate coverage
        # Adjust lower (e.g., -90 dBm) for weaker baseline scenarios
        # Adjust higher (e.g., -70 dBm) for stronger baseline scenarios

        # P90 THRESHOLD: Minimum acceptable 90th percentile
        'p90_min_dbm': -130.0,
        # Zones with p90 < -130 dBm are rejected (too weak/dead)
        # -130 dBm ensures strongest 10% of zone has detectable signal
        # Adjust higher (e.g., -120 dBm) for stronger requirements
        # Adjust lower (e.g., -140 dBm) to allow weaker zones

        # PERCENTILE RANGE: Minimum dynamic range between p10 and p90
        'min_percentile_range_db': 15.0,
        # Zones with (p90 - p10) < 15 dB are rejected (too uniform)
        # 15 dB ensures significant variation between weak and strong areas
        # Higher values (20+ dB) = more challenging, complex propagation
        # Lower values (10 dB) = more uniform, less challenging
    }

    if verbose:
        print(f"\n{'='*70}")
        print("HIGH STAKES SCENARIO THRESHOLDS")
        print(f"{'='*70}")
        print("\nRecommended settings for challenging optimization scenarios:")

        print(f"\n1. 10th Percentile Range: [{thresholds['p10_min_dbm']:.1f}, {thresholds['p10_max_dbm']:.1f}] dBm")
        print("   - Zones with p10 outside this range are REJECTED")
        print("   - Lower bound (-140 dBm): Ensures weak areas aren't completely dead")
        print("   - Upper bound (-80 dBm): Ensures weak spots exist for optimization")
        print("   - Adjust to control baseline strength:")
        print("     * Weaker: [-150, -90] dBm")
        print("     * Moderate: [-140, -80] dBm (recommended)")
        print("     * Stronger: [-130, -70] dBm")

        print(f"\n2. 90th Percentile Min: {thresholds['p90_min_dbm']:.1f} dBm")
        print("   - Zones with p90 < this are REJECTED (too weak/dead)")
        print("   - Ensures strong spots exist in the zone")
        print("   - Prevents completely blocked/dead zones")
        print("   - Adjust to control minimum viable signal:")
        print("     * -140 dBm = very permissive (accept very weak zones)")
        print("     * -130 dBm = permissive (recommended)")
        print("     * -120 dBm = strict (require decent signal)")

        print(f"\n3. Percentile Range Min: {thresholds['min_percentile_range_db']:.1f} dB")
        print("   - Zones with (p90 - p10) < this are REJECTED (too uniform)")
        print("   - Ensures significant variation between weak and strong areas")
        print("   - High variation = complex propagation, challenging optimization")
        print("   - Adjust to control scenario difficulty:")
        print("     * 10 dB = moderate variation")
        print("     * 15 dB = high variation (recommended)")
        print("     * 20+ dB = extreme variation (very challenging)")

        print(f"\n{'='*70}")
        print("EXAMPLE SCENARIOS:")
        print(f"{'='*70}")

        print("\nMODERATE Challenge (recommended):")
        print("  p10_min_dbm = -140.0")
        print("  p10_max_dbm = -80.0")
        print("  p90_min_dbm = -130.0")
        print("  min_percentile_range_db = 15.0")
        print("  → Balanced scenarios with clear optimization potential")

        print("\nHIGH Challenge:")
        print("  p10_min_dbm = -145.0  (allow weaker weak areas)")
        print("  p10_max_dbm = -90.0  (weaker baseline)")
        print("  p90_min_dbm = -120.0  (require decent strong spots)")
        print("  min_percentile_range_db = 20.0  (extreme variation)")
        print("  → Difficult scenarios with large coverage gaps")

        print("\nLOW Challenge:")
        print("  p10_min_dbm = -135.0  (require better weak areas)")
        print("  p10_max_dbm = -70.0  (stronger baseline)")
        print("  p90_min_dbm = -140.0  (very permissive)")
        print("  min_percentile_range_db = 10.0  (moderate variation)")
        print("  → Easier scenarios with better baseline coverage")

        print(f"\n{'='*70}\n")

    return thresholds

"""
Diagnostic utilities for radio map optimization
Automatically adapts to loss function type and target distribution
"""

import numpy as np
import matplotlib.pyplot as plt


def analyze_coverage(signal_strength_dBm, target_map, loss_type, map_config):
    """
    Analyze coverage and compare with target based on loss function type

    Parameters:
    -----------
    signal_strength_dBm : np.ndarray
        Achieved signal strength in dBm (2D array)
    target_map : np.ndarray
        Target distribution in dB (2D array)
    loss_type : str
        'mse' or 'cross_entropy'
    map_config : dict
        Map configuration with center, size, etc.

    Returns:
    --------
    dict : Analysis results and interpretation
    """

    # Detect target distribution type
    target_std = np.std(target_map)
    target_range = np.max(target_map) - np.min(target_map)

    if target_range < 5:
        target_type = 'uniform'
    elif target_std > 5:
        # Check if it's Gaussian-like (center hot, edges cold)
        n_y, n_x = target_map.shape
        center_val = target_map[n_y//2, n_x//2]
        corners = [target_map[0, 0], target_map[0, -1],
                   target_map[-1, 0], target_map[-1, -1]]
        avg_corner = np.mean(corners)

        if center_val > avg_corner + 5:
            target_type = 'gaussian'
        else:
            target_type = 'custom'
    else:
        target_type = 'custom'

    # Compute basic statistics
    valid_mask = np.isfinite(signal_strength_dBm)
    valid_signals = signal_strength_dBm[valid_mask]

    stats = {
        'mean': np.mean(valid_signals),
        'std': np.std(valid_signals),
        'min': np.min(valid_signals),
        'max': np.max(valid_signals),
        'range': np.max(valid_signals) - np.min(valid_signals),
        'valid_cells': len(valid_signals),
        'total_cells': signal_strength_dBm.size
    }

    # Target statistics
    target_stats = {
        'mean': np.mean(target_map),
        'std': np.std(target_map),
        'min': np.min(target_map),
        'max': np.max(target_map),
        'range': np.max(target_map) - np.min(target_map)
    }

    # Type-specific analysis
    if target_type == 'uniform':
        analysis = _analyze_uniform(signal_strength_dBm, target_map, stats, target_stats)
    elif target_type == 'gaussian':
        analysis = _analyze_gaussian(signal_strength_dBm, target_map, stats, target_stats)
    else:
        analysis = _analyze_custom(signal_strength_dBm, target_map, stats, target_stats)

    # Add loss-specific metrics
    if loss_type == 'mse':
        # MSE cares about spatial matching
        mse = np.mean((signal_strength_dBm - target_map)**2)
        rmse = np.sqrt(mse)
        analysis['mse'] = mse
        analysis['rmse'] = rmse
    elif loss_type == 'cross_entropy':
        # Cross-entropy cares about distribution matching
        # (Would need softmax computation here for true CE)
        analysis['distribution_match'] = _compare_distributions(
            signal_strength_dBm.flatten(),
            target_map.flatten()
        )

    analysis['target_type'] = target_type
    analysis['loss_type'] = loss_type
    analysis['stats'] = stats
    analysis['target_stats'] = target_stats

    return analysis


def _analyze_uniform(achieved, target, stats, target_stats):
    """Analyze uniform coverage targets"""
    target_value = target_stats['mean']

    interpretation = []

    # Check uniformity
    if stats['std'] < 5:
        interpretation.append(f"✓ Excellent uniformity! (std = {stats['std']:.1f} dB)")
    elif stats['std'] < 10:
        interpretation.append(f"✓ Good uniformity (std = {stats['std']:.1f} dB)")
    else:
        interpretation.append(f"⚠ Poor uniformity (std = {stats['std']:.1f} dB)")

    # Check if mean matches target
    mean_error = abs(stats['mean'] - target_value)
    if mean_error < 3:
        interpretation.append(f"✓ Mean power matches target ({stats['mean']:.1f} vs {target_value:.1f} dB)")
    else:
        interpretation.append(f"⚠ Mean power off by {mean_error:.1f} dB")

    # Check dynamic range
    if stats['range'] < 15:
        interpretation.append(f"✓ Tight dynamic range ({stats['range']:.1f} dB)")
    else:
        interpretation.append(f"⚠ Wide dynamic range ({stats['range']:.1f} dB)")

    return {
        'interpretation': interpretation,
        'success': stats['std'] < 10 and mean_error < 5
    }


def _analyze_gaussian(achieved, target, stats, target_stats):
    """Analyze Gaussian (hot center) coverage targets"""
    n_y, n_x = achieved.shape

    # Center cell
    center_power = achieved[n_y//2, n_x//2]
    target_center = target[n_y//2, n_x//2]

    # Corner cells
    corners = [achieved[0, 0], achieved[0, -1],
               achieved[-1, 0], achieved[-1, -1]]
    avg_corner = np.mean(corners)

    target_corners = [target[0, 0], target[0, -1],
                      target[-1, 0], target[-1, -1]]
    target_avg_corner = np.mean(target_corners)

    # Gradient
    center_to_edge_drop = center_power - avg_corner
    target_drop = target_center - target_avg_corner

    interpretation = []

    # Check hot zone
    if center_to_edge_drop > 10:
        interpretation.append(f"✓ Hot zone detected! ({center_to_edge_drop:.1f} dB drop)")
    elif center_to_edge_drop > 5:
        interpretation.append(f"~ Moderate gradient ({center_to_edge_drop:.1f} dB drop)")
    else:
        interpretation.append(f"✗ No hot zone (only {center_to_edge_drop:.1f} dB drop)")

    # Check center power
    center_error = abs(center_power - target_center)
    if center_error < 5:
        interpretation.append(f"✓ Center power matches target ({center_power:.1f} vs {target_center:.1f} dB)")
    else:
        interpretation.append(f"⚠ Center power off by {center_error:.1f} dB")

    # Check gradient match
    gradient_error = abs(center_to_edge_drop - target_drop)
    if gradient_error < 5:
        interpretation.append(f"✓ Gradient matches target ({center_to_edge_drop:.1f} vs {target_drop:.1f} dB)")
    else:
        interpretation.append(f"⚠ Gradient off by {gradient_error:.1f} dB")

    # Check spatial variation
    if stats['std'] > 5:
        interpretation.append(f"✓ Spatial variation present (std = {stats['std']:.1f} dB)")
    else:
        interpretation.append(f"⚠ Low spatial variation (std = {stats['std']:.1f} dB)")

    return {
        'interpretation': interpretation,
        'center_power': center_power,
        'target_center': target_center,
        'avg_corner': avg_corner,
        'target_avg_corner': target_avg_corner,
        'center_to_edge_drop': center_to_edge_drop,
        'target_drop': target_drop,
        'success': center_to_edge_drop > 10 and center_error < 5
    }


def _analyze_custom(achieved, target, stats, target_stats):
    """Analyze custom distribution targets"""
    # Generic correlation-based analysis
    correlation = np.corrcoef(achieved.flatten(), target.flatten())[0, 1]

    interpretation = []

    if correlation > 0.8:
        interpretation.append(f"✓ Strong spatial correlation with target (r = {correlation:.2f})")
    elif correlation > 0.5:
        interpretation.append(f"~ Moderate spatial correlation (r = {correlation:.2f})")
    else:
        interpretation.append(f"⚠ Weak spatial correlation (r = {correlation:.2f})")

    # Check mean match
    mean_error = abs(stats['mean'] - target_stats['mean'])
    if mean_error < 5:
        interpretation.append(f"✓ Mean matches target")
    else:
        interpretation.append(f"⚠ Mean off by {mean_error:.1f} dB")

    return {
        'interpretation': interpretation,
        'correlation': correlation,
        'success': correlation > 0.7
    }


def _compare_distributions(achieved, target):
    """Compare distribution similarity (for cross-entropy)"""
    # Create histograms
    bins = np.linspace(min(target.min(), achieved.min()),
                      max(target.max(), achieved.max()), 20)

    hist_achieved, _ = np.histogram(achieved, bins=bins, density=True)
    hist_target, _ = np.histogram(target, bins=bins, density=True)

    # KL divergence approximation
    epsilon = 1e-10
    kl_div = np.sum(hist_target * np.log((hist_target + epsilon) / (hist_achieved + epsilon)))

    return {
        'kl_divergence': kl_div,
        'match_quality': 'good' if kl_div < 0.5 else 'poor'
    }


def print_analysis(analysis):
    """Pretty print analysis results"""
    print("="*70)
    print(f"COVERAGE ANALYSIS ({analysis['target_type'].upper()} TARGET, {analysis['loss_type'].upper()} LOSS)")
    print("="*70)

    stats = analysis['stats']
    print(f"\nACHIEVED COVERAGE STATISTICS:")
    print(f"  Mean:   {stats['mean']:.1f} dB")
    print(f"  Std:    {stats['std']:.1f} dB")
    print(f"  Range:  [{stats['min']:.1f}, {stats['max']:.1f}] dB (span: {stats['range']:.1f} dB)")
    print(f"  Valid cells: {stats['valid_cells']}/{stats['total_cells']}")

    if analysis['target_type'] == 'gaussian':
        print(f"\nGAUSSIAN DISTRIBUTION CHECK:")
        print(f"  Center cell power: {analysis['center_power']:.1f} dB")
        print(f"  Average corner power: {analysis['avg_corner']:.1f} dB")
        print(f"  Center-to-edge drop: {analysis['center_to_edge_drop']:.1f} dB")
        print(f"\nTARGET vs ACHIEVED:")
        print(f"  Target center: {analysis['target_center']:.1f} dB  →  Achieved: {analysis['center_power']:.1f} dB")
        print(f"  Target edges: {analysis['target_avg_corner']:.1f} dB  →  Achieved: {analysis['avg_corner']:.1f} dB")
        print(f"  Target drop: {analysis['target_drop']:.1f} dB  →  Achieved: {analysis['center_to_edge_drop']:.1f} dB")

    if analysis['loss_type'] == 'mse':
        print(f"\nMSE METRICS:")
        print(f"  MSE:  {analysis['mse']:.2f}")
        print(f"  RMSE: {analysis['rmse']:.2f} dB")

    print(f"\nINTERPRETATION:")
    for item in analysis['interpretation']:
        print(f"  {item}")

    print("="*70)

    return analysis['success']


def visualize_results(achieved, target, analysis, save_path=None):
    """Create visualization plots based on target type"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    vmin = min(target.min(), achieved.min())
    vmax = max(target.max(), achieved.max())

    # Target map
    ax = axes[0, 0]
    im = ax.imshow(target, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title(f'Target: {analysis["target_type"].title()} Distribution', fontsize=12)
    ax.set_xlabel('X cell index')
    ax.set_ylabel('Y cell index')
    plt.colorbar(im, ax=ax, label='Target Power (dB)')

    # Achieved map
    ax = axes[0, 1]
    im = ax.imshow(achieved, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    stats = analysis['stats']
    ax.set_title(f'Achieved Coverage\n(Mean: {stats["mean"]:.1f} dB, Std: {stats["std"]:.1f} dB)', fontsize=12)
    ax.set_xlabel('X cell index')
    ax.set_ylabel('Y cell index')
    plt.colorbar(im, ax=ax, label='Received Power (dBm)')

    # Target histogram
    ax = axes[1, 0]
    ax.hist(target.flatten(), bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(target), color='red', linestyle='--', label=f'Mean: {np.mean(target):.1f} dB')
    ax.set_xlabel('Target Power (dB)')
    ax.set_ylabel('Number of cells')
    ax.set_title('Target Distribution Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Achieved histogram
    ax = axes[1, 1]
    ax.hist(achieved.flatten(), bins=20, edgecolor='black', alpha=0.7, color='red')
    ax.axvline(stats['mean'], color='darkred', linestyle='--', label=f'Mean: {stats["mean"]:.1f} dB')
    ax.set_xlabel('Achieved Power (dBm)')
    ax.set_ylabel('Number of cells')
    ax.set_title('Achieved Distribution Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

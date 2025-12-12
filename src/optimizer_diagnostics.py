"""
Optimizer diagnostic utilities to determine if optimization is stuck or converged

Helps distinguish between:
- True convergence (found global/local minimum)
- Stuck optimizer (vanishing gradients, learning rate issues, constraints)
- Poorly configured optimization
"""

import numpy as np
import matplotlib.pyplot as plt


def diagnose_optimization(loss_history, boresight_history, gradient_history=None):
    """
    Comprehensive diagnostic analysis of optimization results

    Parameters:
    -----------
    loss_history : list
        Loss values at each iteration
    boresight_history : list
        Boresight coordinates at each iteration [[x,y,z], ...]
    gradient_history : list, optional
        Gradient norms at each iteration

    Returns:
    --------
    dict : Diagnostic results with recommendations
    """

    loss_arr = np.array(loss_history)
    boresight_arr = np.array(boresight_history)

    diagnostics = {
        'converged': False,
        'stuck': False,
        'oscillating': False,
        'issues': [],
        'recommendations': []
    }

    # 1. Check for convergence (loss plateaus)
    if len(loss_arr) >= 5:
        last_5_losses = loss_arr[-5:]
        loss_variance = np.var(last_5_losses)
        loss_change = abs(last_5_losses[-1] - last_5_losses[0])
        relative_change = loss_change / (abs(last_5_losses[0]) + 1e-10)

        if relative_change < 0.001:  # Less than 0.1% change
            diagnostics['converged'] = True
            diagnostics['loss_variance'] = loss_variance
            diagnostics['relative_change'] = relative_change

    # 2. Check for stuck optimization (no movement)
    if len(boresight_arr) >= 5:
        last_5_positions = boresight_arr[-5:]
        position_movement = np.max(np.std(last_5_positions, axis=0))

        if position_movement < 0.01:  # Less than 1cm movement
            diagnostics['stuck'] = True
            diagnostics['position_movement'] = position_movement
            diagnostics['issues'].append(
                f"Parameters barely moving (std: {position_movement:.3f}m)"
            )

    # 3. Check for oscillation (loss going up and down)
    if len(loss_arr) >= 10:
        last_10 = loss_arr[-10:]
        direction_changes = 0
        for i in range(1, len(last_10) - 1):
            if (last_10[i] > last_10[i-1] and last_10[i] > last_10[i+1]) or \
               (last_10[i] < last_10[i-1] and last_10[i] < last_10[i+1]):
                direction_changes += 1

        if direction_changes >= 3:
            diagnostics['oscillating'] = True
            diagnostics['direction_changes'] = direction_changes
            diagnostics['issues'].append(
                f"Loss oscillating ({direction_changes} direction changes in last 10 iterations)"
            )

    # 4. Analyze loss trajectory
    if len(loss_arr) >= 3:
        early_loss = loss_arr[0]
        mid_loss = loss_arr[len(loss_arr)//2]
        final_loss = loss_arr[-1]

        early_improvement = (early_loss - mid_loss) / abs(early_loss)
        late_improvement = (mid_loss - final_loss) / abs(mid_loss)

        diagnostics['early_improvement'] = early_improvement
        diagnostics['late_improvement'] = late_improvement

        if early_improvement < 0.01:
            diagnostics['issues'].append(
                "Little improvement in first half (early_improvement: {:.1%})".format(early_improvement)
            )

        if abs(late_improvement) < 0.001:
            diagnostics['issues'].append(
                "No improvement in second half (late_improvement: {:.1%})".format(late_improvement)
            )

    # 5. Check gradient behavior (if available)
    if gradient_history is not None and len(gradient_history) > 0:
        grad_arr = np.array(gradient_history)
        final_grad = grad_arr[-1] if len(grad_arr) > 0 else 0
        avg_grad = np.mean(grad_arr)

        diagnostics['final_gradient'] = final_grad
        diagnostics['avg_gradient'] = avg_grad

        if final_grad < 1e-6:
            diagnostics['issues'].append(
                f"Vanishing gradients (final: {final_grad:.2e})"
            )
        elif final_grad > avg_grad * 10:
            diagnostics['issues'].append(
                f"Exploding gradients (final: {final_grad:.2e}, avg: {avg_grad:.2e})"
            )

    # 6. Generate recommendations
    if diagnostics['converged'] and not diagnostics['stuck']:
        diagnostics['recommendations'].append(
            "✓ LIKELY CONVERGED: Loss plateaued and parameters stabilized"
        )
        diagnostics['recommendations'].append(
            "  Try: Multi-start optimization to verify this is global minimum"
        )

    if diagnostics['stuck'] and not diagnostics['converged']:
        diagnostics['recommendations'].append(
            "⚠ OPTIMIZER STUCK: Parameters not moving but loss not converged"
        )
        diagnostics['recommendations'].append(
            "  Try: Increase learning rate (2-5x)"
        )
        diagnostics['recommendations'].append(
            "  Try: Different optimizer (switch Adam ↔ SGD with momentum)"
        )
        diagnostics['recommendations'].append(
            "  Try: Check if hitting constraint boundaries"
        )

    if diagnostics['oscillating']:
        diagnostics['recommendations'].append(
            "⚠ OSCILLATING: Loss bouncing around minimum"
        )
        diagnostics['recommendations'].append(
            "  Try: Decrease learning rate (0.3-0.5x)"
        )
        diagnostics['recommendations'].append(
            "  Try: Add learning rate scheduler (exponential decay)"
        )

    if len(loss_arr) >= 3:
        if diagnostics['early_improvement'] < 0.05:
            diagnostics['recommendations'].append(
                "⚠ SLOW START: Poor initial improvement"
            )
            diagnostics['recommendations'].append(
                "  Try: Better initialization (closer to coverage center)"
            )
            diagnostics['recommendations'].append(
                "  Try: Increase learning rate for first few iterations"
            )

    # Check if no issues found
    if len(diagnostics['issues']) == 0 and not diagnostics['converged']:
        diagnostics['recommendations'].append(
            "~ INCONCLUSIVE: Need more iterations or better diagnostics"
        )
        diagnostics['recommendations'].append(
            "  Try: Run for more iterations (2-3x current)"
        )

    return diagnostics


def suggest_multi_start(map_config, num_starts=5):
    """
    Generate multiple starting points for multi-start optimization

    Parameters:
    -----------
    map_config : dict
        Map configuration with center, size, etc.
    num_starts : int
        Number of random starting points to generate

    Returns:
    --------
    list : List of [x, y, z] starting points
    """

    center = map_config['center']
    size = map_config['size']

    # Generate points in a grid around the coverage area
    starts = []

    # 1. Center of coverage (most obvious choice)
    starts.append([center[0], center[1], center[2]])

    # 2. Four corners of coverage area
    half_x = size[0] / 2
    half_y = size[1] / 2

    corners = [
        [center[0] + half_x, center[1] + half_y, center[2]],  # Top-right
        [center[0] - half_x, center[1] + half_y, center[2]],  # Top-left
        [center[0] + half_x, center[1] - half_y, center[2]],  # Bottom-right
        [center[0] - half_x, center[1] - half_y, center[2]],  # Bottom-left
    ]

    starts.extend(corners[:num_starts-1])

    # 3. Random points within extended area (if more needed)
    while len(starts) < num_starts:
        random_x = center[0] + np.random.uniform(-half_x * 1.5, half_x * 1.5)
        random_y = center[1] + np.random.uniform(-half_y * 1.5, half_y * 1.5)
        random_z = center[2] + np.random.uniform(-5, 5)
        starts.append([random_x, random_y, random_z])

    return starts


def compare_learning_rates(loss_histories, learning_rates, boresight_histories=None):
    """
    Compare optimization runs with different learning rates

    Parameters:
    -----------
    loss_histories : dict
        Dictionary of {lr: loss_history} pairs
    learning_rates : list
        List of learning rates tested
    boresight_histories : dict, optional
        Dictionary of {lr: boresight_history} pairs

    Returns:
    --------
    dict : Analysis of which learning rate performed best
    """

    results = {}

    for lr in learning_rates:
        if lr not in loss_histories:
            continue

        loss_hist = np.array(loss_histories[lr])

        results[lr] = {
            'final_loss': loss_hist[-1],
            'best_loss': np.min(loss_hist),
            'total_improvement': loss_hist[0] - loss_hist[-1],
            'relative_improvement': (loss_hist[0] - loss_hist[-1]) / abs(loss_hist[0]),
            'converged': abs(loss_hist[-1] - loss_hist[-5]) / abs(loss_hist[-5]) < 0.001 if len(loss_hist) >= 5 else False
        }

    # Find best learning rate
    best_lr = min(results.keys(), key=lambda lr: results[lr]['final_loss'])

    return {
        'results': results,
        'best_lr': best_lr,
        'best_final_loss': results[best_lr]['final_loss']
    }


def plot_diagnostics(loss_history, boresight_history, gradient_history=None, tx_history=None, save_path=None):
    """
    Visualize optimization diagnostics

    Parameters:
    -----------
    loss_history : list
        Loss values at each iteration
    boresight_history : list
        Boresight coordinates at each iteration
    gradient_history : list, optional
        Gradient norms at each iteration
    tx_history : list, optional
        TX position coordinates at each iteration [[x, y], ...]
    save_path : str, optional
        Path to save figure
    """

    # Determine number of plots
    num_plots = 2  # Loss + Boresight
    if gradient_history is not None:
        num_plots += 1
    if tx_history is not None:
        num_plots += 1

    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))

    # Handle single axis case
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # 1. Loss trajectory
    ax = axes[plot_idx]
    plot_idx += 1
    iterations = np.arange(len(loss_history))
    ax.plot(iterations, loss_history, 'b-', linewidth=2)
    ax.scatter(iterations, loss_history, c='blue', s=30, alpha=0.6)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss vs Iteration', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add convergence indicator
    if len(loss_history) >= 5:
        last_5 = loss_history[-5:]
        if np.std(last_5) < 0.01 * abs(np.mean(last_5)):
            ax.axhspan(min(last_5), max(last_5), alpha=0.2, color='green',
                      label='Converged region')
            ax.legend()

    # 2. Boresight trajectory
    ax = axes[plot_idx]
    plot_idx += 1
    boresight_arr = np.array(boresight_history)

    # Plot X and Y movement
    ax.plot(boresight_arr[:, 0], boresight_arr[:, 1], 'ro-', linewidth=2,
            markersize=6, label='Trajectory')
    ax.scatter(boresight_arr[0, 0], boresight_arr[0, 1],
              c='green', s=200, marker='*', label='Start', zorder=5)
    ax.scatter(boresight_arr[-1, 0], boresight_arr[-1, 1],
              c='red', s=200, marker='*', label='End', zorder=5)

    # Annotate iterations
    for i in [0, len(boresight_arr)//2, len(boresight_arr)-1]:
        ax.annotate(f'iter {i}',
                   (boresight_arr[i, 0], boresight_arr[i, 1]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Boresight X (m)', fontsize=12)
    ax.set_ylabel('Boresight Y (m)', fontsize=12)
    ax.set_title('Boresight Trajectory', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # 3. TX Position trajectory (if available)
    if tx_history is not None and len(tx_history) > 0:
        ax = axes[plot_idx]
        plot_idx += 1
        tx_arr = np.array(tx_history)

        # Plot X and Y movement
        ax.plot(tx_arr[:, 0], tx_arr[:, 1], 'mo-', linewidth=2,
                markersize=6, label='Trajectory')
        ax.scatter(tx_arr[0, 0], tx_arr[0, 1],
                  c='green', s=200, marker='*', label='Start', zorder=5)
        ax.scatter(tx_arr[-1, 0], tx_arr[-1, 1],
                  c='red', s=200, marker='*', label='End', zorder=5)

        # Annotate iterations
        for i in [0, len(tx_arr)//2, len(tx_arr)-1]:
            ax.annotate(f'iter {i}',
                       (tx_arr[i, 0], tx_arr[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.set_xlabel('TX X (m)', fontsize=12)
        ax.set_ylabel('TX Y (m)', fontsize=12)
        ax.set_title('TX Position Trajectory', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    # 4. Gradient history (if available)
    if gradient_history is not None and len(gradient_history) > 0:
        ax = axes[plot_idx]
        plot_idx += 1
        iterations = np.arange(len(gradient_history))
        ax.semilogy(iterations, gradient_history, 'g-', linewidth=2)
        ax.scatter(iterations, gradient_history, c='green', s=30, alpha=0.6)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Gradient Norm (log scale)', fontsize=12)
        ax.set_title('Gradient Magnitude', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add warning zones
        avg_grad = np.mean(gradient_history)
        ax.axhline(1e-6, color='red', linestyle='--', alpha=0.5,
                  label='Vanishing threshold')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def print_diagnostics(diagnostics):
    """
    Pretty print diagnostic results

    Parameters:
    -----------
    diagnostics : dict
        Results from diagnose_optimization()
    """

    print("="*70)
    print("OPTIMIZER DIAGNOSTICS")
    print("="*70)

    # Status indicators
    status_flags = []
    if diagnostics['converged']:
        status_flags.append("✓ CONVERGED")
    if diagnostics['stuck']:
        status_flags.append("⚠ STUCK")
    if diagnostics['oscillating']:
        status_flags.append("⚠ OSCILLATING")

    if status_flags:
        print(f"\nStatus: {' | '.join(status_flags)}")

    # Numerical metrics
    print(f"\nMetrics:")
    if 'relative_change' in diagnostics:
        print(f"  Loss change (last 5 iters): {diagnostics['relative_change']:.4%}")
    if 'position_movement' in diagnostics:
        print(f"  Parameter movement (std): {diagnostics['position_movement']:.4f}m")
    if 'early_improvement' in diagnostics:
        print(f"  Early improvement: {diagnostics['early_improvement']:.2%}")
    if 'late_improvement' in diagnostics:
        print(f"  Late improvement: {diagnostics['late_improvement']:.2%}")
    if 'final_gradient' in diagnostics:
        print(f"  Final gradient norm: {diagnostics['final_gradient']:.2e}")

    # Issues
    if diagnostics['issues']:
        print(f"\nIssues Detected:")
        for issue in diagnostics['issues']:
            print(f"  • {issue}")

    # Recommendations
    if diagnostics['recommendations']:
        print(f"\nRecommendations:")
        for rec in diagnostics['recommendations']:
            print(f"  {rec}")

    print("="*70)


if __name__ == "__main__":
    # Example usage
    print("Optimizer Diagnostics Module")
    print("Import and use: diagnose_optimization(), plot_diagnostics(), print_diagnostics()")

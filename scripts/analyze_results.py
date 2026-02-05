"""
Analysis Script for Boresight Optimization Results

Loads results from validation.py and computes aggregate metrics and CDFs.

Usage:
    python analyze_results.py validation_results.pkl
    python analyze_results.py validation_results.pkl --filter-sampler CDT --filter-freq 5.2e9
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path


def load_results(filepath):
    """Load results from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {data['metadata']['num_results']} results from {filepath}")
    print(f"  Timestamp: {data['metadata']['timestamp']}")
    print(f"  Successful: {data['metadata']['num_successful']}")
    return data['results'], data['metadata']


def compute_metrics(zone_power_initial, zone_power_optimized):
    """Compute all improvement metrics for a single result."""
    return {
        'mean_initial': np.mean(zone_power_initial),
        'mean_optimized': np.mean(zone_power_optimized),
        'mean_improvement': np.mean(zone_power_optimized) - np.mean(zone_power_initial),

        'median_initial': np.median(zone_power_initial),
        'median_optimized': np.median(zone_power_optimized),
        'median_improvement': np.median(zone_power_optimized) - np.median(zone_power_initial),

        'p10_initial': np.percentile(zone_power_initial, 10),
        'p10_optimized': np.percentile(zone_power_optimized, 10),
        'p10_improvement': np.percentile(zone_power_optimized, 10) - np.percentile(zone_power_initial, 10),

        'p90_initial': np.percentile(zone_power_initial, 90),
        'p90_optimized': np.percentile(zone_power_optimized, 90),
        'p90_improvement': np.percentile(zone_power_optimized, 90) - np.percentile(zone_power_initial, 90),

        'std_initial': np.std(zone_power_initial),
        'std_optimized': np.std(zone_power_optimized),
        'std_change': np.std(zone_power_optimized) - np.std(zone_power_initial),

        'min_initial': np.min(zone_power_initial),
        'min_optimized': np.min(zone_power_optimized),
        'min_improvement': np.min(zone_power_optimized) - np.min(zone_power_initial),
    }


def results_to_dataframe(results, filters=None):
    """Convert results dict to pandas DataFrame with computed metrics."""
    rows = []

    for key, data in results.items():
        if data.get('status') != 'success':
            continue

        # Apply filters
        if filters:
            if filters.get('sampler') and data['sampler'] != filters['sampler']:
                continue
            if filters.get('frequency') and data['frequency'] != filters['frequency']:
                continue
            if filters.get('lds') and data['lds'] != filters['lds']:
                continue
            if filters.get('scene') and data['scene_name'] != filters['scene']:
                continue

        metrics = compute_metrics(data['zone_power_initial'], data['zone_power_optimized'])

        row = {
            'key': key,
            'scene': data['scene_name'],
            'sampler': data['sampler'],
            'frequency': data['frequency'],
            'freq_ghz': data['frequency'] / 1e9,
            'lds': data['lds'],
            'combination': f"{data['sampler']}/{data['lds']}",
            'zone_distance': data['zone_distance_from_tx'],
            'initial_azimuth': data['initial_angles'][0],
            'initial_elevation': data['initial_angles'][1],
            'best_azimuth': data['best_angles'][0],
            'best_elevation': data['best_angles'][1],
            'final_loss': data['loss_hist'][-1] if data.get('loss_hist') else None,
            **metrics
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_cdf(data, label, ax=None, **kwargs):
    """Plot CDF of data."""
    if ax is None:
        ax = plt.gca()

    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cdf, label=label, **kwargs)
    return ax


def plot_improvement_cdfs(df, metric='median_improvement', group_by=None, title=None, ax=None):
    """Plot CDFs of improvement metrics, optionally grouped."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if group_by is None:
        plot_cdf(df[metric].values, 'All', ax=ax, linewidth=2)
    else:
        for group_val in df[group_by].unique():
            group_data = df[df[group_by] == group_val][metric].values
            plot_cdf(group_data, str(group_val), ax=ax, linewidth=2)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(f'{metric.replace("_", " ").title()} (dB)')
    ax.set_ylabel('CDF')
    ax.set_title(title or f'CDF of {metric.replace("_", " ").title()}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax


def print_summary_stats(df, metric='median_improvement'):
    """Print summary statistics for a metric."""
    data = df[metric].values
    print(f"\n{metric.replace('_', ' ').title()}:")
    print(f"  Mean:   {np.mean(data):+.2f} dB")
    print(f"  Median: {np.median(data):+.2f} dB")
    print(f"  Std:    {np.std(data):.2f} dB")
    print(f"  Min:    {np.min(data):+.2f} dB")
    print(f"  Max:    {np.max(data):+.2f} dB")
    print(f"  P10:    {np.percentile(data, 10):+.2f} dB")
    print(f"  P90:    {np.percentile(data, 90):+.2f} dB")
    positive_pct = 100 * np.sum(data > 0) / len(data)
    print(f"  Positive: {positive_pct:.1f}% ({np.sum(data > 0)}/{len(data)})")


def generate_report(df, output_dir=None):
    """Generate a full analysis report with plots and statistics."""
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    print(f"\nTotal experiments: {len(df)}")
    print(f"Scenes: {df['scene'].nunique()}")
    print(f"Frequencies: {sorted(df['freq_ghz'].unique())}")
    print(f"Combinations: {sorted(df['combination'].unique())}")

    # Key metrics
    for metric in ['mean_improvement', 'median_improvement', 'p10_improvement', 'p90_improvement']:
        print_summary_stats(df, metric)

    # By combination
    print("\n" + "=" * 80)
    print("BY COMBINATION")
    print("=" * 80)
    combo_stats = df.groupby('combination').agg({
        'median_improvement': ['mean', 'std', 'median'],
        'p10_improvement': ['mean', 'median'],
    }).round(2)
    print(combo_stats.to_string())

    # By frequency
    print("\n" + "=" * 80)
    print("BY FREQUENCY")
    print("=" * 80)
    freq_stats = df.groupby('freq_ghz').agg({
        'median_improvement': ['mean', 'std', 'median'],
        'p10_improvement': ['mean', 'median'],
    }).round(2)
    print(freq_stats.to_string())

    # Generate plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # CDF of median improvement - overall
    plot_improvement_cdfs(df, 'median_improvement', ax=axes[0, 0],
                          title='CDF of Median Improvement (All)')

    # CDF of median improvement - by combination
    plot_improvement_cdfs(df, 'median_improvement', group_by='combination', ax=axes[0, 1],
                          title='CDF of Median Improvement by Combination')

    # CDF of median improvement - by frequency
    plot_improvement_cdfs(df, 'median_improvement', group_by='freq_ghz', ax=axes[0, 2],
                          title='CDF of Median Improvement by Frequency')

    # CDF of P10 improvement
    plot_improvement_cdfs(df, 'p10_improvement', ax=axes[1, 0],
                          title='CDF of P10 Improvement (All)')

    # CDF of P10 improvement - by combination
    plot_improvement_cdfs(df, 'p10_improvement', group_by='combination', ax=axes[1, 1],
                          title='CDF of P10 Improvement by Combination')

    # CDF of mean improvement - by frequency
    plot_improvement_cdfs(df, 'mean_improvement', group_by='freq_ghz', ax=axes[1, 2],
                          title='CDF of Mean Improvement by Frequency')

    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
        fig.savefig(f"{output_dir}/improvement_cdfs.png", dpi=150, bbox_inches='tight')
        df.to_csv(f"{output_dir}/metrics.csv", index=False)
        print(f"\nSaved plots to {output_dir}/improvement_cdfs.png")
        print(f"Saved metrics to {output_dir}/metrics.csv")

    plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Analyze boresight optimization results")
    parser.add_argument('input', help='Input pickle file from validation.py')
    parser.add_argument('--filter-sampler', help='Filter by sampler (Rejection, CDT)')
    parser.add_argument('--filter-freq', type=float, help='Filter by frequency (e.g., 5.2e9)')
    parser.add_argument('--filter-lds', help='Filter by LDS method (Sobol, Halton, Latin)')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots and CSV')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    results, metadata = load_results(args.input)

    filters = {}
    if args.filter_sampler:
        filters['sampler'] = args.filter_sampler
    if args.filter_freq:
        filters['frequency'] = args.filter_freq
    if args.filter_lds:
        filters['lds'] = args.filter_lds

    df = results_to_dataframe(results, filters if filters else None)

    if len(df) == 0:
        print("No results match the specified filters.")
        return

    print(f"\nAnalyzing {len(df)} results")

    if not args.no_plots:
        generate_report(df, args.output_dir)
    else:
        # Just print stats without plots
        for metric in ['mean_improvement', 'median_improvement', 'p10_improvement']:
            print_summary_stats(df, metric)


if __name__ == "__main__":
    main()

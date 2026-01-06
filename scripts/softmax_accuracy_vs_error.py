#!/usr/bin/env python3
"""
Plot model accuracy vs softmax approximation error
to analyze correlation between softmax error and model performance
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sys
import os

# personal style
try:
    plt.style.use("scripts/thesis_plot_styles.mplstyle")
except IOError:
    pass


def load_and_combine_accuracy_csvs(csv_files):
    """Load multiple accuracy CSV files and combine them."""
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source_file'] = csv_file
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Normalize optim_bounds: empty/NaN -> False, 'true' -> True, 'false' -> False
    combined['softmax_optim_bounds'] = combined['softmax_optim_bounds'].fillna('false')
    combined['softmax_optim_bounds'] = combined['softmax_optim_bounds'].astype(str).str.strip().str.lower()
    combined['softmax_optim_bounds'] = combined['softmax_optim_bounds'].apply(
        lambda x: True if x == 'true' else False
    )
    
    combined['gelu_optim_bounds'] = combined['gelu_optim_bounds'].fillna('false')
    combined['gelu_optim_bounds'] = combined['gelu_optim_bounds'].astype(str).str.strip().str.lower()
    combined['gelu_optim_bounds'] = combined['gelu_optim_bounds'].apply(
        lambda x: True if x == 'true' else False
    )
    
    # Normalize backend names
    combined['softmax_backend'] = combined['softmax_backend'].str.replace('_optim', '', regex=False)
    combined['gelu_backend'] = combined['gelu_backend'].str.replace('_optim', '', regex=False)
    
    # Convert to proper types
    combined['softmax_deg'] = pd.to_numeric(combined['softmax_deg'], errors='coerce')
    combined['softmax_seg'] = pd.to_numeric(combined['softmax_seg'], errors='coerce')
    combined['gelu_deg'] = pd.to_numeric(combined['gelu_deg'], errors='coerce')
    combined['gelu_seg'] = pd.to_numeric(combined['gelu_seg'], errors='coerce')
    combined['top1'] = pd.to_numeric(combined['top1'], errors='coerce')
    
    return combined


def main():
    # Default files
    accuracy_csvs = [
        'results/sweep_results_2025-12-10.csv',
        'results/sweep_results_2025-12-14.csv',
    ]
    error_csv = 'results/softmax_sweep_results.csv'
    
    # Parse command line args
    if len(sys.argv) > 1:
        # First arg is error csv, rest are accuracy csvs
        error_csv = sys.argv[1]
        if len(sys.argv) > 2:
            accuracy_csvs = sys.argv[2:]
    
    # Output directory
    output_dir = 'results/softmax_accuracy_vs_error'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df_acc = load_and_combine_accuracy_csvs(accuracy_csvs)
    df_err = pd.read_csv(error_csv)
    
    print(f"Loaded {len(df_acc)} accuracy results from {len(accuracy_csvs)} files")
    print(f"  optim_bounds=True: {(df_acc['softmax_optim_bounds'] == True).sum()}")
    print(f"  optim_bounds=False: {(df_acc['softmax_optim_bounds'] == False).sum()}")
    
    # Filter error data for ibert backend only (excluding baseline)
    df_err_ibert = df_err[(df_err['backend'] == 'ibert') & (df_err['degree'] != 'N/A (I-BERT)')].copy()
    df_err_ibert['degree'] = pd.to_numeric(df_err_ibert['degree'], errors='coerce')
    df_err_ibert['segments'] = pd.to_numeric(df_err_ibert['segments'], errors='coerce')
    
    # Create lookup for error values (keyed by deg, seg, optim_bounds)
    error_lookup = {}
    for _, row in df_err_ibert.iterrows():
        key = (int(row['degree']), int(row['segments']), row['optim_bounds'])
        error_lookup[key] = {
            'max_err': row['max_err_vs_ibert'],
            'mean_err': row['mean_err_vs_ibert'],
        }
    
    # Merge error data into accuracy data
    def get_error(row, error_type):
        key = (row['softmax_deg'], row['softmax_seg'], row['softmax_optim_bounds'])
        return error_lookup.get(key, {}).get(error_type, np.nan)
    
    df_acc['softmax_max_err'] = df_acc.apply(lambda r: get_error(r, 'max_err'), axis=1)
    df_acc['softmax_mean_err'] = df_acc.apply(lambda r: get_error(r, 'mean_err'), axis=1)
    
    # Calculate softmax complexity
    df_acc['softmax_complexity'] = df_acc['softmax_deg'] * df_acc['softmax_seg']
    
    # Drop rows with missing error data
    df_merged = df_acc.dropna(subset=['softmax_max_err', 'top1'])
    
    print(f"Merged: {len(df_merged)} rows with matching error data")
    
    # Color map for segments
    color_map = {8: 'tab:blue', 16: 'tab:green', 32: 'tab:red'}
    # Marker map for degrees
    marker_map = {1: '*', 2: 'o'}
    marker_size = {1: 200, 2: 120}
    
    # I-BERT baseline accuracy
    ibert_baseline_acc = 71.73
    
    # =========================================================================
    # Plot 1a: Top-1 Accuracy vs Max Error
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for deg in [1, 2]:
        for seg in [8, 16, 32]:
            subset = df_merged[(df_merged['softmax_deg'] == deg) & (df_merged['softmax_seg'] == seg)]
            if not subset.empty:
                ax.scatter(
                    subset['softmax_max_err'],
                    subset['top1'],
                    marker=marker_map[deg],
                    s=marker_size[deg],
                    color=color_map[seg],
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f'Deg {deg}, Seg {seg}'
                )
    
    ax.axhline(y=ibert_baseline_acc, color='black', linestyle='--', linewidth=2,
               label=f'I-BERT Baseline ({ibert_baseline_acc}%)', alpha=0.7)
    
    ax.set_xlabel('Softmax Max Error vs I-BERT', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy vs Softmax Max Error', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_max_error.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_vs_max_error.png")
    plt.close()
    
    # =========================================================================
    # Plot 1b: Top-1 Accuracy vs Mean Error
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for deg in [1, 2]:
        for seg in [8, 16, 32]:
            subset = df_merged[(df_merged['softmax_deg'] == deg) & (df_merged['softmax_seg'] == seg)]
            if not subset.empty:
                ax.scatter(
                    subset['softmax_mean_err'],
                    subset['top1'],
                    marker=marker_map[deg],
                    s=marker_size[deg],
                    color=color_map[seg],
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f'Deg {deg}, Seg {seg}'
                )
    
    ax.axhline(y=ibert_baseline_acc, color='black', linestyle='--', linewidth=2,
               label=f'I-BERT Baseline ({ibert_baseline_acc}%)', alpha=0.7)
    
    ax.set_xlabel('Softmax Mean Error vs I-BERT', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy vs Softmax Mean Error', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_mean_error.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_vs_mean_error.png")
    plt.close()
    
    # =========================================================================
    # Plot 2a: Aggregated - Max Error
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grouped = df_merged.groupby(['softmax_deg', 'softmax_seg']).agg({
        'top1': ['mean', 'std', 'count'],
        'softmax_max_err': 'first',
        'softmax_mean_err': 'first',
    }).reset_index()
    grouped.columns = ['softmax_deg', 'softmax_seg', 'top1_mean', 'top1_std', 'count',
                       'softmax_max_err', 'softmax_mean_err']
    
    for _, row in grouped.iterrows():
        deg = int(row['softmax_deg'])
        seg = int(row['softmax_seg'])
        
        ax.errorbar(
            row['softmax_max_err'],
            row['top1_mean'],
            yerr=row['top1_std'],
            marker=marker_map[deg],
            markersize=12 if deg == 2 else 18,
            color=color_map[seg],
            capsize=5,
            capthick=2,
            elinewidth=2,
            alpha=0.8,
            label=f'Deg {deg}, Seg {seg} (n={int(row["count"])})'
        )
        
        ax.annotate(
            f'd{deg}s{seg}',
            (row['softmax_max_err'], row['top1_mean']),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
            alpha=0.8
        )
    
    ax.axhline(y=ibert_baseline_acc, color='black', linestyle='--', linewidth=2,
               label=f'I-BERT Baseline ({ibert_baseline_acc}%)', alpha=0.7)
    
    ax.set_xlabel('Softmax Max Error vs I-BERT', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%) - Mean ± Std', fontsize=12)
    ax.set_title('Average Accuracy vs Softmax Max Error (by config)', fontsize=14)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_max_error_aggregated.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_vs_max_error_aggregated.png")
    plt.close()
    
    # =========================================================================
    # Plot 2b: Aggregated - Mean Error
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for _, row in grouped.iterrows():
        deg = int(row['softmax_deg'])
        seg = int(row['softmax_seg'])
        
        ax.errorbar(
            row['softmax_mean_err'],
            row['top1_mean'],
            yerr=row['top1_std'],
            marker=marker_map[deg],
            markersize=12 if deg == 2 else 18,
            color=color_map[seg],
            capsize=5,
            capthick=2,
            elinewidth=2,
            alpha=0.8,
            label=f'Deg {deg}, Seg {seg} (n={int(row["count"])})'
        )
        
        ax.annotate(
            f'd{deg}s{seg}',
            (row['softmax_mean_err'], row['top1_mean']),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
            alpha=0.8
        )
    
    ax.axhline(y=ibert_baseline_acc, color='black', linestyle='--', linewidth=2,
               label=f'I-BERT Baseline ({ibert_baseline_acc}%)', alpha=0.7)
    
    ax.set_xlabel('Softmax Mean Error vs I-BERT', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%) - Mean ± Std', fontsize=12)
    ax.set_title('Average Accuracy vs Softmax Mean Error (by config)', fontsize=14)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_mean_error_aggregated.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_vs_mean_error_aggregated.png")
    plt.close()
    
    # =========================================================================
    # Plot 3a: Complexity colored by Max Error
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        df_merged['softmax_complexity'],
        df_merged['top1'],
        c=df_merged['softmax_max_err'],
        cmap='RdYlGn_r',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Softmax Max Error vs I-BERT', fontsize=10)
    
    ax.axhline(y=ibert_baseline_acc, color='black', linestyle='--', linewidth=2,
               label=f'I-BERT Baseline ({ibert_baseline_acc}%)', alpha=0.7)
    
    ax.set_xlabel('Softmax Complexity (degree × segments)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Complexity (colored by max error)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_complexity_max_error.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_vs_complexity_max_error.png")
    plt.close()
    
    # =========================================================================
    # Plot 3b: Complexity colored by Mean Error
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        df_merged['softmax_complexity'],
        df_merged['top1'],
        c=df_merged['softmax_mean_err'],
        cmap='RdYlGn_r',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Softmax Mean Error vs I-BERT', fontsize=10)
    
    ax.axhline(y=ibert_baseline_acc, color='black', linestyle='--', linewidth=2,
               label=f'I-BERT Baseline ({ibert_baseline_acc}%)', alpha=0.7)
    
    ax.set_xlabel('Softmax Complexity (degree × segments)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Complexity (colored by mean error)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_complexity_mean_error.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_vs_complexity_mean_error.png")
    plt.close()
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY: Softmax Config -> Error -> Accuracy")
    print("="*80)
    print(f"\n{'Config':<20} | {'Max Err':>10} | {'Mean Err':>12} | {'Avg Acc':>8} | {'Std':>6} | {'N':>4}")
    print("-"*75)
    
    for _, row in grouped.sort_values('softmax_max_err').iterrows():
        config = f"d{int(row['softmax_deg'])}s{int(row['softmax_seg'])}"
        print(f"{config:<20} | {row['softmax_max_err']:>10.6f} | {row['softmax_mean_err']:>12.2e} | "
              f"{row['top1_mean']:>8.2f} | {row['top1_std']:>6.2f} | {int(row['count']):>4}")


if __name__ == "__main__":
    main()
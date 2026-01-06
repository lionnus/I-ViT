#!/usr/bin/env python3
"""
Plot sweep results: Top-1 accuracy vs complexity
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = 'results/sweep_results_2025-12-10.csv'
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Convert to proper types
    df['gelu_deg'] = pd.to_numeric(df['gelu_deg'], errors='coerce')
    df['gelu_seg'] = pd.to_numeric(df['gelu_seg'], errors='coerce')
    df['softmax_deg'] = pd.to_numeric(df['softmax_deg'], errors='coerce')
    df['softmax_seg'] = pd.to_numeric(df['softmax_seg'], errors='coerce')
    df['top1'] = pd.to_numeric(df['top1'], errors='coerce')
    
    # Filter for gelu_deg=2 and gelu_seg=8
    df_filtered = df[(df['gelu_deg'] == 2) & (df['gelu_seg'] == 8)].copy()
    
    print(f"Found {len(df_filtered)} rows matching gelu_deg=2 and gelu_seg=8")
    
    # Calculate complexity (degree * segments) for softmax
    df_filtered['softmax_complexity'] = df_filtered['softmax_deg'] * df_filtered['softmax_seg']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color map for segments
    color_map = {8: 'blue', 16: 'green', 32: 'red'}
    
    # Plot for each degree (marker) and segment (color)
    for deg in [1, 2]:
        for seg in [8, 16, 32]:
            subset = df_filtered[(df_filtered['softmax_deg'] == deg) & 
                                (df_filtered['softmax_seg'] == seg)]
            if not subset.empty:
                marker = '*' if deg == 1 else 'o'
                label = f'Deg {deg}, Seg {seg}'
                ax.scatter(subset['softmax_complexity'], subset['top1'], 
                          marker=marker, s=150, color=color_map[seg], label=label, alpha=0.7)
                print(f"Plotted {len(subset)} points for deg={deg}, seg={seg}")
    
    ax.set_xlabel('Complexity (degree x segments)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Top-1 Accuracy vs Complexity', fontsize=14)
    
    # Add horizontal line for I-BERT accuracy
    ax.axhline(y=71.73, color='black', linestyle='--', linewidth=2, label='I-BERT (71.73%)', alpha=0.7)
    
    ax.legend()
    
    # Set finer tick spacing
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    
    ax.grid(True, alpha=0.3, which='major', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.15, which='minor', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    
    # Save plot
    output_file = csv_file.replace('.csv', '_plot.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()

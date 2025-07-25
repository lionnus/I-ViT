#!/usr/bin/env python3
# gelu_approx_analysis.py
# --------------------------------------------------------------
# Comparison of the GELU approximations of I-ViT, I-BERT, and Piecewise Polynomial
# Approximations taken from: 
# I-BERT: https://github.com/kssteven418/I-BERT/ and 
# I-ViT: https://github.com/zkkli/I-ViT
# Piecewise Polynomial: Custom implementation
# Author: Based on softmax_approx_analysis.py by Lionnus Kesting
# --------------------------------------------------------------

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / 'models'))

# Import the actual modules
from quantization_utils.ivit_modules import IVITIntGELU
from quantization_utils.ibert_modules import IBERTIntGELU
from quantization_utils.ppoly_modules import PPolyIntGELU

# personal style
try:
    plt.style.use("scripts/thesis_plot_styles.mplstyle")
except IOError:
    pass

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Helper: print error stats
# ------------------------------------------------------------------
def print_stats(name: str, err: torch.Tensor):
    print(
        f"{name:25s} | max {err.max():.6e} "
        f"| mean {err.mean():.6e} "
        f"| median {err.median():.6e}"
    )

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare IntGELU_IVIT / IBERT / Piecewise to float GELU.\n"
        "Plots 256 points (8-bit resolution) between -14 and 14."
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.08,
        help="Scaling factor for quantization",
    )
    parser.add_argument("--output-bit", type=int, default=8)
    
    args = parser.parse_args()
    
    # ----------------------------------------------------------
    # 1.  Generate test data - 256 points between -14 and 14
    # ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate integer values from -128 to 127 (256 values for 8-bit signed)
    x_int = torch.arange(-128, 128, device=device, dtype=torch.float32)
    
    # Convert to float values using scale factor
    scale_tensor = torch.tensor(args.scale_factor, device=device)
    x = x_int * scale_tensor
    
    print(f"Generated {len(x)} points")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Scaling factor: {scale_tensor.item()}")

    # ----------------------------------------------------------
    # 2.  Ground-truth float GELU
    # ----------------------------------------------------------
    gelu_float = nn.GELU()
    y_float = gelu_float(x)

    # ----------------------------------------------------------
    # 3.  Approximations using imported modules
    # ----------------------------------------------------------
    ivit = IVITIntGELU().to(device)
    ibert = IBERTIntGELU().to(device)
    piecewise_ibert = PPolyIntGELU(backend='ibert').to(device)
    piecewise_float = PPolyIntGELU(backend='float').to(device)
    
    # Compute approximations
    y_ivit, scaling_ivit_out = ivit(x, scaling_factor=scale_tensor)
    y_ibert, scaling_ibert_out = ibert(x, scaling_factor=scale_tensor)
    y_piecewise_ibert, scaling_piecewise_ibert_out = piecewise_ibert(x, scaling_factor=scale_tensor)
    y_piecewise_float, scaling_piecewise_float_out = piecewise_float(x, scaling_factor=scale_tensor)
    
    # Calculate absolute errors
    abs_err_ivit = (y_ivit - y_float).abs()
    abs_err_ibert = (y_ibert - y_float).abs()
    abs_err_piecewise_ibert = (y_piecewise_ibert - y_float).abs()
    abs_err_piecewise_float = (y_piecewise_float - y_float).abs()
    
    # ----------------------------------------------------------
    # 4.  Print error statistics
    # ----------------------------------------------------------
    print("\n--- GELU Approximation Error Statistics ---")
    print_stats("I-ViT IntGELU", abs_err_ivit)
    print_stats("I-BERT IntGELU", abs_err_ibert)
    print_stats("Piecewise (I-BERT)", abs_err_piecewise_ibert)
    print_stats("Piecewise (Float)", abs_err_piecewise_float)

    # ----------------------------------------------------------
    # 5. Create comparison plots
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Convert to CPU for plotting
    x_cpu = x.cpu().numpy()
    y_float_cpu = y_float.cpu().numpy()
    y_ivit_cpu = y_ivit.cpu().numpy()
    y_ibert_cpu = y_ibert.cpu().numpy()
    y_piecewise_ibert_cpu = y_piecewise_ibert.cpu().numpy()
    y_piecewise_float_cpu = y_piecewise_float.cpu().numpy()
    
    # 5.1 GELU comparison - output vs input
    ax = axes[0, 0]
    ax.plot(x_cpu, y_float_cpu, 'k-', linewidth=2, alpha=0.8, label="Float GELU")
    ax.plot(x_cpu, y_ivit_cpu, 'o-', markersize=3, linewidth=1, alpha=0.7, label="I-ViT")
    ax.plot(x_cpu, y_ibert_cpu, 's-', markersize=3, linewidth=1, alpha=0.7, label="I-BERT")
    ax.plot(x_cpu, y_piecewise_ibert_cpu, '^-', markersize=3, linewidth=1, alpha=0.7, label="Piecewise (I-BERT)")
    ax.plot(x_cpu, y_piecewise_float_cpu, 'v-', markersize=3, linewidth=1, alpha=0.7, label="Piecewise (Float)")
    ax.set_title("GELU Function Approximations")
    ax.set_xlabel("Input x")
    ax.set_ylabel("GELU(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-14.5, 14.5)
    
    # 5.2 Float vs Integer GELU scatter plot
    ax = axes[0, 1]
    ax.scatter(y_float_cpu, y_ivit_cpu, s=10, alpha=0.7, label="I-ViT")
    ax.scatter(y_float_cpu, y_ibert_cpu, s=10, alpha=0.7, label="I-BERT")
    ax.scatter(y_float_cpu, y_piecewise_ibert_cpu, s=10, alpha=0.7, label="Piecewise (I-BERT)")
    ax.scatter(y_float_cpu, y_piecewise_float_cpu, s=10, alpha=0.7, label="Piecewise (Float)")
    
    # Add diagonal line
    min_val = min(y_float_cpu.min(), y_ivit_cpu.min(), y_ibert_cpu.min(), 
                  y_piecewise_ibert_cpu.min(), y_piecewise_float_cpu.min())
    max_val = max(y_float_cpu.max(), y_ivit_cpu.max(), y_ibert_cpu.max(),
                  y_piecewise_ibert_cpu.max(), y_piecewise_float_cpu.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    
    ax.set_title("Float vs Integer GELU")
    ax.set_xlabel("Float GELU")
    ax.set_ylabel("IntGELU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5.3 Error distribution histogram
    ax = axes[1, 0]
    ax.hist(abs_err_ivit.cpu().numpy(), bins=30, alpha=0.7, label="I-ViT", density=True)
    ax.hist(abs_err_ibert.cpu().numpy(), bins=30, alpha=0.7, label="I-BERT", density=True)
    ax.hist(abs_err_piecewise_ibert.cpu().numpy(), bins=30, alpha=0.7, label="Piecewise (I-BERT)", density=True)
    ax.hist(abs_err_piecewise_float.cpu().numpy(), bins=30, alpha=0.7, label="Piecewise (Float)", density=True)
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 5.4 Error vs input value
    ax = axes[1, 1]
    # Create error arrays
    err_ivit = (y_ivit_cpu - y_float_cpu)
    err_ibert = (y_ibert_cpu - y_float_cpu)
    err_piecewise_ibert = (y_piecewise_ibert_cpu - y_float_cpu)
    err_piecewise_float = (y_piecewise_float_cpu - y_float_cpu)
    
    ax.plot(x_cpu, err_ivit, 'o-', markersize=3, linewidth=1, alpha=0.7, label="I-ViT")
    ax.plot(x_cpu, err_ibert, 's-', markersize=3, linewidth=1, alpha=0.7, label="I-BERT")
    ax.plot(x_cpu, err_piecewise_ibert, '^-', markersize=3, linewidth=1, alpha=0.7, label="Piecewise (I-BERT)")
    ax.plot(x_cpu, err_piecewise_float, 'v-', markersize=3, linewidth=1, alpha=0.7, label="Piecewise (Float)")
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel("Input x")
    ax.set_ylabel("Error (IntGELU - Float GELU)")
    ax.set_title("Error vs Input Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-14.5, 14.5)
    
    plt.tight_layout()
    plt.savefig("gelu_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print scaling factor information
    print(f"\n--- Scaling Factor Information ---")
    print(f"Input scaling factor: {args.scale_factor}")
    print(f"Output scaling factors:")
    print(f"  I-ViT:               {scaling_ivit_out.item():.6f}")
    print(f"  I-BERT:              {scaling_ibert_out.item():.6f}")
    print(f"  Piecewise (I-BERT):  {scaling_piecewise_ibert_out.item():.6f}")
    print(f"  Piecewise (Float):   {scaling_piecewise_float_out.item():.6f}")

if __name__ == "__main__":
    main()
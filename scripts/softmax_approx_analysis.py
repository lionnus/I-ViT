#!/usr/bin/env python3
# softmax_approx_analysis.py
# --------------------------------------------------------------
# Comparison of the softmax approximations of I-ViT, I-BERT, and Piecewise Polynomial
# Approximations taken from: 
# I-BERT: https://github.com/kssteven418/I-BERT/ and 
# I-ViT: https://github.com/zkkli/I-ViT
# Piecewise Polynomial: Custom implementation
# Author: Lionnus Kesting (lkesting@ethz.ch)
# --------------------------------------------------------------

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / 'models'))

# Import the actual modules instead of redefining
from quantization_utils.ivit_modules import IVITIntSoftmax
from quantization_utils.ibert_modules import IBERTIntSoftmax
from quantization_utils.ppoly_modules import PPolyIntSoftmax

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
        description="Compare IntSoftmax_IVIT / IBERT / Piecewise to float softmax.\n"
        "Uses external data files for comparison."
    )
    parser.add_argument(
        "--x-file",
        type=Path,
        default=None,
        help="Text file with flattened x tensor (one value per line)",
    )
    parser.add_argument(
        "--scale-file",
        type=Path,
        default=None,
        help="Text file containing a single scaling-factor value",
    )
    parser.add_argument("--output-bit", type=int, default=8)
    parser.add_argument("--subsample", type=int, default=20, 
                       help="Subsample every nth point for plotting")
    
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Original tensor shape, comma-separated, e.g. 128,3,197,197"
    )
    args = parser.parse_args()
    
    # ----------------------------------------------------------
    # 1.  Load test data
    # ----------------------------------------------------------
    if args.x_file and args.x_file.exists() and args.scale_file and args.scale_file.exists():
        x_np = np.loadtxt(args.x_file, dtype=np.float32)
        scale_value = float(np.loadtxt(args.scale_file))

        if args.shape is None:
            raise ValueError(
                "--shape is required when you pass a batched tensor; "
                "example: --shape 128,3,197,197"
            )

        tgt_shape = tuple(int(s) for s in args.shape.split(","))
        x = torch.tensor(x_np, dtype=torch.float32).view(*tgt_shape)
        print(f"Loaded x shape {x.shape}, scaling_factor {scale_value}")
    else:
        raise ValueError("No valid input files provided. Please provide --x-file and --scale-file.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    scale_tensor = torch.tensor(scale_value).to(device)

    # ----------------------------------------------------------
    # 2.  Ground-truth float softmax
    # ----------------------------------------------------------
    y_float = F.softmax(x, dim=-1)

    # ----------------------------------------------------------
    # 3.  Approximations using imported modules
    # ----------------------------------------------------------
    ivit = IVITIntSoftmax(output_bit=args.output_bit).to(device)
    ibert = IBERTIntSoftmax(output_bit=args.output_bit, quant_mode="symmetric").to(device)
    piecewise_ibert = PPolyIntSoftmax(output_bit=args.output_bit, backend='ibert').to(device)
    piecewise_float = PPolyIntSoftmax(output_bit=args.output_bit, backend='float').to(device)
    
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
    print("\n--- Softmax Approximation Error Statistics ---")
    print_stats("I-ViT IntSoftmax", abs_err_ivit)
    print_stats("I-BERT IntSoftmax", abs_err_ibert)
    print_stats("Piecewise (I-BERT)", abs_err_piecewise_ibert)
    print_stats("Piecewise (Float)", abs_err_piecewise_float)

    # ----------------------------------------------------------
    # 5. Create comparison plots
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Flatten data for plotting
    flat_true = y_float.view(-1).cpu().numpy()
    flat_ivit = y_ivit.view(-1).cpu().numpy()
    flat_ibert = y_ibert.view(-1).cpu().numpy()
    flat_piecewise_ibert = y_piecewise_ibert.view(-1).cpu().numpy()
    flat_piecewise_float = y_piecewise_float.view(-1).cpu().numpy()
    
    # Subsample data for plotting
    keep = slice(None, None, args.subsample)
    
    # 5.1 Softmax comparison - float values
    ax = axes[0]
    ax.scatter(flat_true[keep], flat_ivit[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="I-ViT")
    ax.scatter(flat_true[keep], flat_ibert[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="I-BERT")
    ax.scatter(flat_true[keep], flat_piecewise_ibert[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="Piecewise (I-BERT)")
    ax.scatter(flat_true[keep], flat_piecewise_float[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="Piecewise (Float)")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.7)
    ax.set_title("Float vs Integer Softmax")
    ax.set_xlabel("Float Softmax")
    ax.set_ylabel("IntSoftmax")
    leg = ax.legend(scatterpoints=1, markerscale=6)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # 5.2 Error distribution plot
    ax = axes[1]
    ax.hist(abs_err_ivit.view(-1).cpu().numpy(), bins=50, alpha=0.7, label="I-ViT", density=True)
    ax.hist(abs_err_ibert.view(-1).cpu().numpy(), bins=50, alpha=0.7, label="I-BERT", density=True)
    ax.hist(abs_err_piecewise_ibert.view(-1).cpu().numpy(), bins=50, alpha=0.7, label="Piecewise (I-BERT)", density=True)
    ax.hist(abs_err_piecewise_float.view(-1).cpu().numpy(), bins=50, alpha=0.7, label="Piecewise (Float)", density=True)
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("softmax_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print scaling factor information
    print(f"\n--- Scaling Factor Information ---")
    print(f"Input scaling factor: {scale_value}")
    print(f"Output scaling factors:")
    print(f"  I-ViT:               {scaling_ivit_out.item():.6f}")
    print(f"  I-BERT:              {scaling_ibert_out.item():.6f}")
    print(f"  Piecewise (I-BERT):  {scaling_piecewise_ibert_out.item():.6f}")
    print(f"  Piecewise (Float):   {scaling_piecewise_float_out.item():.6f}")
    
if __name__ == "__main__":
    main()
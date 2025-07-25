#!/usr/bin/env python3
# ppoly_softmax_analysis.py
# --------------------------------------------------------------
# Analysis of Piecewise Polynomial Softmax approximation with parameter sweeps
# Author: Lionnus Kesting (lkesting@ethz.ch)
# --------------------------------------------------------------

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / 'models'))
from quantization_utils.ppoly_modules import PPolyIntSoftmax

def print_stats(name: str, err: torch.Tensor):
    """Print error statistics"""
    print(f"{name:30s} | max {err.max():.6e} | mean {err.mean():.6e} | median {err.median():.6e}")

def run_ppoly_analysis(x, scale_tensor, degrees, segments, N_values, alpha_values, optim_bounds_values, output_bit=8):
    """Run analysis with parameter sweeps"""
    device = x.device
    y_float = F.softmax(x, dim=-1)
    
    print("\n=== PPOLY SOFTMAX PARAMETER SWEEP ===")
    
    # 1. Degree and segments sweep
    print("\n--- Degree & Segments Sweep ---")
    for deg in degrees:
        for seg in segments:
            ppoly = PPolyIntSoftmax(output_bit=output_bit, deg=deg, seg=seg, backend='float').to(device)
            y_pred, _ = ppoly(x, scaling_factor=scale_tensor)
            abs_err = (y_pred - y_float).abs()
            print_stats(f"deg={deg}, seg={seg}", abs_err)
    
    # 2. N sweep (with deg=2, seg=16, alpha=0.0, optim_bounds=False)
    print("\n--- N Parameter Sweep (deg=2, seg=16, alpha=0.0, optim_bounds=False) ---")
    for N in N_values:
        ppoly = PPolyIntSoftmax(output_bit=output_bit, scale_bits=N, deg=2, seg=16, 
                               backend='float', alpha=0.0, optim_bounds=False).to(device)
        y_pred, _ = ppoly(x, scaling_factor=scale_tensor)
        abs_err = (y_pred - y_float).abs()
        print_stats(f"N={N}", abs_err)
    
    # 3. Alpha sweep (with deg=2, seg=16, N=23, optim_bounds=False)
    print("\n--- Alpha Parameter Sweep (deg=2, seg=16, N=28, optim_bounds=False) ---")
    for alpha in alpha_values:
        ppoly = PPolyIntSoftmax(output_bit=output_bit, scale_bits=28, deg=2, seg=16,
                               backend='float', alpha=alpha, optim_bounds=False).to(device)
        y_pred, _ = ppoly(x, scaling_factor=scale_tensor)
        abs_err = (y_pred - y_float).abs()
        print_stats(f"alpha={alpha:.1f}", abs_err)
    
    # 4. Optim bounds sweep (with deg=2, seg=16, N=28, alpha=0.0)
    print("\n--- Optim Bounds Sweep (deg=2, seg=16, N=28, alpha=0.0) ---")
    for optim_bounds in optim_bounds_values:
        ppoly = PPolyIntSoftmax(output_bit=output_bit, scale_bits=28, deg=2, seg=16,
                               backend='float', alpha=0.0, optim_bounds=optim_bounds).to(device)
        y_pred, _ = ppoly(x, scaling_factor=scale_tensor)
        abs_err = (y_pred - y_float).abs()
        print_stats(f"optim_bounds={optim_bounds}", abs_err)

def main():
    parser = argparse.ArgumentParser(description="Analyze PPolyIntSoftmax with parameter sweeps")
    parser.add_argument("--x-file", type=Path, required=True, help="Text file with flattened x tensor")
    parser.add_argument("--scale-file", type=Path, required=True, help="Text file with scaling factor")
    parser.add_argument("--shape", type=str, required=True, help="Tensor shape, e.g. 128,3,197,197")
    parser.add_argument("--output-bit", type=int, default=8, help="Output bit width")
    
    args = parser.parse_args()
    
    # Load data
    x_np = np.loadtxt(args.x_file, dtype=np.float32)
    scale_value = float(np.loadtxt(args.scale_file))
    tgt_shape = tuple(int(s) for s in args.shape.split(","))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x_np, dtype=torch.float32).view(*tgt_shape).to(device)
    scale_tensor = torch.tensor(scale_value).to(device)
    
    print(f"Loaded x shape {x.shape}, scaling_factor {scale_value}")
    
    # Define parameter ranges
    degrees = [1, 2]
    segments = [8, 16, 32, 64]
    N_values = list(range(14, 31))  # 14 to 30
    alpha_values = [i * 0.1 for i in range(11)]  # 0.0 to 1.0 with 0.1 increments
    optim_bounds_values = [True, False]
    
    # Run analysis
    run_ppoly_analysis(x, scale_tensor, degrees, segments, N_values, alpha_values, 
                      optim_bounds_values, args.output_bit)

if __name__ == "__main__":
    main()

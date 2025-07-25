#!/usr/bin/env python3
# ppoly_gelu_analysis.py
# --------------------------------------------------------------
# Analysis of Piecewise Polynomial GELU approximation with parameter sweeps
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
from quantization_utils.ppoly_modules import PPolyIntGELU

def print_stats(name: str, err: torch.Tensor):
    """Print error statistics"""
    print(f"{name:30s} | max {err.max():.6e} | mean {err.mean():.6e} | median {err.median():.6e}")

def run_ppoly_analysis(x, scale_tensor, degrees, segments, N_values, alpha_values, optim_bounds_values, output_bit=8):
    """Run analysis with parameter sweeps"""
    device = x.device
    y_float = F.gelu(x)
    
    print("\n=== PPOLY GELU PARAMETER SWEEP ===")
    
    # 1. Degree and segments sweep
    print("\n--- Degree & Segments Sweep ---")
    for deg in degrees:
        for seg in segments:
            ppoly = PPolyIntGELU(output_bit=output_bit, deg=deg, seg=seg, backend='float').to(device)
            y_pred, _ = ppoly(x, scaling_factor=scale_tensor)
            abs_err = (y_pred - y_float).abs()
            print_stats(f"deg={deg}, seg={seg}", abs_err)
    
    # 2. N sweep (with deg=2, seg=16, alpha=0.0, optim_bounds=False)
    print("\n--- N Parameter Sweep (deg=2, seg=16, alpha=0.0, optim_bounds=False) ---")
    for N in N_values:
        ppoly = PPolyIntGELU(output_bit=output_bit, scale_bits=N, deg=2, seg=16, 
                            backend='float', alpha=0.0, optim_bounds=False).to(device)
        y_pred, _ = ppoly(x, scaling_factor=scale_tensor)
        abs_err = (y_pred - y_float).abs()
        print_stats(f"N={N}", abs_err)
    
    # 3. Alpha sweep (with deg=2, seg=16, N=23, optim_bounds=False)
    print("\n--- Alpha Parameter Sweep (deg=2, seg=16, N=23, optim_bounds=False) ---")
    for alpha in alpha_values:
        ppoly = PPolyIntGELU(output_bit=output_bit, scale_bits=23, deg=2, seg=16,
                            backend='float', alpha=alpha, optim_bounds=False).to(device)
        y_pred, _ = ppoly(x, scaling_factor=scale_tensor)
        abs_err = (y_pred - y_float).abs()
        print_stats(f"alpha={alpha:.1f}", abs_err)
    
    # 4. Optim bounds sweep (with deg=2, seg=16, N=23, alpha=0.0)
    print("\n--- Optim Bounds Sweep (deg=2, seg=16, N=23, alpha=0.0) ---")
    for optim_bounds in optim_bounds_values:
        ppoly = PPolyIntGELU(output_bit=output_bit, scale_bits=23, deg=2, seg=16,
                            backend='float', alpha=0.0, optim_bounds=optim_bounds).to(device)
        y_pred, _ = ppoly(x, scaling_factor=scale_tensor)
        abs_err = (y_pred - y_float).abs()
        print_stats(f"optim_bounds={optim_bounds}", abs_err)

def main():
    parser = argparse.ArgumentParser(description="Analyze PPolyIntGELU with parameter sweeps")
    parser.add_argument("--output-bit", type=int, default=8, help="Output bit width")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scale_factor = 0.08  # Scaling factor for quantization
    # Generate integer values from -128 to 127 (256 values for 8-bit signed)
    x_int = torch.arange(-128, 128, device=device, dtype=torch.float32)
    
    # Convert to float values using scale factor
    scale_tensor = torch.tensor(scale_factor, device=device)
    x = x_int * scale_tensor
    
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

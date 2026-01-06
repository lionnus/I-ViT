#!/usr/bin/env python3
# softmax_sweep.py
# --------------------------------------------------------------
# Sweep softmax approximation parameters and output error metrics to CSV
# Compares piecewise polynomial approximation against I-BERT approximation
# Author: Lionnus Kesting (lkesting@ethz.ch)
# --------------------------------------------------------------

import argparse
import csv
import logging
import sys
from pathlib import Path
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / 'models'))

# Import the actual modules instead of redefining
from quantization_utils.ibert_modules import IBERTIntSoftmax
from quantization_utils.ppoly_modules import PPolyIntSoftmax

logger = logging.getLogger(__name__)


def compute_error_stats(y_approx: torch.Tensor, y_ref: torch.Tensor):
    """Compute error statistics between approximation and reference."""
    abs_err = (y_approx - y_ref).abs()
    sq_err = (y_approx - y_ref) ** 2
    
    # Basic stats
    stats = {
        'max_err': abs_err.max().item(),
        'mean_err': abs_err.mean().item(),
        'median_err': abs_err.median().item(),
        'std_err': abs_err.std().item(),
        'rmse': torch.sqrt(sq_err.mean()).item(),
        'p95_err': torch.quantile(abs_err.view(-1), 0.95).item(),
        'p99_err': torch.quantile(abs_err.view(-1), 0.99).item(),
    }
    
    # KL divergence (only valid for probability distributions like softmax output)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    y_approx_safe = y_approx.clamp(min=eps)
    y_ref_safe = y_ref.clamp(min=eps)
    kl_div = (y_ref_safe * (y_ref_safe.log() - y_approx_safe.log())).sum(dim=-1).mean().item()
    stats['kl_div'] = kl_div
    
    return stats


def run_sweep(x: torch.Tensor, scaling_factor: torch.Tensor, output_bit: int,
              degrees: list, segments: list, optim_bounds_options: list,
              backends: list, output_csv: Path, verbose: bool = False):
    """
    Sweep through all parameter combinations and compute error metrics.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor
    scaling_factor : torch.Tensor
        Scaling factor for quantization
    output_bit : int
        Output bitwidth
    degrees : list
        List of polynomial degrees to sweep
    segments : list
        List of segment counts to sweep
    optim_bounds_options : list
        List of optim_bounds boolean values to sweep
    backends : list
        List of backends ('ibert' or 'float') to use as reference for fitting
    output_csv : Path
        Output CSV file path
    verbose : bool
        Print progress information
    """
    device = x.device
    
    # Compute I-BERT reference (this is our ground truth for comparison)
    ibert = IBERTIntSoftmax(output_bit=output_bit, quant_mode="symmetric").to(device)
    y_ibert, _ = ibert(x, scaling_factor=scaling_factor)
    
    # Also compute float softmax for additional comparison
    y_float = F.softmax(x, dim=-1)
    
    results = []
    total_configs = len(degrees) * len(segments) * len(optim_bounds_options) * len(backends)
    config_num = 0
    
    for deg, seg, optim_bounds, backend in product(degrees, segments, optim_bounds_options, backends):
        config_num += 1
        if verbose:
            print(f"[{config_num}/{total_configs}] Testing: deg={deg}, seg={seg}, "
                  f"optim_bounds={optim_bounds}, backend={backend}")
        
        try:
            # Create piecewise polynomial softmax with current config
            ppoly = PPolyIntSoftmax(
                output_bit=output_bit,
                seg=seg,
                deg=deg,
                backend=backend,
                optim_bounds=optim_bounds
            ).to(device)
            
            # Run forward pass
            y_ppoly, _ = ppoly(x, scaling_factor=scaling_factor)
            
            # Compute errors against I-BERT reference
            err_vs_ibert = compute_error_stats(y_ppoly, y_ibert)
            
            # Compute errors against float softmax (for reference)
            err_vs_float = compute_error_stats(y_ppoly, y_float)
            
            result = {
                'degree': deg,
                'segments': seg,
                'optim_bounds': optim_bounds,
                'backend': backend,
                'max_err_vs_ibert': err_vs_ibert['max_err'],
                'mean_err_vs_ibert': err_vs_ibert['mean_err'],
                'median_err_vs_ibert': err_vs_ibert['median_err'],
                'std_err_vs_ibert': err_vs_ibert['std_err'],
                'rmse_vs_ibert': err_vs_ibert['rmse'],
                'p95_err_vs_ibert': err_vs_ibert['p95_err'],
                'p99_err_vs_ibert': err_vs_ibert['p99_err'],
                'kl_div_vs_ibert': err_vs_ibert['kl_div'],
                'max_err_vs_float': err_vs_float['max_err'],
                'mean_err_vs_float': err_vs_float['mean_err'],
                'median_err_vs_float': err_vs_float['median_err'],
                'std_err_vs_float': err_vs_float['std_err'],
                'rmse_vs_float': err_vs_float['rmse'],
                'p95_err_vs_float': err_vs_float['p95_err'],
                'p99_err_vs_float': err_vs_float['p99_err'],
                'kl_div_vs_float': err_vs_float['kl_div'],
            }
            
        except Exception as e:
            logger.error(f"Error with config deg={deg}, seg={seg}, optim_bounds={optim_bounds}, "
                        f"backend={backend}: {e}")
            result = {
                'degree': deg,
                'segments': seg,
                'optim_bounds': optim_bounds,
                'backend': backend,
                'max_err_vs_ibert': float('nan'),
                'mean_err_vs_ibert': float('nan'),
                'median_err_vs_ibert': float('nan'),
                'std_err_vs_ibert': float('nan'),
                'rmse_vs_ibert': float('nan'),
                'p95_err_vs_ibert': float('nan'),
                'p99_err_vs_ibert': float('nan'),
                'kl_div_vs_ibert': float('nan'),
                'max_err_vs_float': float('nan'),
                'mean_err_vs_float': float('nan'),
                'median_err_vs_float': float('nan'),
                'std_err_vs_float': float('nan'),
                'rmse_vs_float': float('nan'),
                'p95_err_vs_float': float('nan'),
                'p99_err_vs_float': float('nan'),
                'kl_div_vs_float': float('nan'),
            }
        
        results.append(result)
    
    # Also add I-BERT baseline error vs float softmax
    ibert_vs_float = compute_error_stats(y_ibert, y_float)
    baseline_result = {
        'degree': 'N/A (I-BERT)',
        'segments': 'N/A',
        'optim_bounds': 'N/A',
        'backend': 'ibert_baseline',
        'max_err_vs_ibert': 0.0,
        'mean_err_vs_ibert': 0.0,
        'median_err_vs_ibert': 0.0,
        'std_err_vs_ibert': 0.0,
        'rmse_vs_ibert': 0.0,
        'p95_err_vs_ibert': 0.0,
        'p99_err_vs_ibert': 0.0,
        'kl_div_vs_ibert': 0.0,
        'max_err_vs_float': ibert_vs_float['max_err'],
        'mean_err_vs_float': ibert_vs_float['mean_err'],
        'median_err_vs_float': ibert_vs_float['median_err'],
        'std_err_vs_float': ibert_vs_float['std_err'],
        'rmse_vs_float': ibert_vs_float['rmse'],
        'p95_err_vs_float': ibert_vs_float['p95_err'],
        'p99_err_vs_float': ibert_vs_float['p99_err'],
        'kl_div_vs_float': ibert_vs_float['kl_div'],
    }
    results.append(baseline_result)
    
    # Write to CSV
    fieldnames = list(results[0].keys())
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    if verbose:
        print(f"\nResults written to: {output_csv}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sweep softmax approximation parameters and output error metrics to CSV.\n"
        "Compares piecewise polynomial approximation against I-BERT approximation."
    )
    parser.add_argument(
        "--x-file",
        type=Path,
        default="scripts/data_softmax/x_softmax_batch.txt",
        help="Text file with flattened x tensor (one value per line)",
    )
    parser.add_argument(
        "--scale-file",
        type=Path,
        default="scripts/data_softmax/scaling_factor_batch.txt",
        help="Text file containing a single scaling-factor value",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="128,3,197,197",
        help="Original tensor shape, comma-separated, e.g. 128,3,197,197"
    )
    parser.add_argument("--output-bit", type=int, default=8,
                       help="Output bitwidth (default: 8)")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/softmax_sweep_results_8b.csv"),
        help="Output CSV file path"
    )
    parser.add_argument(
        "--degrees",
        type=str,
        default="1,2",
        help="Comma-separated list of polynomial degrees to sweep (default: 1,2)"
    )
    parser.add_argument(
        "--segments",
        type=str,
        default="8,16,32",
        help="Comma-separated list of segment counts to sweep (default: 8,16,32)"
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="ibert,float",
        help="Comma-separated list of backends to use (default: ibert,float)"
    )
    parser.add_argument(
        "--random-data",
        action="store_true",
        help="Use random test data instead of loading from files"
    )
    parser.add_argument(
        "--random-shape",
        type=str,
        default="128,3,197,197",
        help="Shape for random test data (default: 32,3,197,197)"
    )
    parser.add_argument(
        "--random-scale",
        type=float,
        default=0.01,
        help="Scaling factor for random test data (default: 0.01)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information"
    )
    
    args = parser.parse_args()
    
    # Parse sweep parameters
    degrees = [int(d) for d in args.degrees.split(',')]
    segments = [int(s) for s in args.segments.split(',')]
    backends = [b.strip() for b in args.backends.split(',')]
    optim_bounds_options = [True, False]
    
    # Load or generate test data
    if args.random_data:
        tgt_shape = tuple(int(s) for s in args.random_shape.split(","))
        x = torch.randn(*tgt_shape, dtype=torch.float32) * 10  # Scale for realistic range
        scale_value = args.random_scale
        print(f"Generated random x shape {x.shape}, scaling_factor {scale_value}")
    elif args.x_file and args.x_file.exists() and args.scale_file and args.scale_file.exists():
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
        raise ValueError(
            "No valid input data. Either provide --x-file and --scale-file, "
            "or use --random-data flag."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    scale_tensor = torch.tensor(scale_value).to(device)
    
    print(f"\n=== Softmax Approximation Sweep ===")
    print(f"Device: {device}")
    print(f"Input shape: {x.shape}")
    print(f"Scaling factor: {scale_value}")
    print(f"Output bit: {args.output_bit}")
    print(f"Degrees: {degrees}")
    print(f"Segments: {segments}")
    print(f"Optim bounds: {optim_bounds_options}")
    print(f"Backends: {backends}")
    print(f"Output CSV: {args.output_csv}")
    print()
    
    # Run sweep
    results = run_sweep(
        x=x,
        scaling_factor=scale_tensor,
        output_bit=args.output_bit,
        degrees=degrees,
        segments=segments,
        optim_bounds_options=optim_bounds_options,
        backends=backends,
        output_csv=args.output_csv,
        verbose=args.verbose
    )
    
    # Print summary table
    print("\n=== Summary (Error vs I-BERT Reference) ===")
    print(f"{'Config':<45} | {'Max Err':>12} | {'Mean Err':>12}")
    print("-" * 75)
    
    for r in results:
        if r['backend'] == 'ibert_baseline':
            config = "I-BERT Baseline (vs Float)"
        else:
            config = f"deg={r['degree']}, seg={r['segments']}, opt={r['optim_bounds']}, {r['backend']}"
        
        max_err = r['max_err_vs_ibert']
        mean_err = r['mean_err_vs_ibert']
        
        if isinstance(max_err, float) and not np.isnan(max_err):
            print(f"{config:<45} | {max_err:>12.6e} | {mean_err:>12.6e}")
        else:
            print(f"{config:<45} | {'ERROR':>12} | {'ERROR':>12}")
    
    print(f"\nFull results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
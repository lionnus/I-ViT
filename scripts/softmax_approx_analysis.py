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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# personal style
try:
    plt.style.use("scripts/thesis_plot_styles.mplstyle")
except IOError:
    pass

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Simplified QuantAct with min and max from training
# ------------------------------------------------------------------

class QuantAct:
    """
    Simplified QuantAct
    """
    def __init__(self, bit_width=8):
        self.bit_width = bit_width
        
        # For symmetric quantization:
        # 2^(bit_width-1) - 1 range for symmetric quantization
        self.scale = 2 ** (bit_width - 1) - 1
        self.x_min = 20414464
        self.x_max = 756987985920
        
    def __call__(self, x, scaling_factor=None):
        """
        Quantize input tensor to specified bit-width
        """
    
        # Transform and quantize
        x_int = x / scaling_factor
        eps = torch.finfo(torch.float32).eps

        max_val = torch.max(torch.Tensor([-self.x_min]), torch.Tensor([self.x_max]))
        scale = max_val / float(self.scale)
        scale.clamp_(eps)
        
        new_scale = scaling_factor / scale
        quant_act_int = torch.round(x_int * new_scale)

        correct_output_scale = self.scale

        return quant_act_int * correct_output_scale, scale

# ------------------------------------------------------------------
# Piecewise Polynomial Softmax
# ------------------------------------------------------------------
class IntSoftmax_Piecewise(nn.Module):
    """
    Implementation of Integer Softmax using piecewise polynomial approximation for exp()
    """
    
    def __init__(self, output_bit=8, N=24, segments=16, degree=2):
        super().__init__()
        self.output_bit = output_bit
        self.N = N  # Bit shift for integer representation
        self.segments = segments
        self.degree = degree
        
        # Exponential approximation range (typical softmax input range after max subtraction)
        self.input_range = (-10.0, 0.0)
        
        # Fit the piecewise polynomials once during initialization
        self.float_pieces = self._fit_piecewise_polynomials()
        
        # self.register_buffer('act_scaling_factor', torch.zeros(1))
    
    def _exp_func(self, x):
        """Standard exponential function for fitting"""
        return np.exp(x)
    
    def _fit_piecewise_polynomials(self):
        """Fit piecewise polynomials to approximate exp(x)."""
        x_lo, x_hi = self.input_range
        xs = np.linspace(x_lo, x_hi, 10000, dtype=np.float32)
        ys = self._exp_func(xs)
        bounds = np.linspace(x_lo, x_hi, self.segments + 1, dtype=np.float32)
        
        pieces = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mask = (xs >= lo) & (xs <= hi)
            if np.sum(mask) < self.degree + 1:
                # Not enough points for fitting, use neighboring points
                center = (lo + hi) / 2
                distances = np.abs(xs - center)
                indices = np.argsort(distances)[:max(self.degree + 1, 10)]
                x_fit = xs[indices]
                y_fit = ys[indices]
            else:
                x_fit = xs[mask]
                y_fit = ys[mask]
            
            coeffs = np.polyfit(x_fit, y_fit, self.degree).astype(np.float32)
            pieces.append(((lo, hi), coeffs))
        return pieces
    
    def int_exp_poly(self, x_int, scaling_factor):
        """Evaluate piecewise polynomial for exponential approximation."""
        # Build integer bounds and integer coefficients under torch.no_grad
        with torch.no_grad():
            s_in = scaling_factor
            lo_list = []
            hi_list = []
            int_coeffs_list = []
            
            for (lo_f, hi_f), coeffs in self.float_pieces:
                lo_i = torch.floor(torch.tensor(lo_f/s_in, device=x_int.device))
                hi_i = torch.floor(torch.tensor(hi_f/s_in, device=x_int.device))
                lo_list.append(lo_i)
                hi_list.append(hi_i)
                
                # Convert float coeffs to integer coeffs
                deg = len(coeffs) - 1
                this_int_coeffs = []
                for i, coeff in enumerate(coeffs):
                    power = deg - i
                    scaled = coeff * (s_in ** power) * (2 ** self.N)
                    this_int_coeffs.append(torch.floor(torch.tensor(scaled, device=x_int.device)))
                int_coeffs_list.append(torch.stack(this_int_coeffs))
                
            lo_i = torch.stack(lo_list)          # (segments,)
            hi_i = torch.stack(hi_list)          # (segments,)
            coeffs_tensor = torch.stack(int_coeffs_list)  # (segments, degree+1)
        
        # Initialize output
        exp_int = torch.zeros_like(x_int, dtype=torch.float32)
        S = self.segments
        D = self.degree
        
        # Evaluate polynomial without building gradient graph
        with torch.no_grad():
            for i in range(S):
                if i == 0:
                    mask_i = x_int <= hi_i[0]
                elif i == S - 1:
                    mask_i = x_int >= lo_i[-1]
                else:
                    mask_i = (x_int >= lo_i[i]) & (x_int <= hi_i[i])
                    
                if not mask_i.any():
                    continue

                x_seg = x_int[mask_i]
                c = coeffs_tensor[i]
                
                # Horner's rule
                r = c[0].to(x_seg.dtype)
                for idx in range(1, D + 1):
                    r = r * x_seg + c[idx]
                
                exp_int[mask_i] = r
        
        # Clamp to ensure positive values (exp should always be positive)
        # exp_int = torch.clamp(exp_int, min=1e-10)
        
        # The output scaling factor accounts for the 2^N factor
        scaling_factor_out = scaling_factor / (2 ** self.N)
        
        return exp_int, scaling_factor_out
    
    def forward(self, x, scaling_factor):
        """Forward pass implementing integer softmax with polynomial exp approximation."""
        device = x.device
        scaling_factor = scaling_factor.to(device)
        
        # Convert to integer representation
        x_int = torch.floor(x / scaling_factor).to(torch.int32)
        
        # Subtract max for numerical stability (standard softmax trick)
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        
        # Apply polynomial exponential approximation
        exp_int, exp_scaling = self.int_exp_poly(x_int.float(), scaling_factor)
        
        # Sum for normalization
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        exp_int_sum = torch.clamp(exp_int_sum, min=1)  # Avoid division by zero
        
        # Normalize
        max_int32 = 2 ** 31 - 1
        factor = torch.floor(max_int32 / exp_int_sum)
        factor = torch.clamp(factor, max=2 ** (32 - self.output_bit)) # shouldnt happen?
        
        # Apply normalization
        # exp_int scaled with 2**N, factor scaled by 2**32/2**N -> 2**N cancles out
        # divide by 2**32 to remove 2**32 scaling
        # multiply by 2**output_bit to get proper output scaling
        normalized_int = torch.floor(exp_int * factor / 2 ** (32 - self.output_bit))
        
        # Final scaling factor for output
        output_scaling_factor = 1.0 / (2 ** (self.output_bit - 1))
        output_scaling_factor = torch.tensor(output_scaling_factor, device=device, dtype=x.dtype)
        
        # Store scaling factor
        self.act_scaling_factor = output_scaling_factor
        
        # Return final result
        result = normalized_int * output_scaling_factor
        
        return result, output_scaling_factor
# ------------------------------------------------------------------
# I-ViT Softmax
# ------------------------------------------------------------------
class IntSoftmax_IVIT(nn.Module):
    """
    Shiftmax from I-ViT quantisation utilities (CPU-only version)
    """
    def __init__(self, output_bit=8):
        super().__init__()
        self.output_bit = output_bit
        self.n = 15                                # large enough integer
        self.register_buffer('act_scaling_factor', torch.zeros(1))

    # ----------------------------------------------------------
    # integer exponential approximation
    # ----------------------------------------------------------
    def int_exp_shift(self, x_int: torch.Tensor, scaling_factor: torch.Tensor):
        """
        Integer approximation of exp(x) in Q-domain
        """
        # shift approximation of exp(x)
        x_int = x_int + torch.floor(x_int / 2) - torch.floor(x_int / 16)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor).to(x_int.device)

        x_int = torch.max(x_int, self.n * x0_int)

        # quotient / remainder decomposition
        q = torch.floor(x_int / x0_int)
        r = x_int - x0_int * q

        # build exp(r/q)
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(
            torch.floor(exp_int * 2 ** (self.n - q)),
            min=0
        )
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x: torch.Tensor, scaling_factor: torch.Tensor):
        device = x.device
        scaling_factor = scaling_factor.to(device)

        # 1) quantise input
        x_int = x / scaling_factor
        x_int = x_int.to(torch.int32)

        # 2) subtract (per-row) max for numerical stability
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        # 3) integer exp
        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)

        # 4) normalise
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        exp_int_sum.clamp_max_(2 ** 31 - 1)
        factor = torch.floor((2 ** 31 - 1) / exp_int_sum)

        exp_int = torch.floor(exp_int * factor / 2 ** (31 - self.output_bit + 1))
        scaling_factor = torch.tensor(
            1.0 / 2 ** (self.output_bit - 1),
            device=device
        )

        # save scaling factor
        self.act_scaling_factor = scaling_factor.detach()
        return exp_int * scaling_factor, scaling_factor


# ------------------------------------------------------------------
# IBERT Softmax
# ------------------------------------------------------------------
class IntSoftmax_IBERT(nn.Module):
    """
    Polynomial-based int Softmax from IBERT.
    Minor edits: on CPU instead of GPU
    """
    def __init__(self,
                 output_bit=8,
                 quant_mode='symmetric',
                 force_dequant='none'):
        super().__init__()
        self.output_bit = output_bit
        self.quant_mode = quant_mode

        if force_dequant in ['nonlinear', 'softmax']:
            logger.info("Force dequantise softmax")
            self.quant_mode = 'none'

        # QuantAct
        self.act = QuantAct(8) #NOTE Originally 16 bit

        # polynomial / shift parameters
        self.x0 = -0.6931                      # −ln(2)
        self.n = 30                            # sufficiently large integer
        self.coef = [0.35815147, 0.96963238, 1.]
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]

    # ----------------------------------------------------------
    def int_polynomial(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor ** 2)

        z = x_int + b_int
        z = x_int * z
        z = z + c_int

        scaling_factor = self.coef[0] * scaling_factor ** 2
        return z, scaling_factor

    def int_exp(self, x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)

        x_int = torch.max(x_int, self.n * x0_int)

        q = torch.floor(x_int / x0_int)
        r = x_int - x0_int * q

        exp_int, exp_scale = self.int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(
            torch.floor(exp_int * 2 ** (self.n - q)),
            min=0
        )
        scaling_factor = exp_scale / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x: torch.Tensor, scaling_factor: torch.Tensor):
        device = x.device
        scaling_factor = scaling_factor.to(device)

        # float mode passthrough
        if self.quant_mode == 'none':
            return F.softmax(x, dim=-1), None

        assert self.quant_mode == 'symmetric', \
            f"Unsupported quant mode: {self.quant_mode}"

        # 1) quantise input
        x_int = (x / scaling_factor).to(torch.int32)

        # 2) subtract max for numerical stability
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        # 3) integer exp, then a fake-quant (QuantAct)
        exp_int, exp_scale = self.int_exp(x_int, scaling_factor)
        exp_q, exp_scale = self.act(exp_int, exp_scale)
        exp_int = exp_q / exp_scale

        # 4) denominator
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        factor = torch.floor(2 ** 32 / exp_int_sum)

        # 5) scale into desired output bit-width
        exp_int = torch.floor(exp_int * factor / 2 ** (32 - self.output_bit + 1)) # NOTE added +1 because output was not correct bitwidth
        scaling_factor = torch.tensor(
           2.0 * 1.0 / 2 ** self.output_bit, # NOTE added 2.0* to match the output bitwidth
            device=device
        )
        return exp_int * scaling_factor, scaling_factor


# ------------------------------------------------------------------
# Helper: print error stats
# ------------------------------------------------------------------
def print_stats(name: str, err: torch.Tensor):
    print(
        f"{name:18s} | max {err.max():.6e} "
        f"| mean {err.mean():.6e} "
        f"| median {err.median():.6e}"
    )

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare IntSoftmax_IVIT / IBERT / Piecewise to float softmax.\n"
        "If --x-file / --scale-file are provided, the script uses real "
        "tensors captured during debugging; otherwise it falls back to a "
        "synthetic sweep."
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
    
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Original tensor shape, comma-separated, e.g. 128,3,197,197"
    )
    args = parser.parse_args()
    
    # ----------------------------------------------------------
    # 1.  Load or generate test data
    # ----------------------------------------------------------
    if args.x_file and args.x_file.exists() and args.scale_file and args.scale_file.exists():
        x_np = np.loadtxt(args.x_file, dtype=np.float32)
        scale_value = float(np.loadtxt(args.scale_file))
        scale_ivit = scale_ibert = scale_piecewise = scale_value

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

    device = torch.device("cpu")
    x = x.to(device)

    # ----------------------------------------------------------
    # 2.  Ground-truth float softmax
    # ----------------------------------------------------------
    y_float = F.softmax(x, dim=-1)

    # ----------------------------------------------------------
    # 3.  Approximations
    # ----------------------------------------------------------
    ivit = IntSoftmax_IVIT(output_bit=args.output_bit)
    ibert = IntSoftmax_IBERT(output_bit=args.output_bit, quant_mode="symmetric")
    piecewise = IntSoftmax_Piecewise(output_bit=args.output_bit)
    y_ivit, scaling_ivit_out = ivit(x, scaling_factor=torch.tensor(scale_ivit))
    y_ibert, scaling_ibert_out = ibert(x, scaling_factor=torch.tensor(scale_ibert))
    y_piecewise, scaling_piecewise_out = piecewise(x, scaling_factor=torch.tensor(scale_piecewise))
    
    abs_err_ivit = (y_ivit - y_float).abs()
    abs_err_ibert = (y_ibert - y_float).abs()
    abs_err_piecewise = (y_piecewise - y_float).abs()
    
    # ----------------------------------------------------------
    # 4.  Visualization and error analysis
    # ----------------------------------------------------------
    # 4.1 Print existing error‐stats
    print_stats("I-ViT IntSoftmax", abs_err_ivit)
    print_stats("I-BERT IntSoftmax", abs_err_ibert)
    print_stats("Piecewise IntSoftmax", abs_err_piecewise)

        
    # 4.2 Bar chart: random row
    sel = (0, 0, 0)               # (batch, head, token)
    true_row  = y_float [sel].cpu().numpy()
    ivit_row  = y_ivit  [sel].cpu().numpy()
    ibert_row = y_ibert [sel].cpu().numpy()
    piecewise_row = y_piecewise [sel].cpu().numpy()

    classes = np.arange(true_row.size)
    # plt.figure(figsize=(7, 4))
    plt.bar(classes - 0.3, true_row,  width=0.2, label="float")
    plt.bar(classes - 0.1, ivit_row,  width=0.2, label="IViT")
    plt.bar(classes + 0.1, ibert_row, width=0.2, label="IBERT")
    plt.bar(classes + 0.3, piecewise_row, width=0.2, label="Piecewise")
    plt.title(f"Softmax for sample {sel}")
    plt.xlabel("class index"); plt.ylabel("probability")
    plt.legend(); plt.grid(True); plt.show()

    # 4.3 Scatter plot on subset
    flat_true  = y_float .view(-1).cpu().numpy()
    flat_ivit  = y_ivit  .view(-1).cpu().numpy()
    flat_ibert = y_ibert .view(-1).cpu().numpy()
    flat_piecewise = y_piecewise .view(-1).cpu().numpy()
    plt.figure(figsize=(12, 4))
    
    keep = slice(None, None, 20)   # keep every n-th point/downsample
    plt.scatter(flat_true[keep], flat_ivit [keep],
            s=3, alpha=0.9, marker='.', linewidths=0,
            label="I-ViT")
    plt.scatter(flat_true[keep], flat_ibert[keep],
                s=3, alpha=0.9, marker='.', linewidths=0,
                label="I-BERT")
    plt.scatter(flat_true[keep], flat_piecewise[keep],
                s=3, alpha=0.9, marker='.', linewidths=0,
                label="Piecewise")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.7)
    plt.title("Float vs Integer Softmax (128 batch size, 197 model dim)") # TODO: change title duynamically
    plt.xlabel("Float Softmax"); plt.ylabel("IntSoftmax")
    leg = plt.legend(scatterpoints=1, markerscale=6)  # markerscale increases legend dot size
    
    # now override the alpha on the legend
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    plt.grid(True); plt.tight_layout()
    # save plot as png in high dpi
    # plt.savefig("Softmax_comparison_batch_v1.png", dpi=300, bbox_inches='tight')
    plt.show()

    
    # 4.4 Histogram of absolute errors
    err_ivit_vals  = abs_err_ivit .view(-1).cpu().numpy()
    err_ibert_vals = abs_err_ibert.view(-1).cpu().numpy()
    err_piecewise_vals = abs_err_piecewise.view(-1).cpu().numpy()
    # plt.figure(figsize=(6,4))
    plt.hist(err_ivit_vals,  bins=50, density=True, alpha=0.5, label="I-ViT")
    plt.hist(err_ibert_vals, bins=50, density=True, alpha=0.5, label="I-BERT")
    plt.hist(err_piecewise_vals, bins=50, density=True, alpha=0.5, label="Piecewise")
    plt.title("Distribution of |error|")
    plt.xlabel("absolute error")
    plt.ylabel("density")
    plt.legend(); plt.grid(True)
    plt.show()
    
    # # 4.5 Integer value comparison (input vs output)
    # # Convert inputs to integer representation
    x_int_input = torch.floor(x / torch.tensor(scale_value)).to(torch.int32)
    
    # Convert outputs to integer representation
    y_ivit_int = torch.floor(y_ivit / scaling_ivit_out).to(torch.int32)
    y_ibert_int = torch.floor(y_ibert / scaling_ibert_out).to(torch.int32)
    y_piecewise_int = torch.floor(y_piecewise / scaling_piecewise_out).to(torch.int32)
    
    # Select a sample for visualization
    sel_int = (0, 0, 0)  # (batch, head, token)
    
    # Plot integer input range
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    x_int_flat = x_int_input[sel_int].cpu().numpy()
    plt.hist(x_int_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Integer Input Distribution\n(sample {sel_int})")
    plt.xlabel("Integer value")
    plt.ylabel("Count")
    plt.axvline(x=-128, color='red', linestyle='--', alpha=0.5, label='8-bit range')
    plt.axvline(x=127, color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot integer output distributions for each method
    plt.subplot(1, 3, 2)
    methods = ['I-ViT', 'I-BERT', 'Piecewise']
    int_outputs = [y_ivit_int[sel_int].cpu().numpy(), 
                   y_ibert_int[sel_int].cpu().numpy(),
                   y_piecewise_int[sel_int].cpu().numpy()]
    colors = ['orange', 'green', 'purple']
    
    for method, output, color in zip(methods, int_outputs, colors):
        plt.hist(output, bins=30, alpha=0.5, label=method, color=color, edgecolor='black')
    
    plt.title(f"Integer Output Distribution\n(sample {sel_int})")
    plt.xlabel("Integer value")
    plt.ylabel("Count")
    plt.axvline(x=-128, color='red', linestyle='--', alpha=0.5, label='8-bit range')
    plt.axvline(x=127, color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot input vs output integer values scatter
    plt.subplot(1, 3, 3)
    # Take a subset of points for clarity
    n_points = min(100, x_int_flat.size)
    indices = np.random.choice(x_int_flat.size, n_points, replace=False)
    
    x_subset = x_int_flat[indices]
    for method, output, color in zip(methods, int_outputs, colors):
        y_subset = output[indices]
        plt.scatter(x_subset, y_subset, alpha=0.6, s=20, label=method, color=color)
    
    plt.title(f"Integer Input vs Output\n(sample {sel_int}, {n_points} points)")
    plt.xlabel("Input integer value")
    plt.ylabel("Output integer value")
    plt.axhline(y=-128, color='red', linestyle='--', alpha=0.3)
    plt.axhline(y=127, color='red', linestyle='--', alpha=0.3)
    plt.axvline(x=-128, color='red', linestyle='--', alpha=0.3)
    plt.axvline(x=127, color='red', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print integer value statistics
    print("\n--- Integer Value Statistics ---")
    print(f"Input integer range: [{x_int_input.min().item()}, {x_int_input.max().item()}]")
    print(f"Input scaling factor: {scale_value}")
    print(f"\nOutput integer ranges:")
    print(f"  I-ViT:     [{y_ivit_int.min().item()}, {y_ivit_int.max().item()}] (scaling: {scaling_ivit_out.item():.6f})")
    print(f"  I-BERT:    [{y_ibert_int.min().item()}, {y_ibert_int.max().item()}] (scaling: {scaling_ibert_out.item():.6f})")
    print(f"  Piecewise: [{y_piecewise_int.min().item()}, {y_piecewise_int.max().item()}] (scaling: {scaling_piecewise_out.item():.6f})")

if __name__ == "__main__":
    main()
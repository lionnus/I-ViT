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
    Simplified QuantAct with dynamic min/max and scale search
    """
    def __init__(self, bit_width=8):
        self.bit_width = bit_width
        # For symmetric quantization: 2^(bit_width-1) - 1
        self.quant_max = 2 ** (bit_width - 1) - 1
        
    def __call__(self, x, scaling_factor=None):
        """
        Quantize input tensor to specified bit-width with dynamic scale
        """
        # Find actual min/max values dynamically from input
        x_min = x.min()
        x_max = x.max()
        
        # Get maximum absolute value for symmetric quantization
        max_abs_val = torch.max(torch.abs(x_min), torch.abs(x_max))
        
        # Prevent division by zero
        eps = torch.finfo(torch.float32).eps
        max_abs_val = torch.clamp(max_abs_val, min=eps)
        
        # Calculate quantization scale
        scale = max_abs_val / self.quant_max
        
        # Apply scaling factor if provided
        if scaling_factor is not None:
            x_scaled = x / scaling_factor
            new_scale = scale / scaling_factor
        else:
            x_scaled = x
            new_scale = scale
        
        # Quantize to integers
        x_quant = torch.round(x_scaled / new_scale)
        x_quant = torch.clamp(x_quant, -self.quant_max, self.quant_max)
        
        # Dequantize for output
        x_dequant = x_quant * scale
        
        return x_dequant, scale

# ------------------------------------------------------------------
# Piecewise Polynomial Softmax
# ------------------------------------------------------------------

class IntSoftmax_Piecewise(nn.Module):
    """
    Implementation of Integer Softmax using piecewise polynomial approximation for exp()
    """
    
    def __init__(self, output_bit=8, N=16, segments=16, degree=2, ibert_patch=True):
        super().__init__()
        self.output_bit = output_bit
        self.N = N  # Bit shift for integer representation
        self.exp_bitwidth = 16
        self.segments = segments
        self.degree = degree
        self.ibert_patch = ibert_patch
        self.fixed = False  
        
        # Torch module for float softmax backward pass
        self.softmax_module = nn.Softmax(dim=-1)
        
        # Default exponential approximation range for non-ibert mode
        if not self.ibert_patch:
            self.input_range = (-18.0, 0.0)
        
        # Initialize IBERT parameters for exp approximation
        self.x0 = -0.6931  # -ln(2)
        self.n = 30  # sufficiently large integer
        self.coef = [0.35815147, 0.96963238, 1.]
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]
        
        # Buffers for storing fixed coefficients
        self.register_buffer('fixed_lo_bounds', None)
        self.register_buffer('fixed_hi_bounds', None)
        self.register_buffer('fixed_coeffs', None)
        self.register_buffer('fixed_scaling_factor_out', None)
        
        # Store exp_int for analysis
        self.last_exp_int = None
        self.last_exp_scale = None
    
    def _exp_func(self, x):
        """Standard exponential function for fitting"""
        return np.exp(x)
    
    def _ibert_int_polynomial(self, x_int, scaling_factor):
        """IBERT's polynomial approximation"""
        with torch.no_grad():
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor ** 2)

        z = x_int + b_int
        z = x_int * z
        z = z + c_int

        scaling_factor = self.coef[0] * scaling_factor ** 2
        return z, scaling_factor

    def _ibert_int_exp(self, x_int, scaling_factor):
        """IBERT's integer exponential approximation"""
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)

        x_int = torch.max(x_int, self.n * x0_int)

        q = torch.floor(x_int / x0_int)
        r = x_int - x0_int * q

        exp_int, exp_scale = self._ibert_int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(
            torch.floor(exp_int * 2 ** (self.n - q)),
            min=0
        )
        scaling_factor = exp_scale / 2 ** self.n
        return exp_int, scaling_factor
    
    def _fit_piecewise_polynomials(self, x=None, scaling_factor=None):
        """Fit piecewise polynomials to approximate exp(x)."""
        if self.ibert_patch and x is not None and scaling_factor is not None:
            # Use IBERT exponential approximation as the target function
            x_lo = torch.floor(torch.min(x)).item()
            x_hi = torch.ceil(torch.max(x)).item()
            # Ensure we stay in negative range for exp approximation
            x_hi = min(x_hi, 0.0)
            
            # Create xs as a tensor on the same device as scaling_factor
            xs = torch.linspace(x_lo, x_hi, 10000, device=scaling_factor.device, dtype=scaling_factor.dtype)
            
            # Use IBERT's exp approximation as the golden model
            xs_int = torch.floor(xs / scaling_factor)
            ys_int, _ = self._ibert_int_exp(xs_int, scaling_factor)
            ys = ys_int * _
            
            # Convert to numpy for polynomial fitting
            ys_np = ys.detach().cpu().numpy()
            xs_np = xs.detach().cpu().numpy()
        else:
            # Use default input range and standard exponential
            x_lo, x_hi = self.input_range
            xs_np = np.linspace(x_lo, x_hi, 10000, dtype=np.float32)
            ys_np = self._exp_func(xs_np)
        
        bounds = np.linspace(x_lo, x_hi, self.segments + 1, dtype=np.float32)
        
        pieces = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mask = (xs_np >= lo) & (xs_np <= hi)
            x_fit = xs_np[mask]
            y_fit = ys_np[mask]
            
            coeffs = np.polyfit(x_fit, y_fit, self.degree).astype(np.float32)
            pieces.append(((lo, hi), coeffs))
        
        return pieces
    
    def fix(self):
        """Fix the module by storing current coefficients"""
        self.fixed = True
    
    def unfix(self):
        """Unfix the module to allow dynamic coefficient computation"""
        self.fixed = False
        # Clear stored coefficients to save memory
        self.fixed_lo_bounds = None
        self.fixed_hi_bounds = None
        self.fixed_coeffs = None
        self.fixed_scaling_factor_out = None
    
    def _compute_integer_coefficients(self, x, scaling_factor):
        """Compute integer coefficients from float pieces and store if fixed"""
        # Fit float polynomial pieces
        float_pieces = self._fit_piecewise_polynomials(x, scaling_factor)
        
        # Convert to integer representation
        s_in = scaling_factor
        lo_list = []
        hi_list = []
        int_coeffs_list = []
        
        for (lo_f, hi_f), coeffs in float_pieces:
            lo_i = torch.floor(torch.tensor(lo_f, device=x.device) / s_in)
            hi_i = torch.floor(torch.tensor(hi_f, device=x.device) / s_in)
            lo_list.append(lo_i)
            hi_list.append(hi_i)
            
            # Convert float coeffs to integer coeffs
            deg = len(coeffs) - 1
            this_int_coeffs = []
            for i, coeff in enumerate(coeffs):
                power = deg - i
                scaled = coeff * (s_in ** power) * (2 ** self.N)
                this_int_coeffs.append(torch.floor(torch.tensor(scaled, device=x.device)))
            int_coeffs_list.append(torch.stack(this_int_coeffs))
        
        lo_bounds = torch.stack(lo_list)
        hi_bounds = torch.stack(hi_list)
        coeffs_tensor = torch.stack(int_coeffs_list)
        
        scaling_factor_out = scaling_factor / (2 ** self.N)
        
        self.fixed_lo_bounds = lo_bounds
        self.fixed_hi_bounds = hi_bounds
        self.fixed_coeffs = coeffs_tensor
        self.fixed_scaling_factor_out = scaling_factor_out
        
        return lo_bounds, hi_bounds, coeffs_tensor, scaling_factor_out
    
    def int_exp_poly(self, x_int, scaling_factor):
        """Evaluate piecewise polynomial for exponential approximation."""
        # Get integer coefficients
        with torch.no_grad():
            if self.fixed and self.fixed_coeffs is not None:
                # Use stored coefficients
                lo_i = self.fixed_lo_bounds
                hi_i = self.fixed_hi_bounds
                coeffs_tensor = self.fixed_coeffs
                scaling_factor_out = self.fixed_scaling_factor_out
            else:
                # Compute coefficients (and store if fixed)
                # Need to pass original float values for proper fitting
                x_float = x_int * scaling_factor
                lo_i, hi_i, coeffs_tensor, scaling_factor_out = self._compute_integer_coefficients(x_float, scaling_factor)
        
        # Initialize output
        exp_int = torch.zeros_like(x_int, dtype=torch.float32)
        S = self.segments
        D = self.degree
        
        # Evaluate polynomial without building gradient graph
        with torch.no_grad():
            for i in range(S):
                if i == 0:
                    # Force values below range to 0
                    # below_range_mask = x_int < lo_i[0]
                    # exp_int[below_range_mask] = 0
                    mask_i = (x_int <= hi_i[0])
                    # actually just use last segment for below range
                    # mask_i = x_int <= lo_i[0]
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
        exp_int = torch.clamp(exp_int, min=0)
        
        # Store for analysis
        self.last_exp_int = exp_int.clone()
        self.last_exp_scale = scaling_factor_out
        
        return exp_int, scaling_factor_out
    
    def forward(self, x, scaling_factor):
        """Forward pass implementing integer softmax with polynomial exp approximation."""
        device = x.device
        scaling_factor = scaling_factor.to(device)
        
        # Convert to integer representation
        x_int = torch.floor(x / scaling_factor)
        
        # Subtract max for numerical stability
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        
        # Apply polynomial exponential approximation
        exp_int, exp_scale = self.int_exp_poly(x_int, scaling_factor)
        
        # Scale down to fit in output bit
        exp_int = torch.floor(exp_int / 2 ** (30 - self.exp_bitwidth + 1))
        
        # Compute denominator
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        
        # Avoid division by zero
        exp_int_sum = torch.clamp(exp_int_sum, min=1.0)
        
        # Compute softmax values
        with torch.no_grad():
            factor = torch.floor(2**32 / exp_int_sum)
            softmax_int = torch.floor(exp_int * factor / 2 ** (32 - self.output_bit + 1))
        
        # Compute float softmax for gradient path
        softmax_float = self.softmax_module(x)
        
        # Convert integer result back to float
        scaling_factor_out = torch.tensor([2 / 2 ** self.output_bit], device=device)
        softmax_int_float = softmax_int * scaling_factor_out
        
        # Replace forward with integer computation but use float softmax derivatives
        output = softmax_int_float.detach() + (softmax_float - softmax_float.detach())
        
        # Ensure output is quantized
        output = scaling_factor_out * torch.floor(output / scaling_factor_out)
        
        return output, scaling_factor_out
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
        
        # Store exp_int for analysis
        self.last_exp_int = None
        self.last_exp_scale = None

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
        
        # Store for analysis
        self.last_exp_int = exp_int.clone()
        self.last_exp_scale = scaling_factor
        
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
        self.act = QuantAct(16)

        # polynomial / shift parameters
        self.x0 = -0.6931                      # −ln(2)
        self.n = 30                      # sufficiently large integer
        self.coef = [0.35815147, 0.96963238, 1.]
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]
        
        # Store exp_int for analysis
        self.last_exp_int = None
        self.last_exp_scale = None

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
        
        # Store for analysis
        self.last_exp_int = exp_int.clone()
        self.last_exp_scale = scaling_factor
        
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
        scale_ivit = scale_ibert = scale_piecewise = scale_piecewise_float = scale_value

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
    piecewise_ibert = IntSoftmax_Piecewise(output_bit=args.output_bit, ibert_patch=True)
    piecewise_float = IntSoftmax_Piecewise(output_bit=args.output_bit, ibert_patch=False)
    
    y_ivit, scaling_ivit_out = ivit(x, scaling_factor=torch.tensor(scale_ivit))
    y_ibert, scaling_ibert_out = ibert(x, scaling_factor=torch.tensor(scale_ibert))
    y_piecewise_ibert, scaling_piecewise_ibert_out = piecewise_ibert(x, scaling_factor=torch.tensor(scale_piecewise))
    y_piecewise_float, scaling_piecewise_float_out = piecewise_float(x, scaling_factor=torch.tensor(scale_piecewise_float))
    
    # Get integer values for analysis
    x_int_input = torch.floor(x / torch.tensor(scale_value))
    y_ivit_int = torch.floor(y_ivit / scaling_ivit_out)
    y_ibert_int = torch.floor(y_ibert / scaling_ibert_out)
    y_piecewise_ibert_int = torch.floor(y_piecewise_ibert / scaling_piecewise_ibert_out)
    y_piecewise_float_int = torch.floor(y_piecewise_float / scaling_piecewise_float_out)
    
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
    # 5. Exponential Approximation Analysis
    # ----------------------------------------------------------
    print("\n--- Exponential Approximation Analysis ---")
    
    # Get the exp_int values from each method
    exp_int_ivit = ivit.last_exp_int
    exp_scale_ivit = ivit.last_exp_scale
    
    exp_int_ibert = ibert.last_exp_int
    exp_scale_ibert = ibert.last_exp_scale
    
    exp_int_piecewise_ibert = piecewise_ibert.last_exp_int
    exp_scale_piecewise_ibert = piecewise_ibert.last_exp_scale
    
    exp_int_piecewise_float = piecewise_float.last_exp_int
    exp_scale_piecewise_float = piecewise_float.last_exp_scale
    
    # Convert to float values
    exp_float_ivit = exp_int_ivit * exp_scale_ivit
    exp_float_ibert = exp_int_ibert * exp_scale_ibert
    exp_float_piecewise_ibert = exp_int_piecewise_ibert * exp_scale_piecewise_ibert
    exp_float_piecewise_float = exp_int_piecewise_float * exp_scale_piecewise_float
    
    # Compute ground truth exponential
    x_int = torch.floor(x / torch.tensor(scale_value)).to(torch.int32)
    x_int_max, _ = x_int.max(dim=-1, keepdim=True)
    x_int_shifted = x_int - x_int_max
    x_shifted = x_int_shifted * torch.tensor(scale_value)
    exp_float_true = torch.exp(x_shifted)
    
    # Print statistics
    print(f"\nExponential Integer Ranges:")
    print(f"  I-ViT:               [{exp_int_ivit.min().item():.0f}, {exp_int_ivit.max().item():.0f}] (scaling: {exp_scale_ivit.item():.6e})")
    print(f"  I-BERT:              [{exp_int_ibert.min().item():.0f}, {exp_int_ibert.max().item():.0f}] (scaling: {exp_scale_ibert.item():.6e})")
    print(f"  Piecewise (I-BERT):  [{exp_int_piecewise_ibert.min().item():.0f}, {exp_int_piecewise_ibert.max().item():.0f}] (scaling: {exp_scale_piecewise_ibert.item():.6e})")
    print(f"  Piecewise (Float):   [{exp_int_piecewise_float.min().item():.0f}, {exp_int_piecewise_float.max().item():.0f}] (scaling: {exp_scale_piecewise_float.item():.6e})")
    
    # Exponential approximation errors
    exp_err_ivit = (exp_float_ivit - exp_float_true).abs()
    exp_err_ibert = (exp_float_ibert - exp_float_true).abs()
    exp_err_piecewise_ibert = (exp_float_piecewise_ibert - exp_float_true).abs()
    exp_err_piecewise_float = (exp_float_piecewise_float - exp_float_true).abs()
    
    print("\nExponential Approximation Error Statistics:")
    print_stats("  I-ViT exp error", exp_err_ivit)
    print_stats("  I-BERT exp error", exp_err_ibert)
    print_stats("  Piecewise (I-BERT)", exp_err_piecewise_ibert)
    print_stats("  Piecewise (Float)", exp_err_piecewise_float)
    
    # ----------------------------------------------------------
    # 6. Create Plots
    # ----------------------------------------------------------
    
    # 6.1 Exponential Approximation Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sample selection for visualization
    sel = (0, 0, 0)  # (batch, head, token)
    x_shifted_row = x_shifted[sel].cpu().numpy()
    sort_idx = np.argsort(x_shifted_row)
    x_sorted = x_shifted_row[sort_idx]
    
    # 6.1.1 Integer exponential values
    ax = axes[0, 0]
    exp_int_ivit_row = exp_int_ivit[sel].cpu().numpy()[sort_idx]
    exp_int_ibert_row = exp_int_ibert[sel].cpu().numpy()[sort_idx]
    exp_int_piecewise_ibert_row = exp_int_piecewise_ibert[sel].cpu().numpy()[sort_idx]
    exp_int_piecewise_float_row = exp_int_piecewise_float[sel].cpu().numpy()[sort_idx]
    
    ax.plot(x_sorted, exp_int_ivit_row, 'o-', markersize=2, label='I-ViT', alpha=0.7)
    ax.plot(x_sorted, exp_int_ibert_row, 's-', markersize=2, label='I-BERT', alpha=0.7)
    ax.plot(x_sorted, exp_int_piecewise_ibert_row, '^-', markersize=2, label='Piecewise (I-BERT)', alpha=0.7)
    ax.plot(x_sorted, exp_int_piecewise_float_row, 'v-', markersize=2, label='Piecewise (Float)', alpha=0.7)
    ax.set_title(f'Integer Exponential Values\n(sample {sel})')
    ax.set_xlabel('x (after max subtraction)')
    ax.set_ylabel('Integer Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6.1.2 Float exponential values
    ax = axes[0, 1]
    exp_float_true_row = exp_float_true[sel].cpu().numpy()[sort_idx]
    exp_float_ivit_row = exp_float_ivit[sel].cpu().numpy()[sort_idx]
    exp_float_ibert_row = exp_float_ibert[sel].cpu().numpy()[sort_idx]
    exp_float_piecewise_ibert_row = exp_float_piecewise_ibert[sel].cpu().numpy()[sort_idx]
    exp_float_piecewise_float_row = exp_float_piecewise_float[sel].cpu().numpy()[sort_idx]
    
    ax.plot(x_sorted, exp_float_true_row, 'k-', linewidth=2, label='True exp()', alpha=0.8)
    ax.plot(x_sorted, exp_float_ivit_row, 'o-', markersize=2, label='I-ViT', alpha=0.7)
    ax.plot(x_sorted, exp_float_ibert_row, 's-', markersize=2, label='I-BERT', alpha=0.7)
    ax.plot(x_sorted, exp_float_piecewise_ibert_row, '^-', markersize=2, label='Piecewise (I-BERT)', alpha=0.7)
    ax.plot(x_sorted, exp_float_piecewise_float_row, 'v-', markersize=2, label='Piecewise (Float)', alpha=0.7)
    ax.set_title(f'Float Exponential Values\n(sample {sel})')
    ax.set_xlabel('x (after max subtraction)')
    ax.set_ylabel('exp(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6.1.3 Softmax comparison - float values
    ax = axes[1, 0]
    flat_true = y_float.view(-1).cpu().numpy()
    flat_ivit = y_ivit.view(-1).cpu().numpy()
    flat_ibert = y_ibert.view(-1).cpu().numpy()
    flat_piecewise_ibert = y_piecewise_ibert.view(-1).cpu().numpy()
    flat_piecewise_float = y_piecewise_float.view(-1).cpu().numpy()
    
    keep = slice(None, None, 20)  # Subsample every 20th point
    ax.scatter(flat_true[keep], flat_ivit[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="I-ViT")
    ax.scatter(flat_true[keep], flat_ibert[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="I-BERT")
    ax.scatter(flat_true[keep], flat_piecewise_ibert[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="Piecewise (I-BERT)")
    ax.scatter(flat_true[keep], flat_piecewise_float[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="Piecewise (Float)")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.7)
    ax.set_title("Float vs Integer Softmax (Full Batch)")
    ax.set_xlabel("Float Softmax")
    ax.set_ylabel("IntSoftmax")
    leg = ax.legend(scatterpoints=1, markerscale=6)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # 6.1.4 Softmax comparison - integer scale
    ax = axes[1, 1]
    flat_true_int = y_float.view(-1).cpu().numpy() / scaling_ibert_out.item()  # Use I-BERT as reference
    flat_ivit_int = y_ivit.view(-1).cpu().numpy() / scaling_ivit_out.item()
    flat_ibert_int = y_ibert.view(-1).cpu().numpy() / scaling_ibert_out.item()
    flat_piecewise_ibert_int = y_piecewise_ibert.view(-1).cpu().numpy() / scaling_piecewise_ibert_out.item()
    flat_piecewise_float_int = y_piecewise_float.view(-1).cpu().numpy() / scaling_piecewise_float_out.item()
    
    ax.scatter(flat_true_int[keep], flat_ivit_int[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="I-ViT")
    ax.scatter(flat_true_int[keep], flat_ibert_int[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="I-BERT")
    ax.scatter(flat_true_int[keep], flat_piecewise_ibert_int[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="Piecewise (I-BERT)")
    ax.scatter(flat_true_int[keep], flat_piecewise_float_int[keep], s=3, alpha=0.5, marker='.', linewidths=0, label="Piecewise (Float)")
    
    # Find appropriate axis limits
    max_val = max(flat_true_int[keep].max(), flat_ivit_int[keep].max(), 
                  flat_ibert_int[keep].max(), flat_piecewise_ibert_int[keep].max(),
                  flat_piecewise_float_int[keep].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.7)
    ax.set_title("Float vs Integer Softmax (Integer Scale)")
    ax.set_xlabel("Float Softmax (Integer Scale)")
    ax.set_ylabel("IntSoftmax (Integer Scale)")
    leg = ax.legend(scatterpoints=1, markerscale=6)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("softmax_exponential_comparison.png", dpi=300, bbox_inches='tight')
    plt.show(block=True)
    
    # Print integer value statistics
    print("\n--- Integer Value Statistics ---")
    print(f"Input integer range: [{x_int_input.min().item()}, {x_int_input.max().item()}]")
    print(f"Input scaling factor: {scale_value}")
    print(f"\nOutput integer ranges:")
    print(f"  I-ViT:               [{y_ivit_int.min().item()}, {y_ivit_int.max().item()}] (scaling: {scaling_ivit_out.item():.6f})")
    print(f"  I-BERT:              [{y_ibert_int.min().item()}, {y_ibert_int.max().item()}] (scaling: {scaling_ibert_out.item():.6f})")
    print(f"  Piecewise (I-BERT):  [{y_piecewise_ibert_int.min().item()}, {y_piecewise_ibert_int.max().item()}] (scaling: {scaling_piecewise_ibert_out.item():.6f})")
    print(f"  Piecewise (Float):   [{y_piecewise_float_int.min().item()}, {y_piecewise_float_int.max().item()}] (scaling: {scaling_piecewise_float_out.item():.6f})")
    
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# gelu_approx_analysis.py
# --------------------------------------------------------------
# Comparison of GELU approximations from I-ViT, I-BERT, and Custom Piecewise Polynomial
# Approximations taken from: 
# I-BERT: https://github.com/kssteven418/I-BERT/
# I-ViT: https://github.com/zkkli/I-ViT
# Custom: Piecewise polynomial approximation
# Author: Lionnus Kesting (lkesting@ethz.ch)
# --------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.style.use('scripts/thesis_plot_styles.mplstyle')
import logging
import numpy as np
from scipy.special import erf

logger = logging.getLogger(__name__)

class IntGELU_IViT(nn.Module):
    """
    ShiftGELU from I-ViT quantization_utils, tweaked to be CPU-only
    """
    def __init__(self, output_bit=8):
        super().__init__()
        self.output_bit = output_bit
        self.n = 23  # sufficiently large integer
        self.register_buffer('act_scaling_factor', torch.zeros(1))

    def int_exp_shift(self, x_int, scaling_factor):
        device = x_int.device
        x_int = x_int.to(torch.int32)

        x_int = x_int + torch.floor(x_int / 2) \
                     - torch.floor(x_int / 16)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor).to(device)

        x_int = torch.max(x_int, self.n * x0_int)

        q = torch.floor(x_int / x0_int)
        r = x_int - x0_int * q

        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(
            torch.floor(exp_int * (2 ** (self.n - q))),
            min=0
        )

        scaling_factor = scaling_factor / (2 ** self.n)
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        device = x.device
        scaling_factor = scaling_factor.to(device)

        # quantize input
        pre_x_int = x / scaling_factor

        # subtract max for numerical stability
        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        # approximate exp(x−max) and exp(−max)
        sig_scale = scaling_factor * 1.702
        exp_int, _     = self.int_exp_shift(x_int, sig_scale)
        exp_int_max, _ = self.int_exp_shift(-x_int_max, sig_scale)

        # sum & normalize
        exp_sum = exp_int + exp_int_max
        temp_exp_sum = exp_sum
        exp_sum = exp_sum.clamp_max(2**31 - 1)
        factor  = torch.floor((2**31 - 1) / exp_sum)

        # build integer sigmoid
        sigmoid_int   = torch.floor(
            exp_int * factor / (2 ** (31 - self.output_bit + 1))
        )
        sigmoid_scale = torch.tensor(
            1 / (2 ** (self.output_bit - 1)),
            device=device
        )

        # multiply scaling
        out_int   = pre_x_int * sigmoid_int
        out_scale = scaling_factor * sigmoid_scale

        self.act_scaling_factor = out_scale.detach()
        return out_int * out_scale, out_scale

class IntGELU_IBERT(nn.Module):
    """
    IntGELU from IBERT
    """
    def __init__(self,
                 quant_mode='symmetric'):
        super(IntGELU_IBERT, self).__init__()
        self.register_buffer('input_scaling_factor', torch.ones(1))
        self.quant_mode = quant_mode

        self.k = 1.4142
        self.n = 6  # sufficiently large integer
        self.coeff = [-0.2888, -1.769, 1]  # a(x+b)**2 + c
        self.coeff[2] /= self.coeff[0] # c = 1/a

    def fix(self):  pass
    def unfix(self):  pass

    def int_erf(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coeff[1] / scaling_factor)
            c_int = torch.floor(self.coeff[2] / (scaling_factor ** 2))

        sign = torch.sign(x_int)
        abs_int = torch.abs(x_int)
        abs_int = torch.min(abs_int, -b_int)
        y_int = (abs_int + b_int) ** 2 + c_int
        y_int = sign * y_int
        scaling_factor = (scaling_factor ** 2) * self.coeff[0]

        y_int = torch.floor(y_int / (2 ** self.n))
        scaling_factor = scaling_factor * (2 ** self.n)
        return y_int, scaling_factor

    def forward(self, x, scaling_factor):
            
        x_int = x / scaling_factor
        sigmoid_int, sigmoid_scaling_factor = self.int_erf(
            x_int, scaling_factor / self.k
        )

        shift_int = torch.floor(1.0 / sigmoid_scaling_factor) # 1/(scale_in^2*-0.2888*2^n), where n is self.n set in __init__

        x_int = x_int * (sigmoid_int + shift_int)
        scaling_factor = scaling_factor * sigmoid_scaling_factor / 2
        
        return x_int * scaling_factor, scaling_factor

class IntGELU_Custom(nn.Module):
    """
    Implementation of Integer GELU using piecewise polynomial approximation
    Uses float GELU for gradient computation to avoid OOM issues
    """
    
    def __init__(self, output_bit=8, N=20, segments=16, degree=2, ibert_patch=True):
        super(IntGELU_Custom, self).__init__()
        self.N = N  # Bit shift for integer representation
        self.segments = segments
        self.degree = degree
        self.ibert_patch = ibert_patch

        # Torch module for float gelu backward pass
        self.gelu_module = nn.GELU()
        if not self.ibert_patch:
            # GELU approx range
            self.input_range = (-5.0, 5.0)
            # Fit the piecewise polynomials once during initialization
            self.float_pieces = self._fit_piecewise_polynomials()
        else:
            self.float_pieces = None  # Placeholder, will be set in forward pass if ibert_patch is True
        
        self.register_buffer('act_scaling_factor', torch.zeros(1))
    
    def _gelu_func(self, x):
        """Standard GELU function for fitting"""
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    
    def _fit_piecewise_polynomials(self, x=None, scaling_factor=None, ibert_patch=False):
        """Fit piecewise polynomials to approximate GELU."""
        if ibert_patch and x is not None and scaling_factor is not None:
            # Use provided x and scaling_factor for fitting
            x_lo = torch.floor(torch.min(x)).item()
            x_hi = torch.ceil(torch.max(x)).item()
            self.input_range = (x_lo, x_hi)
            
            # Create xs as a tensor on the same device as scaling_factor
            xs = torch.linspace(x_lo, x_hi, 10000, device=scaling_factor.device, dtype=scaling_factor.dtype)
            
            # Golden model - IBERTIntGELU expects tensor input
            ibert = IntGELU_IBERT()
            ys, scaling_factor_ibert = ibert(xs, scaling_factor)
            
            # Convert ys to numpy for polynomial fitting
            ys_np = ys.detach().cpu().numpy()
            xs_np = xs.detach().cpu().numpy()
        else:
            # Use default input range
            x_lo, x_hi = self.input_range
            xs_np = np.linspace(x_lo, x_hi, 10000, dtype=np.float32)
            ys_np = self._gelu_func(xs_np)
        
        bounds = np.linspace(x_lo, x_hi, self.segments + 1, dtype=np.float32)
        
        pieces = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mask = (xs_np >= lo) & (xs_np <= hi)
            coeffs = np.polyfit(xs_np[mask], ys_np[mask], self.degree).astype(np.float32)
            pieces.append(((lo, hi), coeffs))
        
        if ibert_patch:
            # If using IBERT patch, return the pieces and scaling factor
            return pieces, scaling_factor_ibert
        else:
            return pieces
    
    def fix(self):
        pass
    
    def unfix(self):
        pass
    
    def forward(self, x, scaling_factor):
        """Evaluate piecewise polynomial for integer inputs with float GELU gradients."""
        
        # Convert input to integer representation
        x_int = torch.floor(x / scaling_factor)
        
        if self.ibert_patch:
            # Use IBERT patch for GELU approximation
            self.float_pieces, scaling_factor_ibert = self._fit_piecewise_polynomials(x, scaling_factor, ibert_patch=True)
        
        # Build integer bounds and integer coefficients with torch.no_grad
        # to avoid building gradient graph for the polynomial fitting
        with torch.no_grad():
            s_in = scaling_factor
            lo_list = []
            hi_list = []
            int_coeffs_list = []
            for (lo_f, hi_f), coeffs in self.float_pieces:
                lo_i = torch.floor(torch.tensor(lo_f, device=x_int.device) / s_in)
                hi_i = torch.floor(torch.tensor(hi_f, device=x_int.device) / s_in)
                lo_list.append(lo_i)
                hi_list.append(hi_i)
                # Convert float coeffs into integer coeffs
                deg = len(coeffs) - 1
                this_int_coeffs = []
                for i, coeff in enumerate(coeffs):
                    power = deg - i
                    scaled = coeff * (s_in ** power) * (2 ** self.N)
                    this_int_coeffs.append(torch.floor(torch.tensor(scaled, device=x_int.device)))
                int_coeffs_list.append(torch.stack(this_int_coeffs))  # shape (degree+1,)
            lo_i = torch.stack(lo_list)          # (segments,)
            hi_i = torch.stack(hi_list)          # (segments,)
            coeffs_tensor = torch.stack(int_coeffs_list)  # (segments, degree+1)
        
        # Initialize output
        y_int = torch.zeros_like(x_int, dtype=torch.float32)
        S = self.segments
        D = self.degree
        
        # Evaluate polynomial, again without building gradient graph
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

                x_seg = x_int[mask_i]  # all elements in segment i
                c = coeffs_tensor[i]   # shape (degree+1,)
                
                # Horner's rule
                r = c[0].to(x_seg.dtype)
                for idx in range(1, D + 1):
                    r = r * x_seg + c[idx]

                # Store result
                y_int[mask_i] = r
        
        # Compute float GELU for gradient path
        y_float_gelu = self.gelu_module(x)
        
        # Convert integer result back to float
        y_float = y_int / (2 ** self.N)
        
        # replaces the forward pass with integer computation but uses float GELU derivatives for backprop
        y_float = y_float.detach() + (y_float_gelu - y_float_gelu.detach())

        if self.ibert_patch:
            # If using IBERT patch, calculate using IBERT Mystery Formula
            k = 1.4142
            n = 6 # sufficiently large integer
            coeff = [-0.2888, -1.769, 1]
            scaling_factor_out = scaling_factor/k
            scaling_factor_out = scaling_factor_out ** 2 / coeff[0]
            scaling_factor_out = scaling_factor_out * (2**n)
            scaling_factor_out = scaling_factor * scaling_factor_out / 2
            scaling_factor_out = -scaling_factor_out # For some reason it inverts
        else:   
            scaling_factor_out = scaling_factor / (2 ** self.N)
        
        # Make sure y_float/scaling_factor_out is integer
        y_float = scaling_factor_out * torch.floor(y_float/scaling_factor_out)

        return y_float, scaling_factor_out


# ------------------------------------------------------------------
# Plot helper function
# ------------------------------------------------------------------

def new_figure(nrows: int = 1, height_mul: float = 1.0):
    """Return a (fig, axes) tuple sized for nrows stacked subplots."""
    width, base_h = plt.rcParams['figure.figsize']
    fig_height = base_h * height_mul
    return plt.subplots(nrows=nrows, ncols=1 if nrows == 1 else 2,
                        figsize=(width, fig_height))

# ------------------------------------------------------------------
# Comparison & plotting routine
# ------------------------------------------------------------------

def run_comparison():
    # 1. Input range & scaling factors -----------------------------
    full_scale   = 40
    bit_width    = 8
    scale_ivit   = full_scale / (2 ** bit_width)
    scale_ibert  = full_scale / (2 ** bit_width)
    scale_custom = full_scale / (2 ** bit_width)
    x = torch.linspace(-full_scale / 2, full_scale / 2, 2**(bit_width-1)+1)
    y_float = F.gelu(x)

    # 2. Integer approximations ------------------------------------
    ivit  = IntGELU_IViT(bit_width)
    y_ivit, _ = ivit(x, scaling_factor=torch.tensor(scale_ivit))

    ibert = IntGELU_IBERT(bit_width)
    y_ibert, _ = ibert(x, scaling_factor=torch.tensor(scale_ibert))

    custom = IntGELU_Custom(bit_width, N=20, segments=16, degree=2)
    y_custom, _ = custom(x, scaling_factor=torch.tensor(scale_custom))

    # 3. Errors ------------------------------------------------------
    abs_err_ivit  = (y_ivit  - y_float).abs()
    abs_err_ibert = (y_ibert - y_float).abs()
    abs_err_custom = (y_custom - y_float).abs()
    rel_err_ivit  = abs_err_ivit  / torch.clamp(y_float.abs(), min=1e-3) * 100.0
    rel_err_ibert = abs_err_ibert / torch.clamp(y_float.abs(), min=1e-3) * 100.0
    rel_err_custom = abs_err_custom / torch.clamp(y_float.abs(), min=1e-3) * 100.0

    # 4. Metrics printout -------------------------------------------
    def _stats(name, ae, re):
        print(f"=== {name} ===\n"
              f"Max abs error : {ae.max():.6f}\n"
              f"Mean abs error: {ae.mean():.6f}\n"
              f"Max % error  : {re.max():.2f}%\n"
              f"Mean % error : {re.mean():.2f}%\n")
    _stats("I-ViT IntGELU",  abs_err_ivit,  rel_err_ivit)
    _stats("I-BERT IntGELU", abs_err_ibert, rel_err_ibert)
    _stats("PP IntGELU", abs_err_custom, rel_err_custom)

    # 5. Figure with 4 panels --------------------------------------
    fig, axes = new_figure(nrows=2, height_mul=2) 
    ax11, ax12 = axes[0]
    ax21, ax22 = axes[1]

    # GELU & approximations
    ax11.plot(x, y_float, 'k-', label="Float GELU", linewidth=2)
    ax11.plot(x, y_ivit,  '--', label="I-ViT IntGELU", linewidth=1.5)
    ax11.plot(x, y_ibert, ':',  label="I-BERT IntGELU", linewidth=1.5)
    ax11.plot(x, y_custom, '-.', label="PP IntGELU", linewidth=1.5)
    ax11.set_title("GELU vs Integer Approximations")
    ax11.set_xlabel("Input x")
    ax11.set_ylabel("GELU(x)")
    ax11.legend()

    # absolute error
    ax12.plot(x, abs_err_ivit,  label="I-ViT abs err", linewidth=1.5)
    ax12.plot(x, abs_err_ibert, label="I-BERT abs err", linewidth=1.5)
    ax12.plot(x, abs_err_custom, label="PP abs err", linewidth=1.5)
    ax12.set_title("Absolute Error")
    ax12.set_xlabel("Input x")
    ax12.set_ylabel("|error|")
    ax12.legend()

    # percentage error
    ax21.plot(x, rel_err_ivit,  label="I-ViT % err", linewidth=1.5)
    ax21.plot(x, rel_err_ibert, label="I-BERT % err", linewidth=1.5)
    ax21.plot(x, rel_err_custom, label="PP % err", linewidth=1.5)
    ax21.set_title("Percentage Error")
    ax21.set_xlabel("Input x")
    ax21.set_ylabel("Error (%)")
    ax21.legend()

    # log-scale absolute error
    ax22.semilogy(x, abs_err_ivit,  '--', label="I-ViT abs err", linewidth=1.5)
    ax22.semilogy(x, abs_err_ibert, ':',  label="I-BERT abs err", linewidth=1.5)
    ax22.semilogy(x, abs_err_custom, '-.', label="PP abs err", linewidth=1.5)
    ax22.set_title("Absolute Error (log-scale)")
    ax22.set_xlabel("Input x")
    ax22.set_ylabel("|error|")
    ax22.legend()

    fig.show()

    # 6. Bin-wise error distribution -------------------------------
    fig2, ax = new_figure()
    bins = np.linspace(-full_scale / 2, full_scale / 2, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ivit_bin_err  = []
    ibert_bin_err = []
    custom_bin_err = []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if torch.any(mask):
            ivit_bin_err.append(rel_err_ivit[mask].mean().item())
            ibert_bin_err.append(rel_err_ibert[mask].mean().item())
            custom_bin_err.append(rel_err_custom[mask].mean().item())
        else:
            ivit_bin_err.append(0.0)
            ibert_bin_err.append(0.0)
            custom_bin_err.append(0.0)
    width = 0.25
    ax.bar(bin_centers - width, ivit_bin_err,  width, label='I-ViT')
    ax.bar(bin_centers, ibert_bin_err, width, label='I-BERT')
    ax.bar(bin_centers + width, custom_bin_err, width, label='PP')
    ax.set_xlabel('Input range')
    ax.set_ylabel('Average % error')
    ax.set_title('Error distribution across input range')
    ax.legend()
    fig2.show()

    # 7. Additional visualization: Show piecewise boundaries -------
    fig3, ax = new_figure()
    ax.plot(x, y_float, 'k-', label="Float GELU", linewidth=2)
    ax.plot(x, y_custom, 'r-', label="PP Piecewise Poly", linewidth=1.5, alpha=0.8)
    
    # Mark segment boundaries
    bounds = np.linspace(-full_scale / 2, full_scale / 2, custom.segments + 1)
    for b in bounds[1:-1]:
        ax.axvline(x=b, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    ax.legend()
    ax.set_title(f"Piecewise Polynomial GELU (segments={custom.segments}, degree={custom.degree})")
    ax.set_xlabel("Input x")
    ax.set_ylabel("GELU(x)")
    fig3.show()


if __name__ == "__main__":
    run_comparison()
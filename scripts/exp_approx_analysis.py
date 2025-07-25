#!/usr/bin/env python3
# exp_approx_analysis.py
# --------------------------------------------------------------
# Comparison of integer exponential approximations from I-ViT, I-BERT, and Custom Piecewise Polynomial
# Approximations taken from: 
# I-BERT: https://github.com/kssteven418/I-BERT/
# I-ViT: https://github.com/zkkli/I-ViT
# Custom: Piecewise polynomial approximation
# Author: Lionnus Kesting (lkesting@ethz.ch)
# --------------------------------------------------------------

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('scripts/thesis_plot_styles.mplstyle')

# ------------------------------------------------------------------
# Isolated I-ViT int_exp_shift implementation
# ------------------------------------------------------------------
def int_exp_shift(x_int: torch.Tensor, scaling_factor: torch.Tensor, n=15):
    """
    Integer approximation of exp(x) using the shift approximation from I-ViT
    """
    # --- Shift approximation of exp(x) ---------------------
    x_int = x_int + torch.floor(x_int / 2) - torch.floor(x_int / 16)

    with torch.no_grad():
        x0_int = torch.floor(-1.0 / scaling_factor).to(x_int.device)

    x_int = torch.max(x_int, n * x0_int)

    # quotient / remainder decomposition
    q = torch.floor(x_int / x0_int)
    r = x_int - x0_int * q

    # build exp(r/q)
    exp_int = r / 2 - x0_int
    exp_int = torch.clamp(
        torch.floor(exp_int * 2 ** (n - q)),
        min=0
    )
    scaling_factor_out = scaling_factor / 2 ** n
    return exp_int, scaling_factor_out

# ------------------------------------------------------------------
# Isolated IBERT int_exp implementation
# ------------------------------------------------------------------
def int_polynomial(x_int, scaling_factor):
    """Polynomial approximation used in IBERT's exponential function"""
    coef = [0.35815147, 0.96963238, 1.]
    coef[1] /= coef[0]
    coef[2] /= coef[0]
    
    with torch.no_grad():
        b_int = torch.floor(coef[1] / scaling_factor)
        c_int = torch.floor(coef[2] / scaling_factor ** 2)

    z = x_int + b_int
    z = x_int * z
    z = z + c_int

    scaling_factor_out = coef[0] * scaling_factor ** 2
    return z, scaling_factor_out

def int_exp_ibert(x_int, scaling_factor, n=30):
    """
    Integer approximation of exp(x) using the polynomial method from IBERT
    """
    x0 = -0.6931  # −ln(2)
    
    with torch.no_grad():
        x0_int = torch.floor(x0 / scaling_factor)

    x_int = torch.max(x_int, n * x0_int)

    q = torch.floor(x_int / x0_int)
    r = x_int - x0_int * q

    exp_int, exp_scale = int_polynomial(r, scaling_factor)
    exp_int = torch.clamp(
        torch.floor(exp_int * 2 ** (n - q)),
        min=0
    )
    scaling_factor_out = exp_scale / 2 ** n
    return exp_int, scaling_factor_out

# ------------------------------------------------------------------
# Custom Piecewise Polynomial Implementation
# ------------------------------------------------------------------
class CustomPiecewisePolynomial:
    def __init__(self, N=20, segments=16, degree=2, ibert_patch=False):
        self.N = N  # Bit shift for integer representation
        self.segments = segments
        self.degree = degree
        self.ibert_patch = ibert_patch
        self.input_range = (-18.0, 0.0)
        
        if not self.ibert_patch:
            # Fit to standard exponential function
            self.float_pieces = self._fit_piecewise_polynomials()
        else:
            # Initialize IBERT parameters
            self.x0 = -0.6931  # −ln(2)
            self.n = 30  # sufficiently large integer
            self.coef = [0.35815147, 0.96963238, 1.]
            self.coef[1] /= self.coef[0]
            self.coef[2] /= self.coef[0]
            self.float_pieces = None  # Will be set when needed
    
    def _exp_func(self, x):
        """Standard exponential function for fitting"""
        return np.exp(np.clip(x, -50, 0))  # Clip to avoid overflow/underflow
    
    def _fit_piecewise_polynomials(self, x_range=None, scaling_factor=None):
        """Fit piecewise polynomials to approximate exp(x)."""
        if self.ibert_patch and x_range is not None and scaling_factor is not None:
            # Use IBERT exponential approximation as the target function
            x_lo = torch.floor(torch.min(x_range)).item()
            x_hi = torch.ceil(torch.max(x_range)).item()
            # Ensure we stay in reasonable range for exp approximation
            x_hi = min(x_hi, 0.0)
            x_lo = max(x_lo, -50.0)
            self.input_range = (x_lo, x_hi)
            
            # Create xs as a tensor
            xs = torch.linspace(x_lo, x_hi, 10000, device=scaling_factor.device, dtype=scaling_factor.dtype)
            
            # Use IBERT's exp approximation as the golden model
            xs_int = torch.floor(xs / scaling_factor).to(torch.int32)
            ys_int, scaling_factor_out = int_exp_ibert(xs_int, scaling_factor, n=self.n)
            ys = ys_int * scaling_factor_out
            
            # Convert to numpy for polynomial fitting
            ys_np = ys.detach().cpu().numpy()
            xs_np = xs.detach().cpu().numpy()
        else:
            # Use standard exponential
            x_lo, x_hi = self.input_range
            xs_np = np.linspace(x_lo, x_hi, 10000, dtype=np.float32)
            ys_np = self._exp_func(xs_np)
            
        bounds = np.linspace(x_lo, x_hi, self.segments + 1, dtype=np.float32)
        
        pieces = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mask = (xs_np >= lo) & (xs_np <= hi)
            if np.sum(mask) < self.degree + 1:
                # Not enough points for fitting, use neighboring points
                center = (lo + hi) / 2
                distances = np.abs(xs_np - center)
                indices = np.argsort(distances)[:max(self.degree + 1, 10)]
                x_fit = xs_np[indices]
                y_fit = ys_np[indices]
            else:
                x_fit = xs_np[mask]
                y_fit = ys_np[mask]
            
            coeffs = np.polyfit(x_fit, y_fit, self.degree).astype(np.float32)
            pieces.append(((lo, hi), coeffs))
            
        if self.ibert_patch:
            return pieces, scaling_factor_out
        else:
            return pieces
    
    def int_exp_poly(self, x_int, scaling_factor):
        """Evaluate piecewise polynomial for exponential approximation."""
        
        if self.ibert_patch and self.float_pieces is None:
            # Fit polynomials to IBERT's exp approximation on first use
            x_range = torch.linspace(self.input_range[0], self.input_range[1], 1000, device=x_int.device, dtype=scaling_factor.dtype)
            result = self._fit_piecewise_polynomials(x_range, scaling_factor)
            if isinstance(result, tuple):
                self.float_pieces, _ = result
            else:
                self.float_pieces = result
        
        # Build integer bounds and integer coefficients under torch.no_grad
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
                
                # Convert float coeffs → integer coeffs
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
                if i==0:
                    below_range_mask = x_int < lo_i[0]
                    exp_int[below_range_mask] = 0    
                    
                if i == S - 1:
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
        
        # Convert integer result back to float
        exp_int = torch.clamp(exp_int, min=0)
        exp_float = exp_int / (2 ** self.N)
        scaling_factor_out = scaling_factor / (2 ** self.N)
        
        # Ensure non-negative (exp should always be positive)
        # exp_float = torch.clamp(exp_float, min=1e-10)
        
        return exp_float, scaling_factor_out

# ------------------------------------------------------------------
# Comparison function
# ------------------------------------------------------------------
def run_comparison():
    # Setup input range from -11 to 0, with 8 bit quantization
    full_range = 140
    bit_width = 8
    x = torch.linspace(-full_range/2, 0, 2**(bit_width)+1)
    
    # Set scaling factors
    scale_ivit = full_range/2**bit_width
    scale_ibert = full_range/2**bit_width
    scale_custom = full_range/2**bit_width
    scale_custom_ibert = full_range/2**bit_width
    
    # Compute float reference
    y_float = torch.exp(x)
    
    # Quantize inputs for integer domain
    x_int_ivit = (x / scale_ivit).to(torch.int32)
    x_int_ibert = (x / scale_ibert).to(torch.int32)
    x_int_custom = (x / scale_custom).to(torch.int32)
    x_int_custom_ibert = (x / scale_custom_ibert).to(torch.int32)
    
    # Apply integer exponential approximations
    exp_int_ivit, scale_out_ivit = int_exp_shift(x_int_ivit, torch.tensor(scale_ivit))
    y_ivit = exp_int_ivit * scale_out_ivit
    
    exp_int_ibert, scale_out_ibert = int_exp_ibert(x_int_ibert, torch.tensor(scale_ibert))
    y_ibert = exp_int_ibert * scale_out_ibert
    
    # Apply custom piecewise polynomial approximation (standard)
    custom_approx = CustomPiecewisePolynomial(N=20, segments=6, degree=2, ibert_patch=False)
    y_custom, scale_out_custom = custom_approx.int_exp_poly(x_int_custom, torch.tensor(scale_custom))
    
    # Apply custom piecewise polynomial approximation (IBERT patch)
    custom_approx_ibert = CustomPiecewisePolynomial(N=20, segments=6, degree=2, ibert_patch=True)
    y_custom_ibert, scale_out_custom_ibert = custom_approx_ibert.int_exp_poly(x_int_custom_ibert, torch.tensor(scale_custom_ibert))
    
    # Compute errors
    abs_err_ivit = (y_ivit - y_float).abs()
    rel_err_ivit = abs_err_ivit / (y_float + 1e-10) * 100  # percentage
    
    abs_err_ibert = (y_ibert - y_float).abs()
    rel_err_ibert = abs_err_ibert / (y_float + 1e-10) * 100  # percentage
    
    abs_err_custom = (y_custom - y_float).abs()
    rel_err_custom = abs_err_custom / (y_float + 1e-10) * 100  # percentage
    
    abs_err_custom_ibert = (y_custom_ibert - y_float).abs()
    rel_err_custom_ibert = abs_err_custom_ibert / (y_float + 1e-10) * 100  # percentage
    
    # Print error metrics
    print("=== I-ViT int_exp_shift Error Metrics ===")
    print(f"Max absolute error: {abs_err_ivit.max().item():.6f}")
    print(f"Mean absolute error: {abs_err_ivit.mean().item():.6f}")
    print(f"Max percentage error: {rel_err_ivit.max().item():.2f}%")
    print(f"Mean percentage error: {rel_err_ivit.mean().item():.2f}%")
    print()
    
    print("=== I-BERT int_exp Error Metrics ===")
    print(f"Max absolute error: {abs_err_ibert.max().item():.6f}")
    print(f"Mean absolute error: {abs_err_ibert.mean().item():.6f}")
    print(f"Max percentage error: {rel_err_ibert.max().item():.2f}%")
    print(f"Mean percentage error: {rel_err_ibert.mean().item():.2f}%")
    print()
    
    print("=== Custom Piecewise Polynomial Error Metrics ===")
    print(f"Max absolute error: {abs_err_custom.max().item():.6f}")
    print(f"Mean absolute error: {abs_err_custom.mean().item():.6f}")
    print(f"Max percentage error: {rel_err_custom.max().item():.2f}%")
    print(f"Mean percentage error: {rel_err_custom.mean().item():.2f}%")
    print()
    
    print("=== Custom PP (IBERT patch) Error Metrics ===")
    print(f"Max absolute error: {abs_err_custom_ibert.max().item():.6f}")
    print(f"Mean absolute error: {abs_err_custom_ibert.mean().item():.6f}")
    print(f"Max percentage error: {rel_err_custom_ibert.max().item():.2f}%")
    print(f"Mean percentage error: {rel_err_custom_ibert.mean().item():.2f}%")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 10))
    
    # Plot 1: Exponential function and approximations
    plt.subplot(2, 2, 1)
    plt.plot(x.numpy(), y_float.numpy(), 'k-', label="Float exp(x)", linewidth=2)
    plt.plot(x.numpy(), y_ivit.numpy(), '--', label="I-ViT int_exp_shift", linewidth=1.5)
    plt.plot(x.numpy(), y_ibert.numpy(), ':', label="I-BERT int_exp", linewidth=1.5)
    plt.plot(x.numpy(), y_custom.numpy(), '-.', label="Custom PP", linewidth=1.5)
    plt.plot(x.numpy(), y_custom_ibert.numpy(), '-', label="Custom PP (IBERT)", linewidth=1.5, alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("exp(x) vs Integer Approximations")
    plt.xlabel("Input x")
    plt.ylabel("exp(x)")
    
    # Plot 2: Absolute errors
    plt.subplot(2, 2, 2)
    plt.plot(x.numpy(), abs_err_ivit.numpy(), label="I-ViT", linewidth=1.5)
    plt.plot(x.numpy(), abs_err_ibert.numpy(), label="I-BERT", linewidth=1.5)
    plt.plot(x.numpy(), abs_err_custom.numpy(), label="Custom PP", linewidth=1.5)
    plt.plot(x.numpy(), abs_err_custom_ibert.numpy(), label="Custom PP (IBERT)", linewidth=1.5, alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Absolute Error")
    plt.xlabel("Input x")
    plt.ylabel("Absolute Error")
    
    # Plot 3: Percentage errors
    plt.subplot(2, 2, 3)
    plt.plot(x.numpy(), rel_err_ivit.numpy(), label="I-ViT", linewidth=1.5)
    plt.plot(x.numpy(), rel_err_ibert.numpy(), label="I-BERT", linewidth=1.5)
    plt.plot(x.numpy(), rel_err_custom.numpy(), label="Custom PP", linewidth=1.5)
    plt.plot(x.numpy(), rel_err_custom_ibert.numpy(), label="Custom PP (IBERT)", linewidth=1.5, alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Percentage Error")
    plt.xlabel("Input x")
    plt.ylabel("Error (%)")
    plt.ylim(0, min(100, max(rel_err_ivit.max().item(), rel_err_ibert.max().item(), 
                            rel_err_custom.max().item(), rel_err_custom_ibert.max().item()) * 1.1))
    
    # Plot 4: Log scale for better visualization of small values
    plt.subplot(2, 2, 4)
    plt.semilogy(x.numpy(), y_float.numpy(), 'k-', label="Float exp(x)", linewidth=2)
    plt.semilogy(x.numpy(), y_ivit.numpy(), '--', label="I-ViT", linewidth=1.5)
    plt.semilogy(x.numpy(), y_ibert.numpy(), ':', label="I-BERT", linewidth=1.5)
    plt.semilogy(x.numpy(), y_custom.numpy(), '-.', label="Custom PP", linewidth=1.5)
    plt.semilogy(x.numpy(), y_custom_ibert.numpy(), '-', label="Custom PP (IBERT)", linewidth=1.5, alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("exp(x) vs Integer Approximations (Log Scale)")
    plt.xlabel("Input x")
    plt.ylabel("exp(x)")
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: Show piecewise boundaries comparison
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Standard piecewise polynomial
    ax1.plot(x.numpy(), y_float.numpy(), 'k-', label="Float exp(x)", linewidth=2)
    ax1.plot(x.numpy(), y_custom.numpy(), 'r-', label="Custom PP (Standard)", linewidth=1.5, alpha=0.8)
    
    # Mark segment boundaries
    bounds = np.linspace(-full_range/2, 0, custom_approx.segments + 1)
    for b in bounds[1:-1]:
        ax1.axvline(x=b, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"PP Approximation to exp(x)\n(segments={custom_approx.segments}, degree={custom_approx.degree})")
    ax1.set_xlabel("Input x")
    ax1.set_ylabel("exp(x)")
    
    # Right plot: IBERT-based piecewise polynomial
    ax2.plot(x.numpy(), y_float.numpy(), 'k-', label="Float exp(x)", linewidth=2)
    ax2.plot(x.numpy(), y_ibert.numpy(), 'b:', label="I-BERT exp", linewidth=2)
    ax2.plot(x.numpy(), y_custom_ibert.numpy(), 'g-', label="Custom PP (IBERT)", linewidth=1.5, alpha=0.8)
    
    # Mark segment boundaries
    for b in bounds[1:-1]:
        ax2.axvline(x=b, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"PP Approximation to IBERT exp\n(segments={custom_approx_ibert.segments}, degree={custom_approx_ibert.degree})")
    ax2.set_xlabel("Input x")
    ax2.set_ylabel("exp(x)")
    
    plt.tight_layout()
    plt.show()
    
    # Print integer summary
    print("\n=== Integer Summary ===")
    print(f"I-ViT scaling factor: {scale_ivit:.6f}")
    print(f"I-BERT scaling factor: {scale_ibert:.6f}")
    print(f"Custom scaling factor: {scale_custom:.6f}")
    print(f"Custom (IBERT) scaling factor: {scale_custom_ibert:.6f}")
    
    # Integer ranges
    print(f"\nI-ViT int output range: [{y_ivit.min().item()/scale_out_ivit:.6f}, {y_ivit.max().item()/scale_out_ivit:.6f}]")
    print(f"I-BERT int range: [{y_ibert.min().item()/scale_out_ibert:.6f}, {y_ibert.max().item()/scale_out_ibert:.6f}]")
    print(f"Custom int range: [{y_custom.min().item()/scale_out_custom:.6f}, {y_custom.max().item()/scale_out_custom:.6f}]")
    print(f"Custom (IBERT) int range: [{y_custom_ibert.min().item()/scale_out_custom_ibert:.6f}, {y_custom_ibert.max().item()/scale_out_custom_ibert:.6f}]")
    
    # Output ranges
    print(f"\nI-ViT output range: [{y_ivit.min().item()}, {y_ivit.max().item()}]")
    print(f"I-BERT output range: [{y_ibert.min().item()}, {y_ibert.max().item()}]")
    print(f"Custom output range: [{y_custom.min().item()}, {y_custom.max().item()}]")
    print(f"Custom (IBERT) output range: [{y_custom_ibert.min().item()}, {y_custom_ibert.max().item()}]")
    
    # Bits needed for output
    bits_needed_ivit = (y_ivit.max() / scale_out_ivit)/2**32
    bits_needed_ibert = (y_ibert.max() / scale_out_ibert)/2**32
    bits_needed_custom = (y_custom.max() / scale_out_custom)/2**32
    bits_needed_custom_ibert = (y_custom_ibert.max() / scale_out_custom_ibert)/2**32
    
    print(f"\nBits needed for output:")
    print(f"  I-ViT: {bits_needed_ivit:.6f}")
    print(f"  I-BERT: {bits_needed_ibert:.6f}")
    print(f"  Custom: {bits_needed_custom:.6f}")
    print(f"  Custom (IBERT): {bits_needed_custom_ibert:.6f}")

if __name__ == "__main__":
    run_comparison()
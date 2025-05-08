#!/usr/bin/env python3
# int_exp_comparison.py
# --------------------------------------------------------------
# Comparison of integer exponential approximations from I-ViT and IBERT
# Focuses on comparing only the exp approximation components
# --------------------------------------------------------------

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Utility functions for STE
# ------------------------------------------------------------------
class _FloorSTE(torch.autograd.Function):
    """Straight-through estimator for floor()"""
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, g):
        # identity gradient → straight-through
        return g

floor_ste = _FloorSTE.apply

# ------------------------------------------------------------------
# Isolated I-ViT int_exp_shift implementation
# ------------------------------------------------------------------
def int_exp_shift(x_int: torch.Tensor, scaling_factor: torch.Tensor, n=15):
    """
    Integer approximation of exp(x) in Q-domain using the shift approximation from I-ViT
    """
    # --- Shift approximation of exp(x) ---------------------
    x_int = x_int + floor_ste(x_int / 2) - floor_ste(x_int / 16)

    with torch.no_grad():
        x0_int = torch.floor(-1.0 / scaling_factor).to(x_int.device)

    x_int = torch.max(x_int, n * x0_int)

    # quotient / remainder decomposition
    q = floor_ste(x_int / x0_int)
    r = x_int - x0_int * q

    # build exp(r/q)
    exp_int = r / 2 - x0_int
    exp_int = torch.clamp(
        floor_ste(exp_int * 2 ** (n - q)),
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
    Integer approximation of exp(x) in Q-domain using the polynomial method from IBERT
    """
    x0 = -0.6931  # −ln(2)
    
    with torch.no_grad():
        x0_int = torch.floor(x0 / scaling_factor)

    x_int = torch.max(x_int, n * x0_int)

    q = floor_ste(x_int / x0_int)
    r = x_int - x0_int * q

    exp_int, exp_scale = int_polynomial(r, scaling_factor)
    exp_int = torch.clamp(
        floor_ste(exp_int * 2 ** (n - q)),
        min=0
    )
    scaling_factor_out = exp_scale / 2 ** n
    return exp_int, scaling_factor_out

# ------------------------------------------------------------------
# Comparison function
# ------------------------------------------------------------------
def run_comparison():
    # Setup input range from -11 to 0
    x = torch.linspace(-11, 0, 1000)
    
    # Set scaling factors
    scale_ivit = 22/2**16
    scale_ibert = 22/2**16
    
    # Compute float reference
    y_float = torch.exp(x)
    
    # Quantize inputs for integer domain
    x_int_ivit = (x / scale_ivit).to(torch.int32)
    x_int_ibert = (x / scale_ibert).to(torch.int32)
    
    # Apply integer exponential approximations
    exp_int_ivit, scale_out_ivit = int_exp_shift(x_int_ivit, torch.tensor(scale_ivit))
    y_ivit = exp_int_ivit * scale_out_ivit
    
    exp_int_ibert, scale_out_ibert = int_exp_ibert(x_int_ibert, torch.tensor(scale_ibert))
    y_ibert = exp_int_ibert * scale_out_ibert
    
    # Compute errors
    abs_err_ivit = (y_ivit - y_float).abs()
    rel_err_ivit = abs_err_ivit / (y_float + 1e-10) * 100  # percentage
    
    abs_err_ibert = (y_ibert - y_float).abs()
    rel_err_ibert = abs_err_ibert / (y_float + 1e-10) * 100  # percentage
    
    # Print error metrics
    print("=== I-ViT int_exp_shift Error Metrics ===")
    print(f"Max absolute error: {abs_err_ivit.max().item():.6f}")
    print(f"Mean absolute error: {abs_err_ivit.mean().item():.6f}")
    print(f"Max percentage error: {rel_err_ivit.max().item():.2f}%")
    print(f"Mean percentage error: {rel_err_ivit.mean().item():.2f}%")
    print()
    
    print("=== IBERT int_exp Error Metrics ===")
    print(f"Max absolute error: {abs_err_ibert.max().item():.6f}")
    print(f"Mean absolute error: {abs_err_ibert.mean().item():.6f}")
    print(f"Max percentage error: {rel_err_ibert.max().item():.2f}%")
    print(f"Mean percentage error: {rel_err_ibert.mean().item():.2f}%")
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Exponential function and approximations
    plt.subplot(2, 2, 1)
    plt.plot(x.numpy(), y_float.numpy(), label="Float exp(x)")
    plt.plot(x.numpy(), y_ivit.numpy(), '--', label="I-ViT int_exp_shift")
    plt.plot(x.numpy(), y_ibert.numpy(), ':', label="IBERT int_exp")
    plt.legend()
    plt.grid(True)
    plt.title("exp(x) vs Integer Approximations")
    plt.xlabel("Input x")
    plt.ylabel("exp(x)")
    
    # Plot 2: Absolute errors
    plt.subplot(2, 2, 2)
    plt.plot(x.numpy(), abs_err_ivit.numpy(), label="I-ViT absolute error")
    plt.plot(x.numpy(), abs_err_ibert.numpy(), label="IBERT absolute error")
    plt.legend()
    plt.grid(True)
    plt.title("Absolute Error")
    plt.xlabel("Input x")
    plt.ylabel("Absolute Error")
    
    # Plot 3: Percentage errors
    plt.subplot(2, 2, 3)
    plt.plot(x.numpy(), rel_err_ivit.numpy(), label="I-ViT percentage error")
    plt.plot(x.numpy(), rel_err_ibert.numpy(), label="IBERT percentage error")
    plt.legend()
    plt.grid(True)
    plt.title("Percentage Error")
    plt.xlabel("Input x")
    plt.ylabel("Error (%)")
    
    # Plot 4: Log scale for better visualization of small values
    plt.subplot(2, 2, 4)
    plt.semilogy(x.numpy(), y_float.numpy(), label="Float exp(x)")
    plt.semilogy(x.numpy(), y_ivit.numpy(), '--', label="I-ViT int_exp_shift")
    plt.semilogy(x.numpy(), y_ibert.numpy(), ':', label="IBERT int_exp")
    plt.legend()
    plt.grid(True)
    plt.title("exp(x) vs Integer Approximations (Log Scale)")
    plt.xlabel("Input x")
    plt.ylabel("exp(x)")
    
    plt.tight_layout()
    plt.show()

    # Additional analysis: error distribution across the input range
    plt.figure(figsize=(10, 6))
    
    # Create 10 equally spaced bins in our x range
    bins = np.linspace(-11, 0, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Compute average percentage error in each bin
    ivit_bin_errors = []
    ibert_bin_errors = []
    
    for i in range(len(bins)-1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if torch.any(mask):
            ivit_bin_errors.append(rel_err_ivit[mask].mean().item())
            ibert_bin_errors.append(rel_err_ibert[mask].mean().item())
        else:
            ivit_bin_errors.append(0)
            ibert_bin_errors.append(0)
    
    width = 0.35
    plt.bar(bin_centers - width/2, ivit_bin_errors, width, label='I-ViT')
    plt.bar(bin_centers + width/2, ibert_bin_errors, width, label='IBERT')
    
    plt.xlabel('Input Range')
    plt.ylabel('Average Percentage Error (%)')
    plt.title('Error Distribution Across Input Range')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()
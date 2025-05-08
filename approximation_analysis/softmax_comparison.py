#!/usr/bin/env python3
# softmax_int_approximations.py
# --------------------------------------------------------------
#  Integer‑domain Softmax approximations from
#   ‑ I‑ViT  (“Shiftmax”)  -> IntSoftmax_IVIT
#   ‑ IBERT  (“Polynomial+Shift”) -> IntSoftmax_IBERT
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
    plt.style.use("thesis_plot_styles.mplstyle")
except IOError:
    pass

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Small utility stubs so the script is self‑contained
# ------------------------------------------------------------------
class _FloorSTE(torch.autograd.Function):
    """Straight‑through estimator for floor()"""
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, g):
        # identity gradient → straight‑through
        return g


floor_ste = _FloorSTE.apply


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
# I‑ViT Shiftmax
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
    # integer exponential (same as in IntGELU_IVIT)
    # ----------------------------------------------------------
    def int_exp_shift(self, x_int: torch.Tensor, scaling_factor: torch.Tensor):
        """
        Integer approximation of exp(x) in Q-domain
        """
        # --- Shift approximation of exp(x) ---------------------
        x_int = x_int + floor_ste(x_int / 2) - floor_ste(x_int / 16)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor).to(x_int.device)

        x_int = torch.max(x_int, self.n * x0_int)

        # quotient / remainder decomposition
        q = floor_ste(x_int / x0_int)
        r = x_int - x0_int * q

        # build exp(r/q)
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(
            floor_ste(exp_int * 2 ** (self.n - q)),
            min=0
        )
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    # ----------------------------------------------------------
    # forward
    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor, scaling_factor: torch.Tensor):
        """
        Parameters
        ----------
        x               : (…, N) tensor of *floating* activations
        scaling_factor  : scalar tensor, same as in I‑ViT paper
        """
        device = x.device
        scaling_factor = scaling_factor.to(device)

        # 1) quantise input
        x_int = x / scaling_factor
        x_int = x_int.to(torch.int32)

        # 2) subtract (per‑row) max for numerical stability
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        # 3) integer exp
        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)

        # 4) normalise
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        exp_int_sum.clamp_max_(2 ** 31 - 1)
        factor = floor_ste((2 ** 31 - 1) / exp_int_sum)

        exp_int = floor_ste(exp_int * factor / 2 ** (31 - self.output_bit + 1))
        scaling_factor = torch.tensor(
            1.0 / 2 ** (self.output_bit - 1),
            device=device
        )

        # save scaling factor (nice for tensorboard / debugging)
        self.act_scaling_factor = scaling_factor.detach()
        return exp_int * scaling_factor, scaling_factor


# ------------------------------------------------------------------
# IBERT Softmax
# ------------------------------------------------------------------
class IntSoftmax_IBERT(nn.Module):
    """
    Polynomial‑based int Softmax from IBERT.
    Minor edits: CPU‑friendly & QuantAct stub
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

        # `QuantAct` could fuse a straight‑through fake‑quant
        self.act = QuantAct(16)

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

        q = floor_ste(x_int / x0_int)
        r = x_int - x0_int * q

        exp_int, exp_scale = self.int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(
            floor_ste(exp_int * 2 ** (self.n - q)),
            min=0
        )
        scaling_factor = exp_scale / 2 ** self.n
        return exp_int, scaling_factor

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor, scaling_factor: torch.Tensor):
        device = x.device
        scaling_factor = scaling_factor.to(device)

        # optional “float mode” passthrough
        if self.quant_mode == 'none':
            return F.softmax(x, dim=-1), None

        assert self.quant_mode == 'symmetric', \
            f"Unsupported quant mode: {self.quant_mode}"

        # 1) quantise input
        x_int = (x / scaling_factor).to(torch.int32)

        # 2) subtract max for numerical stability
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        # 3) integer exp, then a fake‑quant (QuantAct)
        exp_int, exp_scale = self.int_exp(x_int, scaling_factor)
        exp_q, exp_scale = self.act(exp_int, exp_scale)  # identity in stub
        exp_int = exp_q / exp_scale

        # 4) denominator
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        factor = floor_ste(2 ** 32 / exp_int_sum)

        # 5) scale into desired output bit‑width
        exp_int = floor_ste(exp_int * factor / 2 ** (32 - self.output_bit))
        scaling_factor = torch.tensor(
            1.0 / 2 ** self.output_bit,
            device=device
        )
        return exp_int * scaling_factor, scaling_factor


# ------------------------------------------------------------------
# Helper: pretty error stats
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
        description="Compare IntSoftmax_IVIT / IBERT to float softmax.\n"
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
    args = parser.parse_args()

    # ----------------------------------------------------------
    # 1.  Load or generate test data
    # ----------------------------------------------------------
    if args.x_file and args.x_file.exists() and args.scale_file and args.scale_file.exists():
        x_np = np.loadtxt(args.x_file, dtype=np.float32)
        scale_value = float(np.loadtxt(args.scale_file))
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # shape (1, N)
        scale_ivit = scale_ibert = scale_value
        print(f"Loaded x shape {x.shape}, scaling_factor {scale_value}")
    else:
        # throw error
        raise ValueError("No valid input files provided. Please provide --x-file and --scale-file.")

    device = torch.device("cpu")
    x = x.to(device)

    # ----------------------------------------------------------
    # 2.  Ground‑truth float softmax
    # ----------------------------------------------------------
    y_float = F.softmax(x, dim=-1)

    # ----------------------------------------------------------
    # 3.  Approximations
    # ----------------------------------------------------------
    ivit = IntSoftmax_IVIT(output_bit=args.output_bit)
    ibert = IntSoftmax_IBERT(output_bit=args.output_bit, quant_mode="symmetric")

    y_ivit, _ = ivit(x, scaling_factor=torch.tensor(scale_ivit))
    y_ibert, _ = ibert(x, scaling_factor=torch.tensor(scale_ibert))

    abs_err_ivit = (y_ivit - y_float).abs()
    abs_err_ibert = (y_ibert - y_float).abs()

   # ── 4) Visualization and error analysis ───────────────────────────────────────
    # ── 4) Visualization and error analysis ───────────────────────────────────────

    # 4.1 Print existing error‐stats
    print_stats("IViT IntSoftmax", abs_err_ivit)
    print_stats("IBERT IntSoftmax", abs_err_ibert)

    # 4.2 Bar chart for the (single) sample
    N = x.size(-1)
    classes = np.arange(N)
    idx = 0  # only one row in x
    plt.figure(figsize=(6,4))
    plt.bar(classes - 0.2, y_float[idx].cpu().numpy(),   width=0.2, label="float")
    plt.bar(classes      , y_ivit[idx].cpu().numpy(),    width=0.2, label="IViT")
    plt.bar(classes + 0.2, y_ibert[idx].cpu().numpy(),   width=0.2, label="IBERT")
    plt.title("Softmax distributions")
    plt.xlabel("class index")
    plt.ylabel("probability")
    plt.legend(); plt.grid(True)
    plt.show()

    # 4.3 Scatter plot: true vs approx probabilities
    flat_true   = y_float .view(-1).cpu().numpy()
    flat_ivit   = y_ivit  .view(-1).cpu().numpy()
    flat_ibert  = y_ibert .view(-1).cpu().numpy()
    plt.figure(figsize=(6,4))
    plt.scatter(flat_true, flat_ivit,  alpha=0.3, s=5, label="IViT")
    plt.scatter(flat_true, flat_ibert, alpha=0.3, s=5, label="IBERT")
    plt.plot([0,1],[0,1], 'k--')
    plt.title("True vs Approx Softmax")
    plt.xlabel("soft_true")
    plt.ylabel("soft_approx")
    plt.legend(); plt.grid(True)
    plt.show()

    # 4.4 Histogram of absolute errors
    err_ivit_vals  = abs_err_ivit .view(-1).cpu().numpy()
    err_ibert_vals = abs_err_ibert.view(-1).cpu().numpy()
    plt.figure(figsize=(6,4))
    plt.hist(err_ivit_vals,  bins=50, density=True, alpha=0.5, label="IViT")
    plt.hist(err_ibert_vals, bins=50, density=True, alpha=0.5, label="IBERT")
    plt.title("Distribution of |error|")
    plt.xlabel("absolute error")
    plt.ylabel("density")
    plt.legend(); plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()

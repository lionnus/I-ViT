#!/usr/bin/env python3
# softmax_approx_analysis.py
# --------------------------------------------------------------
# Comparison of the softmax approximations of I-ViT and I-BERT
# Approximations taken from: 
# I-BERT: https://github.com/kssteven418/I-BERT/ and 
# I-ViT: https://github.com/zkkli/I-ViT
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
        # --- Shift approximation of exp(x) ---------------------
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

    # ----------------------------------------------------------
    # forward
    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor, scaling_factor: torch.Tensor):
        """
        Parameters
        ----------
        x               : (…, N) tensor of *floating* activations
        scaling_factor  : scalar tensor
        """
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

        # save scaling factor (nice for tensorboard / debugging)
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

        # 3) integer exp, then a fake-quant (QuantAct)
        exp_int, exp_scale = self.int_exp(x_int, scaling_factor)
        exp_q, exp_scale = self.act(exp_int, exp_scale)  # identity in stub
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
        scale_ivit = scale_ibert = scale_value

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

    y_ivit, scaling_ivit_out = ivit(x, scaling_factor=torch.tensor(scale_ivit))
    y_ibert, scaling_ibert_out = ibert(x, scaling_factor=torch.tensor(scale_ibert))
    
    abs_err_ivit = (y_ivit - y_float).abs()
    abs_err_ibert = (y_ibert - y_float).abs()
    
    # ----------------------------------------------------------
    # 4.  Visualization and error analysis
    # ----------------------------------------------------------
    # 4.1 Print existing error‐stats
    print_stats("I-ViT IntSoftmax", abs_err_ivit)
    print_stats("I-BERT IntSoftmax", abs_err_ibert)

        
    # 4.2 Bar chart: pick a representative/random row
    sel = (0, 0, 0)               # (batch, head, token)
    true_row  = y_float [sel].cpu().numpy()
    ivit_row  = y_ivit  [sel].cpu().numpy()
    ibert_row = y_ibert [sel].cpu().numpy()

    classes = np.arange(true_row.size)
    # plt.figure(figsize=(7, 4))
    plt.bar(classes - 0.2, true_row,  width=0.2, label="float")
    plt.bar(classes      , ivit_row,  width=0.2, label="IViT")
    plt.bar(classes + 0.2, ibert_row, width=0.2, label="IBERT")
    plt.title(f"Softmax for sample {sel}")
    plt.xlabel("class index"); plt.ylabel("probability")
    plt.legend(); plt.grid(True); plt.show()

    # 4.3 Scatter plot on subset
    flat_true  = y_float .view(-1).cpu().numpy()
    flat_ivit  = y_ivit  .view(-1).cpu().numpy()
    flat_ibert = y_ibert .view(-1).cpu().numpy()

    keep = slice(None, None, 20)   # keep every n-th point/downsample
    plt.scatter(flat_true[keep], flat_ivit [keep],
            s=3, alpha=0.9, marker='.', linewidths=0,
            label="I-ViT")
    plt.scatter(flat_true[keep], flat_ibert[keep],
                s=3, alpha=0.9, marker='.', linewidths=0,
                label="I-BERT")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.7)
    plt.title("Float vs Integer Softmax (128 batch size, 197 model dim)")
    plt.xlabel("Float Softmax"); plt.ylabel("IntSoftmax")
    leg = plt.legend(scatterpoints=1, markerscale=6)  # markerscale increases legend dot size
    
    # now override the alpha on the legend
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    plt.grid(True); plt.tight_layout()
    # save plot as png in high dpi
    # plt.savefig("Softmax_comparison_batch_v1.png", dpi=300, bbox_inches='tight')
    plt.show()

    
    # 4.3 Density plot with hexbin
    plt.hexbin(flat_true, flat_ivit, gridsize=60, cmap='Blues', mincnt=1, linewidths=0.2, alpha=0.9)
    plt.hexbin(flat_true, flat_ibert, gridsize=60, cmap='Oranges', mincnt=1, linewidths=0.2, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.7)
    plt.title("Float vs Integer Softmax - density view")
    plt.xlabel("Float Softmax"); plt.ylabel("IntSoftmax")
    cb = plt.colorbar(label="count per bin")
    plt.grid(True); plt.tight_layout()
    plt.show()
    # 4.4 Histogram of absolute errors
    err_ivit_vals  = abs_err_ivit .view(-1).cpu().numpy()
    err_ibert_vals = abs_err_ibert.view(-1).cpu().numpy()
    # plt.figure(figsize=(6,4))
    plt.hist(err_ivit_vals,  bins=50, density=True, alpha=0.5, label="I-ViT")
    plt.hist(err_ibert_vals, bins=50, density=True, alpha=0.5, label="I-BERT")
    plt.title("Distribution of |error|")
    plt.xlabel("absolute error")
    plt.ylabel("density")
    plt.legend(); plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
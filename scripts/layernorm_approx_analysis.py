#!/usr/bin/env python3
# layernorm_approx_analysis.py
# --------------------------------------------------------------
# Comparison of the layernorm approximations of I-ViT and I-BERT
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

# thesis style
try:
    plt.style.use("scripts/thesis_plot_styles.mplstyle")
except IOError:
    pass

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# STE helpers
# ------------------------------------------------------------------
class round_ste(torch.autograd.Function):
    """
    STE for rounding operation
    """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class floor_ste(torch.autograd.Function):
    """
    STE for floor operation
    """
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# ------------------------------------------------------------------
# I-ViT LayerNorm
# ------------------------------------------------------------------
class IVITIntLayerNorm(nn.LayerNorm):
    """
    Implementation of I-LayerNorm
    Class to quantize given LayerNorm layer
    """
    def __init__(self,
                normalized_shape,
                eps=1e-6, # default value from I-ViT code
                elementwise_affine=True):
        super(IVITIntLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.dim_sqrt = None
        self.register_buffer('norm_scaling_factor', torch.zeros(normalized_shape))
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))
        
    def fix(self):
        pass
        
    def unfix(self):
        pass
        
    def forward(self, x, scaling_factor=None):
        if self.dim_sqrt is None:
            # Get last dimension size (feature dimension)
            n = torch.tensor(x.shape[-1], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).to(device=x.device)
            
        # Normalization: computes mean and variance
        x_int = x / scaling_factor
        # Use -1 as axis for the last dimension
        mean_int = round_ste.apply(x_int.mean(axis=-1, keepdim=True))
        x_int = x_int.to(dtype=torch.int32)
        mean_int = mean_int.to(dtype=torch.int32)
        y_int = x_int - mean_int
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=-1, keepdim=True)
        
        # Integer Iteration
        k = 2 ** 16
        for _ in range(10):
            k_1 = floor_ste.apply((k + floor_ste.apply(var_int/k))/2)
            k = k_1
        std_int = k
        
        factor = floor_ste.apply((2 ** 31-1) / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2 ** 30
        
        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)
        self.bias_integer = bias_int
        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor
        
        self.norm_scaling_factor = scaling_factor
        return x, scaling_factor


# ------------------------------------------------------------------
# I-BERT LayerNorm
# ------------------------------------------------------------------
class IBERTIntLayerNorm(nn.Module):
    """
    Class to quantize given LayerNorm layer

    Parameters:
    ----------
    output_bit : int
        Bitwidth for the LayerNorm output.
    overflow_handling : bool, default True
        Whether to do overflow handling if the intermediate values are larger than 32-bit.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    force_dequant : str, default 'none'
        Force dequantize LayerNorm if either 'layernorm' or 'nonlinear' is given.
    """
    def __init__(self,
                 normalized_shape,
                 output_bit=8,
                 overflow_handling=True,
                 quant_mode='symmetric',
                 force_dequant='none',
                 elementwise_affine=True,
                 eps=1e-5):
        super(IBERTIntLayerNorm, self).__init__()
        self.quant_mode = quant_mode
        if force_dequant in ['nonlinear', 'layernorm']:
            logging.info("Force dequantize layernorm")
            self.quant_mode = 'none'
        self.overflow_handling = overflow_handling
        self.register_buffer('shift', torch.zeros(1))
        self.output_bit = output_bit
        self.dim_sqrt = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = eps

    def fix(self):
        self.overflow_handling = False

    def unfix(self):
        self.overflow_handling = True

    def set_shift(self, y_int):
        with torch.no_grad():
            y_sq_int = y_int ** 2
            var_int = torch.sum(y_sq_int, axis=-1, keepdim=True)
            shift = (torch.log2(torch.sqrt(var_int / 2**32)).ceil()).max()
            shift_old = self.shift
            self.shift = torch.max(self.shift, shift)
            logging.info("Dynamic shift adjustment: {} -> {}".format(
                int(shift_old), int(self.shift)))

    def overflow_fallback(self, y_int):
        self.set_shift(y_int)
        y_int_shifted = floor_ste.apply(y_int / 2 ** self.shift)
        y_sq_int = y_int_shifted ** 2
        var_int = torch.sum(y_sq_int, axis=-1, keepdim=True)
        return var_int

    def forward(self, x, scaling_factor=None, exponents=None):
        if self.quant_mode == 'none':
            mean = x.mean(axis=-1, keepdim=True)
            y = x - mean
            var = torch.mean(y ** 2, axis=-1, keepdim=True)
            x = y / torch.sqrt(self.eps + var)
            x = x * self.weight + self.bias
            return x, None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(self.quant_mode)

        if self.dim_sqrt is None:
            # Get last dimension size (feature dimension)
            n = torch.tensor(x.shape[-1], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).to(device=x.device)

        # Normalization: computes mean and variance(std)
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=-1, keepdim=True))
        y_int = x_int - mean_int
        y_int_shifted = floor_ste.apply(y_int / 2 ** self.shift) # avoid overflow
        y_sq_int = y_int_shifted ** 2
        var_int = torch.sum(y_sq_int, axis=-1, keepdim=True)
        
        # overflow handling in training stage
        if self.overflow_handling:
            if var_int.max() >= 2**32:
                var_int = self.overflow_fallback(y_int)
                assert var_int.max() < 2**32
        
        # To be replaced with integer-sqrt kernel that produces the same output
        std_int = floor_ste.apply(torch.sqrt(var_int)) * 2 ** self.shift 
        factor = floor_ste.apply(2**31 / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2**30

        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor

        return x, scaling_factor


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
# Helper: check if values are within int8 range
# ------------------------------------------------------------------
def check_int8_range(tensor, name):
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    int8_min = -128
    int8_max = 127
    within_range = (min_val >= int8_min) and (max_val <= int8_max)
    
    print(f"{name:18s} | min: {min_val:.4f} | max: {max_val:.4f} | within int8 range: {within_range}")
    
    # If not within range, calculate scaling factor needed to fit in int8
    if not within_range:
        abs_max = max(abs(min_val), abs(max_val))
        suggested_scale = abs_max / 127.0  # Scale to fit within int8 range
        print(f"{' ':18s} | suggested scaling factor: {suggested_scale:.6f}")
    
    return within_range


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare IVITIntLayerNorm / IBERTIntLayerNorm to float layernorm.\n"
        "If --x-file / --scale-file are provided, the script uses real "
        "tensors captured during debugging; otherwise it falls back to a "
        "fake sweep(doesn't make sense tho)."
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
        help="Original tensor shape, comma-separated, e.g. 1,197,768"
    )
    
    parser.add_argument(
        "--weights-file",
        type=Path,
        default=None,
        help="Text file with flattened weights tensor for layer norm"
    )
    
    parser.add_argument(
        "--bias-file",
        type=Path,
        default=None,
        help="Text file with flattened bias tensor for layer norm"
    )
    
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data if no input files are provided"
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
                "example: --shape 128,3,197,768"
            )

        tgt_shape = tuple(int(s) for s in args.shape.split(","))
        x = torch.tensor(x_np, dtype=torch.float32).view(*tgt_shape)
        print(f"Loaded x shape {x.shape}, scaling_factor {scale_value}")
    elif args.synthetic:
        # Generate synthetic data if requested
        if args.shape is None:
            tgt_shape = (1, 197, 192)  # Default synthetic shape
        else:
            tgt_shape = tuple(int(s) for s in args.shape.split(","))
            
        # Generate random data following a normal distribution
        x = torch.randn(tgt_shape, dtype=torch.float32)
        scale_value = 0.1  # Arbitrary scaling factor
        scale_ivit = scale_ibert = scale_value
        print(f"Generated synthetic data with shape {x.shape}, scaling_factor {scale_value}")
    else:
        raise ValueError("No valid input files provided, provide --x-file and --scale-file or use --synthetic.")

    device = torch.device("cpu")
    x = x.to(device)

    # ----------------------------------------------------------
    # 2.  Ground-truth float layernorm
    # ----------------------------------------------------------
    # PyTorch LayerNorm
    layernorm_shape = x.shape[-1]
    layernorm_float = nn.LayerNorm(layernorm_shape).to(device)
    
    # Load weights and bias from files if provided, otherwise initialize to default values
    if args.weights_file and args.weights_file.exists() and args.bias_file and args.bias_file.exists():
        weights_np = np.loadtxt(args.weights_file, dtype=np.float32)
        bias_np = np.loadtxt(args.bias_file, dtype=np.float32)
        
        # Make sure weights and bias match the expected shape
        if weights_np.shape[0] != layernorm_shape or bias_np.shape[0] != layernorm_shape:
            print(f"Warning: weights/bias dimensions ({weights_np.shape[0]}/{bias_np.shape[0]}) don't match layer shape ({layernorm_shape})")
            print("Falling back to default initialization")
            layernorm_float.weight.data.fill_(1.0)  # Initialize weight to 1
            layernorm_float.bias.data.fill_(0.0)   # Initialize bias to 0
        else:
            # Load weights and bias
            layernorm_float.weight.data = torch.tensor(weights_np, dtype=torch.float32).to(device)
            layernorm_float.bias.data = torch.tensor(bias_np, dtype=torch.float32).to(device)
            print(f"Loaded weights and bias from files")
    else:
        layernorm_float.weight.data.fill_(1.0)  # Initialize weight to 1
        layernorm_float.bias.data.fill_(0.0)   # Initialize bias to 0
        print("Using default weights (1.0) and bias (0.0)")
    
    y_float = layernorm_float(x)

    # ----------------------------------------------------------
    # 3.  Approximations
    # ----------------------------------------------------------
    ivit = IVITIntLayerNorm(layernorm_shape).to(device)
    ibert = IBERTIntLayerNorm(layernorm_shape, output_bit=args.output_bit, quant_mode="symmetric").to(device)
    
    # Make sure the parameters match the float version
    ivit.weight.data.copy_(layernorm_float.weight.data)
    ivit.bias.data.copy_(layernorm_float.bias.data)
    ibert.weight.data.copy_(layernorm_float.weight.data)
    ibert.bias.data.copy_(layernorm_float.bias.data)
    ivit.fix()
    ibert.fix()

    y_ivit, scaling_ivit_out = ivit(x, scaling_factor=torch.tensor(scale_ivit))
    y_ibert, scaling_ibert_out = ibert(x, scaling_factor=torch.tensor(scale_ibert))
    
    abs_err_ivit = (y_ivit - y_float).abs()
    abs_err_ibert = (y_ibert - y_float).abs()
    
    # ----------------------------------------------------------
    # 4.  Check int8 range compliance
    # ----------------------------------------------------------
    # Check input values
    x_scaled = x / scale_value
    check_int8_range(x_scaled, "Input (x_scaled)")
    
    # Check weights and bias
    check_int8_range(layernorm_float.weight.data, "LayerNorm weights")
    check_int8_range(layernorm_float.bias.data, "LayerNorm bias")
    
    # Check intermediate values
    check_int8_range(ivit.bias_integer, "I-ViT bias_integer")
    
    # Check output values (before scaling back)
    y_ivit_int = y_ivit / scaling_ivit_out if scaling_ivit_out is not None else y_ivit
    y_ibert_int = y_ibert / scaling_ibert_out if scaling_ibert_out is not None else y_ibert
    
    check_int8_range(y_ivit_int, "I-ViT output (int)")
    check_int8_range(y_ibert_int, "I-BERT output (int)")
    
    print("\n===== Error statistics =====")
    # ----------------------------------------------------------
    # 5.  Visualization and error analysis
    # ----------------------------------------------------------
    # 5.1 Print error‐stats
    print_stats("I-ViT IntLayerNorm", abs_err_ivit)
    print_stats("I-BERT IntLayerNorm", abs_err_ibert)

    # 5.2 Bar chart random sample
    # Adjust selection based on tensor dimensionality
    if len(x.shape) == 3:  # For 3D tensors (batch, seq_len, hidden_dim)
        sel = (0, 0)  # (batch, token)
    else:  # For 4D tensors (batch, head, seq_len, hidden_dim)
        sel = (0, 0, 0)  # (batch, head, token)
        
    true_row = y_float[sel].detach().cpu().numpy()
    ivit_row = y_ivit[sel].detach().cpu().numpy()
    ibert_row = y_ibert[sel].detach().cpu().numpy()

    # Plotting just a subset of features for clarity
    num_features = min(20, true_row.size)
    feature_indices = np.arange(num_features)
    
    plt.figure(figsize=(10, 5))
    plt.bar(feature_indices - 0.2, true_row[:num_features], width=0.2, label="float")
    plt.bar(feature_indices, ivit_row[:num_features], width=0.2, label="IViT")
    plt.bar(feature_indices + 0.2, ibert_row[:num_features], width=0.2, label="IBERT")
    plt.title(f"LayerNorm for sample {sel} (first {num_features} features)")
    plt.xlabel("feature index")
    plt.ylabel("normalized value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5.3 Scatter plot on subset
    flat_true = y_float.view(-1).detach().cpu().numpy()
    flat_ivit = y_ivit.view(-1).detach().cpu().numpy()
    flat_ibert = y_ibert.view(-1).detach().cpu().numpy()

    keep = slice(None, None, 20)  # keep every n-th point/downsample
    plt.figure(figsize=(8, 6))
    plt.scatter(flat_true[keep], flat_ivit[keep],
            s=3, alpha=0.9, marker='.', linewidths=0,
            label="I-ViT")
    plt.scatter(flat_true[keep], flat_ibert[keep],
                s=3, alpha=0.9, marker='.', linewidths=0,
                label="I-BERT")
                
    # Add diagonal reference line
    y_min = min(flat_true.min(), flat_ivit.min(), flat_ibert.min())
    y_max = max(flat_true.max(), flat_ivit.max(), flat_ibert.max())
    plt.plot([y_min, y_max], [y_min, y_max], 'k--', linewidth=0.7)
    
    plt.title(f"Float vs Integer LayerNorm ({x.shape[0]} batch size, {x.shape[-1]} feature dim)")
    plt.xlabel("Float LayerNorm")
    plt.ylabel("IntLayerNorm")
    leg = plt.legend(scatterpoints=1, markerscale=6)  # increases legend dot size
    
    # Override alpha on the legend
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("LayerNorm_comparison_batch.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 5.4 Histogram of absolute errors
    err_ivit_vals = abs_err_ivit.view(-1).detach().cpu().numpy()
    err_ibert_vals = abs_err_ibert.view(-1).detach().cpu().numpy()
    
    plt.figure(figsize=(8, 5))
    plt.hist(err_ivit_vals, bins=50, density=True, alpha=0.5, label="I-ViT")
    plt.hist(err_ibert_vals, bins=50, density=True, alpha=0.5, label="I-BERT")
    plt.title("Distribution of |error|")
    plt.xlabel("absolute error")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # One-liner commands to export trained model data:
    # 1. Export input tensor: np.savetxt("input_tensor.txt", model_input.detach().cpu().numpy().flatten())
    # 2. Export scaling factor: np.savetxt("scale_factor.txt", [scaling_factor.item()])
    # 3. Export LayerNorm weights: np.savetxt("layernorm_weights.txt", model.layernorm.weight.data.cpu().numpy())
    # 4. Export LayerNorm bias: np.savetxt("layernorm_bias.txt", model.layernorm.bias.data.cpu().numpy())
    # 
    # Then run this script with:
    # python layernorm_approx_analysis.py --x-file input_tensor.txt --scale-file scale_factor.txt --weights-file layernorm_weights.txt --bias-file layernorm_bias.txt --shape batch,seq_len,hidden_dim
    
    main()
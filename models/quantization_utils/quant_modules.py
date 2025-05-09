"""
Extended quantized layers with a light-weight I/O-statistics collector.
"""

import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Parameter
import pandas as pd  # used only at export time of IO stats
from functools import partial

from .quant_utils import *

# -------------------------------------------------------------------------
#                           I/O-stat collector
# -------------------------------------------------------------------------
_LAYER_IO_STATS = []        # global buffer (unchanged)

def _collect_io_stats(module, inputs, output, module_name):
    """
    Record integer extrema + shapes for every layer call.
    For layers with two runtime inputs (e.g. QuantMatMul) it also logs
    per-input min/max + shape.
    """
    try:
        # ----------- unpack main in/out tensors & scales -----------
        x_scaled = inputs[0]
        scale_x  = inputs[1] if len(inputs) > 1 and torch.is_tensor(inputs[1]) else None

        y_scaled = output[0] if isinstance(output, (tuple, list)) else output
        scale_y  = output[1] if isinstance(output, (tuple, list)) and torch.is_tensor(output[1]) else None

        xs, ys = x_scaled.detach(), y_scaled.detach()
        xs_int = xs / scale_x if scale_x is not None else None
        ys_int = ys / scale_y if scale_y is not None else None

        rec = {
            "layer":       module_name,
            "type":        module.__class__.__name__,
            "min_in":    xs.min().item()  if xs is not None else None,
            "max_in":    xs.max().item()  if xs is not None else None,
            "min_out":   ys.min().item()  if ys is not None else None,
            "max_out":   ys.max().item()  if ys is not None else None,
            "scale_in":  scale_x.item() if scale_x is not None else None,
            "scale_out": scale_y.item() if scale_y is not None else None,
            "min_in_int":  xs_int.min().item()  if xs_int is not None else None,
            "max_in_int":  xs_int.max().item()  if xs_int is not None else None,
            "min_out_int": ys_int.min().item()  if ys_int is not None else None,
            "max_out_int": ys_int.max().item()  if ys_int is not None else None,
            "shape_in":    tuple(xs.shape),
            "shape_out":   tuple(ys.shape),
        }

        # ----------- extra block for two-input layers -----------
        if isinstance(module, QuantMatMul):
            # A == inputs[0],  scale_A == inputs[1]
            # B == inputs[2],  scale_B == inputs[3]
            A, sA = inputs[0].detach(), inputs[1]
            B, sB = inputs[2].detach(), inputs[3]
            A_int, B_int = A / sA, B / sB

            rec.update({
                "min_A_int":  A_int.min().item(),
                "max_A_int":  A_int.max().item(),
                "shape_A":    tuple(A.shape),
                "min_B_int":  B_int.min().item(),
                "max_B_int":  B_int.max().item(),
                "shape_B":    tuple(B.shape),
            })

        _LAYER_IO_STATS.append(rec)

    except Exception:
        # swallow any hook errors so evaluation never breaks
        pass

def attach_io_stat_hooks(model: nn.Module):
    """Recursively attach `_collect_io_stats` as a forward hook to every sub-module.
    """
    for name, module in model.named_modules():
        if module is model:  # skip the top-level container to avoid double-logging
            continue
        module.register_forward_hook(partial(_collect_io_stats, module_name=name))


def get_io_stats_df() -> pd.DataFrame:
    """Return a new DataFrame of all gathered statistics."""
    return pd.DataFrame(_LAYER_IO_STATS)


def save_io_stats_df(path: str = "io_stats.pkl", to_csv: bool = False) -> pd.DataFrame:
    """Export collected statistics to `path` (Pickle), plus optional CSV.

    Returns the DataFrame so callers may inspect / pretty-print immediately.
    """
    df = get_io_stats_df()
    df.to_pickle(path)
    if to_csv:
        csv_path = path.rsplit(".", 1)[0] + ".csv"
        df.to_csv(csv_path, index=False)
    return df

# -------------------------------------------------------------------------
#                           Original quantized layers
# -------------------------------------------------------------------------

class QuantLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer

    Parameters:
    ----------
    weight_bit : int
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 per_channel=True,
                 quant_mode='symmetric'):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        if self.bias is not None:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(
            self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, prev_act_scaling_factor=None):
        """
        using quantized weights to forward activation x
        """
        if self.quant_mode == 'none':
            return F.linear(x, weight=self.weight, bias=self.bias), None

    	# x / prev_act_scaling_factor = int
        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)

        # assert that prev_act_scaling_factor is a scalar tensor
        # e.g. all input tensors have the same scalar factor
        assert prev_act_scaling_factor is not None and \
              prev_act_scaling_factor.shape == (1,) 

        w = self.weight
        w_transform = w.data.detach()
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        self.fc_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, w_min, w_max)
        self.weight_integer = self.weight_function(
                self.weight, self.weight_bit, 
                self.fc_scaling_factor, True)

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        self.bias_integer = self.weight_function(self.bias, 
                self.bias_bit, bias_scaling_factor, True)

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) \
                * bias_scaling_factor, bias_scaling_factor


class QuantAct(nn.Module):
    """
    Class to quantize given activations

    Parameters:
    ----------
    activation_bit : int
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    channel_len : int, default None
        Specify the channel length when using the per_channel mode.
    quant_mode : 'none' or 'symmetric', default 'symmetric'
        The mode for quantization. 'none' for no quantization.
    """
    def __init__(self,
                 activation_bit=8,
                 act_range_momentum=0.95,
                 running_stat=True,
                 per_channel=False,
                 channel_len=None,
                 quant_mode="symmetric"):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.percentile = False

        if not per_channel:
            self.register_buffer('x_min', torch.zeros(1))
            self.register_buffer('x_max', torch.zeros(1))
            self.register_buffer('act_scaling_factor', torch.zeros(1))
        else:
            assert channel_len is not None
            self.register_buffer('x_min', torch.zeros(channel_len))
            self.register_buffer('x_max', torch.zeros(channel_len))
            self.register_buffer('act_scaling_factor', torch.zeros(channel_len))

        self.quant_mode = quant_mode
        self.per_channel = per_channel

        if self.quant_mode == "none":
            self.act_function = None
        elif self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        return "{0}(activation_bit={1}, " \
               "quant_mode: {2}, Act_min: {3:.2f}, " \
               "Act_max: {4:.2f})".format(self.__class__.__name__, self.activation_bit,
                                          self.quant_mode, self.x_min.item(), self.x_max.item())
    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False
        
    def unfix(self):
        """
        unfix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(self, x, 
                pre_act_scaling_factor=None, 
                identity=None, 
                identity_scaling_factor=None,
                specified_min=None,
                specified_max=None):
        # collect runnng stats
        x_act = x if identity is None else identity + x
        if self.running_stat:
            if not self.percentile:
                if not self.per_channel:
                    x_min = x_act.data.min()
                    x_max = x_act.data.max()
                else:
                    x_min = x_act.data.min(axis=0).values.min(axis=0).values
                    x_max = x_act.data.max(axis=0).values.max(axis=0).values
            else:
                raise NotImplementedError("percentile mode is not currently supported.")

            # Initialization
            if torch.eq(self.x_min, self.x_max).all():
                self.x_min = self.x_min + x_min
                self.x_max = self.x_max + x_max

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every iteration
            elif self.act_range_momentum == -1:
                self.x_min = torch.min(self.x_min, x_min)
                self.x_max = torch.max(self.x_max, x_max)
            else:
                self.x_min = self.x_min * self.act_range_momentum +\
                        x_min * (1 - self.act_range_momentum)
                self.x_max = self.x_max * self.act_range_momentum +\
                        x_max * (1 - self.act_range_momentum)

        if self.quant_mode == 'none':
            return x_act, None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)
        
        x_min = self.x_min if specified_min is None else specified_min
        x_max = self.x_max if specified_max is None else specified_max

        self.act_scaling_factor = symmetric_linear_quantization_params(
            self.activation_bit, x_min, x_max)

        if pre_act_scaling_factor is None:
            # this is for the input quantization 
            quant_act_int = self.act_function(x, self.activation_bit, \
                    self.act_scaling_factor, False)
        else:
            quant_act_int = fixedpoint_mul.apply(
                    x, pre_act_scaling_factor, 
                    self.activation_bit, self.quant_mode, 
                    self.act_scaling_factor, 
                    identity, identity_scaling_factor)

        correct_output_scale = self.act_scaling_factor.view(-1)

        return quant_act_int * correct_output_scale, self.act_scaling_factor


class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given matmul layer
    """
    def __init__(self):
        super(QuantMatMul, self).__init__()
        self.register_buffer('act_scaling_factor', torch.zeros(1))
    
    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        A_int = A / pre_act_scaling_factor_A
        B_int = B / pre_act_scaling_factor_B
        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        self.act_scaling_factor = act_scaling_factor
        return (A_int @ B_int) * act_scaling_factor, act_scaling_factor


class QuantConv2d(nn.Conv2d):
    """
    Class to quantize weights of given convolutional layer
    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 quant_mode="symmetric",
                 per_channel=True,
                 weight_percentile=0):
        super(QuantConv2d, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias
                                          )
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)

        self.register_buffer('conv_scaling_factor', torch.zeros(self.out_channels))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, pre_act_scaling_factor=None):
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception('For weight, we only support per_channel quantization.')

            self.conv_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val)

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.conv_scaling_factor, True)
        bias_scaling_factor = self.conv_scaling_factor * pre_act_scaling_factor
        self.bias_integer = self.weight_function(
            self.bias, self.bias_bit, bias_scaling_factor, True)

        pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
        x_int = x / pre_act_scaling_factor
        x_int.to(dtype=torch.int32)
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

        return (F.conv2d(x_int, self.weight_integer, self.bias_integer, self.stride, self.padding,
                         self.dilation, self.groups) * correct_output_scale, correct_output_scale)


class IntLayerNorm(nn.Module):
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
                 quant_mode='none',
                 force_dequant='none',
                 elementwise_affine=True,
                 eps=1e-5):
        super(IntLayerNorm, self).__init__()
        self.quant_mode = quant_mode
        if force_dequant in ['nonlinear', 'layernorm']:
            logger.info("Force dequantize layernorm")
            self.quant_mode = 'none'
        self.overflow_handling = overflow_handling
        self.register_buffer('shift', torch.zeros(1))
        self.output_bit = output_bit
        self.dim_sqrt = None
        if isinstance(normalized_shape,int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = eps

        self.activation = QuantAct(output_bit, quant_mode=self.quant_mode)
        if self.quant_mode == "none":
            pass
        elif quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def fix(self):
        self.overflow_handling = False

    def unfix(self):
        self.overflow_handling = True

    def set_shift(self, y_int):
        with torch.no_grad():
            y_sq_int = y_int ** 2
            var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
            shift = (torch.log2(torch.sqrt(var_int / 2**32)).ceil()).max()
            shift_old = self.shift
            self.shift = torch.max(self.shift, shift)
            logger.info("Dynamic shift adjustment: {} -> {}".format(
                int(shift_old), int(self.shift)))

    def overflow_fallback(self, y_int):
        self.set_shift(y_int)
        y_int_shifted = floor_ste.apply(y_int / 2 ** self.shift)
        y_sq_int = y_int_shifted ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        return var_int

    def forward(self, x, scaling_factor=None, exponents=None):
        if self.quant_mode == 'none':
            mean = x.mean(axis=2, keepdim=True)
            y = x - mean
            var = torch.mean(y ** 2, axis=2, keepdim=True)
            x = y / torch.sqrt(self.eps + var)
            x = x * self.weight + self.bias
            return x, None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)

        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float) # feature dim(768)
            self.dim_sqrt = torch.sqrt(n).cuda()

        # Normalization: computes mean and variance(std)
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_int_shifted = floor_ste.apply(y_int / 2 ** self.shift) # avoid overflow
        y_sq_int = y_int_shifted ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        
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
    

class IntGELU(nn.Module):
    """
    Class to quantize given GELU layer

    Parameters:
    ----------
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    force_dequant : str, default 'none'
        Force dequantize GELU if either 'gelu' or 'nonlinear' is given.
    """
    def __init__(self,
                 quant_mode='none',
                 force_dequant='none'):
        super(IntGELU, self).__init__()
        self.register_buffer('input_scaling_factor', torch.ones(1))
        self.quant_mode = quant_mode
        if force_dequant in ['nonlinear', 'gelu']:
            logger.info("Force dequantize gelu")
            self.quant_mode = 'none'


        if self.quant_mode == 'none':
            self.activation_fn = nn.GELU()
        elif self.quant_mode == 'symmetric':
            pass
        elif quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

        self.k = 1.4142
        self.n = 14 # sufficiently large integer
        self.coeff = [-0.2888, -1.769, 1] # a(x+b)**2 + c
        self.coeff[2] /= self.coeff[0]

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_erf(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coeff[1] / scaling_factor)
            c_int = torch.floor(self.coeff[2] / scaling_factor ** 2)

        with torch.no_grad():
            sign = torch.sign(x_int)
        abs_int = torch.abs(x_int)
        abs_int = torch.min(abs_int, -b_int)
        y_int = (abs_int + b_int) ** 2 + c_int
        y_int = sign * y_int
        scaling_factor = scaling_factor ** 2 * self.coeff[0]
        y_int = floor_ste.apply(y_int / 2 ** self.n)
        scaling_factor = scaling_factor * 2 ** self.n
        
        return y_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        if self.quant_mode == 'none':
            return self.activation_fn(x), None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)

        x_int = x / scaling_factor
        sigmoid_int, sigmoid_scaling_factor = self.int_erf(x_int, scaling_factor / self.k)

        shift_int = torch.floor(1. / sigmoid_scaling_factor)

        x_int = x_int * (sigmoid_int + shift_int)
        scaling_factor = scaling_factor * sigmoid_scaling_factor / 2

        return x_int * scaling_factor, scaling_factor

class IntSoftmax(nn.Module):
    """
    Class to quantize given Softmax layer

    Parameters:
    ----------
    output_bit : int
        Bitwidth for the Softmax output.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    force_dequant : str, default 'none'
        Force dequantize Softmax if either 'softmax' or 'nonlinear' is given.
    """
    def __init__(self,
                 output_bit,
                 quant_mode='symmetric',
                 force_dequant='none'):
        super(IntSoftmax, self).__init__()
        self.output_bit = output_bit
        self.quant_mode = quant_mode
        if force_dequant in ['nonlinear', 'softmax']:
            logger.info("Force dequantize softmax")
            self.quant_mode = 'none'


        self.act = QuantAct(16, quant_mode=self.quant_mode)
        self.x0 = -0.6931 # -ln2
        self.n = 30 # sufficiently large integer
        self.coef = [0.35815147, 0.96963238, 1.] # ax**2 + bx + c
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]

    def fix(self):
        pass

    def unfix(self):
        pass

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

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int, exp_scaling_factor = self.int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = exp_scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        if self.quant_mode == 'none':
            return softmax(x, dim=-1, onnx_trace=False), None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)

        x_int = x / scaling_factor

        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max


        exp_int, exp_scaling_factor = self.int_exp(x_int, scaling_factor)
        exp, exp_scaling_factor = self.act(exp_int, exp_scaling_factor)
        exp_int = exp / exp_scaling_factor
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        factor = floor_ste.apply(2**32 / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (32 - self.output_bit + 1))
        # scaling_factor = 1 / 2 ** self.output_bit / 2 # TODO Integrate /2-> now fixing the output to 16 bit signed, instead of 16 bit unsigned.
        scaling_factor = torch.Tensor([2 / 2 ** (self.output_bit)]).cuda(device=x.device)
        return exp_int * scaling_factor, scaling_factor

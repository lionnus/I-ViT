"""
Piecewise Polynomial Approximation Modules
"""

import torch
import torch.nn as nn
import numpy as np
import math
from scipy.special import erf
from .quant_utils import floor_ste
from .ibert_modules import IBERTIntGELU
from .quant_modules import QuantAct
from .ppoly_backend import fit_piecewise_polynomials, compute_integer_coefficients, evaluate_piecewise_polynomial


class PPolyIntGELU(nn.Module):
    """
    Implementation of Integer GELU using piecewise polynomial approximation
    Uses float GELU for gradient computation to avoid OOM issues
    """

    def __init__(self, output_bit=8, scale_bits=22, seg=16, deg=2, backend='ibert',
                 alpha=0.0, optim_bounds=True):
        super(PPolyIntGELU, self).__init__()
        self.N = scale_bits  # kept for backward compatibility
        self.segments = seg
        self.degree = deg
        self.backend = backend
        self.alpha = alpha
        self.optim_bounds = optim_bounds
        self.fixed = False

        self.gelu_module = nn.GELU()
        self.register_buffer('act_scaling_factor', torch.zeros(1))

        self.register_buffer('fixed_bounds', None)
        self.register_buffer('fixed_coeffs', None)
        self.register_buffer('fixed_scaling_factor_out', None)

    def _gelu_func(self, x):
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))

    def fix(self):
        self.fixed = True

    def unfix(self):
        self.fixed = False
        self.fixed_bounds = None
        self.fixed_coeffs = None
        self.fixed_scaling_factor_out = None

    def _compute_integer_coefficients(self, x, scaling_factor):
        x_lo = torch.floor(torch.min(x)).item()
        x_hi = torch.ceil(torch.max(x)).item()

        if self.backend == 'ibert' and x is not None and scaling_factor is not None:
            xs = torch.linspace(x_lo, x_hi, 10000, device=scaling_factor.device, dtype=scaling_factor.dtype)
            ibert = IBERTIntGELU()
            ys, _ = ibert(xs, scaling_factor)
            ys_np = ys.detach().cpu().numpy()
            xs_np = xs.detach().cpu().numpy()
        else:
            xs_np = np.linspace(x_lo, x_hi, 10000, dtype=np.float32)
            ys_np = self._gelu_func(xs_np)

        float_pieces = fit_piecewise_polynomials(
            xs_np, ys_np, x_lo, x_hi,
            self.segments, self.degree,
            self.alpha,
            optim_bounds=self.optim_bounds
        )

        bounds, coeffs_tensor, output_scale = compute_integer_coefficients(
            float_pieces,
            scaling_factor=scaling_factor,
            N=None,          # NEW MODE
            device=x.device,
            verbose=False
        )

        if self.backend == 'ibert':
            k = 1.4142
            n = 6
            coeff = [-0.2888, -1.769, 1]
            scaling_factor_out = scaling_factor / k
            scaling_factor_out = scaling_factor_out ** 2 * coeff[0]
            scaling_factor_out = scaling_factor_out * (2 ** n)
            scaling_factor_out = scaling_factor * scaling_factor_out / 2
        else:
            # Use power-of-two output_scale chosen by backend
            scaling_factor_out = scaling_factor / output_scale

        self.fixed_bounds = bounds
        self.fixed_coeffs = coeffs_tensor
        self.fixed_scaling_factor_out = scaling_factor_out

        return bounds, coeffs_tensor, scaling_factor_out

    def forward(self, x, scaling_factor):
        x_int = floor_ste.apply(x / scaling_factor)

        with torch.no_grad():
            if self.fixed and self.fixed_coeffs is not None:
                bounds_int = self.fixed_bounds
                coeffs_tensor = self.fixed_coeffs
                scaling_factor_out = self.fixed_scaling_factor_out
            else:
                bounds_int, coeffs_tensor, scaling_factor_out = self._compute_integer_coefficients(x, scaling_factor)

        with torch.no_grad():
            y_int = evaluate_piecewise_polynomial(x_int, bounds_int, coeffs_tensor, self.segments, self.degree)

        y_float_gelu = self.gelu_module(x)

        # integer -> float using scaling_factor_out
        y_float = y_int.float() * scaling_factor_out

        y_float = y_float.detach() + (y_float_gelu - y_float_gelu.detach())
        y_float = scaling_factor_out * floor_ste.apply(y_float / scaling_factor_out)
        return y_float, scaling_factor_out


class PPolyIntSoftmax(nn.Module):
    """
    Implementation of Integer Softmax using piecewise polynomial approximation for exp()
    """

    def __init__(self, output_bit=8, scale_bits=28, exp_bits=16, seg=16, deg=2,
                 backend='float', alpha=0.0, optim_bounds=False):
        super().__init__()
        self.output_bit = output_bit
        self.N = scale_bits  # kept for backward compatibility
        self.exp_bitwidth = exp_bits
        self.segments = seg
        self.degree = deg
        self.backend = backend
        self.alpha = alpha
        self.optim_bounds = optim_bounds
        self.fixed = False

        self.softmax_module = nn.Softmax(dim=-1)

        if self.backend == 'ibert':
            self.x0 = -0.6931
            self.n = 30
            self.coef = [0.35815147, 0.96963238, 1.]
            self.coef[1] /= self.coef[0]
            self.coef[2] /= self.coef[0]

        self.register_buffer('fixed_bounds', None)
        self.register_buffer('fixed_coeffs', None)
        self.register_buffer('fixed_output_scale', None)

    def _exp_func(self, x):
        return np.exp(x)

    def _ibert_int_exp(self, x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)

        x_int = torch.max(x_int, self.n * x0_int)

        q = torch.floor(x_int / x0_int)
        r = x_int - x0_int * q

        b_int = floor_ste.apply(self.coef[1] / scaling_factor)
        c_int = floor_ste.apply(self.coef[2] / scaling_factor ** 2)

        z = r + b_int
        z = r * z
        z = z + c_int
        exp_int = z
        exp_scale = self.coef[0] * scaling_factor ** 2

        exp_int = torch.clamp(torch.floor(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = exp_scale / 2 ** self.n
        return exp_int, scaling_factor

    def fix(self):
        self.fixed = True

    def unfix(self):
        self.fixed = False
        self.fixed_bounds = None
        self.fixed_coeffs = None
        self.fixed_output_scale = None

    def _compute_integer_coefficients(self, x_int, scaling_factor):
        x_lo_int = torch.floor(torch.min(x_int)).item()
        x_hi_int = torch.ceil(torch.max(x_int)).item()

        x_lo = x_lo_int * scaling_factor.cpu().numpy()
        x_hi = x_hi_int * scaling_factor.cpu().numpy()

        if self.backend == 'ibert':
            xs_int_offset = torch.linspace(x_lo_int, x_hi_int, 10000, device=x_int.device, dtype=x_int.dtype)
            xs_int_original = xs_int_offset - 127
            ys_int, scaling_factor_out = self._ibert_int_exp(xs_int_original, scaling_factor)
            ys = ys_int * scaling_factor_out
            xs_np = (xs_int_offset * scaling_factor).detach().cpu().numpy()
            ys_np = ys.detach().cpu().numpy()
        else:
            xs_int_offset = np.linspace(x_lo_int, x_hi_int, 10000)
            xs_original = (xs_int_offset - 127) * scaling_factor.cpu().numpy()
            ys_np = self._exp_func(xs_original)
            xs_np = xs_int_offset * scaling_factor.cpu().numpy()

        x_lo = x_lo_int * scaling_factor.cpu().numpy()
        x_hi = x_hi_int * scaling_factor.cpu().numpy()

        float_pieces = fit_piecewise_polynomials(
            xs_np, ys_np, x_lo, x_hi,
            self.segments, self.degree,
            self.alpha,
            optim_bounds=self.optim_bounds
        )

        bounds, coeffs_tensor, output_scale = compute_integer_coefficients(
            float_pieces,
            scaling_factor=scaling_factor,
            N=None,          # NEW MODE
            device=x_int.device,
            verbose=False
        )

        self.fixed_bounds = bounds
        self.fixed_coeffs = coeffs_tensor
        self.fixed_output_scale = torch.tensor(output_scale, device=x_int.device, dtype=torch.float32)

        return bounds, coeffs_tensor, output_scale

    def int_exp_poly(self, x_int, scaling_factor):
        with torch.no_grad():
            if self.fixed and self.fixed_coeffs is not None:
                bounds_int = self.fixed_bounds
                coeffs_int = self.fixed_coeffs
                output_scale = self.fixed_output_scale.item()
            else:
                bounds_int, coeffs_int, output_scale = self._compute_integer_coefficients(x_int, scaling_factor)

        with torch.no_grad():
            exp_int = evaluate_piecewise_polynomial(x_int, bounds_int, coeffs_int, self.segments, self.degree)

        exp_int = torch.clamp(exp_int, min=0)
        return exp_int, output_scale

    def exp_debug(self, x, scaling_factor):
        device = x.device
        s = scaling_factor.to(device)
        with torch.no_grad():
            x_int = floor_ste.apply(x / s)
            x_int = x_int - x_int.max(dim=-1, keepdim=True).values + 127

            exp_int, output_scale = self.int_exp_poly(x_int, s)
            exp_approx_float = exp_int / output_scale

            exp_true_float = torch.exp(x_int.to(torch.float32) * s)
        return exp_approx_float, exp_true_float

    def forward(self, x, scaling_factor, return_exp_debug: bool = False):
        device = x.device<
        scaling_factor = scaling_factor.to(device)

        x_int = floor_ste.apply(x / scaling_factor).to(torch.int32)

        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max + 127

        exp_int, output_scale = self.int_exp_poly(x_int, scaling_factor)

        # =========================
        # FIX: shift based on output_scale (2^N_eff), not hardcoded "30"
        # =========================
        N_eff = int(round(math.log2(output_scale))) if output_scale > 0 else 0
        shift = max(0, N_eff - self.exp_bitwidth + 1)
        if shift > 0:
            exp_int = floor_ste.apply(exp_int / (2 ** shift))

        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        exp_int_sum = torch.clamp(exp_int_sum, min=1.0)

        with torch.no_grad():
            factor = floor_ste.apply(2 ** 32 / exp_int_sum)
            softmax_int = floor_ste.apply(exp_int * factor / 2 ** (32 - self.output_bit + 1))

        softmax_float = self.softmax_module(x)

        scaling_factor_out = torch.tensor([2 / 2 ** self.output_bit], device=device)
        softmax_int_float = softmax_int * scaling_factor_out

        output = softmax_int_float.detach() + (softmax_float - softmax_float.detach())
        output = scaling_factor_out * floor_ste.apply(output / scaling_factor_out)

        if return_exp_debug:
            with torch.no_grad():
                exp_approx_float, exp_true_float = self.exp_debug(x, scaling_factor)
            return output, scaling_factor_out, {"exp_true": exp_true_float, "exp_approx": exp_approx_float}

        return output, scaling_factor_out

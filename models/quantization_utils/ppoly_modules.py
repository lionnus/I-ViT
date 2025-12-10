"""
Custom Integer GELU implementation using piecewise polynomial approximation
"""

import torch
import torch.nn as nn
import numpy as np
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
    
    def __init__(self, output_bit=8, scale_bits=22, seg=16, deg=2, backend='ibert', alpha=0.0, optim_bounds=True):
        super(PPolyIntGELU, self).__init__()
        self.N = scale_bits  # Bit shift for integer representation
        self.segments = seg
        self.degree = deg
        self.backend = backend  # 'ibert' or 'float'
        self.alpha = alpha  # Overlap parameter for extending bounds
        self.optim_bounds = optim_bounds  # Whether to optimize segment boundaries
        self.fixed = False  # Track whether module is in fixed/eval mode

        # Torch module for float gelu backward pass
        self.gelu_module = nn.GELU()
        
        self.register_buffer('act_scaling_factor', torch.zeros(1))
        
        # Buffers for storing fixed integer coefficients and bounds
        self.register_buffer('fixed_lo_bounds', None)
        self.register_buffer('fixed_hi_bounds', None)
        self.register_buffer('fixed_coeffs', None)
        self.register_buffer('fixed_scaling_factor_out', None)
    
    def _gelu_func(self, x):
        """Standard GELU function for fitting"""
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    
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
        # Prepare data for fitting
        x_lo = torch.floor(torch.min(x)).item()
        x_hi = torch.ceil(torch.max(x)).item()
        if self.backend == 'ibert' and x is not None and scaling_factor is not None:
            # Create xs as a tensor on the same device as scaling_factor
            xs = torch.linspace(x_lo, x_hi, 10000, device=scaling_factor.device, dtype=scaling_factor.dtype)
            
            # Golden model IBERTIntGELU
            ibert = IBERTIntGELU()
            ys, _ = ibert(xs, scaling_factor)
            # Convert ys to numpy for polynomial fitting
            ys_np = ys.detach().cpu().numpy()
            xs_np = xs.detach().cpu().numpy()
        else:
            xs_np = np.linspace(x_lo, x_hi, 10000, dtype=np.float32)
            ys_np = self._gelu_func(xs_np)
        
        # Fit float polynomial pieces using shared function with alpha parameter
        float_pieces = fit_piecewise_polynomials(xs_np, ys_np, x_lo, x_hi, self.segments, self.degree, self.alpha, optim_bounds=self.optim_bounds)
        
        # Convert to integer representation
        lo_bounds, hi_bounds, coeffs_tensor = compute_integer_coefficients(
            float_pieces, scaling_factor, self.N, x.device
        )
        
        # Compute output scaling factor
        if self.backend == 'ibert':
            # IBERT Mystery Formula
            k = 1.4142
            n = 6
            coeff = [-0.2888, -1.769, 1]
            scaling_factor_out = scaling_factor/k
            scaling_factor_out = scaling_factor_out ** 2 * coeff[0]
            scaling_factor_out = scaling_factor_out * (2**n)
            scaling_factor_out = scaling_factor * scaling_factor_out / 2
        else:   
            scaling_factor_out = scaling_factor / (2 ** self.N)
        
        self.fixed_lo_bounds = lo_bounds
        self.fixed_hi_bounds = hi_bounds
        self.fixed_coeffs = coeffs_tensor
        self.fixed_scaling_factor_out = scaling_factor_out
        
        return lo_bounds, hi_bounds, coeffs_tensor, scaling_factor_out

    def forward(self, x, scaling_factor):
        """Evaluate piecewise polynomial for integer inputs with float GELU gradients."""
        
        # Convert input to integer representation
        x_int = floor_ste.apply(x / scaling_factor)
        
        # Get integer coefficients - either compute fresh or use stored
        with torch.no_grad():
            if self.fixed and self.fixed_coeffs is not None:
                # Use stored coefficients
                lo_i = self.fixed_lo_bounds
                hi_i = self.fixed_hi_bounds
                coeffs_tensor = self.fixed_coeffs
                scaling_factor_out = self.fixed_scaling_factor_out
            else:
                # Compute coefficients (and store if fixed)
                lo_i, hi_i, coeffs_tensor, scaling_factor_out = self._compute_integer_coefficients(x, scaling_factor)
        
        # Evaluate polynomial using shared function
        with torch.no_grad():
            y_int = evaluate_piecewise_polynomial(x_int, lo_i, hi_i, coeffs_tensor, self.segments, self.degree)
        
        # Compute float GELU for gradient path
        y_float_gelu = self.gelu_module(x)
        
        # Convert integer result back to float
        y_float = y_int / (2 ** self.N)
        
        # Replace forward with integer computation but use float GELU derivatives
        y_float = y_float.detach() + (y_float_gelu - y_float_gelu.detach())
        
        # Ensure output is quantized
        y_float = scaling_factor_out * floor_ste.apply(y_float/scaling_factor_out)

        return y_float, scaling_factor_out
    
class PPolyIntSoftmax(nn.Module):
    """
    Implementation of Integer Softmax using piecewise polynomial approximation for exp()
    """

    def __init__(self, output_bit=8, scale_bits=28, exp_bits=16, seg=16, deg=2, backend='float', alpha=0.0, optim_bounds=False):
        super().__init__()
        self.output_bit = output_bit
        self.N = scale_bits  # Bit shift for integer representation
        self.exp_bitwidth = exp_bits
        self.segments = seg
        self.degree = deg
        self.backend = backend  # 'ibert' or 'float'
        self.alpha = alpha  # Overlap parameter for extending bounds
        self.optim_bounds = optim_bounds  # Whether to optimize segment boundaries
        self.fixed = False
        
        # Torch module for float softmax backward pass
        self.softmax_module = nn.Softmax(dim=-1)
        
        if self.backend == 'ibert':
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
    
    def _exp_func(self, x):
        """Standard exponential function for fitting"""
        return np.exp(x)
    
    def _ibert_int_exp(self, x_int, scaling_factor):
        """IBERT's integer exponential approximation"""
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)

        x_int = torch.max(x_int, self.n * x0_int)

        q = torch.floor(x_int / x0_int)
        r = x_int - x0_int * q
        
        # IBERT polynomial approx
        b_int = floor_ste.apply(self.coef[1] / scaling_factor)
        c_int = floor_ste.apply(self.coef[2] / scaling_factor ** 2)

        z = r + b_int
        z = r * z
        z = z + c_int
        exp_int = z
        exp_scale = self.coef[0] * scaling_factor ** 2
        
        exp_int = torch.clamp(
            torch.floor(exp_int * 2 ** (self.n - q)),
            min=0
        )
        scaling_factor = exp_scale / 2 ** self.n
        return exp_int, scaling_factor
    
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
    
    def _compute_integer_coefficients(self, x_int, scaling_factor):
        """Compute integer coefficients from float pieces and store if fixed"""
        # The input x_int here is already offset by 128, in range [-127, 128]
        # But we need to fit exp((x - 128) * scaling_factor) = exp(x_original * scaling_factor)
        # TODO Parameterize
        x_lo_int = -128  # Hardware constraint
        x_hi_int = 127   # Hardware constraint
        
        # Approximating exp((x_int_offset - 127) * scaling_factor)
        if self.backend == 'ibert':
            # Use IBERT's integer exponential approximation
            xs_int_offset = torch.linspace(x_lo_int, x_hi_int, 10000, device=x_int.device, dtype=x_int.dtype)
            # Remove offset to get original values for IBERT computation
            xs_int_original = xs_int_offset - 127
            ys_int, scaling_factor_out = self._ibert_int_exp(xs_int_original, scaling_factor)
            ys = ys_int * scaling_factor_out
            # fit against the offset x values
            xs_np = (xs_int_offset * scaling_factor).detach().cpu().numpy()
            ys_np = ys.detach().cpu().numpy()
        else:
            # Use float exponential function
            xs_int_offset = np.linspace(x_lo_int, x_hi_int, 10000)
            # Compute exp((x_offset - 127) * s) which is the actual function
            xs_original = (xs_int_offset - 127) * scaling_factor.cpu().numpy()
            ys_np = self._exp_func(xs_original)
            xs_np = xs_int_offset * scaling_factor.cpu().numpy()
        # Convert integer bounds to float
        x_lo = x_lo_int * scaling_factor.cpu().numpy()
        x_hi = x_hi_int * scaling_factor.cpu().numpy()
        
        # Fit float polynomial pieces
        float_pieces = fit_piecewise_polynomials(xs_np, ys_np, x_lo, x_hi, self.segments, self.degree, self.alpha, optim_bounds=self.optim_bounds)
        
        # Convert to integer representation
        lo_bounds, hi_bounds, coeffs_tensor = compute_integer_coefficients(
            float_pieces, scaling_factor, self.N, x_int.device
        )
        
        self.fixed_lo_bounds = lo_bounds
        self.fixed_hi_bounds = hi_bounds
        self.fixed_coeffs = coeffs_tensor
        
        return lo_bounds, hi_bounds, coeffs_tensor

    def int_exp_poly(self, x_int, scaling_factor):
        """Evaluate piecewise polynomial for exponential approximation."""
        # Get integer coefficients
        with torch.no_grad():
            if self.fixed and self.fixed_coeffs is not None:
                # Use stored coefficients
                lo_i = self.fixed_lo_bounds
                hi_i = self.fixed_hi_bounds
                coeffs_tensor = self.fixed_coeffs
            else:
                # Compute coefficients
                lo_i, hi_i, coeffs_tensor = self._compute_integer_coefficients(x_int, scaling_factor)

        # Evaluate polynomial using shared function
        with torch.no_grad():
            exp_int = evaluate_piecewise_polynomial(x_int, lo_i, hi_i, coeffs_tensor, self.segments, self.degree)
        
        # Clamp to ensure positive values (exp should always be positive)
        exp_int = torch.clamp(exp_int, min=0)
        
        return exp_int
    
    def exp_debug(self, x, scaling_factor):
        """
        Returns (exp_approx_float, exp_true_float)
        """
        device = x.device
        s = scaling_factor.to(device)
        with torch.no_grad():
            x_int = floor_ste.apply(x / s)
            x_int = x_int - x_int.max(dim=-1, keepdim=True).values + 127  # Offset input
            
            exp_int = self.int_exp_poly(x_int, s)
            exp_approx_float = exp_int / (2 ** self.N)
            
            # True exponential is still of the original x_int values
            exp_true_float = torch.exp(x_int.to(torch.float32) * s)
        return exp_approx_float, exp_true_float

    def forward(self, x, scaling_factor, return_exp_debug: bool = False):
        """Forward pass implementing integer softmax with polynomial exp approximation."""
        device = x.device
        scaling_factor = scaling_factor.to(device)
        
        # Convert to integer representation
        x_int = floor_ste.apply(x / scaling_factor)
        
        # Subtract max for numerical stability
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max + 127  # Shift to avoid negative values

        # Apply polynomial exponential approximation with offset input
        exp_int = self.int_exp_poly(x_int, scaling_factor)
        
        # Scale down to fit in output bit
        exp_int = floor_ste.apply(exp_int / 2 ** (30 - self.exp_bitwidth + 1))
        
        # Compute denominator
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        
        # Avoid division by zero
        exp_int_sum = torch.clamp(exp_int_sum, min=1.0)
        
        # Compute softmax values
        with torch.no_grad():
            factor = floor_ste.apply(2**32 / exp_int_sum)
            softmax_int = floor_ste.apply(exp_int * factor / 2 ** (32 - self.output_bit + 1))
        
        # Compute float softmax for gradient path
        softmax_float = self.softmax_module(x)
        
        # Convert integer result back to float
        scaling_factor_out = torch.tensor([2 / 2 ** self.output_bit], device=device)
        softmax_int_float = softmax_int * scaling_factor_out
        
        # Replace forward with integer computation but use float softmax derivatives
        output = softmax_int_float.detach() + (softmax_float - softmax_float.detach())
        
        # Ensure output is quantized
        output = scaling_factor_out * floor_ste.apply(output / scaling_factor_out)
        
        if return_exp_debug:
            with torch.no_grad():
                exp_approx_float, exp_true_float = self.exp_debug(x, scaling_factor)
            return output, scaling_factor_out, {"exp_true": exp_true_float, "exp_approx": exp_approx_float}

        return output, scaling_factor_out
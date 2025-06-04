"""
Custom Integer GELU implementation using piecewise polynomial approximation
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import erf
from .quant_utils import floor_ste

class ICUSTOMIntGELU(nn.Module):
    """
    Implementation of Integer GELU using piecewise polynomial approximation
    Uses float GELU for gradient computation to avoid OOM issues
    """
    
    def __init__(self, output_bit=8, N=24, segments=16, degree=1):
        super(ICUSTOMIntGELU, self).__init__()
        self.output_bit = 16  # ignore for now, handled in quantAct after?
        self.N = N  # Bit shift for integer representation
        self.segments = segments
        self.degree = degree
        
        # GELU approx range
        self.input_range = (-5.0, 5.0)

        # Torch module for float gelu backward pass
        self.gelu_module = nn.GELU()
        
        # Fit the piecewise polynomials once during initialization
        self.float_pieces = self._fit_piecewise_polynomials()
        
        self.register_buffer('act_scaling_factor', torch.zeros(1))
    
    def _gelu_func(self, x):
        """Standard GELU function for fitting"""
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    
    def _fit_piecewise_polynomials(self):
        """Fit piecewise polynomials to approximate GELU."""
        x_lo, x_hi = self.input_range
        xs = np.linspace(x_lo, x_hi, 10000, dtype=np.float32)
        ys = self._gelu_func(xs)
        bounds = np.linspace(x_lo, x_hi, self.segments + 1, dtype=np.float32)
        
        pieces = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mask = (xs >= lo) & (xs <= hi)
            coeffs = np.polyfit(xs[mask], ys[mask], self.degree).astype(np.float32)
            pieces.append(((lo, hi), coeffs))
        return pieces
    
    def fix(self):
        pass
    
    def unfix(self):
        pass
    
    def forward(self, x, scaling_factor):
        """Evaluate piecewise polynomial for integer inputs with float GELU gradients."""
        
        # Convert input to integer representation
        x_int = floor_ste.apply(x / scaling_factor)
        
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

        scaling_factor_out = scaling_factor / (2 ** self.N)

        return y_float, scaling_factor_out
    
class ICUSTOMIntSoftmax(nn.Module):
    """
    Implementation of Integer Softmax using piecewise polynomial approximation for exp()
    Class to quantize given Softmax layer using polynomial segments for exponential
    """
    
    def __init__(self, output_bit=16, N=20, segments=16, degree=2):
        super(ICUSTOMIntSoftmax, self).__init__()
        self.output_bit = 8  # ignore for now, handled in quantAct after?
        self.N = N  # Bit shift for integer representation
        self.segments = segments
        self.degree = degree
        
        # Exponential approximation range
        self.input_range = (-11.0, 0.0)
        
        # Fit the piecewise polynomials once during initialization
        self.float_pieces = self._fit_piecewise_polynomials()
        
        self.register_buffer('act_scaling_factor', torch.zeros(1))
    
    def _exp_func(self, x):
        """Standard exponential function for fitting"""
        return np.exp(np.clip(x, -50, 50))  # Clip to avoid overflow/underflow?
    
    def _fit_piecewise_polynomials(self):
        """Fit piecewise polynomials to approximate GELU."""
        x_lo, x_hi = self.input_range
        xs = np.linspace(x_lo, x_hi, 10000, dtype=np.float32)
        ys = self._exp_func(xs)
        bounds = np.linspace(x_lo, x_hi, self.segments + 1, dtype=np.float32)
        
        pieces = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mask = (xs >= lo) & (xs <= hi)
            coeffs = np.polyfit(xs[mask], ys[mask], self.degree).astype(np.float32)
            pieces.append(((lo, hi), coeffs))
        return pieces
    
    
    def fix(self):
        pass
    
    def unfix(self):
        pass
    
    def int_exp_poly(self, x_int, scaling_factor):
        """Evaluate piecewise polynomial for exponential approximation."""
        
        # Build integer bounds and integer coefficients without building gradient graph
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
                if i == 0:
                    mask_i = x_int <= hi_i[0]
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
        
        exp_int = exp_int.detach()
        
        # Convert integer result back to float
        exp_float = exp_int / (2 ** self.N)
        scaling_factor_out = scaling_factor / (2 ** self.N)
        
        # Ensure non-negative (exp should always be positive)
        exp_float = torch.clamp(exp_float, min=1e-10)
        
        return exp_float, scaling_factor_out
    
    def forward(self, x, scaling_factor):
        """Forward pass implementing integer softmax with polynomial exp approximation."""
        
        # Convert to integer representation
        x_int = floor_ste.apply(x / scaling_factor)
        
        # Subtract max
        with torch.no_grad():
            x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        
        # Apply polynomial exponential approximation
        exp_result, exp_scaling = self.int_exp_poly(x_int, scaling_factor) # TODO: Take care of backward prop since int_exp doesnt store gradents?
        
        # Convert back to integer for summation
        exp_int = floor_ste.apply(exp_result / exp_scaling)
        
        # Sum for normalization
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        exp_int_sum = torch.clamp(exp_int_sum, min=1)  # Avoid div by zero
        
        # Normalize
        max_val = 2 ** (31 - 1) - 1  # Max positive int32
        factor = floor_ste.apply(max_val / exp_int_sum)
        
        # Apply normalization factor
        normalized_int = floor_ste.apply(exp_int * factor / (2 ** (31 - self.output_bit)))
        
        # Final scaling factor for output
        output_scaling_factor = torch.tensor(1.0 / (2 ** (self.output_bit)), 
                                           device=x.device, dtype=x.dtype)
        
        # Store scaling factor
        self.act_scaling_factor = output_scaling_factor
        
        # Return final result
        result = normalized_int * output_scaling_factor
        
        return result, output_scaling_factor
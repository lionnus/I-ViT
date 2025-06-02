"""
Custom Integer GELU implementation using piecewise polynomial approximation
Following the I-ViT interface pattern with proper scaling factor handling
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import erf
from .quant_utils import floor_ste


class ICUSTOMIntGELU(nn.Module):
    """
    Implementation of Integer GELU using piecewise polynomial approximation
    Class to quantize given GELU layer using polynomial segments
    """
    
    def __init__(self, output_bit=8, N=20, segments=16, degree=2):
        super(ICUSTOMIntGELU, self).__init__()
        self.output_bit = output_bit  # ignore for now, handled in quantAct after?
        self.N = N  # Bit shift for integer representation
        self.segments = segments
        self.degree = degree
        
        # GELU approx range
        self.input_range = (-5.0, 5.0)
        
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
        """Evaluate piecewise polynomial for integer inputs."""
        
        # Convert input to integer representation (STE still tracked)
        x_int = floor_ste.apply(x / scaling_factor)
        
        # Build integer bounds and integer coefficients under torch.no_grad (only once per forward)
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
                int_coeffs_list.append(torch.stack(this_int_coeffs))  # shape (degree+1,)
            lo_i = torch.stack(lo_list)          # (segments,)
            hi_i = torch.stack(hi_list)          # (segments,)
            coeffs_tensor = torch.stack(int_coeffs_list)  # (segments, degree+1)
        
        # Initialize output as float on same device
        y_int = torch.zeros_like(x_int, dtype=torch.float32)
        S = self.segments
        D = self.degree
        
        #  ─── Now, evaluate polynomial **without** building a gradient graph ───
        # By putting the Horner‐loop under torch.no_grad(), we do not store any of these
        # multiplications/additions in the autograd buffer.  Only floor_ste.apply(...) 
        # (earlier) will carry gradient (STE).
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

                x_seg = x_int[mask_i]  # 1D slice of all elements in segment i
                c = coeffs_tensor[i]   # shape (degree+1,)
                
                # Horner's rule (all in no_grad)
                r = c[0].to(x_seg.dtype)
                for idx in range(1, D + 1):
                    r = r * x_seg + c[idx]

                # Store the “integer” result back into y_int (also no_grad)
                y_int[mask_i] = r
        
        # Now y_int is available as a float‐tensor but with no “grad graph” from the polynomial.
        # We only need gradient w.r.t. x_int via floor_ste, so detach is fine here:
        y_int = y_int.detach()  # ensure no leftover graph (optional)
        
        # Convert integer result back to float by dividing by 2^N
        y_float = y_int / (2 ** self.N)
        scaling_factor_out = scaling_factor / (2 ** self.N)

        return y_float, scaling_factor_out
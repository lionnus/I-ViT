"""
Custom Integer GELU implementation using piecewise polynomial approximation
Following the I-ViT interface pattern with proper scaling factor handling
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import erf
from .quant_utils import floor_ste, round_ste


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
        xs = np.linspace(x_lo, x_hi, 10000)
        ys = self._gelu_func(xs)
        bounds = np.linspace(x_lo, x_hi, self.segments + 1)
        
        pieces = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mask = (xs >= lo) & (xs <= hi)
            coeffs = np.polyfit(xs[mask], ys[mask], self.degree)
            pieces.append(((lo, hi), coeffs))
        return pieces
    
    def fix(self):
        pass
    
    def unfix(self):
        pass
    
    def forward(self, x, scaling_factor):
        """Evaluate piecewise polynomial for integer inputs."""
        
        # Convert input to integer representation
        x_int = floor_ste.apply(x / scaling_factor) # floor not needed
        
        # Convert float pieces to integer pieces with correct scaling
        int_pieces = []
        s_in = scaling_factor
        for (lo_f, hi_f), coeffs in self.float_pieces:
            lo_i = floor_ste.apply(lo_f / s_in)
            hi_i = floor_ste.apply(hi_f / s_in)
            
            # Convert coefficients with correct scaling
            int_coeffs = []
            deg = len(coeffs) - 1
            for i, coeff in enumerate(coeffs):
                power = deg - i  # Power of x for this coefficient
                # Scale by s_in^power * 2^N
                int_coeff = floor_ste.apply(coeff * (s_in ** power) * (2 ** self.N))
                int_coeffs.append(int_coeff)
            
            int_pieces.append((lo_i, hi_i, int_coeffs))
        
        # Initialize output as float (change to int?)
        y_int = torch.zeros_like(x_int, dtype=torch.float32)
        
        # Evaluate piecewise polynomial
        for seg_lo, seg_hi, int_coeffs in int_pieces:
            mask = (x_int >= seg_lo) & (x_int <= seg_hi)     # bool[*, …]
            if not mask.any():                               # nothing in this seg
                continue

            # Horner's rule: r = (((c0 * x) + c1) * x + c2) …
            r = torch.zeros_like(x_int, dtype=torch.float32)
            for c in int_coeffs:
                r = r * x_int + c
            y_int = torch.where(mask, r, y_int)              # write only masked

        # Handle boundary cases after main loop
        below = x_int < int_pieces[0][0]
        above = x_int > int_pieces[-1][1]

        if below.any():
            _, _, int_coeffs = int_pieces[0]
            r = torch.zeros_like(x_int, dtype=torch.float32)
            for c in int_coeffs:
                r = r * x_int + c
            y_int = torch.where(below, r, y_int)

        if above.any():
            _, _, int_coeffs = int_pieces[-1]
            r = torch.zeros_like(x_int, dtype=torch.float32)
            for c in int_coeffs:
                r = r * x_int + c
            y_int = torch.where(above, r, y_int)
            
        # Scale output back by 2^N
        y_float = y_int / (2 ** self.N)
        scaling_factor_out = scaling_factor / (2 ** self.N)  # Adjust scaling factor for output
        
        # Output scaling factor is the same as input for GELU
        return y_float, scaling_factor_out
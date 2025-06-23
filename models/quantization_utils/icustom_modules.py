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

class ICUSTOMIntGELU(nn.Module):
    """
    Implementation of Integer GELU using piecewise polynomial approximation
    Uses float GELU for gradient computation to avoid OOM issues
    """
    
    def __init__(self, output_bit=8, N=20, segments=16, degree=2, ibert_patch=True):
        super(ICUSTOMIntGELU, self).__init__()
        self.N = N  # Bit shift for integer representation
        self.segments = segments
        self.degree = degree
        self.ibert_patch = ibert_patch

        # Torch module for float gelu backward pass
        self.gelu_module = nn.GELU()
        if not self.ibert_patch:
            # GELU approx range
            self.input_range = (-5.0, 5.0)
            # Fit the piecewise polynomials once during initialization
            self.float_pieces = self._fit_piecewise_polynomials()
        else:
            self.float_pieces = None  # Placeholder, will be set in forward pass if ibert_patch is True
        
        self.register_buffer('act_scaling_factor', torch.zeros(1))
    
    def _gelu_func(self, x):
        """Standard GELU function for fitting"""
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    
    def _fit_piecewise_polynomials(self, x=None, scaling_factor=None, ibert_patch=False):
        """Fit piecewise polynomials to approximate GELU."""
        if ibert_patch and x is not None and scaling_factor is not None:
            # Use provided x and scaling_factor for fitting
            x_lo = torch.floor(torch.min(x)).item()
            x_hi = torch.ceil(torch.max(x)).item()
            self.input_range = (x_lo, x_hi)
            
            # Create xs as a tensor on the same device as scaling_factor
            xs = torch.linspace(x_lo, x_hi, 10000, device=scaling_factor.device, dtype=scaling_factor.dtype)
            
            # Golden model - IBERTIntGELU expects tensor input
            ibert = IBERTIntGELU()
            ys, scaling_factor_ibert = ibert(xs, scaling_factor)
            
            # Convert ys to numpy for polynomial fitting
            ys_np = ys.detach().cpu().numpy()
            xs_np = xs.detach().cpu().numpy()
        else:
            # Use default input range
            x_lo, x_hi = self.input_range
            xs_np = np.linspace(x_lo, x_hi, 10000, dtype=np.float32)
            ys_np = self._gelu_func(xs_np)
        
        bounds = np.linspace(x_lo, x_hi, self.segments + 1, dtype=np.float32)
        
        pieces = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mask = (xs_np >= lo) & (xs_np <= hi)
            coeffs = np.polyfit(xs_np[mask], ys_np[mask], self.degree).astype(np.float32)
            pieces.append(((lo, hi), coeffs))
        
        if ibert_patch:
            # If using IBERT patch, return the pieces and scaling factor
            return pieces, scaling_factor_ibert
        else:
            return pieces
    
    def fix(self):
        pass
    
    def unfix(self):
        pass
    
    def forward(self, x, scaling_factor):
        """Evaluate piecewise polynomial for integer inputs with float GELU gradients."""
        
        # Convert input to integer representation
        x_int = floor_ste.apply(x / scaling_factor)
        
        if self.ibert_patch:
            # Use IBERT patch for GELU approximation
            self.float_pieces, scaling_factor_ibert = self._fit_piecewise_polynomials(x, scaling_factor, ibert_patch=True)
        
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

        if self.ibert_patch:
            # If using IBERT patch, calculate using IBERT Mystery Formula
            k = 1.4142
            n = 6 # sufficiently large integer
            coeff = [-0.2888, -1.769, 1]
            scaling_factor_out = scaling_factor/k
            scaling_factor_out = scaling_factor_out ** 2 / coeff[0]
            scaling_factor_out = scaling_factor_out * (2**n)
            scaling_factor_out = scaling_factor * scaling_factor_out / 2
            scaling_factor_out = -scaling_factor_out # For some reason it inverts
        else:   
            scaling_factor_out = scaling_factor / (2 ** self.N)
        
        # Make sure y_float/scaling_factor_out is integer
        y_float = scaling_factor_out * floor_ste.apply(y_float/scaling_factor_out)

        return y_float, scaling_factor_out
    
class ICUSTOMIntSoftmax(nn.Module):
    """
    Implementation of Integer Softmax using piecewise polynomial approximation for exp()
    """
    
    def __init__(self, output_bit=8, N=30, segments=16, degree=2, ibert_patch=True):
        super().__init__()
        self.output_bit = 8
        self.N = N  # Bit shift for integer representation
        self.segments = segments
        self.degree = degree
        self.ibert_patch = ibert_patch
        self.act = QuantAct(8, quant_mode="symmetric")
        
        # Exponential approximation range (typical softmax input range after max subtraction)
        self.input_range = (-18.0, 0.0)
        
        if not self.ibert_patch:
            # Fit the piecewise polynomials once during initialization
            self.float_pieces = self._fit_piecewise_polynomials()
        else:
            # Initialize IBERT parameters for exp approximation
            self.x0 = -0.6931  # -ln(2)
            self.n = 30  # sufficiently large integer
            self.coef = [0.35815147, 0.96963238, 1.]
            self.coef[1] /= self.coef[0]
            self.coef[2] /= self.coef[0]
            self.float_pieces = None  # Will be set in forward pass
        
        # self.register_buffer('act_scaling_factor', torch.zeros(1))
    
    def _exp_func(self, x):
        """Standard exponential function for fitting"""
        return np.exp(x)
    
    def _ibert_int_polynomial(self, x_int, scaling_factor):
        """IBERT's polynomial approximation"""
        with torch.no_grad():
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor ** 2)

        z = x_int + b_int
        z = x_int * z
        z = z + c_int

        scaling_factor = self.coef[0] * scaling_factor ** 2
        return z, scaling_factor

    def _ibert_int_exp(self, x_int, scaling_factor):
        """IBERT's integer exponential approximation"""
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)

        x_int = torch.max(x_int, self.n * x0_int)

        q = torch.floor(x_int / x0_int)
        r = x_int - x0_int * q

        exp_int, exp_scale = self._ibert_int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(
            torch.floor(exp_int * 2 ** (self.n - q)),
            min=0
        )
        scaling_factor = exp_scale / 2 ** self.n
        return exp_int, scaling_factor
    
    def _fit_piecewise_polynomials(self, x=None, scaling_factor=None, ibert_patch=True):
        """Fit piecewise polynomials to approximate exp(x)."""
        if ibert_patch and x is not None and scaling_factor is not None:
            # Use IBERT exponential approximation as the target function
            x_lo = torch.floor(torch.min(x)*scaling_factor).item()
            x_hi = torch.ceil(torch.max(x)*scaling_factor).item()
            # Ensure we stay in negative range for exp approximation
            x_hi = min(x_hi, 0.0)
            self.input_range = (x_lo, x_hi)
            
            # Create xs as a tensor
            xs = torch.linspace(x_lo, x_hi, 10000, device=scaling_factor.device, dtype=scaling_factor.dtype)
            
            # Use IBERT's exp approximation as the golden model
            xs_int = torch.floor(xs / scaling_factor)
            ys_int, scaling_factor_out = self._ibert_int_exp(xs_int, scaling_factor)
            ys = ys_int * scaling_factor_out
            
            # Convert to numpy for polynomial fitting
            ys_np = ys.detach().cpu().numpy()
            xs_np = xs.detach().cpu().numpy()
        else:
            # Use standard exponential
            x_lo, x_hi = self.input_range
            xs_np = np.linspace(x_lo, x_hi, 10000, dtype=np.float32)
            ys_np = self._exp_func(xs_np)
        
        bounds = np.linspace(x_lo, x_hi, self.segments + 1, dtype=np.float32)
        
        pieces = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mask = (xs_np >= lo) & (xs_np <= hi)
            if np.sum(mask) < self.degree + 1:
                # Not enough points for fitting, use neighboring points
                center = (lo + hi) / 2
                distances = np.abs(xs_np - center)
                indices = np.argsort(distances)[:max(self.degree + 1, 10)]
                x_fit = xs_np[indices]
                y_fit = ys_np[indices]
            else:
                x_fit = xs_np[mask]
                y_fit = ys_np[mask]
            
            coeffs = np.polyfit(x_fit, y_fit, self.degree).astype(np.float32)
            pieces.append(((lo, hi), coeffs))
            
        if ibert_patch:
            return pieces, scaling_factor / 2 ** self.N # scaling_factor_out
        else:
            return pieces
    
    def int_exp_poly(self, x_int, scaling_factor):
        """Evaluate piecewise polynomial for exponential approximation."""
        if self.ibert_patch:
            # Fit polynomials to IBERT's exp approximation every time
            # Create a representative input range for fitting
            x_min = torch.floor(torch.min(x_int)).item()
            x_max = torch.ceil(torch.max(x_int)).item()
            x_range = torch.linspace(x_min, x_max, 1000, device=x_int.device, dtype=scaling_factor.dtype)
            self.float_pieces, _ = self._fit_piecewise_polynomials(
                x_range, scaling_factor, ibert_patch=True
            )
        
        # Build integer bounds and integer coefficients under torch.no_grad
        with torch.no_grad():
            s_in = scaling_factor
            lo_list = []
            hi_list = []
            int_coeffs_list = []
            
            for (lo_f, hi_f), coeffs in self.float_pieces:
                lo_i = torch.floor(torch.tensor(lo_f/s_in, device=x_int.device))
                hi_i = torch.floor(torch.tensor(hi_f/s_in, device=x_int.device))
                lo_list.append(lo_i)
                hi_list.append(hi_i)
                
                # Convert float coeffs to integer coeffs
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
                    # Force values below range to 0
                    below_range_mask = x_int < lo_i[0]
                    exp_int[below_range_mask] = 0
                    
                if i == S - 1:
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
        
        # Clamp to ensure positive values (exp should always be positive)
        exp_int = torch.clamp(exp_int, min=0)
        
        # The output scaling factor accounts for the 2^N factor
        scaling_factor_out = scaling_factor / (2 ** self.N)
        
        return exp_int, scaling_factor_out
    
    def forward(self, x, scaling_factor):
        """Forward pass implementing integer softmax with polynomial exp approximation."""
        device = x.device
        scaling_factor = scaling_factor.to(device)
        
        # Convert to integer representation
        x_int = torch.floor(x / scaling_factor).to(torch.int32)
        
        # Subtract max for numerical stability
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        
        # Apply polynomial exponential approximation
        exp_int, exp_scale = self.int_exp_poly(x_int, scaling_factor)
        # scaling_factor = self.coef[0] * scaling_factor ** 2
        # exp_scaling_factor = torch.Tensor([1 / (2 ** self.N)]).to(device)
        # exp_q, exp_scale = self.act(exp_int, exp_scaling_factor)
        # exp_int = exp_q / exp_scale
        exp_int = floor_ste.apply(exp_int / 2 ** (30 - self.output_bit + 1))  # scale down to fit in output bit

        # 4) denominator
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        factor = floor_ste.apply(2**32 / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (32 - self.output_bit + 1))
        # scaling_factor = 1 / 2 ** self.output_bit / 2 # TODO Integrate /2-> now fixing the output to 16 bit signed, instead of 16 bit unsigned.
        scaling_factor = torch.Tensor([2 / 2 ** (self.output_bit)]).cuda(device)

        return exp_int * scaling_factor, scaling_factor

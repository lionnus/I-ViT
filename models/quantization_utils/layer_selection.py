# layer_selection.py
# Author: Lionnus Kesting (lkesting@ethz.ch)
import torch
import torch.nn as nn
from .ivit_modules import IVITIntGELU, IVITIntSoftmax, IVITIntLayerNorm
from .ibert_modules import IBERTIntGELU, IBERTIntSoftmax, IBERTIntLayerNorm
from .ppoly_modules import PPolyIntGELU, PPolyIntSoftmax
from .quant_utils import floor_ste

# Float wrapper to match the interface of the quantized versions
class FloatGELU(nn.Module):
    """Float GELU with quantized output - isolates approximation error"""
    def __init__(self, bitwidth=8, **kwargs):
        super().__init__()
        self.gelu = nn.GELU()
        self.bitwidth = bitwidth
    
    def forward(self, x, scaling_factor):
        # Apply true GELU function
        output = self.gelu(x)
        
        # Quantize the output to match the bitwidth
        # This ensures fair comparison with approximated versions
        output_int = floor_ste.apply(output / scaling_factor)
        # Clamp to bitwidth range
        qmin = -(2**(self.bitwidth - 1))
        qmax = 2**(self.bitwidth - 1) - 1
        output_int = torch.clamp(output_int, qmin, qmax)
        output_quant = output_int * scaling_factor
        
        return output_quant, scaling_factor
    
    def fix(self):
        pass
    
    def unfix(self):
        pass


class FloatSoftmax(nn.Module):
    """Float Softmax with quantized output"""
    def __init__(self, bitwidth=8, dim=-1, **kwargs):
        super().__init__()
        self.dim = dim
        self.bitwidth = bitwidth
    
    def forward(self, x, scaling_factor):
        # Apply true softmax
        output = torch.softmax(x, dim=self.dim)
        
        # Softmax output is [0,1], need new scaling factor
        # Use the same scaling as IBERTIntSoftmax for fair comparison
        output_scaling_factor = torch.tensor(2.0 / 2**self.bitwidth, device=x.device)
        
        # Quantize output
        output_int = floor_ste.apply(output / output_scaling_factor)
        qmax = 2**(self.bitwidth - 1) - 1
        output_int = torch.clamp(output_int, 0, qmax)  # Softmax is non-negative
        output_quant = output_int * output_scaling_factor
        
        return output_quant, output_scaling_factor
    
    def fix(self):
        pass
    
    def unfix(self):
        pass


class FloatLayerNorm(nn.Module):
    """Float LayerNorm with quantized output"""
    def __init__(self, normalized_shape, eps=1e-5, bitwidth=8, **kwargs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.bitwidth = bitwidth
        
        # Similar to IBERTIntLayerNorm, we need to handle the scaling
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.dim_sqrt = None
    
    def forward(self, x, scaling_factor):
        # Get the feature dimension for scaling
        if self.dim_sqrt is None:
            n = x.shape[-1]  # Last dimension is normalized
            self.dim_sqrt = torch.sqrt(torch.tensor(n, dtype=torch.float32, device=x.device))
        
        # Apply true LayerNorm
        output = self.layer_norm(x)
        
        # LayerNorm output has different scale, compute new scaling factor
        # Following IBERTIntLayerNorm logic: scaling_factor = dim_sqrt / 2^30
        output_scaling_factor = self.dim_sqrt / 2**30
        
        # Scale by layer_norm weights if they exist
        if hasattr(self.layer_norm, 'weight'):
            output_scaling_factor = output_scaling_factor * self.layer_norm.weight
        
        # Quantize output
        output_int = floor_ste.apply(output / output_scaling_factor)
        qmin = -(2**(self.bitwidth - 1))
        qmax = 2**(self.bitwidth - 1) - 1
        output_int = torch.clamp(output_int, qmin, qmax)
        output_quant = output_int * output_scaling_factor
        
        return output_quant, output_scaling_factor
    
    def fix(self):
        pass
    
    def unfix(self):
        pass

# ---- GELU ------------------------------------------------------------
GELU_REGISTRY = {
    "float": FloatGELU,
    "ivit": IVITIntGELU,
    "ibert": IBERTIntGELU,
    "ppoly": PPolyIntGELU,
}

# ---- Softmax ---------------------------------------------------------
SOFTMAX_REGISTRY = {
    "float": FloatSoftmax,
    "ivit": IVITIntSoftmax,
    "ibert": IBERTIntSoftmax,
    "ppoly": PPolyIntSoftmax,
}

# ---- LayerNorm -------------------------------------------------------
LN_REGISTRY = {
    "float": FloatLayerNorm,
    "ivit": IVITIntLayerNorm,
    "ibert": IBERTIntLayerNorm,
}

def _parse_layer_name(name: str):
    """
    Parse layer names with parameters following the pattern:
    basename_arg1_value1_arg2_value2_...
    
    Examples:
    - ppoly_scale-bits_20_seg_16_backend_float -> base='ppoly', params={'scale_bits': 20, 'seg': 16, 'backend': 'float'}
    - ibert_use-int-sqrt_true -> base='ibert', params={'use_int_sqrt': True}
    - ivit_quant-mode_symmetric -> base='ivit', params={'quant_mode': 'symmetric'}

    Returns:
        tuple: (base_name, params_dict) or (name, {}) if no parameters found
    """
    parts = name.lower().split('_')
    
    if len(parts) < 3:  # Need at least base_arg_value
        return name.lower(), {}
    
    base_name = parts[0]
    params = {}
    
    # Parse pairs of arg_value
    i = 1
    while i < len(parts) - 1:
        arg = parts[i].replace('-', '_')  # Convert hyphens to underscores
        value_str = parts[i + 1]
        
        # Try to convert to appropriate type
        if value_str.lower() in ['true', 'false']:
            value = value_str.lower() == 'true'
        elif value_str.isdigit():
            value = int(value_str)
        else:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str  # Keep as string
        
        params[arg] = value
        i += 2
    
    return base_name, params

def get_gelu(name: str):
    base_name, params = _parse_layer_name(name)
    
    # Check if base name exists in registry
    if base_name in GELU_REGISTRY:
        base_class = GELU_REGISTRY[base_name]
        if params:
            # Create parameterized version
            class ParameterizedGELU(base_class):
                def __init__(self, *args, **kwargs):
                    for key, value in params.items():
                        kwargs.setdefault(key, value)
                    super().__init__(*args, **kwargs)
            return ParameterizedGELU
        return base_class
    
    # If base name not found, try original name
    return GELU_REGISTRY[name.lower()]

def get_softmax(name: str):
    base_name, params = _parse_layer_name(name)
    
    # Check if base name exists in registry
    if base_name in SOFTMAX_REGISTRY:
        base_class = SOFTMAX_REGISTRY[base_name]
        if params:
            # Create parameterized version
            class ParameterizedSoftmax(base_class):
                def __init__(self, *args, **kwargs):
                    for key, value in params.items():
                        kwargs.setdefault(key, value)
                    super().__init__(*args, **kwargs)
            return ParameterizedSoftmax
        return base_class
    
    # If base name not found, try original name
    return SOFTMAX_REGISTRY[name.lower()]

def get_layernorm(name: str):
    base_name, params = _parse_layer_name(name)
    
    # Check if base name exists in registry
    if base_name in LN_REGISTRY:
        base_class = LN_REGISTRY[base_name]
        if params:
            # Create parameterized version
            class ParameterizedLayerNorm(base_class):
                def __init__(self, *args, **kwargs):
                    for key, value in params.items():
                        kwargs.setdefault(key, value)
                    super().__init__(*args, **kwargs)
            return ParameterizedLayerNorm
        return base_class
    
    # If base name not found, try original name
    return LN_REGISTRY[name.lower()]
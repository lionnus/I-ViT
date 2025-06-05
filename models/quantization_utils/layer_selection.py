# activations_registry.py
import torch
import torch.nn as nn
from .ivit_modules import IVITIntGELU, IVITIntSoftmax, IVITIntLayerNorm
from .ibert_modules import IBERTIntGELU, IBERTIntSoftmax, IBERTIntLayerNorm
from .icustom_modules import ICUSTOMIntGELU, ICUSTOMIntSoftmax
from .quant_utils import floor_ste

# ---- Float wrapper to match the interface of the others ----
class FloatGELU(nn.Module):
    def __init__(self, **kwargs):  # Accept any kwargs to match quantized interface
        super(FloatGELU, self).__init__()
        self.gelu = nn.GELU()
        self.bitwidth = 8
    
    def forward(self, x, scaling_factor=None):
        # Just apply GELU, ignore scaling_factor for float operations
        float_y = self.gelu(x)
        # Return the same scaling factor or a default one
        scale_factor = 2*torch.max(torch.abs(float_y))/(2**self.bitwidth -1)
        # scaling_factor = scaling_factor / 2**16 # increase to get some accuracy?
        y_int= floor_ste.apply(float_y/(scale_factor)) 

        return y_int * scale_factor, scaling_factor
    
    def fix(self):
        """Match the interface of quantized modules"""
        pass
    
    def unfix(self):
        """Match the interface of quantized modules"""
        pass

class FloatSoftmax(nn.Module):
    def __init__(self, dim=-1, **kwargs):  # Accept any kwargs to match quantized interface
        super(FloatSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=dim)
        self.dim = dim
        # Initialize a dummy scaling factor buffer to match quantized modules
        self.register_buffer('act_scaling_factor', torch.ones(1))
    
    def forward(self, x, scaling_factor=None):
        # Just apply softmax, ignore scaling_factor for float operations
        output = self.softmax(x)
        # Return the same scaling factor or a default one
        if scaling_factor is None:
            scaling_factor = torch.ones(1, device=x.device, dtype=x.dtype)
        return output, scaling_factor
    
    def fix(self):
        """Match the interface of quantized modules"""
        pass
    
    def unfix(self):
        """Match the interface of quantized modules"""
        pass

class FloatLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, **kwargs):  # Accept any kwargs
        super(FloatLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
        # Initialize a dummy scaling factor buffer to match quantized modules
        self.register_buffer('act_scaling_factor', torch.ones(1))
    
    def forward(self, x, scaling_factor=None):
        # Just apply layer norm, ignore scaling_factor for float operations
        output = self.layer_norm(x)
        # Return the same scaling factor or a default one
        if scaling_factor is None:
            scaling_factor = torch.ones(1, device=x.device, dtype=x.dtype)
        return output, scaling_factor
    
    def fix(self):
        """Match the interface of quantized modules"""
        pass
    
    def unfix(self):
        """Match the interface of quantized modules"""
        pass

# ---- GELU ------------------------------------------------------------
GELU_REGISTRY = {
    "float": FloatGELU,
    "ivit": IVITIntGELU,
    "ibert": IBERTIntGELU,
    "icustom-v1": ICUSTOMIntGELU,
}

# ---- Softmax ---------------------------------------------------------
SOFTMAX_REGISTRY = {
    "float": FloatSoftmax,  # Fixed: use FloatSoftmax instead of nn.Softmax
    "ivit": IVITIntSoftmax,
    "ibert": IBERTIntSoftmax,
    "icustom-v1": ICUSTOMIntSoftmax,
}

# ---- LayerNorm -------------------------------------------------------
LN_REGISTRY = {
    "float": FloatLayerNorm,  # Fixed: use FloatLayerNorm instead of nn.LayerNorm
    "ivit": IVITIntLayerNorm,
    "ibert": IBERTIntLayerNorm,
}

def get_gelu(name: str):
    return GELU_REGISTRY[name.lower()]

def get_softmax(name: str):
    return SOFTMAX_REGISTRY[name.lower()]

def get_layernorm(name: str):
    return LN_REGISTRY[name.lower()]
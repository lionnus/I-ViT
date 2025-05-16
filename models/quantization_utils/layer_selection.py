# activations_registry.py
import torch.nn as nn
from .ivit_modules import IVITIntGELU, IVITIntSoftmax, IVITIntLayerNorm
from .ibert_modules   import IBERTIntGELU, IBERTIntSoftmax, IBERTIntLayerNorm

# ---- GELU ------------------------------------------------------------
GELU_REGISTRY = {
    "float":  nn.GELU,
    "ivit":   IVITIntGELU,
    "ibert":  IBERTIntGELU,
}

# ---- Softmax ---------------------------------------------------------
SOFTMAX_REGISTRY = {
    "float": nn.Softmax,
    "ivit":  IVITIntSoftmax,
    "ibert": IBERTIntSoftmax,
}

# ---- LayerNorm -------------------------------------------------------
LN_REGISTRY = {
    "float":  nn.LayerNorm,
    "ivit":   IVITIntLayerNorm,
    "ibert":  IBERTIntLayerNorm,
}

def get_gelu(name: str):
    return GELU_REGISTRY[name.lower()]

def get_softmax(name: str):
    return SOFTMAX_REGISTRY[name.lower()]

def get_layernorm(name: str):
    return LN_REGISTRY[name.lower()]

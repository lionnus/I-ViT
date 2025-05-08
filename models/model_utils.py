import torch.nn as nn
from .quantization_utils import *


def freeze_model(model: nn.Module):
    """
    Call .fix() on every sub-module that provides it.

    **Call this *after* you have run one calibration / warm-up pass** so that
    each QuantAct has a valid act_scaling_factor.
    """
    model.eval()                     # just to be safe
    for m in model.modules():        # recursion is built-in
        if hasattr(m, "fix") and callable(m.fix):
            m.fix()



def unfreeze_model(model):
    """
    unfreeze the activation range. Resursively invokes layer.unfix()
    """
    if type(model) in [QuantAct]:
        model.unfix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            unfreeze_model(m)
    elif type(model) == nn.ModuleList:
        for n in model:
            unfreeze_model(n)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                unfreeze_model(mod)


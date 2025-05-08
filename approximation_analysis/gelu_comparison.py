import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.style.use('thesis_plot_styles.mplstyle')
import logging

logger = logging.getLogger(__name__)

class IntGELU_IViT(nn.Module):
    """
    ShiftGELU from I-ViT quantization_utils, tweaked to be CPU-only
    """
    def __init__(self, output_bit=8):
        super().__init__()
        self.output_bit = output_bit
        self.n = 23  # sufficiently large integer
        self.register_buffer('act_scaling_factor', torch.zeros(1))

    def int_exp_shift(self, x_int, scaling_factor):
        device = x_int.device
        x_int = x_int.to(torch.int32)

        # the “shift” approximation
        x_int = x_int + torch.floor(x_int / 2) \
                     - torch.floor(x_int / 16)

        # floor(-1/scale) on correct device
        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor).to(device)

        x_int = torch.max(x_int, self.n * x0_int)

        # quotient & remainder
        q = torch.floor(x_int / x0_int)
        r = x_int - x0_int * q

        # build exp(r/q) approximation
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(
            torch.floor(exp_int * (2 ** (self.n - q))),
            min=0
        )

        # update scale
        scaling_factor = scaling_factor / (2 ** self.n)
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        device = x.device
        scaling_factor = scaling_factor.to(device)

        # 1) quantize input
        pre_x_int = x / scaling_factor

        # 2) subtract max for numerical stability
        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        # 3) approximate exp(x−max) and exp(−max)
        sig_scale = scaling_factor * 1.702
        exp_int, _     = self.int_exp_shift(x_int, sig_scale)
        exp_int_max, _ = self.int_exp_shift(-x_int_max, sig_scale)

        # 4) sum & normalize
        exp_sum = exp_int + exp_int_max
        temp_exp_sum = exp_sum
        exp_sum = exp_sum.clamp_max(2**31 - 1)
        factor  = torch.floor((2**31 - 1) / exp_sum)

        # 5) build integer sigmoid
        sigmoid_int   = torch.floor(
            exp_int * factor / (2 ** (31 - self.output_bit + 1))
        )
        sigmoid_scale = torch.tensor(
            1 / (2 ** (self.output_bit - 1)),
            device=device
        )

        # 6) multiply through
        out_int   = pre_x_int * sigmoid_int
        out_scale = scaling_factor * sigmoid_scale

        # save for introspection
        self.act_scaling_factor = out_scale.detach()
        return out_int * out_scale, out_scale

class IntGELU_IBERT(nn.Module):
    """
    IntGELU from IBERT
    """
    def __init__(self,
                 quant_mode='symmetric'):
        super(IntGELU_IBERT, self).__init__()
        self.register_buffer('input_scaling_factor', torch.ones(1))
        self.quant_mode = quant_mode

        self.k = 1.4142
        self.n = 14  # sufficiently large integer
        self.coeff = [-0.2888, -1.769, 1]  # a(x+b)**2 + c
        self.coeff[2] /= self.coeff[0]

    def fix(self):  pass
    def unfix(self):  pass

    def int_erf(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coeff[1] / scaling_factor)
            c_int = torch.floor(self.coeff[2] / (scaling_factor ** 2))

        sign = torch.sign(x_int)
        abs_int = torch.abs(x_int)
        abs_int = torch.min(abs_int, -b_int)
        y_int = (abs_int + b_int) ** 2 + c_int
        y_int = sign * y_int
        scaling_factor = (scaling_factor ** 2) * self.coeff[0]

        y_int = torch.floor(y_int / (2 ** self.n))
        scaling_factor = scaling_factor * (2 ** self.n)
        return y_int, scaling_factor

    def forward(self, x, scaling_factor):
            
        x_int = x / scaling_factor
        sigmoid_int, sigmoid_scaling_factor = self.int_erf(
            x_int, scaling_factor / self.k
        )

        shift_int = torch.floor(1.0 / sigmoid_scaling_factor) # 1/(scale_in^2*-0.2888*2^14)

        x_int = x_int * (sigmoid_int + shift_int)
        scaling_factor = scaling_factor * sigmoid_scaling_factor / 2

        return x_int * scaling_factor, scaling_factor

if __name__ == '__main__':
    # 1. choose scale and inputs
    full_scale = 6
    scale_ivit =  full_scale / (2 ** (8))
    scale_ibert = full_scale / (2 ** (16))
    x = torch.linspace(-full_scale/2, full_scale/2, 128+1)
    y_float = F.gelu(x)

    # 2. IViT approximation
    ivit = IntGELU_IViT(8)
    y_ivit, _ = ivit(x, scaling_factor=torch.tensor(scale_ivit))
    # y_ivit = y_ivit*2**(8 - 1
    abs_err_ivit = (y_ivit - y_float).abs()

    # 3. IBERT approximation
    ibert = IntGELU_IBERT()
    y_iber, _ = ibert(x, scaling_factor=torch.tensor(scale_ibert))
    abs_err_iber = (y_iber - y_float).abs()

    # Print metrics
    print("=== IViT IntGELU ===")
    print(f"max error: {abs_err_ivit.max():.5f}")
    print(f"mean error: {abs_err_ivit.mean():.5f}\n")
    # Percentage error
    print("=== IViT IntGELU (percentage) ===")
    print(f"max error: {100*abs_err_ivit.max() / y_float.max():.5f}")
    print(f"mean error: {100*abs_err_ivit.mean() / y_float.mean():.5f}\n")
    # Percentage error
    print("=== IBERT IntGELU (percentage) ===")
    print(f"max error: {100*abs_err_iber.max() / y_float.max():.5f}")
    print(f"mean error: {100*abs_err_iber.mean() / y_float.mean():.5f}\n")

    print("=== IBERT IntGELU ===")
    print(f"max error: {abs_err_iber.max():.5f}")
    print(f"mean error: {abs_err_iber.mean():.5f}\n")

    # Plot both approximations and error
    plt.figure(figsize=(6,4))
    plt.plot(x, y_float, label="float GELU")
    plt.plot(x, y_ivit, '--', label="IntGELU_IViT")
    plt.plot(x, y_iber, ':', label="IntGELU_IBERT")
    plt.legend(); plt.grid(True); plt.title("GELU vs Integer Approximations")
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(x, abs_err_ivit, '--', label="err IViT")
    plt.plot(x, abs_err_iber, ':', label="err IBERT")
    plt.legend(); plt.grid(True); plt.title("Absolute Error")
    plt.show()

    # ------------------------------------------------------------------
    # 4. Exponent approximation (IViT only)
    # ------------------------------------------------------------------
    sig_scale = torch.tensor(scale_ivit * 1.702)       # same scaling used inside forward()
    x_int_exp = torch.floor(x / sig_scale).to(torch.int32)

    exp_int, exp_scale = ivit.int_exp_shift(x_int_exp, sig_scale)
    y_exp_ivit = exp_int * exp_scale                   # integer‑domain exp approximation
    y_exp_true = torch.exp(x)                          # reference

    abs_err_exp = (y_exp_ivit - y_exp_true).abs()

    print("=== IViT int_exp_shift (exp) ===")
    print(f"max error: {abs_err_exp.max():.5f}")
    print(f"mean error: {abs_err_exp.mean():.5f}\n")

    # Plot exponent approximation and its error
    plt.plot(x, y_exp_true, label="exp(x)")
    plt.plot(x, y_exp_ivit, '--', label="IntGELU_IViT exp approx")
    plt.legend(); plt.grid(True); plt.title("Exponent Approximation")
    plt.show()

    plt.plot(x, abs_err_exp, '--', label="abs error exp")
    plt.legend(); plt.grid(True); plt.title("Absolute Error (Exponent Approximation)")
    plt.show()

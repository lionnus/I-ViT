import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.style.use('approximation_analysis/thesis_plot_styles.mplstyle')
import logging
import numpy as np

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

# ------------------------------------------------------------------
# Helper: figure sizing that respects the stylesheet ----------------
# ------------------------------------------------------------------

def new_figure(nrows: int = 1, height_mul: float = 1.0):
    """Return a (fig, axes) tuple sized for *nrows* stacked subplots."""
    width, base_h = plt.rcParams['figure.figsize']
    fig_height = base_h * height_mul
    return plt.subplots(nrows=nrows, ncols=1 if nrows == 1 else 2,
                        figsize=(width, fig_height))

# ------------------------------------------------------------------
# Comparison & plotting routine ------------------------------------
# ------------------------------------------------------------------

def run_comparison():
    # 1. Input range & scaling factors -----------------------------
    full_scale   = 5.6
    scale_ivit   = full_scale / (2 ** 8)
    scale_ibert  = full_scale / (2 ** 16)
    x = torch.linspace(-full_scale / 2, full_scale / 2, 257)
    y_float = F.gelu(x)

    # 2. Integer approximations ------------------------------------
    ivit  = IntGELU_IViT(8)
    y_ivit, _ = ivit(x, scaling_factor=torch.tensor(scale_ivit))

    ibert = IntGELU_IBERT(8)
    y_ibert, _ = ibert(x, scaling_factor=torch.tensor(scale_ibert))

    # 3. Errors ------------------------------------------------------
    abs_err_ivit  = (y_ivit  - y_float).abs()
    abs_err_ibert = (y_ibert - y_float).abs()
    rel_err_ivit  = abs_err_ivit  / (y_float.abs() + 1e-10) * 100.0
    rel_err_ibert = abs_err_ibert / (y_float.abs() + 1e-10) * 100.0

    # 4. Metrics printout -------------------------------------------
    def _stats(name, ae, re):
        print(f"=== {name} ===\n"
              f"Max abs error : {ae.max():.6f}\n"
              f"Mean abs error: {ae.mean():.6f}\n"
              f"Max % error  : {re.max():.2f}%\n"
              f"Mean % error : {re.mean():.2f}%\n")
    _stats("I-ViT IntGELU",  abs_err_ivit,  rel_err_ivit)
    _stats("I-BERT IntGELU", abs_err_ibert, rel_err_ibert)

    # 5. Figure with 4 panels --------------------------------------
    fig, axes = new_figure(nrows=2, height_mul=2)  # height≈5 in
    ax11, ax12 = axes[0]
    ax21, ax22 = axes[1]

    # (a) GELU & approximations
    ax11.plot(x, y_float, label="Float GELU")
    ax11.plot(x, y_ivit,  '--', label="I-ViT IntGELU")
    ax11.plot(x, y_ibert, ':',  label="I-BERT IntGELU")
    ax11.set_title("GELU vs Integer Approximations")
    ax11.set_xlabel("Input x")
    ax11.set_ylabel("GELU(x)")
    ax11.legend()

    # (b) absolute error
    ax12.plot(x, abs_err_ivit,  label="I-ViT abs err")
    ax12.plot(x, abs_err_ibert, label="I-BERT abs err")
    ax12.set_title("Absolute Error")
    ax12.set_xlabel("Input x")
    ax12.set_ylabel("|error|")
    ax12.legend()

    # (c) percentage error
    ax21.plot(x, rel_err_ivit,  label="I-ViT % err")
    ax21.plot(x, rel_err_ibert, label="I-BERT % err")
    ax21.set_title("Percentage Error")
    ax21.set_xlabel("Input x")
    ax21.set_ylabel("Error (%)")
    ax21.legend()

    # (d) log-scale absolute error
    ax22.semilogy(x, abs_err_ivit,  '--', label="I-ViT abs err")
    ax22.semilogy(x, abs_err_ibert, ':',  label="I-BERT abs err")
    ax22.set_title("Absolute Error (log-scale)")
    ax22.set_xlabel("Input x")
    ax22.set_ylabel("|error|")
    ax22.legend()

    fig.show()

    # 6. Bin-wise error distribution -------------------------------
    fig2, ax = new_figure()  # single-panel figure (3.5×2.5 in)
    bins = np.linspace(-full_scale / 2, full_scale / 2, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ivit_bin_err  = []
    ibert_bin_err = []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if torch.any(mask):
            ivit_bin_err.append(rel_err_ivit[mask].mean().item())
            ibert_bin_err.append(rel_err_ibert[mask].mean().item())
        else:
            ivit_bin_err.append(0.0)
            ibert_bin_err.append(0.0)
    width = 0.35
    ax.bar(bin_centers - width / 2, ivit_bin_err,  width, label='I-ViT')
    ax.bar(bin_centers + width / 2, ibert_bin_err, width, label='I-BERT')
    ax.set_xlabel('Input range')
    ax.set_ylabel('Average % error')
    ax.set_title('Error distribution across input range')
    ax.legend()
    fig2.show()


if __name__ == "__main__":
    run_comparison()

#!/usr/bin/env python3
# softmax_comparison.py
# --------------------------------------------------------------
# Comprehensive comparison of integer softmax approximations from I-ViT and IBERT
# against floating point softmax on vector inputs
# --------------------------------------------------------------

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------
# Utility functions for STE
# ------------------------------------------------------------------
class _FloorSTE(torch.autograd.Function):
    """Straight-through estimator for floor()"""
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, g):
        # identity gradient → straight-through
        return g

floor_ste = _FloorSTE.apply

class QuantAct:
    """Minimal QuantAct stub for IBERT softmax"""
    def __init__(self, bit_width=16, quant_mode='symmetric'):
        self.bit_width = bit_width
        self.quant_mode = quant_mode

    def __call__(self, x, scaling_factor):
        return x, scaling_factor

# ------------------------------------------------------------------
# I-ViT Shiftmax Implementation
# ------------------------------------------------------------------
class IntSoftmax_IVIT:
    """
    Shiftmax from I-ViT quantisation utilities
    """
    def __init__(self, output_bit=8):
        self.output_bit = output_bit
        self.n = 15  # large enough integer
        self.act_scaling_factor = torch.zeros(1)

    def int_exp_shift(self, x_int: torch.Tensor, scaling_factor: torch.Tensor):
        """
        Integer approximation of exp(x) in Q-domain
        """
        # --- Shift approximation of exp(x) ---------------------
        x_int = x_int + floor_ste(x_int / 2) - floor_ste(x_int / 16)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor).to(x_int.device)

        x_int = torch.max(x_int, self.n * x0_int)

        # quotient / remainder decomposition
        q = floor_ste(x_int / x0_int)
        r = x_int - x0_int * q

        # build exp(r/q)
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(
            floor_ste(exp_int * 2 ** (self.n - q)),
            min=0
        )
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x: torch.Tensor, scaling_factor: torch.Tensor):
        """
        Parameters
        ----------
        x               : (…, N) tensor of *floating* activations
        scaling_factor  : scalar tensor, same as in I-ViT paper
        """
        device = x.device
        scaling_factor = scaling_factor.to(device)

        # 1) quantise input
        x_int = x / scaling_factor
        x_int = x_int.to(torch.int32)

        # 2) subtract (per-row) max for numerical stability
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        # 3) integer exp
        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)

        # 4) normalise
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        exp_int_sum.clamp_max_(2 ** 31 - 1)
        factor = floor_ste((2 ** 31 - 1) / exp_int_sum)

        exp_int = floor_ste(exp_int * factor / 2 ** (31 - self.output_bit + 1))
        scaling_factor = torch.tensor(
            1.0 / 2 ** (self.output_bit - 1),
            device=device
        )

        # save scaling factor (nice for tensorboard / debugging)
        self.act_scaling_factor = scaling_factor.detach()
        return exp_int * scaling_factor, scaling_factor

# ------------------------------------------------------------------
# IBERT Softmax Implementation
# ------------------------------------------------------------------
class IntSoftmax_IBERT:
    """
    Polynomial-based int Softmax from IBERT.
    """
    def __init__(self, output_bit=8, quant_mode='symmetric'):
        self.output_bit = output_bit
        self.quant_mode = quant_mode
        self.act = QuantAct(16, quant_mode=self.quant_mode)

        # polynomial / shift parameters
        self.x0 = -0.6931                      # −ln(2)
        self.n = 30                            # sufficiently large integer
        self.coef = [0.35815147, 0.96963238, 1.]
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]

    def int_polynomial(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor ** 2)

        z = x_int + b_int
        z = x_int * z
        z = z + c_int

        scaling_factor = self.coef[0] * scaling_factor ** 2
        return z, scaling_factor

    def int_exp(self, x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)

        x_int = torch.max(x_int, self.n * x0_int) # limits min value?

        q = floor_ste(x_int / x0_int)
        r = x_int - x0_int * q

        exp_int, exp_scale = self.int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(
            floor_ste(exp_int * 2 ** (self.n - q)),
            min=0
        )
        scaling_factor = exp_scale / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x: torch.Tensor, scaling_factor: torch.Tensor):
        device = x.device
        scaling_factor = scaling_factor.to(device)

        if self.quant_mode == 'none':
            return F.softmax(x, dim=-1), None

        # 1) quantise input
        x_int = (x / scaling_factor).to(torch.int32)

        # 2) subtract max for numerical stability
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        # 3) integer exp, then a fake-quant (QuantAct)
        exp_int, exp_scale = self.int_exp(x_int, scaling_factor)
        exp_q, exp_scale = self.act(exp_int, exp_scale)  # identity in stub
        exp_int = exp_q / exp_scale

        # 4) denominator
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        factor = floor_ste(2 ** 32 / exp_int_sum)

        # 5) scale into desired output bit-width
        exp_int = floor_ste(exp_int * factor / 2 ** (32 - self.output_bit))
        scaling_factor = torch.tensor(
            1.0 / 2 ** self.output_bit,
            device=device
        )
        return exp_int * scaling_factor, scaling_factor

# ------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------
def calculate_metrics(softmax_float, softmax_approx):
    """Calculate various error metrics between float softmax and approximation"""
    
    # 1. Mean Absolute Error (MAE)
    mae = (softmax_float - softmax_approx).abs().mean().item()
    
    # 2. Max Absolute Error
    max_err = (softmax_float - softmax_approx).abs().max().item()
    
    # 3. Mean Relative Error (percentage)
    # Adding small epsilon to avoid division by zero
    rel_err = ((softmax_float - softmax_approx).abs() / (softmax_float + 1e-10)) * 100
    mean_rel_err = rel_err.mean().item()
    max_rel_err = rel_err.max().item()
    
    # 4. Cosine Similarity 
    norm_float = torch.norm(softmax_float)
    norm_approx = torch.norm(softmax_approx)
    cos_sim = torch.dot(softmax_float, softmax_approx) / (norm_float * norm_approx)
    
    # 5. Maximum Logit Difference
    # Check if the argmax is the same (classification accuracy)
    argmax_float = softmax_float.argmax()
    argmax_approx = softmax_approx.argmax()
    same_argmax = argmax_float == argmax_approx
    
    metrics = {
        'MAE': mae,
        'Max Absolute Error': max_err,
        'Mean Relative Error (%)': mean_rel_err,
        'Max Relative Error (%)': max_rel_err,
        'Cosine Similarity': cos_sim.item(),
        'Same Argmax': same_argmax.item()
    }
    
    return metrics

# ------------------------------------------------------------------
# Comprehensive Evaluation Functions
# ------------------------------------------------------------------
def run_softmax_comparison():
    """Compare softmax implementations on different types of input vectors"""
    
    # Initialize softmax implementations
    ivit = IntSoftmax_IVIT(output_bit=8)
    ibert = IntSoftmax_IBERT(output_bit=8, quant_mode='symmetric')
    
    # Set scaling factors
    scale_ivit = 10/2**8
    scale_ibert = 10/2**8
    
    # -------------------------------------------------------------
    # 1. Evaluate on uniformly distributed scores
    # -------------------------------------------------------------
    print("\n=== TEST CASE 1: Uniformly Distributed Scores ===")
    
    # Create uniform vector from -5 to 5
    x_uniform = torch.linspace(-5, 5, 100)
    
    # Float softmax reference
    softmax_float = F.softmax(x_uniform, dim=-1)
    
    # IVIT approximation
    softmax_ivit, _ = ivit.forward(x_uniform, torch.tensor(scale_ivit))
    
    # IBERT approximation
    softmax_ibert, _ = ibert.forward(x_uniform, torch.tensor(scale_ibert))
    
    # Calculate metrics
    print("\nI-ViT Metrics:")
    metrics_ivit = calculate_metrics(softmax_float, softmax_ivit)
    for k, v in metrics_ivit.items():
        print(f"{k}: {v}")
        
    print("\nIBERT Metrics:")
    metrics_ibert = calculate_metrics(softmax_float, softmax_ibert)
    for k, v in metrics_ibert.items():
        print(f"{k}: {v}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(x_uniform.numpy(), softmax_float.numpy(), label='Float Softmax')
    plt.plot(x_uniform.numpy(), softmax_ivit.numpy(), '--', label='I-ViT Softmax')
    plt.plot(x_uniform.numpy(), softmax_ibert.numpy(), ':', label='IBERT Softmax')
    plt.xlabel('Input Value')
    plt.ylabel('Softmax Output')
    plt.title('Softmax on Uniform Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Plot errors
    plt.figure(figsize=(10, 6))
    plt.plot(x_uniform.numpy(), (softmax_float - softmax_ivit).abs().numpy(), label='I-ViT Error')
    plt.plot(x_uniform.numpy(), (softmax_float - softmax_ibert).abs().numpy(), label='IBERT Error')
    plt.xlabel('Input Value')
    plt.ylabel('Absolute Error')
    plt.title('Softmax Approximation Error on Uniform Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # -------------------------------------------------------------
    # 2. Evaluate on tokens with a strong preference (common in NLP)
    # -------------------------------------------------------------
    print("\n=== TEST CASE 2: Peaked Distribution (Strong Preference) ===")
    
    # Vector with one dominant value (simulating a confident prediction)
    x_peaked = torch.zeros(100)
    x_peaked[50] = 10.0  # Strong signal at position 50
    
    # Float softmax reference
    softmax_float = F.softmax(x_peaked, dim=-1)
    
    # IVIT approximation
    softmax_ivit, _ = ivit.forward(x_peaked, torch.tensor(scale_ivit))
    
    # IBERT approximation
    softmax_ibert, _ = ibert.forward(x_peaked, torch.tensor(scale_ibert))
    
    # Calculate metrics
    print("\nI-ViT Metrics:")
    metrics_ivit = calculate_metrics(softmax_float, softmax_ivit)
    for k, v in metrics_ivit.items():
        print(f"{k}: {v}")
        
    print("\nIBERT Metrics:")
    metrics_ibert = calculate_metrics(softmax_float, softmax_ibert)
    for k, v in metrics_ibert.items():
        print(f"{k}: {v}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(softmax_float.numpy(), label='Float Softmax')
    plt.plot(softmax_ivit.numpy(), '--', label='I-ViT Softmax')
    plt.plot(softmax_ibert.numpy(), ':', label='IBERT Softmax')
    plt.xlabel('Position')
    plt.ylabel('Softmax Output')
    plt.title('Softmax on Peaked Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # -------------------------------------------------------------
    # 3. Evaluate on typical token logits (random with varying magnitudes)
    # -------------------------------------------------------------
    print("\n=== TEST CASE 3: Typical Token Logits (Random Distribution) ===")
    
    # Random vector with varying magnitudes (more realistic token logits)
    torch.manual_seed(42)  # For reproducibility
    x_random = torch.randn(100) * 2  # Random with std=2
    
    # Float softmax reference
    softmax_float = F.softmax(x_random, dim=-1)
    
    # IVIT approximation
    softmax_ivit, _ = ivit.forward(x_random, torch.tensor(scale_ivit))
    
    # IBERT approximation
    softmax_ibert, _ = ibert.forward(x_random, torch.tensor(scale_ibert))
    
    # Calculate metrics
    print("\nI-ViT Metrics:")
    metrics_ivit = calculate_metrics(softmax_float, softmax_ivit)
    for k, v in metrics_ivit.items():
        print(f"{k}: {v}")
        
    print("\nIBERT Metrics:")
    metrics_ibert = calculate_metrics(softmax_float, softmax_ibert)
    for k, v in metrics_ibert.items():
        print(f"{k}: {v}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(softmax_float.numpy(), label='Float Softmax')
    plt.plot(softmax_ivit.numpy(), '--', label='I-ViT Softmax')
    plt.plot(softmax_ibert.numpy(), ':', label='IBERT Softmax')
    plt.xlabel('Position')
    plt.ylabel('Softmax Output')
    plt.title('Softmax on Random Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # -------------------------------------------------------------
    # 4. Evaluate on 2D batch of vectors (typical transformer output)
    # -------------------------------------------------------------
    print("\n=== TEST CASE 4: 2D Batch of Vectors (Transformer Simulation) ===")
    
    # Create a batch of random vectors (batch_size=8, seq_len=100)
    torch.manual_seed(42)  # For reproducibility
    x_batch = torch.randn(8, 100) * 2
    
    # Float softmax reference (dim=-1 applies softmax to each row)
    softmax_float = F.softmax(x_batch, dim=-1)
    
    # IVIT approximation
    softmax_ivit, _ = ivit.forward(x_batch, torch.tensor(scale_ivit))
    
    # IBERT approximation
    softmax_ibert, _ = ibert.forward(x_batch, torch.tensor(scale_ibert))
    
    # Calculate metrics for each batch element and average
    ivit_metrics_list = []
    ibert_metrics_list = []
    
    for i in range(x_batch.size(0)):
        ivit_metrics_list.append(calculate_metrics(softmax_float[i], softmax_ivit[i]))
        ibert_metrics_list.append(calculate_metrics(softmax_float[i], softmax_ibert[i]))
    
    # Average metrics across batch
    avg_ivit_metrics = {k: sum(d[k] for d in ivit_metrics_list) / len(ivit_metrics_list) 
                        for k in ivit_metrics_list[0]}
    
    avg_ibert_metrics = {k: sum(d[k] for d in ibert_metrics_list) / len(ibert_metrics_list) 
                         for k in ibert_metrics_list[0]}
    
    print("\nBatch Average - I-ViT Metrics:")
    for k, v in avg_ivit_metrics.items():
        print(f"{k}: {v}")
        
    print("\nBatch Average - IBERT Metrics:")
    for k, v in avg_ibert_metrics.items():
        print(f"{k}: {v}")
    
    # Plot one example from the batch
    batch_idx = 0
    plt.figure(figsize=(10, 6))
    plt.plot(softmax_float[batch_idx].numpy(), label='Float Softmax')
    plt.plot(softmax_ivit[batch_idx].numpy(), '--', label='I-ViT Softmax')
    plt.plot(softmax_ibert[batch_idx].numpy(), ':', label='IBERT Softmax')
    plt.xlabel('Position')
    plt.ylabel('Softmax Output')
    plt.title(f'Softmax on Batch Element {batch_idx}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # -------------------------------------------------------------
    # 5. Summary comparison visualization
    # -------------------------------------------------------------
    # Create a summary bar chart of key metrics
    metrics = ['MAE', 'Mean Relative Error (%)', 'Cosine Similarity']
    ivit_values = [avg_ivit_metrics['MAE'], avg_ivit_metrics['Mean Relative Error (%)'], 
                  avg_ivit_metrics['Cosine Similarity']]
    ibert_values = [avg_ibert_metrics['MAE'], avg_ibert_metrics['Mean Relative Error (%)'], 
                   avg_ibert_metrics['Cosine Similarity']]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, ivit_values, width, label='I-ViT')
    plt.bar(x + width/2, ibert_values, width, label='IBERT')
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Softmax Approximation Performance Summary')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    run_softmax_comparison()
"""
Piecewise polynomial fitting utilities for integer approximations
"""

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings

def optimize_segment_bounds(xs_np, ys_np, x_lo, x_hi, segments, degree, max_iter=10):
    """
    Optimize segment boundaries using coordinate descent.
    
    Args:
        xs_np: Input x values
        ys_np: Target y values  
        x_lo: Lower bound
        x_hi: Upper bound
        segments: Number of segments
        degree: Polynomial degree
        max_iter: Maximum iterations
    
    Returns:
        Optimized boundaries array (segments+1 points including endpoints)
    """
    # Optimization parameters
    MIN_WIDTH_DIVISOR = 4  # Minimum segment width = total_range / (segments * MIN_WIDTH_DIVISOR)
    SEARCH_RANGE_FACTOR = 0.3  # Maximum movement as fraction of neighbor span
    SEARCH_STEPS = 10  # Number of positions to test in search range
    
    # Initialize uniform boundaries
    bounds = np.linspace(x_lo, x_hi, segments + 1, dtype=np.float32)
    min_width = (x_hi - x_lo) / (segments * MIN_WIDTH_DIVISOR)  # Minimum segment width
    
    # Optimize boundaries using coordinate descent
    for iteration in range(max_iter):
        for i in range(1, segments):
            # Search range for boundary i
            lo_search = max(bounds[i-1] + min_width, bounds[i] - SEARCH_RANGE_FACTOR * (bounds[i+1] - bounds[i-1]))
            hi_search = min(bounds[i+1] - min_width, bounds[i] + SEARCH_RANGE_FACTOR * (bounds[i+1] - bounds[i-1]))
            
            if lo_search >= hi_search:
                continue
            
            # Find best position
            best_pos = bounds[i]
            best_error = float('inf')
            
            for pos in np.linspace(lo_search, hi_search, SEARCH_STEPS):
                bounds_test = bounds.copy()
                bounds_test[i] = pos
                
                # Compute total error
                total_error = 0
                for j in range(segments):
                    mask = (xs_np >= bounds_test[j]) & (xs_np <= bounds_test[j+1])
                    if mask.any():
                        x_seg = xs_np[mask]
                        y_seg = ys_np[mask]
                        with warnings.catch_warnings():
                            # warnings.simplefilter("ignore", np.RankWarning)
                            coeffs = np.polyfit(x_seg, y_seg, degree)
                        y_pred = np.polyval(coeffs, x_seg)
                        total_error += np.sum((y_seg - y_pred) ** 2)
                
                if total_error < best_error:
                    best_error = total_error
                    best_pos = pos
            
            bounds[i] = best_pos
    
    return bounds


def fit_piecewise_polynomials(xs_np, ys_np, x_lo, x_hi, segments, degree, alpha=0.0, debug_plot=False, optim_bounds=True):
    """
    Fit piecewise polynomials to approximate a function.
    
    Args:
        xs_np: Input x values as numpy array
        ys_np: Target y values as numpy array  
        x_lo: Lower bound for fitting range
        x_hi: Upper bound for fitting range
        segments: Number of polynomial segments
        degree: Degree of each polynomial
        alpha: Overlap parameter (0.0-1.0) - percentage of segment width to extend bounds
        debug_plot: If True, plot the approximation with segment visualization
        optim_bounds: If True, optimize segment boundaries; if False, use uniform boundaries
    
    Returns:
        List of ((lo, hi), coeffs) tuples for each segment
    """
    # Convert to float64 for better numerical stability
    xs_np = xs_np.astype(np.float64)
    ys_np = ys_np.astype(np.float64)
    x_lo = float(x_lo)
    x_hi = float(x_hi)
    
    # Get boundaries - optimized or uniform
    if optim_bounds:
        bounds = optimize_segment_bounds(xs_np, ys_np, x_lo, x_hi, segments, degree)
    else:
        bounds = np.linspace(x_lo, x_hi, segments + 1, dtype=np.float32)
    
    # Create bounds for piecewise segments
    segment_width = (x_hi - x_lo) / segments
    overlap_width = alpha * segment_width
    
    pieces = []
    
    # Debug plotting setup
    if debug_plot:
        plt.figure(figsize=(12, 8))
        plt.scatter(xs_np, ys_np, alpha=0.5, s=10, c='black', label='Original data', zorder=1)
        colors = plt.cm.tab10(np.linspace(0, 1, segments))
    
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        # Extend bounds by alpha percentage for fitting data
        fit_lo = lo - overlap_width if i > 0 else lo
        fit_hi = hi + overlap_width if i < segments - 1 else hi
        
        # Select data points within the extended range for fitting
        mask = (xs_np >= fit_lo) & (xs_np <= fit_hi)
        x_fit = xs_np[mask]
        y_fit = ys_np[mask]
        
        if len(x_fit) > degree:  # Ensure we have enough points for fitting
            # Normalize x values to [-1, 1] for better numerical conditioning
            x_center = (fit_lo + fit_hi) / 2.0
            x_scale = (fit_hi - fit_lo) / 2.0
            
            # Avoid division by zero for degenerate segments
            if abs(x_scale) < 1e-10:
                x_scale = 1.0
                x_normalized = x_fit - x_center
            else:
                x_normalized = (x_fit - x_center) / x_scale
            
            with warnings.catch_warnings():
                # warnings.simplefilter("ignore", np.RankWarning)
                # warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')
                
                # Fit polynomial on normalized coordinates
                coeffs_norm = np.polyfit(x_normalized, y_fit, degree)
                
                # Convert coefficients back to original coordinate system
                # We need to transform polynomial p(x_norm) to p((x - x_center)/x_scale)
                coeffs = np.zeros(degree + 1, dtype=np.float64)
                
                # Use binomial expansion to convert coefficients
                for j in range(degree + 1):
                    # For each power in the normalized polynomial
                    poly_power = degree - j
                    coeff_norm = coeffs_norm[j]
                    
                    # Expand (x/x_scale - x_center/x_scale)^poly_power
                    # This is equivalent to ((x - x_center)/x_scale)^poly_power
                    for k in range(poly_power + 1):
                        # Binomial coefficient
                        binom = math.factorial(poly_power) / (math.factorial(k) * math.factorial(poly_power - k))
                        # Contribution to x^k term
                        contrib = coeff_norm * binom * ((-x_center/x_scale) ** (poly_power - k)) / (x_scale ** k)
                        coeffs[degree - k] += contrib
                
                coeffs = coeffs.astype(np.float32)
        else:
            # If no points or too few points in this segment, use zero coefficients or constant
            print(f"[WARNING] Not enough points to fit polynomial in segment {i}: {len(x_fit)} points")
            coeffs = np.zeros(degree + 1, dtype=np.float32)
            if len(y_fit) > 0:
                coeffs[-1] = np.mean(y_fit)  # Use mean value as constant term
        
        # Store original bounds (not extended) for evaluation
        pieces.append(((lo, hi), coeffs))
        
        # Debug plotting for this segment
        if debug_plot:
            color = colors[i]
            
            # Plot extended fitting region as shaded area
            if len(x_fit) > 0:
                plt.axvspan(float(fit_lo), float(fit_hi), alpha=0.1, color=color, 
                           label=f'Segment {i} fit region' if i == 0 else "")
            
            # Plot polynomial approximation over the evaluation bounds (not extended)
            x_eval = np.linspace(float(lo), float(hi), 100)
            y_eval = np.polyval(coeffs, x_eval)
            plt.plot(x_eval, y_eval, color=color, linewidth=2, 
                    label=f'Segment {i} polynomial')
            
            # Plot vertical bounds (dotted lines)
            plt.axvline(float(lo), color=color, linestyle='--', alpha=0.7, linewidth=1)
            if i == segments - 1:  # Plot the final bound
                plt.axvline(float(hi), color=color, linestyle='--', alpha=0.7, linewidth=1)
            
            # Highlight fitting data points for this segment
            if len(x_fit) > 0:
                plt.scatter(x_fit, y_fit, color=color, alpha=0.8, s=15, 
                           edgecolors='white', linewidth=0.5, zorder=3)
    
    if debug_plot:
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Piecewise Polynomial Approximation\n'
                 f'{segments} segments, degree {degree}, alpha={alpha:.2f}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    return pieces


def compute_integer_coefficients(float_pieces, scaling_factor, device, accumulator_bitwidth=None, N=None, verbose=True):
    """
    Convert float polynomial pieces to integer representation.
    Args:
        float_pieces: List of ((lo, hi), coeffs) tuples for each segment
        scaling_factor: Scaling factor for input x
        device: Torch device for tensors
        accumulator_bitwidth: If provided, use accumulator-based scaling
        N: If provided, use fixed N-bit scaling
        verbose: If True, print debug information
    Returns:
        bounds: Tensor of shape [segments-1] with internal boundary points
        coeffs_tensor: Tensor of shape [segments, degree+1] with integer coefficients
        K or N: Scaling exponent used (K for accumulator-based, N for fixed N-bit
    """
    bounds_list = []
    int_coeffs_list = []
    assert accumulator_bitwidth is not None or N is not None, "Either accumulator_bitwidth or N must be provided"

    degree = len(float_pieces[0][1]) - 1
    BW_X = 8  # TODO Pass the bitwidth of x as argument

    use_acc_bw = accumulator_bitwidth is not None
    if use_acc_bw:
        if degree == 1:
            BW_C = {
                1: accumulator_bitwidth - 1,
                0: accumulator_bitwidth - 1 - BW_X,
            }
        elif degree == 2:
            BW_C = {
                2: accumulator_bitwidth - 2 - 2 * BW_X,
                1: accumulator_bitwidth - 2 - BW_X,
                0: accumulator_bitwidth - 1,
            }
        else:
            raise ValueError(f"Accumulator-based scaling only supports degree 1 or 2 (got {degree})")

        # Max magnitude of base-scaled coefficients per power across all segments
        max_abs_by_power = {p: 0.0 for p in range(degree + 1)}
        for (_, coeffs) in float_pieces:
            deg = len(coeffs) - 1
            for i, coeff in enumerate(coeffs):
                power = deg - i
                base_scaled = float(coeff) * float(scaling_factor ** power)
                max_abs_by_power[power] = max(max_abs_by_power[power], abs(base_scaled))

        # Compute global feasible gain upper bound
        g_max = float("inf")
        for power, bw in BW_C.items():
            M = max_abs_by_power.get(power, 0.0)
            if bw <= 0:
                g_max = 0.0
                continue
            L = float((2 ** bw) - 1)
            if M == 0.0:
                continue
            g_max = min(g_max, L / M)

        # Choose power-of-two gain G = 2^K <= g_max
        if not math.isfinite(g_max) or g_max <= 0.0:
            K = 0
        else:
            K = int(math.floor(math.log2(g_max)))
            if K < 0:
                K = 0

        if verbose:
            for p in sorted(BW_C.keys(), reverse=True):
                print(f"[INFO] power={p}, BW={BW_C[p]}, max_abs={max_abs_by_power[p]:.6g}")
            print(f"[INFO] g_max={g_max:.6g}, chosen K={K}, G=2^{K}={2**K}")

    for idx, ((lo_f, hi_f), coeffs) in enumerate(float_pieces):
        if idx > 0:
            lo_i = torch.floor(torch.tensor(lo_f, device=device) / scaling_factor)
            bounds_list.append(lo_i)

        deg = len(coeffs) - 1
        this_int_coeffs = []
        for i, coeff in enumerate(coeffs):
            power = deg - i
            if use_acc_bw:
                base_scaled = float(coeff) * float(scaling_factor ** power)
                scaled = base_scaled * float(2 ** K)
                int_coeff = torch.floor(torch.tensor(scaled, device=device, dtype=torch.float64))
            else:
                scaled = coeff * (scaling_factor ** power) * (2 ** N)
                int_coeff = torch.floor(scaled.clone().detach())

            this_int_coeffs.append(int_coeff)

        int_coeffs_list.append(torch.stack(this_int_coeffs))

    if len(bounds_list) > 0:
        bounds = torch.stack(bounds_list).to(torch.int32)
    else:
        bounds = torch.tensor([], dtype=torch.int32, device=device)

    coeffs_tensor = torch.stack(int_coeffs_list).to(torch.int64)
    if verbose:
        abs_coeffs = torch.abs(coeffs_tensor.to(torch.int64))
        used_bits = torch.where(abs_coeffs == 0, torch.ones_like(abs_coeffs), torch.floor(torch.log2(abs_coeffs)) + 2).to(torch.int64)
        print("[INFO] used bits per coeff (max over segments):", used_bits.max(dim=0).values.tolist())

    return bounds, coeffs_tensor, K if use_acc_bw else N


def evaluate_piecewise_polynomial(x_int, bounds, coeffs_tensor, segments, degree, accumulator_bitwidth=32, verbose=True):
    """
    Evaluate piecewise polynomial using Horner's rule.
    
    Args:
        x_int: Integer input tensor
        bounds: Bounds tensor of shape [segments-1] containing internal boundary points [b1, b2, ..., b_{segments-1}]
        coeffs_tensor: Polynomial coefficients tensor of shape [segments, degree+1]
        segments: Number of segments
        degree: Polynomial degree
        accumulator_bitwidth: Bitwidth for accumulator overflow checks (default: 32)
        verbose: If True, print warnings about accumulator overflow (default: True)
    
    Returns:
        Output tensor with polynomial evaluation results
        
    Segments are defined as:
        - Segment 0: x < bounds[0]
        - Segment i (0 < i < segments-1): bounds[i-1] <= x < bounds[i]
        - Segment segments-1: bounds[segments-2] <= x
    """
    # Initialize output
    y_int = torch.zeros_like(x_int, dtype=torch.int64)
    x_int = x_int.to(torch.int64)
    
    # Evaluate polynomial for each segment
    for i in range(segments):
        if segments == 1:
            # Only one segment, use all points
            mask_i = torch.ones_like(x_int, dtype=torch.bool)
        elif i == 0:
            # First segment: x < bounds[0]
            mask_i = x_int < bounds[0]
        elif i == segments - 1:
            # Last segment: bounds[segments-2] <= x
            mask_i = x_int >= bounds[segments - 2]
        else:
            # Middle segments: bounds[i-1] <= x < bounds[i]
            mask_i = (x_int >= bounds[i - 1]) & (x_int < bounds[i])

        if not mask_i.any():
            continue

        x_seg = x_int[mask_i]
        c = coeffs_tensor[i].to(torch.int64)

        # Horner's rule for polynomial evaluation
        r = c[0].to(x_int.dtype)
        for idx in range(1, degree + 1):
            r_mult = r * x_seg.to(x_int.dtype)
            r_add = r_mult + c[idx]

            # Check if value exceeds accumulator bitwidth
            if verbose:
                if torch.max(torch.abs(r_mult)).item() >= 2 ** (accumulator_bitwidth - 1):
                    print(f"[WARNING] Mult accumulator exceeds {accumulator_bitwidth} bits signed in segment {i}, degree {idx}")
                if torch.max(torch.abs(r_add)).item() >= 2 ** (accumulator_bitwidth - 1):
                    print(f"[WARNING] Add accumulator exceeds {accumulator_bitwidth} bits signed in segment {i}, degree {idx}")
            r = r_add

        y_int[mask_i] = r.to(torch.int64)

    return y_int
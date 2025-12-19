"""
Piecewise polynomial fitting utilities for integer approximations
"""

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings

# Accumulator bitwidth constant
ACCUMULATOR_BITWIDTH = 32


def optimize_segment_bounds(xs_np, ys_np, x_lo, x_hi, segments, degree, max_iter=10):
    """
    Optimize segment boundaries using coordinate descent.
    """
    MIN_WIDTH_DIVISOR = 4
    SEARCH_RANGE_FACTOR = 0.3
    SEARCH_STEPS = 10

    bounds = np.linspace(x_lo, x_hi, segments + 1, dtype=np.float32)
    min_width = (x_hi - x_lo) / (segments * MIN_WIDTH_DIVISOR)

    for iteration in range(max_iter):
        for i in range(1, segments):
            lo_search = max(
                bounds[i - 1] + min_width,
                bounds[i] - SEARCH_RANGE_FACTOR * (bounds[i + 1] - bounds[i - 1]),
            )
            hi_search = min(
                bounds[i + 1] - min_width,
                bounds[i] + SEARCH_RANGE_FACTOR * (bounds[i + 1] - bounds[i - 1]),
            )

            if lo_search >= hi_search:
                continue

            best_pos = bounds[i]
            best_error = float("inf")

            for pos in np.linspace(lo_search, hi_search, SEARCH_STEPS):
                bounds_test = bounds.copy()
                bounds_test[i] = pos

                total_error = 0.0
                for j in range(segments):
                    mask = (xs_np >= bounds_test[j]) & (xs_np <= bounds_test[j + 1])
                    if mask.any():
                        x_seg = xs_np[mask]
                        y_seg = ys_np[mask]
                        with warnings.catch_warnings():
                            coeffs = np.polyfit(x_seg, y_seg, degree)
                        y_pred = np.polyval(coeffs, x_seg)
                        total_error += np.sum((y_seg - y_pred) ** 2)

                if total_error < best_error:
                    best_error = total_error
                    best_pos = pos

            bounds[i] = best_pos

    return bounds


def fit_piecewise_polynomials(
    xs_np,
    ys_np,
    x_lo,
    x_hi,
    segments,
    degree,
    alpha=0.0,
    debug_plot=False,
    optim_bounds=True,
):
    """
    Fit piecewise polynomials to approximate a function.
    """
    xs_np = xs_np.astype(np.float64)
    ys_np = ys_np.astype(np.float64)
    x_lo = float(x_lo)
    x_hi = float(x_hi)

    if optim_bounds:
        bounds = optimize_segment_bounds(xs_np, ys_np, x_lo, x_hi, segments, degree)
    else:
        bounds = np.linspace(x_lo, x_hi, segments + 1, dtype=np.float32)

    segment_width = (x_hi - x_lo) / segments
    overlap_width = alpha * segment_width

    pieces = []

    if debug_plot:
        plt.figure(figsize=(12, 8))
        plt.scatter(xs_np, ys_np, alpha=0.5, s=10, c="black", label="Original data", zorder=1)
        colors = plt.cm.tab10(np.linspace(0, 1, segments))

    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        fit_lo = lo - overlap_width if i > 0 else lo
        fit_hi = hi + overlap_width if i < segments - 1 else hi

        mask = (xs_np >= fit_lo) & (xs_np <= fit_hi)
        x_fit = xs_np[mask]
        y_fit = ys_np[mask]

        if len(x_fit) > degree:
            x_center = (fit_lo + fit_hi) / 2.0
            x_scale = (fit_hi - fit_lo) / 2.0

            if abs(x_scale) < 1e-10:
                x_scale = 1.0
                x_normalized = x_fit - x_center
            else:
                x_normalized = (x_fit - x_center) / x_scale

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.RankWarning)
                warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

                coeffs_norm = np.polyfit(x_normalized, y_fit, degree)

                coeffs = np.zeros(degree + 1, dtype=np.float64)

                for j in range(degree + 1):
                    poly_power = degree - j
                    coeff_norm = coeffs_norm[j]

                    for k in range(poly_power + 1):
                        binom = math.factorial(poly_power) / (
                            math.factorial(k) * math.factorial(poly_power - k)
                        )
                        contrib = (
                            coeff_norm
                            * binom
                            * ((-x_center / x_scale) ** (poly_power - k))
                            / (x_scale**k)
                        )
                        coeffs[degree - k] += contrib

                coeffs = coeffs.astype(np.float32)
        else:
            print(f"[WARNING] Not enough points to fit polynomial in segment {i}: {len(x_fit)} points")
            coeffs = np.zeros(degree + 1, dtype=np.float32)
            if len(y_fit) > 0:
                coeffs[-1] = np.mean(y_fit)

        pieces.append(((lo, hi), coeffs))

        if debug_plot:
            color = colors[i]

            if len(x_fit) > 0:
                plt.axvspan(float(fit_lo), float(fit_hi), alpha=0.1, color=color,
                           label=f"Segment {i} fit region" if i == 0 else "")

            x_eval = np.linspace(float(lo), float(hi), 100)
            y_eval = np.polyval(coeffs, x_eval)
            plt.plot(x_eval, y_eval, color=color, linewidth=2, label=f"Segment {i} polynomial")

            plt.axvline(float(lo), color=color, linestyle="--", alpha=0.7, linewidth=1)
            if i == segments - 1:
                plt.axvline(float(hi), color=color, linestyle="--", alpha=0.7, linewidth=1)

            if len(x_fit) > 0:
                plt.scatter(x_fit, y_fit, color=color, alpha=0.8, s=15,
                           edgecolors="white", linewidth=0.5, zorder=3)

    if debug_plot:
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(
            f"Piecewise Polynomial Approximation\n{segments} segments, degree {degree}, alpha={alpha:.2f}"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return pieces


def compute_integer_coefficients(
    float_pieces,
    scaling_factor=None,
    N=None,
    x_bitwidth=8,
    acc_bitwidth=ACCUMULATOR_BITWIDTH,
    device="cpu",
    verbose=True,
):
    """
    Convert float polynomial pieces to integer representation.

    Two modes:
    1. Legacy mode (N provided): Use fixed 2^N output scaling
    2. New mode (N=None): Choose the largest POWER-OF-TWO scale that fits the accumulator
       (so downstream shift-based logic stays valid).
    """
    bounds_list = []
    int_coeffs_list = []

    # Legacy mode unchanged
    if N is not None:
        max_c2_bitwidth = 0
        max_c1_bitwidth = 0
        max_c0_bitwidth = 0

        for idx, ((lo_f, hi_f), coeffs) in enumerate(float_pieces):
            if idx > 0:
                lo_i = torch.floor(torch.tensor(lo_f, device=device) / scaling_factor)
                bounds_list.append(lo_i)

            deg = len(coeffs) - 1
            this_int_coeffs = []
            for i, coeff in enumerate(coeffs):
                power = deg - i
                scaled = coeff * (scaling_factor ** power) * (2 ** N)
                int_coeff = torch.floor(torch.tensor(scaled, device=device))

                abs_val = torch.abs(int_coeff).item()
                if abs_val == 0:
                    bitwidth = 1
                else:
                    bitwidth = int(torch.ceil(torch.log2(torch.tensor(abs_val + 1))).item()) + 1

                coeff_index = deg - i
                if coeff_index == 2:
                    max_c2_bitwidth = max(max_c2_bitwidth, bitwidth)
                elif coeff_index == 1:
                    max_c1_bitwidth = max(max_c1_bitwidth, bitwidth)
                elif coeff_index == 0:
                    max_c0_bitwidth = max(max_c0_bitwidth, bitwidth)

                this_int_coeffs.append(int_coeff)

            int_coeffs_list.append(torch.stack(this_int_coeffs))

        output_scale = float(2 ** N)

        if verbose:
            print(f"[INFO] Legacy mode: scaling_factor={scaling_factor}, N={N}")
            print(f"[INFO] Maximum c2 bitwidth (signed): {max_c2_bitwidth} bits")
            print(f"[INFO] Maximum c1 bitwidth (signed): {max_c1_bitwidth} bits")
            print(f"[INFO] Maximum c0 bitwidth (signed): {max_c0_bitwidth} bits")

    else:
        # =========================
        # NEW MODE (FIXED)
        # =========================
        x_max_abs = 2 ** (x_bitwidth - 1)  # e.g. 128 for int8 signed magnitude bound
        acc_max = 2 ** (acc_bitwidth - 1)  # e.g. 2^31 for int32 signed magnitude bound

        sf = scaling_factor.item() if isinstance(scaling_factor, torch.Tensor) else float(scaling_factor)

        # 1) Compute a worst-case bound on |p(x_int)| in *ACCUMULATOR UNITS*
        #    where x_float = x_int * sf and poly is expressed in x_float.
        max_weighted_norm = 0.0
        for _, ((lo_f, hi_f), coeffs) in enumerate(float_pieces):
            degree = len(coeffs) - 1
            if degree == 2:
                weighted_norm = (
                    abs(float(coeffs[0])) * (sf ** 2) * (x_max_abs ** 2)
                    + abs(float(coeffs[1])) * sf * x_max_abs
                    + abs(float(coeffs[2]))
                )
            elif degree == 1:
                weighted_norm = abs(float(coeffs[0])) * sf * x_max_abs + abs(float(coeffs[1]))
            else:
                raise ValueError(f"Unsupported degree: {degree}")

            max_weighted_norm = max(max_weighted_norm, weighted_norm)

        # 2) Choose biggest POWER-OF-TWO output_scale that fits.
        #    This preserves the old "divide by 2^N" structure and fixes softmax shifts.
        if max_weighted_norm > 0:
            ratio = (acc_max - 1) / max_weighted_norm
            if ratio <= 1.0:
                N_eff = 0
            else:
                N_eff = int(math.floor(math.log2(ratio)))

            # Small safety margin helps when intermediate Horner terms get close to the bound.
            SAFETY_BITS = 1
            N_eff = max(0, N_eff - SAFETY_BITS)
        else:
            N_eff = 0

        output_scale = float(2 ** N_eff)

        max_c2_bitwidth = 0
        max_c1_bitwidth = 0
        max_c0_bitwidth = 0

        # 3) Convert coefficients using output_scale (power-of-two) and the same sf^power as legacy mode.
        for idx, ((lo_f, hi_f), coeffs) in enumerate(float_pieces):
            if idx > 0:
                lo_i = torch.floor(torch.tensor(lo_f, device=device) / sf)
                bounds_list.append(lo_i)

            degree = len(coeffs) - 1
            this_int_coeffs = []

            for i, coeff in enumerate(coeffs):
                power = degree - i
                scaled = float(coeff) * (sf ** power) * output_scale
                # Match legacy behavior: floor (not round)
                int_coeff = torch.floor(torch.tensor(scaled, device=device, dtype=torch.float32)).to(torch.int32)

                abs_val = abs(int_coeff.item())
                if abs_val == 0:
                    bitwidth = 1
                else:
                    bitwidth = int(np.ceil(np.log2(abs_val + 1))) + 1

                coeff_index = degree - i
                if coeff_index == 2:
                    max_c2_bitwidth = max(max_c2_bitwidth, bitwidth)
                elif coeff_index == 1:
                    max_c1_bitwidth = max(max_c1_bitwidth, bitwidth)
                elif coeff_index == 0:
                    max_c0_bitwidth = max(max_c0_bitwidth, bitwidth)

                this_int_coeffs.append(int_coeff)

            int_coeffs_list.append(torch.stack(this_int_coeffs))

        if verbose:
            util_bits = (
                math.log2(max_weighted_norm * output_scale) if max_weighted_norm > 0 else 0.0
            )
            print(f"[INFO] New mode: Power-of-two scaling to maximize accumulator usage")
            print(f"[INFO] scaling_factor: {sf}")
            print(f"[INFO] x_bitwidth: {x_bitwidth}, x_range approx: [-{x_max_abs}, {x_max_abs-1}]")
            print(f"[INFO] Accumulator bitwidth: {acc_bitwidth}")
            print(f"[INFO] Maximum weighted norm (unscaled): {max_weighted_norm:.4e}")
            print(f"[INFO] Chosen N_eff: {int(round(math.log2(output_scale)))} (output_scale={output_scale:.4e})")
            print(f"[INFO] Accumulator utilization: {util_bits:.2f}/{acc_bitwidth} bits")
            print(f"[INFO] Maximum c2 bitwidth (signed): {max_c2_bitwidth} bits")
            print(f"[INFO] Maximum c1 bitwidth (signed): {max_c1_bitwidth} bits")
            print(f"[INFO] Maximum c0 bitwidth (signed): {max_c0_bitwidth} bits")

    if len(bounds_list) > 0:
        bounds = torch.stack(bounds_list).to(torch.int32)
    else:
        bounds = torch.tensor([], dtype=torch.int32, device=device)

    coeffs_tensor = torch.stack(int_coeffs_list).to(torch.int32)
    return bounds, coeffs_tensor, output_scale


def evaluate_piecewise_polynomial(x_int, bounds, coeffs_tensor, segments, degree):
    """
    Evaluate piecewise polynomial using Horner's rule.
    """
    y_int = torch.zeros_like(x_int, dtype=torch.int32)
    x_int = x_int.to(torch.int64)

    for i in range(segments):
        if segments == 1:
            mask_i = torch.ones_like(x_int, dtype=torch.bool)
        elif i == 0:
            mask_i = x_int < bounds[0]
        elif i == segments - 1:
            mask_i = x_int >= bounds[segments - 2]
        else:
            mask_i = (x_int >= bounds[i - 1]) & (x_int < bounds[i])

        if not mask_i.any():
            continue

        x_seg = x_int[mask_i]
        c = coeffs_tensor[i].to(torch.int64)

        r = c[0]
        for idx in range(1, degree + 1):
            r_mult = r * x_seg
            r_add = r_mult + c[idx]

            max_mult = torch.max(torch.abs(r_mult)).item()
            max_add = torch.max(torch.abs(r_add)).item()

            if max_mult >= 2 ** (ACCUMULATOR_BITWIDTH - 1):
                print(f"[WARNING] Mult accumulator exceeds {ACCUMULATOR_BITWIDTH} bits signed in segment {i}, degree {idx}")
                print(f"           max value: {max_mult} vs limit: {2**(ACCUMULATOR_BITWIDTH-1)}")
            if max_add >= 2 ** (ACCUMULATOR_BITWIDTH - 1):
                print(f"[WARNING] Add accumulator exceeds {ACCUMULATOR_BITWIDTH} bits signed in segment {i}, degree {idx}")
                print(f"           max value: {max_add} vs limit: {2**(ACCUMULATOR_BITWIDTH-1)}")

            r = r_add

        y_int[mask_i] = r.to(torch.int32)

    return y_int

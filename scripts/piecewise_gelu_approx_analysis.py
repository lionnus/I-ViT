import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Define the function to approximate
def gelu(x):
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

# Parameters
s_in = 5/127  # 8-bit scaling
N = 20

# Fit piecewise polynomials
def fit_piecewise(func, x_range, segments, degree):
    """Fit equal-width polynomials using least squares."""
    x_lo, x_hi = x_range
    xs = np.linspace(x_lo, x_hi, 10000)
    ys = func(xs)
    bounds = np.linspace(x_lo, x_hi, segments + 1)
    
    pieces = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (xs >= lo) & (xs <= hi)
        coeffs = np.polyfit(xs[mask], ys[mask], degree)
        pieces.append(((lo, hi), coeffs))
    return pieces

# Fit GELU with piecewise polynomials
segments = 16
degree = 2
float_pieces = fit_piecewise(gelu, (-3, 3), segments, degree)

# Convert to integer pieces with correct scaling
int_pieces = []
for (lo_f, hi_f), coeffs in float_pieces:
    lo_i = int(round(lo_f / s_in))
    hi_i = int(round(hi_f / s_in))
    
    # Convert coefficients with correct scaling (like in simple polynomial)
    # For polynomial a*x^3 + b*x^2 + c*x + d
    int_coeffs = []
    deg = len(coeffs) - 1
    for i, coeff in enumerate(coeffs):
        power = deg - i  # Power of x for this coefficient
        # Scale by s_in^power * 2^N
        int_coeff = int(round(coeff * (s_in ** power) * (2 ** N)))
        int_coeffs.append(int_coeff)
    
    int_pieces.append((lo_i, hi_i, int_coeffs))

# X values
x_float = np.linspace(-3, 3, 1000)
x_int = np.round(x_float / s_in).astype(int)

# Float evaluation
y_float = gelu(x_float)

# Integer evaluation
y_int = np.zeros_like(x_int, dtype=np.int64)
for i, x_i in enumerate(x_int):
    # Find which piece this x belongs to
    for lo_i, hi_i, int_coeffs in int_pieces:
        if lo_i <= x_i <= hi_i:
            # Evaluate polynomial
            result = 0
            for coeff in int_coeffs:
                result = result * x_i + coeff
            y_int[i] = result
            break
    # If not within any segment, use first or last segmetn
    else:
        if x_i < int_pieces[0][0]:        # below all segments
            _, _, int_coeffs = int_pieces[0]
        else:                             # above all segments
            _, _, int_coeffs = int_pieces[-1]

        result = 0
        for c in int_coeffs:
            result = result * x_i + c
        y_int[i] = result
y_int_scaling = 1/(2**N)  # Output needs to be scaled back by 2^N

# Plot
plt.figure(figsize=(12, 5))

# Compare y_float with y_int/2^N
plt.subplot(1, 2, 1)
plt.plot(x_float, y_float, 'b-', label='y_float (true GELU)', linewidth=2)
plt.plot(x_float, y_int*y_int_scaling, 'r--', label=f'y_int / 2^{N}', linewidth=2)
plt.xlabel('x_float')
plt.ylabel('y')
plt.title('Float vs Scaled Integer')
plt.legend()
plt.grid(True, alpha=0.3)

# Show the raw integer values
plt.subplot(1, 2, 2)
plt.plot(x_int, y_int, 'g-', linewidth=2)
plt.xlabel('x_int')
plt.ylabel('y_int (before shift)')
plt.title(f'Raw Integer Values\n{segments} segments, degree {degree}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('piecewise_gelu_comparison.png', dpi=150)
plt.show()

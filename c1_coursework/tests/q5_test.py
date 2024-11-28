import math
from dual_autodiff.dual import Dual

# Define the function: f(x) = log(sin(x)) + x^2 * cos(x)
def f(x):
    return math.log(math.sin(x)) + x**2 * math.cos(x)

# Analytical derivative of f(x)
def analytical_derivative(x):
    return (math.cos(x) / math.sin(x)) + 2 * x * math.cos(x) - x**2 * math.sin(x)

# Using Dual Numbers to compute the derivative
x = 1.5
b = 1.0
sin_dual = Dual(x, b).sin()
cos_dual = Dual(x, b).cos()
term1 = sin_dual.log()
term2 = Dual(x, b) * Dual(x, b) * cos_dual
dualdiff = term1 + term2

print("Dual Number Derivative:", dualdiff.dual)

# Forward Differentiation using finite differences
deltas = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
print("\nForward Differentiation Derivatives:")
for delta in deltas:
    forward_diff = (f(x + delta) - f(x)) / delta
    print(f"Delta = {delta}: {forward_diff}")

# Analytical Derivative
analytical = analytical_derivative(x)
print("\nAnalytical Derivative:", analytical)

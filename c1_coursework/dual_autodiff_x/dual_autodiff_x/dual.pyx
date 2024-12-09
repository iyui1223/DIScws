import math
cdef class Dual:
    """Dual number representation for automatic differentiation"""
    # type declaration. Can be float?
    cdef double real
    cdef double dual

    def __init__(self, double real, double dual):
        self.real = real
        self.dual = dual

    def __repr__(self):
        return f"Dual(real={self.real}, dual={self.dual})"

    def __add__(self, Dual other):
        """Add two dual numbers."""
        return Dual(self.real + other.real, self.dual + other.dual)

    def __sub__(self, Dual other):
        """Subtract two dual numbers."""
        return Dual(self.real - other.real, self.dual - other.dual)

    def __mul__(self, Dual other):
        """Multiply two dual numbers."""
        return Dual(
            self.real * other.real,
            self.dual * other.real + self.real * other.dual
        )

    def __truediv__(self, Dual other):
        """Divide two dual numbers."""
        if other.real == 0:
            raise ZeroDivisionError("Division by zero in the real component")
        real_part = self.real / other.real
        dual_part = (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
        return Dual(real_part, dual_part)

    def sin(self):
        """Sine function of a dual number."""
        return Dual(
            math.sin(self.real),
            math.cos(self.real) * self.dual
        )

    def cos(self):
        """Cosine function of a dual number."""
        return Dual(
            math.cos(self.real),
            -math.sin(self.real) * self.dual
        )

    def tan(self):
        """Tangent function of a dual number."""
        if math.cos(self.real) == 0:
            raise ValueError("Tangent undefined at π/2 + kπ.")
        return Dual(
            math.tan(self.real),
            self.dual / (math.cos(self.real) ** 2)
        )

    def log(self):
        """Natural logarithm of a dual number."""
        if self.real <= 0:
            raise ValueError("Logarithm undefined for non-positive real parts.")
        return Dual(
            math.log(self.real),
            self.dual / self.real
        )

    def exp(self):
        """Exponential function of a dual number."""
        real_part = math.exp(self.real)
        return Dual(
            real_part,
            real_part * self.dual
        )

    def __eq__(self, Dual other):
        """Equality check for dual numbers."""
        return self.real == other.real and self.dual == other.dual

    def __neg__(self):
        """Negation of a dual number."""
        return Dual(-self.real, -self.dual)

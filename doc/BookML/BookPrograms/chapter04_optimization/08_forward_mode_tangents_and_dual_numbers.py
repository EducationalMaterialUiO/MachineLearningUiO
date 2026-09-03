"""Chapter 4: listing 8, from the section on forward mode tangents and dual numbers.

Extracted from doc/BookML/chapter4.tex.
"""

import math

class Dual:
    """A dual number a + b*eps with eps**2 = 0: a value and its derivative."""
    def __init__(self, val, dot=0.0):
        self.val, self.dot = val, dot
    def _lift(o):                       # promote plain floats to duals
        return o if isinstance(o, Dual) else Dual(o)
    def __add__(self, o):
        o = Dual._lift(o); return Dual(self.val + o.val, self.dot + o.dot)
    __radd__ = __add__
    def __sub__(self, o):
        o = Dual._lift(o); return Dual(self.val - o.val, self.dot - o.dot)
    def __mul__(self, o):               # the product rule, Eq. (4.dualmult)
        o = Dual._lift(o)
        return Dual(self.val * o.val, self.val * o.dot + self.dot * o.val)
    __rmul__ = __mul__
    def __truediv__(self, o):           # the quotient rule
        o = Dual._lift(o)
        return Dual(self.val / o.val,
                    (self.dot * o.val - self.val * o.dot) / o.val**2)
    def __pow__(self, k):
        return Dual(self.val**k, k * self.val**(k - 1) * self.dot)

# elementary functions: value and local derivative, Eq. (4.dualtaylor)
def sin(d): return Dual(math.sin(d.val), math.cos(d.val) * d.dot)
def exp(d): return Dual(math.exp(d.val), math.exp(d.val) * d.dot)
def log(d): return Dual(math.log(d.val), d.dot / d.val)

def f(x1, x2):                          # Eq. (4.adexample), ordinary code
    return log(x1) + x1 * x2 - sin(x2)

# seed the tangent of the variable we differentiate with respect to
print("df/dx1 =", f(Dual(2.0, 1.0), Dual(5.0, 0.0)).dot)   # 1/x1 + x2 = 5.5
print("df/dx2 =", f(Dual(2.0, 0.0), Dual(5.0, 1.0)).dot)   # x1 - cos(x2)
print("exact  =", 0.5 + 5.0, 2.0 - math.cos(5.0))

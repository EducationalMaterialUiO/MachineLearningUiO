"""Chapter 4: listing 8, from the section on automatic differentiation.

Extracted from doc/BookML/chapter4.tex.
"""

import autograd.numpy as np
from autograd import grad

def f(x):
    return np.sin(2 * np.pi * x + x**2)

df = grad(f)                 # df is a Python function, the derivative of f
print(df(1.0))

# It composes: the second derivative is just grad applied twice
d2f = grad(grad(f))
print(d2f(1.0))

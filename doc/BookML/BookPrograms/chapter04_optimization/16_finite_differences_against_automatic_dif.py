"""Chapter 4: listing 16, from the section on finite differences against automatic dif.

Extracted from doc/BookML/chapter4.tex.
"""

import numpy as np

def f(x):
    return np.sin(2 * np.pi * x + x**2)

exact = (2 * np.pi + 2) * np.cos(2 * np.pi + 1)
for h in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    fd = (f(1.0 + h) - f(1.0 - h)) / (2 * h)       # Eq. (4.centraldiff)
    print(f"h = {h:.0e}   error = {abs(fd - exact):.1e}")

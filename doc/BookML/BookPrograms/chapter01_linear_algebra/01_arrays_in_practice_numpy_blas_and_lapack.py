"""Chapter 1: listing 1, from the section on arrays in practice numpy blas and lapack.

Extracted from doc/BookML/chapter1.tex.
"""

import numpy as np

n = 10
x = np.random.normal(size=n)      # n samples from N(0, 1)
print(x)

x = np.array([1, 2, 3])           # explicit entries: x_0=1, x_1=2, x_2=3
print(x)

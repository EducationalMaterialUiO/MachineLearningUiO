"""Chapter 1: listing 2, from the section on arrays in practice numpy blas and lapack.

Extracted from doc/BookML/chapter1.tex.
"""

import numpy as np
from math import log

x = np.array([4, 7, 8])
for i in range(len(x)):
    x[i] = log(x[i])              # integer array: results are truncated
print(x)                          # prints [1 1 2]

x = np.log(np.array([4.0, 7.0, 8.0]))   # float array, vectorised
print(x)                                # prints [1.386... 1.945... 2.079...]
print(x.itemsize)                       # 8 bytes = 64 bits per element

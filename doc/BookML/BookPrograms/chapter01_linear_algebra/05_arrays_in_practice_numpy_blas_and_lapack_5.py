"""Chapter 1: listing 5, from the section on arrays in practice numpy blas and lapack.

Extracted from doc/BookML/chapter1.tex.
"""

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float64)

print(np.mean(a))                              # mean over all elements
print(np.mean(a, axis=0, keepdims=True))       # one mean per feature (column)
print(np.mean(a, axis=1, keepdims=True))       # one mean per observation (row)

# Standardising the design matrix: zero mean and unit variance per feature
X = (a - np.mean(a, axis=0)) / np.std(a, axis=0)

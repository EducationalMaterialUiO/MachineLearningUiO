"""Chapter 1: listing 3, from the section on arrays in practice numpy blas and lapack.

Extracted from doc/BookML/chapter1.tex.
"""

import numpy as np

A = np.array([[4.0, 7.0, 8.0], [3.0, 10.0, 11.0], [4.0, 5.0, 7.0]])
print(A.shape)                    # (3, 3)
print(A[:, 0])                    # first column  -- all rows, column 0
print(A[1, :])                    # second row

n = 10
print(np.zeros((n, n)))           # all elements zero
print(np.ones((n, n)))            # all elements one
print(np.random.rand(n, n))       # uniform on [0, 1)
print(np.eye(n))                  # identity matrix

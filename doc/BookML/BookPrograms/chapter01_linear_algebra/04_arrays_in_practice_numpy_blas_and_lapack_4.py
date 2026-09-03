"""Chapter 1: listing 4, from the section on arrays in practice numpy blas and lapack.

Extracted from doc/BookML/chapter1.tex.
"""

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float64)
print(np.ravel(a))                # 'C' (row-major) order, the default
print(np.ravel(a, order='F'))     # 'F' (column-major) order
print(a.reshape(-1))              # same as np.ravel(a)

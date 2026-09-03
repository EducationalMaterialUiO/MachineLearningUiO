"""Chapter 1: listing 6, from the section on the covariance matrix.

Extracted from doc/BookML/chapter1.tex.
"""

import numpy as np

n = 100
x = np.random.normal(size=n)
y = 4.0 + 3.0 * x + np.random.normal(size=n)      # strongly correlated with x
z = x**3 + np.random.normal(size=n)

# np.cov expects variables in rows, so stack the three vectors vertically
W = np.vstack((x, y, z))
Sigma = np.cov(W)                                 # 3 x 3 covariance matrix
print(Sigma)

eigvals, eigvecs = np.linalg.eigh(Sigma)          # eigh: symmetric matrices
print(eigvals)
print(eigvals / np.sum(eigvals))                  # fraction of variance each

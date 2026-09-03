"""Chapter 2: listing 2, from the section on covariance correlation and independence.

Extracted from doc/BookML/chapter2.tex.
"""

import numpy as np

def covariance(x, y):
    """Sample covariance of two vectors, using the 1/n convention."""
    return np.mean((x - np.mean(x)) * (y - np.mean(y)))

rng = np.random.default_rng(2024)
n = 1000
x = rng.normal(size=n)
y = 4.0 + 3.0 * x + rng.normal(size=n)      # correlated with x by construction
z = rng.normal(size=n)                      # independent of both

print(covariance(x, y), covariance(x, z))
print(np.cov(np.vstack((x, y, z))))         # note: numpy uses 1/(n-1)
print(np.corrcoef(np.vstack((x, y, z))))    # the correlation matrix

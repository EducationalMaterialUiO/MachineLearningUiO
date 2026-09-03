"""Chapter 1: listing 10, from the section on rank the pseudoinverse and ill condition.

Extracted from doc/BookML/chapter1.tex.
"""

import numpy as np

def lstsq_svd(X, y, rcond=1.0e-12):
    """Least-squares fit through the SVD, with explicit truncation.

    Returns the coefficients, the singular values, and the numerical rank.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    keep = s > rcond * s[0]                 # numerical rank
    theta = (Vt[keep].T * (1.0 / s[keep])) @ (U[:, keep].T @ y)
    return theta, s, int(np.sum(keep))

# A deliberately collinear design matrix: the third column is nearly the
# sum of the first two.
rng = np.random.default_rng(2024)
n = 200
x1, x2 = rng.normal(size=n), rng.normal(size=n)
X = np.column_stack([x1, x2, x1 + x2 + 1.0e-6 * rng.normal(size=n)])
y = 1.0 + 2.0 * x1 - x2 + 0.1 * rng.normal(size=n)

theta, s, rank = lstsq_svd(X, y)
print("singular values:", s)
print("condition number:", s[0] / s[-1])
print("numerical rank:", rank)

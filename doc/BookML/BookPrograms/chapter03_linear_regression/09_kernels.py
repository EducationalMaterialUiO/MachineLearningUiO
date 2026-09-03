"""Chapter 3: listing 9, from the section on kernels.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np

def gaussian_kernel(A, B, gamma):
    """k(x,x') = exp(-gamma ||x - x'||^2), the Gaussian kernel of Section 6.mercer."""
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def kernel_ridge_fit(K, y, lmbda):
    """alpha = (K + lambda I)^{-1} y, Eq. (3.krr).  One n x n solve."""
    return np.linalg.solve(K + lmbda * np.eye(len(y)), y)


def kernel_ridge_predict(alpha, Xtrain, Xnew, gamma):
    """f(x) = sum_i alpha_i k(x_i, x), Eq. (3.krr)."""
    return gaussian_kernel(Xnew, Xtrain, gamma) @ alpha

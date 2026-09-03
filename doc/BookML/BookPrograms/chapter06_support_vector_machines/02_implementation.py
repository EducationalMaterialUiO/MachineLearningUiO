"""Chapter 6: listing 2, from the section on implementation.

Extracted from doc/BookML/chapter6.tex.
"""

import numpy as np


def linear_kernel(X, Z):
    return X @ Z.T


def polynomial_kernel(X, Z, degree=3, gamma=1.0, coef0=1.0):
    return (gamma * (X @ Z.T) + coef0) ** degree


def rbf_kernel(X, Z, gamma=1.0):
    """Gaussian kernel, computed through the identity
    ||x - z||^2 = ||x||^2 + ||z||^2 - 2 x.z  (Section 1.vectors)."""
    d2 = (np.sum(X**2, axis=1)[:, None] + np.sum(Z**2, axis=1)[None, :]
          - 2.0 * (X @ Z.T))
    return np.exp(-gamma * np.maximum(d2, 0.0))

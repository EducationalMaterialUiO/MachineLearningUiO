"""Chapter 1: listing 11, from the section on low rank approximation.

Extracted from doc/BookML/chapter1.tex.
"""

import numpy as np

def truncated_svd(X, chi):
    """Best rank-chi approximation to X, with the retained variance."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    X_chi = (U[:, :chi] * s[:chi]) @ Vt[:chi]
    retained = np.sum(s[:chi]**2) / np.sum(s**2)
    return X_chi, retained

# Storage: n*p entries become chi*(n + p + 1)
n, p, chi = 512, 512, 40
print("compression ratio:", chi * (n + p + 1) / (n * p))

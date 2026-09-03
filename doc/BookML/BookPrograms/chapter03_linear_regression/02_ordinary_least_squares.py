"""Chapter 3: listing 2, from the section on ordinary least squares.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np

def ols(X, y):
    """Ordinary least squares through the SVD -- never the normal equations."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ (U.T @ y / s)


def ols_normal_equations(X, y):
    """The textbook formula.  Shown for comparison; do not use it."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

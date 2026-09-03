"""Chapter 3: listing 4, from the section on the lasso.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np

def soft_threshold(z, gamma):
    """Soft thresholding operator S_gamma(z), Eq. (3.softthresholdop)."""
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


def lasso_coordinate_descent(X, y, lmbda, n_iter=1000, tol=1e-8):
    """Lasso by cyclic coordinate descent.

    Minimises ||y - X theta||^2 / n + lmbda * ||theta||_1.
    The columns of X are assumed centred and standardised, and no
    intercept is penalised -- see Section on scaling and the intercept.
    """
    n, p = X.shape
    theta = np.zeros(p)
    col_norms = np.sum(X**2, axis=0)
    r = y - X @ theta                              # full residual

    for _ in range(n_iter):
        theta_old = theta.copy()
        for j in range(p):
            # partial residual: add back the current contribution of column j
            r += X[:, j] * theta[j]
            rho = X[:, j] @ r
            theta[j] = soft_threshold(rho, lmbda * n / 2.0) / col_norms[j]
            r -= X[:, j] * theta[j]                # remove the updated one
        if np.max(np.abs(theta - theta_old)) < tol:
            break

    return theta

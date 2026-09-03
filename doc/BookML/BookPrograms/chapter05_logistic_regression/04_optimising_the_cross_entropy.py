"""Chapter 5: listing 4, from the section on optimising the cross entropy.

Extracted from doc/BookML/chapter5.tex.
"""

import numpy as np

def sigmoid(z):
    """Numerically stable logistic function, Eq. (5.sigmoid)."""
    out = np.empty_like(z, dtype=float)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])                       # avoids overflow for z << 0
    out[neg] = ez / (1.0 + ez)
    return out


def logreg_newton(X, y, n_iter=25, tol=1e-10, ridge=1e-8):
    """Logistic regression by Newton-Raphson, Eq. (5.newton).

    A tiny ridge term keeps X^T W X invertible when the classes are
    separable or W becomes numerically singular.
    """
    n, p = X.shape
    theta = np.zeros(p)
    for it in range(n_iter):
        prob = sigmoid(X @ theta)
        gradient = X.T @ (y - prob)                       # Eq. (5.gradient)
        W = prob * (1.0 - prob)                           # Eq. (5.Wmatrix)
        H = X.T @ (W[:, None] * X) + ridge * np.eye(p)    # Eq. (5.hessian)
        step = np.linalg.solve(H, gradient)               # never invert H
        theta += step
        if np.linalg.norm(step) < tol:
            break
    return theta, it + 1

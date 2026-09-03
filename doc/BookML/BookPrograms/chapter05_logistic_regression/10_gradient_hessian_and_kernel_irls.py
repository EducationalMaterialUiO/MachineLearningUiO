"""Chapter 5: listing 10, from the section on gradient hessian and kernel irls.

Extracted from doc/BookML/chapter5.tex.
"""

import numpy as np

def kernel_irls(K, y, lmbda, n_iter=100, tol=1e-13, jitter=1e-12):
    """Kernel logistic regression by Newton's method, Eq. (5.kirls).

    Each iteration is a kernel ridge regression, Eq. (3.krr), on the working
    response z with the per-observation penalty lambda / W_ii.
    """
    n = len(y)
    alpha = np.zeros(n)
    for _ in range(n_iter):
        f = K @ alpha
        p = sigmoid(f)
        w = np.maximum(p * (1.0 - p), 1e-10)          # W_ii, Eq. (5.Wmatrix)
        z = f + (y - p) / w                           # Eq. (5.kworking)
        new = np.linalg.solve(K + lmbda * np.diag(1.0 / w)
                              + jitter * np.eye(n), z)
        if np.linalg.norm(new - alpha) < tol:
            return new
        alpha = new
    return alpha


def kernel_logistic_predict(alpha, Xtrain, Xnew, gamma):
    """p(y=1|x) = sigma(sum_j alpha_j k(x_j, x))."""
    return sigmoid(gaussian_kernel(Xnew, Xtrain, gamma) @ alpha)

"""Chapter 3: listing 10, from the section on there is no kernel lasso.

Extracted from doc/BookML/chapter3.tex.
"""

def kernel_lasso(K, y, mu, n_iter=20000, tol=1e-12):
    """Cyclic coordinate descent on ||y - K a||^2 / n + mu ||a||_1.

    Identical to lasso_coordinate_descent above with the Gram matrix as the
    design; the solution is sparse in the training points, Eq. (3.klasso).
    """
    n = len(y)
    a = np.zeros(n)
    cn = (K**2).sum(0)
    r = y - K @ a
    for _ in range(n_iter):
        a_old = a.copy()
        for j in range(n):
            r += K[:, j] * a[j]
            a[j] = soft_threshold(K[:, j] @ r, mu * n / 2.0) / cn[j]
            r -= K[:, j] * a[j]
        if np.abs(a - a_old).max() < tol:
            break
    return a

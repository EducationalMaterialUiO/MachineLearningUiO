"""Chapter 3: listing 3, from the section on weighted least squares and the chi 2 fun.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np

rng = np.random.default_rng(2024)
n, p = 50, 3
x = np.linspace(0, 1, n)
X = np.c_[np.ones(n), x, x**2]
theta_true = np.array([1.0, -2.0, 3.0])
sigmas = 0.05 + 0.3 * x                    # heteroscedastic: error grows with x

chi2_min, theta_wls, theta_ols = [], [], []
for trial in range(20000):
    y = X @ theta_true + sigmas * rng.normal(size=n)
    A, b = X / sigmas[:, None], y / sigmas   # whitened problem, Eq. (3.chisquared)
    th = np.linalg.lstsq(A, b, rcond=None)[0]
    chi2_min.append(np.sum((b - A @ th)**2) / n)
    theta_wls.append(th)
    theta_ols.append(np.linalg.lstsq(X, y, rcond=None)[0])

print("mean of n*chi2_min:", n * np.mean(chi2_min), "   n - p =", n - p)
print("variance of WLS  :", np.var(theta_wls, axis=0))
print("predicted, (X^T S^-2 X)^-1:", np.diag(np.linalg.inv(X.T @ (X / sigmas[:, None]**2))))
print("variance of OLS  :", np.var(theta_ols, axis=0))

"""Chapter 3: listing 6, from the section on statistical properties of the least squa.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np

rng = np.random.default_rng(7)
n = 40
x = np.sort(rng.random(n))
y = np.sin(2 * np.pi * x) + 0.3 * rng.normal(size=n)
X = np.vander(x, 11, increasing=True)                # degree-10 polynomial

def loo_explicit(X, y, lmbda=0.0):
    # leave-one-out by n separate fits
    n, p = X.shape
    errs = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        theta = np.linalg.solve(X[keep].T @ X[keep] + lmbda * np.eye(p),
                                X[keep].T @ y[keep])
        errs[i] = y[i] - X[i] @ theta
    return np.mean(errs**2)

def loo_closed_form(X, y, lmbda=0.0):
    # the same number from one fit, Eq. (3.press)
    n, p = X.shape
    S = X @ np.linalg.solve(X.T @ X + lmbda * np.eye(p), X.T)   # smoother matrix
    e = y - S @ y
    return np.mean((e / (1.0 - np.diag(S)))**2)

def gcv(X, y, lmbda=0.0):
    # generalised cross-validation, Eq. (3.gcv)
    n, p = X.shape
    S = X @ np.linalg.solve(X.T @ X + lmbda * np.eye(p), X.T)
    e = y - S @ y
    return np.mean(e**2) / (1.0 - np.trace(S) / n)**2

for lmbda in [0.0, 1e-4, 1e-2]:
    print(f"lambda = {lmbda:<6}  explicit {loo_explicit(X, y, lmbda):.6f}"
          f"  closed form {loo_closed_form(X, y, lmbda):.6f}"
          f"  GCV {gcv(X, y, lmbda):.6f}")

lambdas = np.logspace(-8, 0, 81)
print("lambda chosen by LOO:", lambdas[np.argmin([loo_closed_form(X, y, l) for l in lambdas])])
print("lambda chosen by GCV:", lambdas[np.argmin([gcv(X, y, l) for l in lambdas])])

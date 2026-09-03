"""Chapter 3: listing 13, from the section on choosing the penalty cross validation fo.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np
from sklearn.linear_model import Lasso

def make_data(n, p, rho, sigma, rng):
    """Correlated Gaussian features, cov[i, j] = rho^|i-j|, and a sparse truth:
    only the first eight of the p coefficients are non-zero."""
    cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    theta = np.zeros(p)
    theta[:8] = [3.0, -2.0, 1.5, -1.0, 1.0, -0.8, 0.6, -0.5]
    y = 2.0 + X @ theta + sigma * rng.normal(size=n)
    return X, y, theta

def ridge_fit(X, y, lmbda):
    """Minimiser of ||y - X theta||^2 / n + lmbda ||theta||_2^2, Eq. (3.ridgeproblem),
    for centred X and y: theta = (X^T X + n lmbda I)^{-1} X^T y, Eq. (3.ridgepern)."""
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lmbda * np.eye(p), X.T @ y)

def lasso_fit(X, y, lmbda):
    """Minimiser of ||y - X theta||^2 / n + lmbda ||theta||_1, Eq. (3.lassoproblem).
    scikit-learn minimises ||y - X theta||^2 / (2n) + alpha ||theta||_1: alpha = lmbda/2."""
    return Lasso(alpha=lmbda / 2.0, fit_intercept=False,
                 max_iter=100000, tol=1e-10).fit(X, y).coef_

def cv_curve(fit, X, y, lambdas, K=5, rng=None):
    """K-fold cross-validation error for every lambda: mean and standard error."""
    rng = np.random.default_rng() if rng is None else rng
    n = X.shape[0]
    folds = np.array_split(rng.permutation(n), K)          # shuffle, then split
    errors = np.empty((K, len(lambdas)))
    for k, held_out in enumerate(folds):
        train = np.setdiff1d(np.arange(n), held_out)
        # standardise with the statistics of the K-1 training folds only
        mu, sd, y_mean = X[train].mean(0), X[train].std(0), y[train].mean()
        X_tr, X_va = (X[train] - mu) / sd, (X[held_out] - mu) / sd
        for j, lmbda in enumerate(lambdas):
            theta = fit(X_tr, y[train] - y_mean, lmbda)
            errors[k, j] = np.mean((y[held_out] - y_mean - X_va @ theta) ** 2)
    return errors.mean(axis=0), errors.std(axis=0, ddof=1) / np.sqrt(K)

def select(lambdas, cv_mean, cv_se):
    """The minimum-CV lambda, and the largest lambda within one standard error of it."""
    i_min = np.argmin(cv_mean)
    within = cv_mean <= cv_mean[i_min] + cv_se[i_min]
    return lambdas[i_min], lambdas[within].max()

def refit(fit, X, y, lmbda):
    """The final model: one fit on all training data, mapped back to the raw scale."""
    mu, sd, y_mean = X.mean(0), X.std(0), y.mean()
    theta = fit((X - mu) / sd, y - y_mean, lmbda)
    theta_0 = y_mean - (mu / sd) @ theta               # intercept, Eq. (3.intercept)
    return theta_0, theta / sd

rng = np.random.default_rng(3155)
X, y, theta_true = make_data(n=150, p=30, rho=0.7, sigma=1.0, rng=rng)
X_train, y_train, X_test, y_test = X[:100], y[:100], X[100:], y[100:]
lambdas = np.logspace(-4, 1, 60)

for name, fit in (("Ridge", ridge_fit), ("Lasso", lasso_fit)):
    cv_mean, cv_se = cv_curve(fit, X_train, y_train, lambdas, K=5,
                              rng=np.random.default_rng(2024))
    for rule, lmbda in zip(("min-CV", "one-SE"), select(lambdas, cv_mean, cv_se)):
        theta_0, theta = refit(fit, X_train, y_train, lmbda)
        test_mse = np.mean((y_test - theta_0 - X_test @ theta) ** 2)   # used once
        j = np.searchsorted(lambdas, lmbda)
        print(f"{name} {rule}: lambda {lmbda:.4f}  CV {cv_mean[j]:.3f} +- {cv_se[j]:.3f}"
              f"  test {test_mse:.3f}  non-zero {np.sum(theta != 0):2d}"
              f"  true ones kept {np.sum(theta[:8] != 0)}")

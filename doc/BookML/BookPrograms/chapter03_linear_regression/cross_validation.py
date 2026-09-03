"""Chapter 3, Section on choosing the penalty by cross-validation.

All listings of the section in order: K-fold cross-validation from scratch for
Ridge and the Lasso with standardisation inside the folds, the minimum-CV and
one-standard-error rules, the refit and the single test evaluation; the
GridSearchCV equivalent; the leave-one-out bootstrap and .632 estimators against
cross-validation and the true error; the repeated experiment behind the bias and
spread table; and the bootstrap selection frequencies of the Lasso support.

Runs as a script and reproduces the numbers quoted in doc/BookML/chapter3.tex.
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

for name, fit in (("Ridge", ridge_fit), ("Lasso", lasso_fit)):
    selected = [select(lambdas, *cv_curve(fit, X_train, y_train, lambdas, K=5,
                                          rng=np.random.default_rng(seed)))[0]
                for seed in range(20)]
    print(f"{name}: lambda_min over 20 shuffles between"
          f" {min(selected):.4f} and {max(selected):.4f}")

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

folds = np.array_split(np.random.default_rng(2024).permutation(100), 5)
cv = [(np.setdiff1d(np.arange(100), f), f) for f in folds]  # the same folds as above

model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
# scikit-learn's Ridge minimises ||y - X theta||^2 + alpha ||theta||^2, so
# alpha = n lambda, with n = 80 the size of the training folds
search = GridSearchCV(model, {"ridge__alpha": 80 * lambdas},
                      scoring="neg_mean_squared_error", cv=cv, refit=True)
search.fit(X_train, y_train)

lambda_star = search.best_params_["ridge__alpha"] / 80
cv_curve_sklearn = -search.cv_results_["mean_test_score"]
final_model = search.best_estimator_    # a new fit on all of X_train, not a fold fit
print(f"lambda* {lambda_star:.4f}, CV {-search.best_score_:.3f}, "
      f"test {np.mean((y_test - final_model.predict(X_test)) ** 2):.3f}")
cv_mean, _ = cv_curve(ridge_fit, X_train, y_train, lambdas, K=5,
                      rng=np.random.default_rng(2024))
print("largest difference from the hand-written curve:",
      np.max(np.abs(cv_curve_sklearn - cv_mean)))

def bootstrap_errors(fit, X, y, lmbda, B=200, rng=None):
    """Training (apparent) error, leave-one-out bootstrap error, .632 estimate."""
    rng = np.random.default_rng() if rng is None else rng
    n = X.shape[0]
    theta_0, theta = refit(fit, X, y, lmbda)
    apparent = np.mean((y - theta_0 - X @ theta) ** 2)
    err_sum, err_count = np.zeros(n), np.zeros(n)
    for _ in range(B):
        idx = rng.integers(0, n, n)                         # draw n with replacement
        out = np.setdiff1d(np.arange(n), idx)               # the points left out
        theta_0, theta = refit(fit, X[idx], y[idx], lmbda)
        err_sum[out] += (y[out] - theta_0 - X[out] @ theta) ** 2
        err_count[out] += 1
    seen = err_count > 0
    loo_boot = np.mean(err_sum[seen] / err_count[seen])    # Eq. (3.loobootstrap)
    return apparent, loo_boot, 0.368 * apparent + 0.632 * loo_boot

rng = np.random.default_rng(3155)
X, y, theta_true = make_data(n=150, p=30, rho=0.7, sigma=1.0, rng=rng)
X_train, y_train = X[:100], y[:100]
# The generating process is known, so the true error of any fit can be measured
# on a large fresh sample -- a luxury never available in practice.
X_big, y_big, _ = make_data(n=20000, p=30, rho=0.7, sigma=1.0,
                            rng=np.random.default_rng(1))
lambdas = np.logspace(-4, 1, 60)

for name, fit in (("Ridge", ridge_fit), ("Lasso", lasso_fit)):
    cv_mean, cv_se = cv_curve(fit, X_train, y_train, lambdas, K=5,
                              rng=np.random.default_rng(2024))
    lambda_min, lambda_1se = select(lambdas, cv_mean, cv_se)
    for lmbda in (1e-4, lambda_min, lambda_1se):
        j = np.searchsorted(lambdas, lmbda)
        apparent, loo_boot, e632 = bootstrap_errors(fit, X_train, y_train, lmbda,
                                                    B=200, rng=np.random.default_rng(2024))
        theta_0, theta = refit(fit, X_train, y_train, lmbda)
        true = np.mean((y_big - theta_0 - X_big @ theta) ** 2)
        print(f"{name} lambda {lmbda:.4f}: train {apparent:.3f}  CV {cv_mean[j]:.3f}"
              f"  LOO-boot {loo_boot:.3f}  .632 {e632:.3f}  true {true:.3f}")

# When p approaches n the bootstrap training sets, with only 0.632 n distinct
# points, reach the singular regime of the Franke study long before the K-fold ones
rng = np.random.default_rng(5)
X, y, _ = make_data(n=100, p=60, rho=0.7, sigma=1.0, rng=rng)
X_big, y_big, _ = make_data(n=20000, p=60, rho=0.7, sigma=1.0,
                            rng=np.random.default_rng(6))
apparent, loo_boot, e632 = bootstrap_errors(ridge_fit, X, y, 1e-4, B=200, rng=rng)
cv_mean, _ = cv_curve(ridge_fit, X, y, np.array([1e-4]), K=5, rng=rng)
theta_0, theta = refit(ridge_fit, X, y, 1e-4)
print(f"p = 60, n = 100, lambda 0.0001: train {apparent:.3f}  CV {cv_mean[0]:.3f}"
      f"  LOO-boot {loo_boot:.3f}  .632 {e632:.3f}"
      f"  true {np.mean((y_big - theta_0 - X_big @ theta) ** 2):.3f}")

X_big, y_big, _ = make_data(n=20000, p=30, rho=0.7, sigma=1.0,
                            rng=np.random.default_rng(1))
# bias and spread of the four estimators over 100 fresh training sets, at fixed lambda
for name, fit, lmbda in (("Ridge", ridge_fit, 0.0131), ("Lasso", lasso_fit, 0.0626)):
    rows = []
    for seed in range(100):
        rng = np.random.default_rng(100 + seed)
        X, y, _ = make_data(n=100, p=30, rho=0.7, sigma=1.0, rng=rng)
        cv_mean, _ = cv_curve(fit, X, y, np.array([lmbda]), K=5, rng=rng)
        apparent, loo_boot, e632 = bootstrap_errors(fit, X, y, lmbda, B=100, rng=rng)
        theta_0, theta = refit(fit, X, y, lmbda)
        true = np.mean((y_big - theta_0 - X_big @ theta) ** 2)
        rows.append((apparent, cv_mean[0], loo_boot, e632, true))
    rows = np.array(rows)
    deviation = rows[:, :4] - rows[:, 4:5]                 # estimate minus true error
    print(f"{name} at lambda = {lmbda}: mean true error {rows[:, 4].mean():.3f}")
    print("  bias (training, CV, LOO-bootstrap, .632):", np.round(deviation.mean(0), 3))
    print("  spread                                 :", np.round(deviation.std(0), 3))

cv_mean, cv_se = cv_curve(lasso_fit, X_train, y_train, lambdas, K=5,
                          rng=np.random.default_rng(2024))
lambda_min, _ = select(lambdas, cv_mean, cv_se)

# selection frequency of every feature over B bootstrap replicas at lambda_min
rng, B, n = np.random.default_rng(2024), 200, 100
count = np.zeros(X_train.shape[1])
for _ in range(B):
    idx = rng.integers(0, n, n)
    theta_0, theta = refit(lasso_fit, X_train[idx], y_train[idx], lambda_min)
    count += theta != 0
frequency = count / B
theta_0, theta = refit(lasso_fit, X_train, y_train, lambda_min)
print("selected on the full training set:", np.flatnonzero(theta))
print("frequency, the eight true features:", np.round(frequency[:8], 2))
print("frequency, the spurious ones selected:")
print(np.round(frequency[8:][theta[8:] != 0], 2))
print("selected in more than 80% of the replicas:", np.flatnonzero(frequency > 0.8))

"""Chapter 3: listing 16, from the section on choosing the penalty cross validation fo.

Extracted from doc/BookML/chapter3.tex.
"""

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

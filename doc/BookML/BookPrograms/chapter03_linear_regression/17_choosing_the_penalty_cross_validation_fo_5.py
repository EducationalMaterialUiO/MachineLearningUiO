"""Chapter 3: listing 17, from the section on choosing the penalty cross validation fo.

Extracted from doc/BookML/chapter3.tex.
"""

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

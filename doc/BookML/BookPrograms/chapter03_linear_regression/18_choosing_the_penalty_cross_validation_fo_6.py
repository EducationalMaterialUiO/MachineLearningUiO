"""Chapter 3: listing 18, from the section on choosing the penalty cross validation fo.

Extracted from doc/BookML/chapter3.tex.
"""

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

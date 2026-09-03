"""Chapter 3: listing 8, from the section on comparing the three estimators.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso

X = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
y = np.array([4.0, 2.0, 3.0])

n = X.shape[0]
lambdas = np.logspace(-3, 2, 200)

# scikit-learn's Ridge minimises ||y - X t||^2 + alpha ||t||^2, so alpha = lambda,
# but its Lasso minimises ||y - X t||^2 / (2n) + alpha ||t||_1, so alpha = lambda / (2n).
ridge_path = np.array([Ridge(alpha=l, fit_intercept=False).fit(X, y).coef_
                       for l in lambdas])
lasso_path = np.array([Lasso(alpha=l / (2 * n), fit_intercept=False,
                             max_iter=100000).fit(X, y).coef_
                       for l in lambdas])

# Analytical results: Eq. (3.toyridge) for Ridge, Eq. (3.softthreshold) for Lasso
ridge_exact = np.column_stack([8.0 / (4.0 + lambdas), 2.0 / (1.0 + lambdas)])
lasso_exact = np.column_stack([np.maximum((16.0 - lambdas) / 8.0, 0.0),
                               np.maximum(2.0 - lambdas / 2.0, 0.0)])
assert np.allclose(ridge_path, ridge_exact) and np.allclose(lasso_path, lasso_exact)

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for k in range(2):
    ax[0].semilogx(lambdas, ridge_path[:, k], label=rf"$\theta_{k}$")
    ax[1].semilogx(lambdas, lasso_path[:, k], label=rf"$\theta_{k}$")
ax[0].set_title("Ridge"); ax[1].set_title("Lasso")
for a in ax:
    a.set_xlabel(r"$\lambda$"); a.legend()
plt.show()

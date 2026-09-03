"""Chapter 3: listing 5, from the section on statistical properties of the least squa.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(2024)
n, p, sigma = 30, 4, 0.5
x = np.linspace(-1, 1, n)
X = np.vander(x, p, increasing=True)                 # cubic polynomial
theta_true = np.array([1.0, 0.5, -2.0, 1.5])
XtX_inv = np.linalg.inv(X.T @ X)
t_q = stats.t.ppf(0.975, n - p)                      # 2.056 for n-p = 26

s2, covered, joint, train, test = [], [], [], [], []
for trial in range(20000):
    y = X @ theta_true + sigma * rng.normal(size=n)
    theta = XtX_inv @ X.T @ y
    e = y - X @ theta
    s2.append(e @ e / (n - p))                        # Eq. (3.sigmahat)
    se = np.sqrt(s2[-1] * np.diag(XtX_inv))
    inside = np.abs(theta - theta_true) <= t_q * se  # Eq. (3.confint)
    covered.append(inside); joint.append(inside.all())
    train.append(e @ e / n)                           # training MSE
    y_new = X @ theta_true + sigma * rng.normal(size=n)
    test.append(np.mean((y_new - X @ theta)**2))     # fresh targets, same X

print("E[sigma_hat^2] =", np.mean(s2), "  sigma^2 =", sigma**2)
print("coverage per coefficient:", np.mean(covered, axis=0))
print("all four covered at once:", np.mean(joint))
print("E[train MSE] =", np.mean(train), "  sigma^2 (1 - p/n) =", sigma**2 * (1 - p/n))
print("E[test  MSE] =", np.mean(test),  "  sigma^2 (1 + p/n) =", sigma**2 * (1 + p/n))

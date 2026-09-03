"""Chapter 4: listing 2, from the section on gradient descent for ridge regression.

Extracted from doc/BookML/chapter4.tex.
"""

import numpy as np

lmbda = 0.001
theta = rng.normal(size=(2, 1))
eta = 0.1
for k in range(1000):
    gradient = 2.0 * (X.T @ (X @ theta - y) / n + lmbda * theta)
    theta -= eta * gradient

# Compare with the closed form of Chapter 3
I = np.eye(X.shape[1])
theta_exact = np.linalg.inv(X.T @ X + n * lmbda * I) @ X.T @ y
print("gradient descent:", theta.ravel())
print("closed form:     ", theta_exact.ravel())

"""Chapter 4: listing 1, from the section on gradient descent for linear regression.

Extracted from doc/BookML/chapter4.tex.
"""

import numpy as np

n = 100
rng = np.random.default_rng(2024)
x = 2.0 * rng.random((n, 1))
y = 4.0 + 3.0 * x + rng.normal(size=(n, 1))
X = np.c_[np.ones((n, 1)), x]

# The analytical solution of Chapter 3, for comparison
theta_exact = np.linalg.pinv(X.T @ X) @ X.T @ y
print("analytical:", theta_exact.ravel())

# Gradient descent, Eq. (4.gditeration)
theta = rng.normal(size=(2, 1))
eta = 0.1
for k in range(1000):
    gradient = (2.0 / n) * X.T @ (X @ theta - y)
    theta -= eta * gradient
    if np.linalg.norm(gradient) < 1.0e-8:
        break
print(f"gradient descent after {k+1} iterations:", theta.ravel())

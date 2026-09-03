"""Chapter 2: listing 3, from the section on the statistics of the least squares esti.

Extracted from doc/BookML/chapter2.tex.
"""

import numpy as np

rng = np.random.default_rng(2024)
x = np.linspace(0.0, 1.0, 5)            # the design is fixed once and for all
X = np.column_stack([np.ones_like(x), x, x**2])
theta, sigma = np.array([1.0, -2.0, 3.0]), 0.5
mu = X @ theta                          # non-random: the same in every repetition

Y = mu + sigma * rng.normal(size=(100000, len(x)))  # 100000 draws of the noise only
print("X theta        :", np.round(mu, 3))
print("mean of y      :", np.round(Y.mean(axis=0), 3))
print("variance of y  :", np.round(Y.var(axis=0), 3), "  sigma^2 =", sigma**2)
print("cov(y_0, y_1)  :", np.round(np.cov(Y[:, 0], Y[:, 1])[0, 1], 4))

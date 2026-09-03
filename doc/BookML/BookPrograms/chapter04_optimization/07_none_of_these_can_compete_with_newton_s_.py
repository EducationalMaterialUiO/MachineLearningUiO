"""Chapter 4: listing 7, from the section on none of these can compete with newton s .

Extracted from doc/BookML/chapter4.tex.
"""

import numpy as np

# One Newton step solves the least-squares problem exactly
H = (2.0 / n) * X.T @ X
theta = rng.normal(size=(2, 1))
gradient = (2.0 / n) * X.T @ (X @ theta - y)
theta -= np.linalg.solve(H, gradient)
print("after one Newton step:", theta.ravel())
print("analytical solution:  ", (np.linalg.pinv(X.T @ X) @ X.T @ y).ravel())

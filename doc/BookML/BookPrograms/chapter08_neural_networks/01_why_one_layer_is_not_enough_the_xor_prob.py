"""Chapter 8: listing 1, from the section on why one layer is not enough the xor prob.

Extracted from doc/BookML/chapter8.tex.
"""

import numpy as np

X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float64)
Xinv = np.linalg.pinv(X.T @ X)

for name, y in [("XOR", [0, 1, 1, 0]), ("OR", [0, 1, 1, 1]), ("AND", [0, 0, 0, 1])]:
    y = np.array(y, dtype=np.float64)
    theta = Xinv @ X.T @ y                     # Eq. (3.olssolution)
    print(f"{name}: theta = {np.round(theta, 3)}, "
          f"prediction = {np.round(X @ theta, 3)}")

"""Chapter 6: listing 1, from the section on kernels and non linearity.

Extracted from doc/BookML/chapter6.tex.
"""

import numpy as np
import matplotlib.pyplot as plt

X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
X2D = np.c_[X1D, X1D**2]                     # the map phi(x) = (x, x^2)
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(X1D[y == 0], np.zeros(4), "bs")
ax[0].plot(X1D[y == 1], np.zeros(5), "g^")
ax[0].set_xlabel(r"$x_1$"); ax[0].set_yticks([]); ax[0].axhline(0, color="k")

ax[1].plot(X2D[y == 0, 0], X2D[y == 0, 1], "bs")
ax[1].plot(X2D[y == 1, 0], X2D[y == 1, 1], "g^")
ax[1].plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)   # the separating line
ax[1].set_xlabel(r"$x_1$"); ax[1].set_ylabel(r"$x_2$")
plt.show()

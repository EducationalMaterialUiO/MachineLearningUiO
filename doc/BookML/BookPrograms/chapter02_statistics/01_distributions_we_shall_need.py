"""Chapter 2: listing 1, from the section on distributions we shall need.

Extracted from doc/BookML/chapter2.tex.
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-20.0, 20.0, 400)

for mu, sigma in [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)]:
    p = np.exp(-(x - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
    plt.plot(x, p, label=rf"$\mu={mu}$, $\sigma={sigma}$")

plt.xlabel(r"$x$"); plt.ylabel(r"$p(x)$")
plt.legend(); plt.show()

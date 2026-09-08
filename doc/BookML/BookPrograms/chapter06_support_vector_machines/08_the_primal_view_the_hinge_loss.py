"""Chapter 6: listing 8, from the section on the primal view the hinge loss.

Extracted from doc/BookML/chapter6.tex.
"""

import numpy as np

def pegasos(X, y, lmbda=0.01, epochs=300, rng=None):
    """Primal SVM by stochastic subgradient descent on Eq. (6.hingeobjective)."""
    rng = np.random.default_rng(0) if rng is None else rng
    n, p = X.shape
    w, b, t = np.zeros(p), 0.0, 0
    for _ in range(epochs):
        for i in rng.permutation(n):
            t += 1
            gamma = 1.0 / (lmbda * t)                  # Eq. (4.timedecay)
            if y[i] * (w @ X[i] + b) < 1:            # inside the margin
                w = (1 - gamma * lmbda) * w + gamma * y[i] * X[i]
                b += gamma * y[i]
            else:                                    # outside: only the penalty
                w = (1 - gamma * lmbda) * w
    return w, b

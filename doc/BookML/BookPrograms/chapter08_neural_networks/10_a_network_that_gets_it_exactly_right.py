"""Chapter 8: listing 10, from the section on a network that gets it exactly right.

Extracted from doc/BookML/chapter8.tex.
"""

import numpy as np

# Proposition 8.isingexact, written down rather than trained
W1 = np.zeros((L, 2 * L))
for j in range(L):
    W1[j, 2 * j], W1[(j + 1) % L, 2 * j] = 1.0, 1.0        # +s_j + s_{j+1}
    W1[j, 2 * j + 1], W1[(j + 1) % L, 2 * j + 1] = -1.0, -1.0
W2, b2 = np.full(2 * L, -Jtrue), Jtrue * L
predict = lambda S: np.maximum(S @ W1, 0.0) @ W2 + b2

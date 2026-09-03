"""Chapter 3: listing 14, from the section on the model and the design matrix.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np

rng = np.random.default_rng(2718)
L, n, Jtrue = 40, 10000, 1.0
spins = rng.choice([-1, 1], size=(n, L))
energies = -Jtrue * np.einsum("ij,ij->i", spins, np.roll(spins, 1, axis=1))

# design matrix, Eq. (3.isingdesign): X[i, j*L+k] = s_j s_k
X = np.einsum("ij,ik->ijk", spins, spins).reshape(n, L * L)

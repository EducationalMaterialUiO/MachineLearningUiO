"""Chapter 3: listing 7, from the section on a complete example the franke function.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np

def franke_function(x, y):
    """The Franke function, Eq. (3.franke)."""
    t1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    t2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    t3 = 0.50 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    t4 = -0.20 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return t1 + t2 + t3 + t4


def make_franke_data(n=400, noise=0.1, rng=None):
    """Sample the Franke function on random points with Gaussian noise."""
    rng = np.random.default_rng() if rng is None else rng
    x, y = rng.random(n), rng.random(n)
    z = franke_function(x, y) + noise * rng.normal(size=n)
    return x, y, z

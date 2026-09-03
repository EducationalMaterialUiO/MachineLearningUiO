"""Chapter 2: listing 7, from the section on random numbers.

Extracted from doc/BookML/chapter2.tex.
"""

import numpy as np

rng = np.random.default_rng(seed=2024)     # a generator object, not global state
print(rng.random(5))                       # uniform on [0, 1)
print(rng.normal(loc=0.0, scale=1.0, size=5))
print(rng.integers(0, 10, size=5))

# Passing the generator explicitly keeps functions reproducible and
# independent of any other code that also draws random numbers.
def experiment(rng):
    return np.mean(rng.normal(size=1000))

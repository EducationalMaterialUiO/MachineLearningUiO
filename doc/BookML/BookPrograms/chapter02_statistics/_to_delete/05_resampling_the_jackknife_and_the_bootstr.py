"""Chapter 2: listing 5, from the section on resampling the jackknife and the bootstr.

Extracted from doc/BookML/chapter2.tex.
"""

import numpy as np

def bootstrap(data, statistic, k=1000, rng=None):
    """Bootstrap estimate of the distribution of a statistic."""
    rng = np.random.default_rng() if rng is None else rng
    n = len(data)
    replicas = np.empty(k)
    for i in range(k):
        sample = data[rng.integers(0, n, n)]      # draw n values with replacement
        replicas[i] = statistic(sample)
    return replicas


def jackknife(data, statistic):
    """Leave-one-out replicas of a statistic."""
    n = len(data)
    return np.array([statistic(np.delete(data, i)) for i in range(n)])


rng = np.random.default_rng(2024)
mu, sigma, n = 100.0, 15.0, 10000
x = rng.normal(mu, sigma, n)

t = bootstrap(x, np.mean, k=1000, rng=rng)
print(f"original {np.mean(x):.5f}  bootstrap mean {np.mean(t):.5f}"
      f"  std. error {np.std(t):.5f}")
print(f"central limit theorem prediction: {np.std(x) / np.sqrt(n):.5f}")

tj = jackknife(x, np.mean)
print(f"jackknife bias {(n - 1) * (np.mean(tj) - np.mean(x)):.3e}"
      f"  std. error {np.sqrt((n - 1) * np.var(tj)):.5f}")

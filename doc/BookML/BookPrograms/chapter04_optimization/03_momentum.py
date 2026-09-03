"""Chapter 4: listing 3, from the section on momentum.

Extracted from doc/BookML/chapter4.tex.
"""

import numpy as np

def objective(x):
    return x**2.0

def derivative(x):
    return 2.0 * x

def gradient_descent(derivative, bounds, n_iter, step_size, momentum=0.0,
                     rng=None):
    """Gradient descent with optional momentum, Eq. (4.momentum)."""
    rng = np.random.default_rng() if rng is None else rng
    solution = bounds[:, 0] + rng.random(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    change = 0.0
    solutions, scores = [], []
    for i in range(n_iter):
        gradient = derivative(solution)
        new_change = step_size * gradient + momentum * change
        solution = solution - new_change
        change = new_change
        solutions.append(solution.copy())
        scores.append(objective(solution))
    return solutions, scores


bounds = np.asarray([[-1.0, 1.0]])
rng = np.random.default_rng(4)
plain = gradient_descent(derivative, bounds, 30, 0.1, momentum=0.0,
                         rng=np.random.default_rng(4))
withmom = gradient_descent(derivative, bounds, 30, 0.1, momentum=0.3,
                           rng=np.random.default_rng(4))
print(f"after 30 steps: plain f = {plain[1][-1][0]:.3e}, "
      f"momentum f = {withmom[1][-1][0]:.3e}")

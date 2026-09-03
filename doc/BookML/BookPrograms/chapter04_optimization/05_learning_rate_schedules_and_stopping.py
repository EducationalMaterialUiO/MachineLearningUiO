"""Chapter 4: listing 5, from the section on learning rate schedules and stopping.

Extracted from doc/BookML/chapter4.tex.
"""

import numpy as np

def step_length(t, t0, t1):
    return t0 / (t + t1)

n, M = 100, 5
m = int(n / M)
n_epochs, t0, t1 = 500, 1.0, 10.0

for epoch in range(1, n_epochs + 1):
    for i in range(m):
        k = np.random.randint(m)
        t = epoch * m + i
        eta = step_length(t, t0, t1)
        # compute the minibatch gradient and update theta
print(f"eta after {n_epochs} epochs: {step_length(n_epochs*m, t0, t1):g}")

"""Chapter 9: listing 3, from the section on exponential decay.

Extracted from doc/BookML/chapter9.tex.
"""

import numpy as np
gamma, g0 = 2.0, 10.0
X = np.linspace(0, 1, 50).reshape(-1, 1)          # collocation points

def residual_decay(P, X):
    """Eq. (9.resdecay)."""
    N, dN = network_derivs(P, X, "tanh", order=1)
    x = X[:, 0]
    g_t  = g0 + x * N                              # Eq. (9.trialode)
    dg_t = N + x * dN                              # Eq. (9.dtrialode)
    return dg_t + gamma * g_t

P, history = solve_de(residual_decay, [1, 40, 40, 1], X, "tanh",
                      n_iter=3000, eta=2e-2, rng=np.random.default_rng(1))

N, dN = network_derivs(P, X, "tanh", order=1)
g_network = g0 + X[:, 0] * N
g_exact = g0 * np.exp(-gamma * X[:, 0])
print(f"max relative error {np.abs(g_network - g_exact).max() / g0:.2e}")

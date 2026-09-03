"""Chapter 9: listing 4, from the section on logistic population growth.

Extracted from doc/BookML/chapter9.tex.
"""

alpha, A, g0 = 2.0, 1.0, 1.2
T = np.linspace(0, 1, 50).reshape(-1, 1)

def residual_logistic(P, X):
    """Eq. (9.reslogistic)."""
    N, dN = network_derivs(P, X, "tanh", order=1)
    t = X[:, 0]
    g_t  = g0 + t * N
    dg_t = N + t * dN
    return dg_t - alpha * g_t * (A - g_t)

P, history = solve_de(residual_logistic, [1, 40, 40, 1], T, "tanh",
                      n_iter=3000, eta=2e-2, rng=np.random.default_rng(1))

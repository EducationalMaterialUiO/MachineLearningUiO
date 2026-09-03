"""Chapter 9: listing 5, from the section on the one dimensional poisson equation.

Extracted from doc/BookML/chapter9.tex.
"""

f = lambda x: (3 * x + x**2) * np.exp(x)
X = np.linspace(0, 1, 60).reshape(-1, 1)

def residual_poisson(P, X):
    """Eqs. (9.trialpoisson) and (9.d2trialpoisson)."""
    N, dN, d2N = network_derivs(P, X, "tanh", order=2)
    x = X[:, 0]
    d2g_t = -2 * N + 2 * (1 - 2 * x) * dN + x * (1 - x) * d2N
    return -d2g_t - f(x)

P, history = solve_de(residual_poisson, [1, 30, 30, 1], X, "tanh",
                      n_iter=4000, eta=1e-2, rng=np.random.default_rng(2))

"""Chapter 9: listing 9, from the section on physics informed neural networks.

Extracted from doc/BookML/chapter9.tex.
"""

def pinn_solve(terms, layer_sizes, activation="tanh", n_iter=2000, eta=1e-2,
               rng=None, every=200):
    """terms: list of (name, weight, residual_fn, points), Eq. (9.pinntotal).

    Each residual_fn(P, X) returns the residual on its own point set, so the
    equation, the initial condition and the boundaries are treated alike.
    """
    P = init_parameters(layer_sizes, activation, rng)

    def cost(P):
        total = 0.0
        for _, w, residual, Xk in terms:
            total = total + w * np.mean(residual(P, Xk) ** 2)
        return total

    return adam_minimise(cost, P, n_iter=n_iter, eta=eta)

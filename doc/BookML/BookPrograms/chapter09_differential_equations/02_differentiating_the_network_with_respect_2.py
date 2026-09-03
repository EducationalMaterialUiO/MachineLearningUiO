"""Chapter 9: listing 2, from the section on differentiating the network with respect.

Extracted from doc/BookML/chapter9.tex.
"""

def init_parameters(layer_sizes, activation="tanh", rng=None):
    """He (8.he) for the ReLU family, Xavier (8.xavier) otherwise."""
    rng = np.random.default_rng(0) if rng is None else rng
    P = []
    for i in range(len(layer_sizes) - 1):
        nin, nout = layer_sizes[i], layer_sizes[i + 1]
        s = np.sqrt(2.0 / nin) if activation in ("relu", "leaky_relu", "elu") \
            else np.sqrt(1.0 / nin)
        P.append([rng.normal(0, s, (nin, nout)), np.zeros(nout)])
    return P


def solve_de(residual, layer_sizes, X, activation="tanh", n_iter=2000,
             eta=1e-2, rng=None):
    """Minimise the mean squared residual, Eq. (9.cost), with Adam."""
    P = init_parameters(layer_sizes, activation, rng)
    cost = lambda P: np.mean(residual(P, X)**2)
    return adam_minimise(cost, P, n_iter=n_iter, eta=eta)

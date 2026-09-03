"""Chapter 9: listing 14, from the section on implementation.

Extracted from doc/BookML/chapter9.tex.
"""

K, T, r, sigma, S_max = 5.0, 1.0, 0.05, 0.3, 20.0
hi = S_max / K                                  # domain in scaled units

def c_net(P, X): return network(P, X, "tanh")   # X = (s, tau)

c_tau = d_dxk(c_net, 1)
c_s = d_dxk(c_net, 0)
c_ss = d_dxk(c_s, 0)

def r_pde(P, X):                                # Eq. (9.bsscaledpde)
    s = X[:, 0]
    return (-c_tau(P, X) + 0.5 * sigma**2 * s**2 * c_ss(P, X)
            + r * s * c_s(P, X) - r * c_net(P, X))

def r_term(P, X):                               # payoff at tau = 0
    return c_net(P, X) - np.maximum(X[:, 0] - 1.0, 0.0)

def r_lo(P, X):                                 # c(0, tau) = 0
    return c_net(P, X)

def r_hi(P, X):                                 # c(hi, tau) = hi - exp(-r tau)
    return c_net(P, X) - (hi - np.exp(-r * X[:, 1]))

terms = [("pde", 1.0, r_pde, X_col), ("term", 10.0, r_term, X_term),
         ("lo", 10.0, r_lo, X_lo),   ("hi", 10.0, r_hi, X_hi)]

P, history = pinn_solve(terms, [2, 40, 40, 1], "tanh", n_iter=4000, eta=5e-3,
                        rng=np.random.default_rng(1))

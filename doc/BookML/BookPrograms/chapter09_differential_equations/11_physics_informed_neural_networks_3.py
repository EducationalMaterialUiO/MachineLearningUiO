"""Chapter 9: listing 11, from the section on physics informed neural networks.

Extracted from doc/BookML/chapter9.tex.
"""

def u_net(P, X): return network(P, X, "tanh")
u_t, u_x = d_dxk(u_net, 1), d_dxk(u_net, 0)
u_tt, u_xx = d_dxk(u_t, 1), d_dxk(u_x, 0)

def r_pde(P, X): return u_tt(P, X) - c**2 * u_xx(P, X)    # Eq. (9.wave)
def r_ic(P, X):  return u_net(P, X) - np.sin(np.pi * X[:, 0])
def r_iv(P, X):  return u_t(P, X)                    # dg/dt = v(x) = 0 at t=0
def r_bc(P, X):  return u_net(P, X)

terms = [("pde", 1.0, r_pde, X_col), ("ic", 10.0, r_ic, X_ic),
         ("iv",  10.0, r_iv,  X_ic),                 # the initial velocity
         ("bcL", 10.0, r_bc,  X_l), ("bcR", 10.0, r_bc, X_r)]

P, history = pinn_solve(terms, [2, 30, 30, 1], "tanh", n_iter=4000, eta=1e-2,
                        rng=np.random.default_rng(1))

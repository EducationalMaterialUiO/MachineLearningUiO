"""Chapter 9: listing 10, from the section on physics informed neural networks.

Extracted from doc/BookML/chapter9.tex.
"""

nx, nt = 20, 20
xs, ts = np.linspace(0, 1, nx), np.linspace(0, 1, nt)

Xi, Ti = np.meshgrid(xs[1:-1], ts[1:-1], indexing="ij")   # interior only
X_col = np.column_stack([Xi.ravel(), Ti.ravel()])         # 324 points
X_ic  = np.column_stack([xs, np.zeros(nx)])               # t = 0
X_l   = np.column_stack([np.zeros(nt), ts])               # x = 0
X_r   = np.column_stack([np.ones(nt),  ts])               # x = 1

def u_net(P, X): return network(P, X, "tanh")             # Eq. (9.pinnu)

u_t = d_dxk(u_net, 1)
u_x = d_dxk(u_net, 0)
u_xx = d_dxk(u_x, 0)

def r_pde(P, X): return u_t(P, X) - u_xx(P, X)            # Eq. (9.pinnpde)
def r_ic(P, X):  return u_net(P, X) - np.sin(np.pi * X[:, 0])
def r_bc(P, X):  return u_net(P, X)

terms = [("pde", 1.0, r_pde, X_col), ("ic",  10.0, r_ic, X_ic),
         ("bcL", 10.0, r_bc, X_l),   ("bcR", 10.0, r_bc, X_r)]

P, history = pinn_solve(terms, [2, 30, 30, 1], "tanh", n_iter=4000, eta=1e-2,
                        rng=np.random.default_rng(1))

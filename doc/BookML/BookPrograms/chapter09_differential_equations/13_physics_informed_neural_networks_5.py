"""Chapter 9: listing 13, from the section on physics informed neural networks.

Extracted from doc/BookML/chapter9.tex.
"""

D_true = 0.5
rng = np.random.default_rng(3)
X_obs = np.column_stack([rng.uniform(0, 1, 40), rng.uniform(0, 1, 40)])
y_obs = np.exp(-D_true * np.pi**2 * X_obs[:, 1]) * np.sin(np.pi * X_obs[:, 0]) \
        + rng.normal(0, 0.01, 40)

def u_net(P, X): return network(P[0], X, "tanh")     # P = [weights, D]
u_t, u_x = d_dxk(u_net, 1), d_dxk(u_net, 0)
u_xx = d_dxk(u_x, 0)

def cost(P):
    D = P[1][0]
    return np.mean((u_t(P, X_col) - D * u_xx(P, X_col)) ** 2) \
         + 10.0 * np.mean((u_net(P, X_obs) - y_obs) ** 2)   # Eq. (9.pinninv)

P = [init_parameters([2, 30, 30, 1], "tanh", np.random.default_rng(1)),
     np.array([2.0])]                                 # initial guess D_0 = 2
P = adam_general(cost, P, n_iter=3000, eta=1e-2)

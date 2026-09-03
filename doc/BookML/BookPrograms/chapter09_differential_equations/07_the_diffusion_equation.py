"""Chapter 9: listing 7, from the section on the diffusion equation.

Extracted from doc/BookML/chapter9.tex.
"""

nx, nt = 20, 20
xs, ts = np.linspace(0, 1, nx), np.linspace(0, 1, nt)
Xg, Tg = np.meshgrid(xs, ts, indexing="ij")
X = np.column_stack([Xg.ravel(), Tg.ravel()])       # M = 400 collocation points

def trial_diffusion(P, X):
    """Eq. (9.trialdiff)."""
    x, t = X[:, 0], X[:, 1]
    return (1 - t) * np.sin(np.pi * x) + x * (1 - x) * t * network(P, X, "tanh")

u_t  = d_dxk(trial_diffusion, 1)
u_x  = d_dxk(trial_diffusion, 0)
u_xx = d_dxk(u_x, 0)

def residual_diffusion(P, X):
    return u_t(P, X) - u_xx(P, X)                   # Eq. (9.diffusion)

P, history = solve_de(residual_diffusion, [2, 30, 30, 1], X, "tanh",
                      n_iter=800, eta=1e-2, rng=np.random.default_rng(1))

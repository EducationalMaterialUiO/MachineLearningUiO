"""Chapter 9: listing 8, from the section on the wave equation.

Extracted from doc/BookML/chapter9.tex.
"""

c = 1.0

def trial_wave(P, X):
    """Eq. (9.trialwave); the t^2 makes dg/dt vanish at t = 0."""
    x, t = X[:, 0], X[:, 1]
    return (1 - t**2) * np.sin(np.pi * x) + x * (1 - x) * t**2 * network(P, X, "tanh")

w_t, w_x = d_dxk(trial_wave, 1), d_dxk(trial_wave, 0)
w_tt, w_xx = d_dxk(w_t, 1), d_dxk(w_x, 0)

def residual_wave(P, X):
    return w_tt(P, X) - c**2 * w_xx(P, X)           # Eq. (9.wave)

P, history = solve_de(residual_wave, [2, 30, 30, 1], X, "tanh",
                      n_iter=800, eta=1e-2, rng=np.random.default_rng(1))

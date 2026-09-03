"""Chapter 16: listing 1, from the section on sampling.

Extracted from doc/BookML/chapter16.tex.
"""

def q_sample(x0, t, abar, eps):
    """x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps,  Eq. (16.marginal)."""
    a = abar[t][:, None]
    return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * eps


def loss(P, x0, t, eps, abar, T):
    """L_simple, Eq. (16.simple): predict the noise from the noisy sample."""
    xt = q_sample(x0, t, abar, eps)
    return np.mean(np.sum((eps - eps_net(P, xt, t, T)) ** 2, axis=1))

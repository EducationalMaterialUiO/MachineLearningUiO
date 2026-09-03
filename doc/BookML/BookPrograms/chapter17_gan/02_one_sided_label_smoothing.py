"""Chapter 17: listing 2, from the section on one sided label smoothing.

Extracted from doc/BookML/chapter17.tex.
"""

def d_loss(Q, P, x_real, z, smooth=1.0):
    """L_D = -V, with one-sided label smoothing s = `smooth`, Eq. (17.dloss)."""
    u_real = discriminator(Q, x_real)
    u_fake = discriminator(Q, generator(P, z))
    real = smooth * softplus(-u_real) + (1.0 - smooth) * softplus(u_real)
    return np.mean(real) + np.mean(softplus(u_fake))


def g_loss_nonsat(P, Q, z):
    """L_G = -E[log D(G(z))],  the non-saturating loss, Eq. (17.nonsat)."""
    return np.mean(softplus(-discriminator(Q, generator(P, z))))


def g_loss_sat(P, Q, z):
    """L_G = E[log(1 - D(G(z)))],  the original minimax loss, Eq. (17.minimax)."""
    return -np.mean(softplus(discriminator(Q, generator(P, z))))

"""Chapter 17: listing 1, from the section on the minimax objective.

Extracted from doc/BookML/chapter17.tex.
"""

def softplus(u):
    """log(1+exp(u)), evaluated without overflow."""
    return np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u)))


def value_function(Q, P, x_real, z):
    """V(G,D) = E[log D(x)] + E[log(1 - D(G(z)))],  Eq. (17.minimax)."""
    u_real = discriminator(Q, x_real)
    u_fake = discriminator(Q, generator(P, z))
    return -np.mean(softplus(-u_real)) - np.mean(softplus(u_fake))

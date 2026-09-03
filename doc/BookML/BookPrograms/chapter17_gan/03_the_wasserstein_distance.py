"""Chapter 17: listing 3, from the section on the wasserstein distance.

Extracted from doc/BookML/chapter17.tex.
"""

def gradient_penalty(Q, x_real, x_fake, rng):
    """E[(||grad f_w(xhat)||_2 - 1)^2] on the segments between real and fake."""
    eps = rng.uniform(0.0, 1.0, (len(x_real), 1))
    xhat = eps * x_real + (1.0 - eps) * x_fake
    g = grad(lambda x: np.sum(critic(Q, x)))(xhat)
    return np.mean((np.sqrt(np.sum(g ** 2, axis=1) + 1e-12) - 1.0) ** 2)


def critic_loss(Q, P, x_real, z, rng, lam=10.0):
    """L_critic = E_pg[f] - E_pr[f] + lambda * GP,  Eq. (17.wgangp)."""
    x_fake = generator(P, z)
    w = np.mean(critic(Q, x_fake)) - np.mean(critic(Q, x_real))
    return w + lam * gradient_penalty(Q, x_real, x_fake, rng)

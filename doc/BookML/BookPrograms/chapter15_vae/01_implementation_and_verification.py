"""Chapter 15: listing 1, from the section on implementation and verification.

Extracted from doc/BookML/chapter15.tex.
"""

def kl_gaussian(mu, logvar):
    """KL(N(mu, sigma^2 I) || N(0, I)) in closed form, Eq. (15.klclosed)."""
    return 0.5 * np.sum(mu ** 2 + np.exp(logvar) - logvar - 1.0, axis=-1)


def elbo(P, X, eps):
    """ELBO with the reparameterisation trick, Eqs. (15.elbo) and (15.reparam).

    eps is supplied from outside so that the randomness is an input rather than
    a side effect: that is exactly what makes the estimator differentiable.
    """
    mu, logvar = encode(P, X)
    H = mu + np.exp(0.5 * logvar) * eps          # Eq. (15.reparam)
    rec = bernoulli_logpdf(decode(P, H), X)
    return np.mean(rec - kl_gaussian(mu, logvar))

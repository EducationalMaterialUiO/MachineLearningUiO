"""Chapter 15: a variational autoencoder from scratch.

Written with autograd, as in Chapters 9 and 13.  The encoder and decoder are
the networks of Chapter 8; everything new is in the objective.
"""
import autograd.numpy as np
from autograd import grad
from autograd.misc import flatten


def init_mlp(sizes, rng, scale=None):
    P = []
    for i in range(len(sizes) - 1):
        s = np.sqrt(1.0 / sizes[i]) if scale is None else scale
        P.append([rng.normal(0, s, (sizes[i], sizes[i + 1])),
                  np.zeros(sizes[i + 1])])
    return P


def mlp(P, X, act=np.tanh):
    a = X
    for l, (W, b) in enumerate(P):
        z = a @ W + b
        a = act(z) if l < len(P) - 1 else z
    return a


def init_vae(d, dh, hidden=32, rng=None):
    """Encoder outputs (mu, log sigma^2); decoder outputs Bernoulli logits."""
    rng = np.random.default_rng(0) if rng is None else rng
    return {"enc": init_mlp([d, hidden, 2 * dh], rng),
            "dec": init_mlp([dh, hidden, d], rng)}


def latent_dim(P):
    return P["enc"][-1][0].shape[1] // 2


def encode(P, X):
    """q_phi(h|x) = N(mu(x), diag sigma^2(x)),  Eq. (15.encoder)."""
    out = mlp(P["enc"], X)
    dh = latent_dim(P)
    return out[:, :dh], out[:, dh:]              # mu, log sigma^2


def decode(P, H):
    return mlp(P["dec"], H)                      # Bernoulli logits


# ---------------------------------------------------------------------------
# the two terms of the ELBO
# ---------------------------------------------------------------------------
def kl_gaussian(mu, logvar):
    """KL(N(mu, sigma^2 I) || N(0, I)) in closed form, Eq. (15.klclosed)."""
    return 0.5 * np.sum(mu ** 2 + np.exp(logvar) - logvar - 1.0, axis=-1)


def bernoulli_logpdf(logits, X):
    """log p_theta(x|h) for a factorised Bernoulli decoder."""
    return np.sum(X * (-np.logaddexp(0.0, -logits))
                  + (1 - X) * (-np.logaddexp(0.0, logits)), axis=-1)


def elbo(P, X, eps):
    """ELBO with the reparameterisation trick, Eqs. (15.elbo) and (15.reparam).

    eps is supplied from outside so that the randomness is an input rather than
    a side effect: that is exactly what makes the estimator differentiable.
    """
    mu, logvar = encode(P, X)
    H = mu + np.exp(0.5 * logvar) * eps          # Eq. (15.reparam)
    rec = bernoulli_logpdf(decode(P, H), X)
    return np.mean(rec - kl_gaussian(mu, logvar))


def train_vae(P, X, n_iter=2000, batch=64, eta=2e-3, rng=None, every=500,
              verbose=False):
    rng = np.random.default_rng(0) if rng is None else rng
    flat, unflatten = flatten(P)
    g = grad(lambda f, xb, e: -elbo(unflatten(f), xb, e))
    m = np.zeros_like(flat); v = np.zeros_like(flat)
    hist = []
    for it in range(1, n_iter + 1):
        idx = rng.integers(0, len(X), batch)
        xb = X[idx]
        e = rng.normal(size=(batch, latent_dim(P)))
        gg = g(flat, xb, e)
        m = 0.9 * m + 0.1 * gg
        v = 0.999 * v + 0.001 * gg ** 2
        flat = flat - eta * (m / (1 - 0.9**it)) / (np.sqrt(v / (1 - 0.999**it)) + 1e-8)
        if it % every == 0 or it == 1:
            Pc = unflatten(flat)
            e2 = rng.normal(size=(len(X), latent_dim(P)))
            hist.append((it, float(elbo(Pc, X, e2))))
            if verbose:
                print(f"  it {it:5d}  ELBO {hist[-1][1]:.4f}")
    return unflatten(flat), hist

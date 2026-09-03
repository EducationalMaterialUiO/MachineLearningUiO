"""Chapter 17: generative adversarial networks from scratch.

Everything here is written against ``autograd.numpy`` so that the two players
can be differentiated separately: the discriminator gradient is taken with the
generator held fixed and vice versa, which is exactly the alternating scheme of
Section 17.algorithm.

The networks are deliberately small.  The point of the two-dimensional
experiments is not sample quality but the *dynamics* of the game: the value of
the discriminator at equilibrium, the difference between the saturating and the
non-saturating generator loss, mode collapse, and what the Wasserstein critic
repairs.
"""
import autograd.numpy as np
from autograd import grad


# ---------------------------------------------------------------------------
# 1.  a minimal multilayer perceptron
# ---------------------------------------------------------------------------
def init_mlp(sizes, rng, scale=None):
    """Gaussian initialisation; `scale=None` uses He scaling."""
    P = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        s = np.sqrt(2.0 / n_in) if scale is None else scale
        P.append((rng.normal(0.0, s, (n_in, n_out)), np.zeros(n_out)))
    return P


def mlp(P, x, slope=0.2):
    """LeakyReLU network; the last layer is linear (a logit or a critic value)."""
    for W, b in P[:-1]:
        x = np.dot(x, W) + b
        x = np.where(x > 0, x, slope * x)
    W, b = P[-1]
    return np.dot(x, W) + b


def generator(P, z):
    """G: R^k -> R^d, Eq. (17.generator)."""
    return mlp(P, z)


def discriminator(Q, x):
    """The discriminator *logit* u(x); D(x) = sigmoid(u(x))."""
    return mlp(Q, x)[:, 0]


def sigmoid(u):
    return 0.5 * (np.tanh(0.5 * u) + 1.0)


def softplus(u):
    """log(1+exp(u)), evaluated without overflow."""
    return np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u)))


# ---------------------------------------------------------------------------
# 2.  the three losses of Section 17.losses
# ---------------------------------------------------------------------------
# log D(x)      = -softplus(-u(x))
# log(1 - D(x)) = -softplus( u(x))
# so the value function V of Eq. (17.minimax) is written with softplus alone.

def value_function(Q, P, x_real, z):
    """V(G,D) = E[log D(x)] + E[log(1 - D(G(z)))],  Eq. (17.minimax)."""
    u_real = discriminator(Q, x_real)
    u_fake = discriminator(Q, generator(P, z))
    return -np.mean(softplus(-u_real)) - np.mean(softplus(u_fake))


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


# ---------------------------------------------------------------------------
# 3.  the Wasserstein critic of Section 17.wgan
# ---------------------------------------------------------------------------
def critic(Q, x):
    """f_w: R^d -> R.  No sigmoid: the critic is not a probability."""
    return mlp(Q, x)[:, 0]


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


def g_loss_wgan(P, Q, z):
    """The generator maximises the critic on its own samples."""
    return -np.mean(critic(Q, generator(P, z)))


# ---------------------------------------------------------------------------
# 4.  Adam, and the alternating training loop of Section 17.algorithm
# ---------------------------------------------------------------------------
def adam_state(P):
    return [(np.zeros_like(W), np.zeros_like(b)) for W, b in P], \
           [(np.zeros_like(W), np.zeros_like(b)) for W, b in P], [0]


def adam_step(P, G, M, V, t, eta, b1=0.5, b2=0.999, eps=1e-8):
    """Adam with beta_1 = 0.5, the DCGAN convention (Section 17.stability)."""
    t[0] += 1
    out = []
    for i, ((W, b), (gW, gb)) in enumerate(zip(P, G)):
        mW = b1 * M[i][0] + (1 - b1) * gW
        mb = b1 * M[i][1] + (1 - b1) * gb
        vW = b2 * V[i][0] + (1 - b2) * gW ** 2
        vb = b2 * V[i][1] + (1 - b2) * gb ** 2
        M[i] = (mW, mb)
        V[i] = (vW, vb)
        c1 = 1 - b1 ** t[0]
        c2 = 1 - b2 ** t[0]
        out.append((W - eta * (mW / c1) / (np.sqrt(vW / c2) + eps),
                    b - eta * (mb / c1) / (np.sqrt(vb / c2) + eps)))
    return out


def train(X, k_z=2, hidden=(64, 64), n_iter=4000, batch=128, eta=2e-3,
          eta_d=None, mode="nonsat", smooth=1.0, n_critic=1, lam=10.0,
          rng=None, record_every=100, hidden_d=None):
    """Alternate: one (or `n_critic`) discriminator steps, then one generator step.

    `mode` selects the generator objective: "nonsat" for Eq. (17.nonsat),
    "sat" for the original Eq. (17.minimax), "wgan" for the critic of
    Eq. (17.wgangp).  `eta_d` and `hidden_d` let the discriminator be given a
    different learning rate and a different width from the generator, which is
    how the two-time-scale rule of Section 17.stability is implemented and how
    the strong-discriminator experiments are set up.
    """
    rng = rng or np.random.default_rng(0)
    eta_d = eta if eta_d is None else eta_d
    hidden_d = hidden if hidden_d is None else hidden
    d = X.shape[1]
    P = init_mlp((k_z,) + tuple(hidden) + (d,), rng)
    Q = init_mlp((d,) + tuple(hidden_d) + (1,), rng)
    MP, VP, tP = adam_state(P)
    MQ, VQ, tQ = adam_state(Q)

    if mode == "wgan":
        dg = grad(critic_loss, 0)
        gg = grad(g_loss_wgan, 0)
    else:
        dg = grad(d_loss, 0)
        gg = grad(g_loss_nonsat if mode == "nonsat" else g_loss_sat, 0)

    hist = []
    for it in range(1, n_iter + 1):
        for _ in range(n_critic):
            xb = X[rng.integers(0, len(X), batch)]
            zb = rng.normal(size=(batch, k_z))
            if mode == "wgan":
                gQ = dg(Q, P, xb, zb, rng, lam)
            else:
                gQ = dg(Q, P, xb, zb, smooth)
            Q = adam_step(Q, gQ, MQ, VQ, tQ, eta_d)

        zb = rng.normal(size=(batch, k_z))
        gP = gg(P, Q, zb)
        P = adam_step(P, gP, MP, VP, tP, eta)

        if it % record_every == 0 or it == 1:
            gnorm = np.sqrt(sum(np.sum(gW ** 2) + np.sum(gb ** 2)
                                for gW, gb in gP))
            xb = X[rng.integers(0, len(X), 512)]
            zb = rng.normal(size=(512, k_z))
            if mode == "wgan":
                hist.append((it, float(np.mean(critic(Q, xb))
                                       - np.mean(critic(Q, generator(P, zb)))),
                             np.nan, np.nan, float(gnorm)))
            else:
                hist.append((it,
                             float(value_function(Q, P, xb, zb)),
                             float(np.mean(sigmoid(discriminator(Q, xb)))),
                             float(np.mean(sigmoid(discriminator(
                                 Q, generator(P, zb))))),
                             float(gnorm)))
    return P, Q, hist


def sample(P, n, k_z=2, rng=None):
    rng = rng or np.random.default_rng(0)
    return generator(P, rng.normal(size=(n, k_z)))


# ---------------------------------------------------------------------------
# 5.  the target distributions and the two-sample statistic
# ---------------------------------------------------------------------------
def eight_gaussians(n, rng, radius=2.0, sigma=0.10):
    """The standard mode-collapse benchmark: eight modes on a circle."""
    ang = 2 * np.pi * np.arange(8) / 8
    C = radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    idx = rng.integers(0, 8, n)
    return C[idx] + sigma * rng.normal(size=(n, 2)), C


def grid_gaussians(n, rng, side=5, spacing=1.0, sigma=0.05):
    """A 5x5 grid of narrow modes: the harder mode-collapse benchmark."""
    g = spacing * (np.arange(side) - (side - 1) / 2)
    C = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    idx = rng.integers(0, len(C), n)
    return C[idx] + sigma * rng.normal(size=(n, 2)), C


def energy_distance(A, B, n=1200, seed=0):
    """A two-sample statistic; lower is closer.  Zero only if the laws agree."""
    r = np.random.default_rng(seed)
    a = A[r.integers(0, len(A), n)]
    b = B[r.integers(0, len(B), n)]
    d = lambda U, V: np.mean(np.sqrt(np.sum((U[:, None, :] - V[None, :, :]) ** 2,
                                            axis=-1)))
    return 2 * d(a, b) - d(a, a) - d(b, b)


def modes_covered(S, C, tol=0.5, frac=0.01):
    """A mode counts as covered if at least `frac` of the samples land near it."""
    dist = np.sqrt(np.sum((S[:, None, :] - C[None, :, :]) ** 2, axis=-1))
    near = dist.min(axis=1) < tol
    which = dist.argmin(axis=1)[near]
    counts = np.array([np.sum(which == j) for j in range(len(C))])
    return int(np.sum(counts >= frac * len(S))), counts, float(np.mean(near))

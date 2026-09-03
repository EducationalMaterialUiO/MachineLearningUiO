"""Chapter 16: a denoising diffusion model from scratch.

The forward process is arithmetic, not a network: everything in Section
16.forward can be checked in closed form, and we do check it.
"""
import autograd.numpy as np
from autograd import grad
from autograd.misc import flatten


# ---------------------------------------------------------------------------
# 1.  the noise schedule and the closed-form forward marginal
# ---------------------------------------------------------------------------
def linear_schedule(T=200, b_min=1e-4, b_max=0.02):
    beta = np.linspace(b_min, b_max, T)
    alpha = 1.0 - beta
    abar = np.cumprod(alpha)
    return beta, alpha, abar


def cosine_schedule(T=200, s=0.008):
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    abar = f[1:] / f[0]
    alpha = abar / np.concatenate([[1.0], abar[:-1]])
    return 1.0 - alpha, alpha, abar


def q_sample(x0, t, abar, eps):
    """x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps,  Eq. (16.marginal)."""
    a = abar[t][:, None]
    return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * eps


def posterior(xt, x0, t, beta, alpha, abar):
    """q(x_{t-1} | x_t, x_0), Eq. (16.posterior): mean and variance."""
    ab_prev = np.where(t > 0, abar[np.maximum(t - 1, 0)], 1.0)[:, None]
    ab = abar[t][:, None]; b = beta[t][:, None]; al = alpha[t][:, None]
    mu = (np.sqrt(al) * (1 - ab_prev) / (1 - ab) * xt
          + np.sqrt(ab_prev) * b / (1 - ab) * x0)
    var = (1 - ab_prev) / (1 - ab) * b
    return mu, var


# ---------------------------------------------------------------------------
# 2.  the noise-prediction network
# ---------------------------------------------------------------------------
def time_embedding(t, T, d=16):
    """Sinusoidal embedding of the step index, as in Eq. (13.posenc)."""
    half = d // 2
    freq = np.exp(-np.log(10000.0) * np.arange(half) / half)
    ang = (t[:, None] / T) * freq[None, :] * 1000.0
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def init_eps_net(d_x=2, d_t=16, hidden=128, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    def L(a, b): return [rng.normal(0, np.sqrt(1.0 / a), (a, b)), np.zeros(b)]
    return {"W1": L(d_x + d_t, hidden), "W2": L(hidden, hidden),
            "W3": L(hidden, d_x)}


def eps_net(P, xt, t, T, d_t=16):
    """eps_theta(x_t, t): predicts the noise that was added, Eq. (16.simple)."""
    h = np.concatenate([xt, time_embedding(t, T, d_t)], axis=1)
    for k in ("W1", "W2"):
        h = np.tanh(h @ P[k][0] + P[k][1])
    return h @ P["W3"][0] + P["W3"][1]


def loss(P, x0, t, eps, abar, T):
    """L_simple, Eq. (16.simple): predict the noise from the noisy sample."""
    xt = q_sample(x0, t, abar, eps)
    return np.mean(np.sum((eps - eps_net(P, xt, t, T)) ** 2, axis=1))


def train(P, X, abar, T, n_iter=3000, batch=128, eta=2e-3, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    flat, unflatten = flatten(P)
    g = grad(lambda f, x0, t, e: loss(unflatten(f), x0, t, e, abar, T))
    m = np.zeros_like(flat); v = np.zeros_like(flat); hist = []
    for it in range(1, n_iter + 1):
        idx = rng.integers(0, len(X), batch)
        t = rng.integers(0, T, batch)
        e = rng.normal(size=(batch, X.shape[1]))
        gg = g(flat, X[idx], t, e)
        m = 0.9 * m + 0.1 * gg
        v = 0.999 * v + 0.001 * gg ** 2
        flat = flat - eta * (m / (1 - 0.9**it)) / (np.sqrt(v / (1 - 0.999**it)) + 1e-8)
        if it % 100 == 0:
            hist.append((it, float(loss(unflatten(flat), X[idx], t, e, abar, T))))
    return unflatten(flat), hist


# ---------------------------------------------------------------------------
# 3.  sampling
# ---------------------------------------------------------------------------
def ddpm_sample(P, n, d, beta, alpha, abar, T, rng):
    """Ancestral sampling, Eq. (16.ddpmsample): T network calls."""
    x = rng.normal(size=(n, d))
    for i in reversed(range(T)):
        t = np.full(n, i)
        e = eps_net(P, x, t, T)
        x0hat = (x - np.sqrt(1 - abar[i]) * e) / np.sqrt(abar[i])
        mu, var = posterior(x, x0hat, t, beta, alpha, abar)
        x = mu + (np.sqrt(var) * rng.normal(size=x.shape) if i > 0 else 0.0)
    return x


def ddim_sample(P, n, d, abar, T, rng, steps=20, eta=0.0):
    """DDIM, Eq. (16.ddim): a subsequence of steps, deterministic at eta = 0."""
    x = rng.normal(size=(n, d))
    ts = np.linspace(T - 1, 0, steps).astype(int)
    for k, i in enumerate(ts):
        t = np.full(n, i)
        e = eps_net(P, x, t, T)
        x0hat = (x - np.sqrt(1 - abar[i]) * e) / np.sqrt(abar[i])
        j = ts[k + 1] if k + 1 < len(ts) else -1
        ab_prev = abar[j] if j >= 0 else 1.0
        sig = eta * np.sqrt((1 - ab_prev) / (1 - abar[i]) * (1 - abar[i] / ab_prev)) \
            if (eta > 0 and j >= 0) else 0.0
        x = (np.sqrt(ab_prev) * x0hat
             + np.sqrt(max(1 - ab_prev - sig**2, 0.0)) * e
             + (sig * rng.normal(size=x.shape) if sig > 0 else 0.0))
    return x

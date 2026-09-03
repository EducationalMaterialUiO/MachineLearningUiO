"""Chapter 16: normalizing flows with an exact log-likelihood."""
import autograd.numpy as np
from autograd import grad
from autograd.misc import flatten


def init_coupling(d, hidden=64, n_layers=6, rng=None):
    """RealNVP: alternating masks, Eq. (16.coupling)."""
    rng = np.random.default_rng(0) if rng is None else rng
    L = []
    for k in range(n_layers):
        mask = np.array([(i + k) % 2 for i in range(d)], dtype=float)
        def lin(a, b, s=0.1):
            return [rng.normal(0, s, (a, b)), np.zeros(b)]
        L.append({"mask": mask,
                  "W1": lin(d, hidden), "W2": lin(hidden, hidden),
                  "Ws": lin(hidden, d), "Wt": lin(hidden, d)})
    return L


def _st(layer, z):
    h = z * layer["mask"]
    for k in ("W1", "W2"):
        h = np.tanh(h @ layer[k][0] + layer[k][1])
    s = np.tanh(h @ layer["Ws"][0] + layer["Ws"][1])   # bounded for stability
    t = h @ layer["Wt"][0] + layer["Wt"][1]
    return s * (1 - layer["mask"]), t * (1 - layer["mask"])


def forward(L, z):
    """z -> x, accumulating the log-determinant, Eq. (16.logdet)."""
    x = z
    logdet = np.zeros(len(z))
    for layer in L:
        s, t = _st(layer, x)
        x = x * np.exp(s) + t
        logdet = logdet + np.sum(s, axis=1)
    return x, logdet


def inverse(L, x):
    """x -> z, exactly invertible one layer at a time."""
    z = x
    logdet = np.zeros(len(x))
    for layer in reversed(L):
        s, t = _st(layer, z)
        z = (z - t) * np.exp(-s)
        logdet = logdet - np.sum(s, axis=1)
    return z, logdet


def log_prob(L, x):
    """log p(x) = log p_0(f^{-1}(x)) + log|det df^{-1}/dx|,  Eq. (16.changevar)."""
    z, logdet_inv = inverse(L, x)
    d = x.shape[1]
    lp0 = -0.5 * np.sum(z ** 2, axis=1) - 0.5 * d * np.log(2 * np.pi)
    return lp0 + logdet_inv


def train(L, X, n_iter=3000, batch=128, eta=3e-3, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    flat, unflatten = flatten(L)
    g = grad(lambda f, xb: -np.mean(log_prob(unflatten(f), xb)))
    m = np.zeros_like(flat); v = np.zeros_like(flat); hist = []
    for it in range(1, n_iter + 1):
        xb = X[rng.integers(0, len(X), batch)]
        gg = g(flat, xb)
        m = 0.9 * m + 0.1 * gg
        v = 0.999 * v + 0.001 * gg ** 2
        flat = flat - eta * (m / (1 - 0.9**it)) / (np.sqrt(v / (1 - 0.999**it)) + 1e-8)
        if it % 100 == 0:
            hist.append((it, float(np.mean(log_prob(unflatten(flat), X)))))
    return unflatten(flat), hist

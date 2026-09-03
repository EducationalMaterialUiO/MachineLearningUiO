"""Chapter 14: a binary-binary restricted Boltzmann machine, from scratch.

Same idiom as Chapters 8 to 13: parameters in a dict, the model is a function,
and everything that can be checked exactly is checked exactly.  Because the
partition function of Eq. (14.Z) is a sum over 2^(M+N) configurations, small
models let us compute it by brute force and compare the sampled gradient
against the true one -- which is the point of Section 14.cdbias.
"""
import itertools
import numpy as np


def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def init_rbm(M, N, rng=None, scale=0.01):
    rng = np.random.default_rng(0) if rng is None else rng
    return {"W": rng.normal(0, scale, (M, N)),
            "a": np.zeros(M), "b": np.zeros(N)}


# ---------------------------------------------------------------------------
# 1.  energy, free energy, and the exact partition function
# ---------------------------------------------------------------------------
def energy(P, x, h):
    """E(x,h) = -a.x - b.h - x^T W h,  Eq. (14.energyBB)."""
    return -(x @ P["a"]) - (h @ P["b"]) - np.einsum("...i,ij,...j->...", x, P["W"], h)


def free_energy(P, X):
    """F(x) = -a.x - sum_j log(1 + exp(b_j + (x^T W)_j)),  Eq. (14.freeenergy).

    The hidden units have been summed out exactly, which is what the
    restriction to a bipartite graph buys.
    """
    z = X @ P["W"] + P["b"]
    return -(X @ P["a"]) - np.sum(np.logaddexp(0.0, z), axis=-1)


def all_states(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)


def log_Z(P):
    """Exact log partition function by enumeration; only for small M."""
    M = len(P["a"])
    F = free_energy(P, all_states(M))
    return float(np.logaddexp.reduce(-F))


def log_likelihood(P, X):
    """(1/n) sum_v log p(v) = -mean F(v) - log Z,  Eq. (14.loglik)."""
    return float(-np.mean(free_energy(P, X)) - log_Z(P))


# ---------------------------------------------------------------------------
# 2.  the conditionals, Eqs. (14.condh) and (14.condx)
# ---------------------------------------------------------------------------
def p_h_given_x(P, X):
    return sigmoid(X @ P["W"] + P["b"])


def p_x_given_h(P, H):
    return sigmoid(H @ P["W"].T + P["a"])


def gibbs_step(P, X, rng):
    """One sweep of block Gibbs: sample all h, then all x, Eq. (14.blockgibbs)."""
    ph = p_h_given_x(P, X)
    H = (rng.random(ph.shape) < ph).astype(float)
    px = p_x_given_h(P, H)
    Xn = (rng.random(px.shape) < px).astype(float)
    return Xn, H, ph, px


# ---------------------------------------------------------------------------
# 3.  gradients
# ---------------------------------------------------------------------------
def positive_phase(P, X):
    """<.>_data:  the term that does not need sampling."""
    ph = p_h_given_x(P, X)
    return {"W": X.T @ ph / len(X), "a": X.mean(0), "b": ph.mean(0)}


def exact_gradient(P, X):
    """The true gradient of the log-likelihood, Eq. (14.gradient).

    The negative phase is computed by enumerating all 2^M visible states and
    weighting by the exact model distribution.  Only feasible for small M, and
    that is exactly why it is useful: it is the ground truth against which
    contrastive divergence can be judged.
    """
    pos = positive_phase(P, X)
    V = all_states(len(P["a"]))
    logp = -free_energy(P, V)
    logp = logp - np.logaddexp.reduce(logp)
    w = np.exp(logp)                                  # exact p(v)
    ph = p_h_given_x(P, V)
    neg = {"W": (V * w[:, None]).T @ ph,
           "a": w @ V,
           "b": w @ ph}
    return {k: pos[k] - neg[k] for k in pos}


def cd_gradient(P, X, k=1, rng=None, persistent=None):
    """Contrastive divergence CD-k, Eq. (14.cdk).

    The negative phase is estimated from k Gibbs sweeps started at the data
    (CD-k) or from a persistent chain (PCD).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    pos = positive_phase(P, X)
    V = X.copy() if persistent is None else persistent
    for _ in range(k):
        V, _, _, _ = gibbs_step(P, V, rng)
    ph = p_h_given_x(P, V)
    neg = {"W": V.T @ ph / len(V), "a": V.mean(0), "b": ph.mean(0)}
    g = {kk: pos[kk] - neg[kk] for kk in pos}
    return (g, V) if persistent is not None else g

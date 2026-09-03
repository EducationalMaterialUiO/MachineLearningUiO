"""Associative recall: the task that separates attention from recurrence.

A sequence presents L key-value pairs and then repeats one key; the model must
emit the value that followed that key.  The information needed sits at one
specific earlier position, and the distance to it grows with L.
"""
import autograd.numpy as np
from autograd import grad
from autograd.misc import flatten
import attention as at

NSYM = 8            # symbols used as keys and as values


def make_batch(n, L, rng):
    """Returns one-hot sequences (n, T, NSYM) and integer targets (n,)."""
    T = 2 * L + 1
    X = np.zeros((n, T, NSYM))
    y = np.zeros(n, dtype=int)
    for b in range(n):
        keys = rng.permutation(NSYM)[:L]
        vals = rng.integers(0, NSYM, L)
        for i in range(L):
            X[b, 2 * i, keys[i]] = 1.0
            X[b, 2 * i + 1, vals[i]] = 1.0
        q = rng.integers(0, L)
        X[b, 2 * L, keys[q]] = 1.0          # the query, repeated key
        y[b] = vals[q]
    return X, y


# --------------------------------------------------------------------------
def init_transformer(d, H, d_ff, rng):
    P = at.init_block(d, H, d_ff, rng)
    P["Win"] = rng.normal(0, np.sqrt(1.0 / NSYM), (NSYM, d))
    P["Wout"] = rng.normal(0, np.sqrt(1.0 / d), (d, NSYM))
    return P


def transformer_logits(P, X, PE):
    Z = X @ P["Win"] + PE                   # (B, T, d), broadcast the encoding
    Z, _ = at.block_batched(P, Z)
    return Z[:, -1, :] @ P["Wout"]          # read out the last position


def init_rnn(d, rng):
    """The vanilla RNN of Chapter 11, Eq. (11.rnn)."""
    s = np.sqrt(1.0 / d)
    return {"U": rng.normal(0, np.sqrt(1.0 / NSYM), (NSYM, d)),
            "W": rng.normal(0, s, (d, d)),
            "b": np.zeros(d),
            "Wout": rng.normal(0, s, (d, NSYM))}


def rnn_logits(P, X, PE=None):
    n, T, _ = X.shape
    h = np.zeros((n, P["W"].shape[0]))
    for t in range(T):
        h = np.tanh(X[:, t] @ P["U"] + h @ P["W"] + P["b"])
    return h @ P["Wout"]


def cross_entropy(logits, y):
    z = logits - np.max(logits, axis=1, keepdims=True)
    return float(np.mean(np.log(np.sum(np.exp(z), axis=1)) - z[np.arange(len(y)), y])) \
        if not hasattr(z, "_value") else \
        np.mean(np.log(np.sum(np.exp(z), axis=1)) - z[np.arange(len(y)), y])


def ce(logits, y):
    z = logits - np.max(logits, axis=1, keepdims=True)
    return np.mean(np.log(np.sum(np.exp(z), axis=1)) - z[np.arange(len(y)), y])


def n_params(P):
    return sum(np.asarray(v).size for v in P.values())


def train(P, logit_fn, L, d, n_iter=600, batch=64, eta=3e-3, seed=0, PE=None):
    rng = np.random.default_rng(seed)
    flat, unflatten = flatten(P)
    def loss(f, X, y):
        return ce(logit_fn(unflatten(f), X, PE), y)
    g = grad(loss)
    m = np.zeros_like(flat); v = np.zeros_like(flat)
    for it in range(1, n_iter + 1):
        X, y = make_batch(batch, L, rng)
        gg = g(flat, X, y)
        m = 0.9 * m + 0.1 * gg
        v = 0.999 * v + 0.001 * gg ** 2
        flat = flat - eta * (m / (1 - 0.9**it)) / (np.sqrt(v / (1 - 0.999**it)) + 1e-8)
    P = unflatten(flat)
    Xt, yt = make_batch(400, L, np.random.default_rng(seed + 999))
    acc = float(np.mean(np.argmax(logit_fn(P, Xt, PE), axis=1) == yt))
    return P, acc


# --------------------------------------------------------------------------
# Two stacked blocks.  Retrieving "the token that FOLLOWED the matching key"
# cannot be done by one attention layer: see Section 13.induction.
# --------------------------------------------------------------------------
def init_transformer2(d, H, d_ff, rng):
    P = {}
    for tag in ("a", "b"):
        for k, v in at.init_block(d, H, d_ff, rng).items():
            P[tag + k] = v
    P["Win"] = rng.normal(0, np.sqrt(1.0 / NSYM), (NSYM, d))
    P["Wout"] = rng.normal(0, np.sqrt(1.0 / d), (d, NSYM))
    return P


def _sub(P, tag):
    return {k[1:]: v for k, v in P.items() if k.startswith(tag) and len(k) > 1
            and k not in ("Win", "Wout")}


def transformer2_logits(P, X, PE):
    Z = X @ P["Win"] + PE
    Z, _ = at.block_batched(_sub(P, "a"), Z)
    Z, _ = at.block_batched(_sub(P, "b"), Z)
    return Z[:, -1, :] @ P["Wout"]

"""Chapter 11: a recurrent network and backpropagation through time, from scratch.

Same idiom as Chapters 8 and 10: parameters are plain arrays in a dict, the
forward pass is a function, and every gradient is checked against finite
differences before anything is trained.
"""
import numpy as np

# ---------------------------------------------------------------------------
# 1.  Forward pass, Eq. (11.rnn)
# ---------------------------------------------------------------------------

def init_rnn(n_in, n_h, n_out, rng=None, scale=None):
    """Xavier initialisation, Eq. (8.xavier); scale overrides the recurrent one."""
    rng = np.random.default_rng(0) if rng is None else rng
    s = np.sqrt(1.0 / n_h) if scale is None else scale
    return {"U": rng.normal(0, np.sqrt(1.0 / n_in), (n_h, n_in)),
            "W": rng.normal(0, s, (n_h, n_h)),
            "V": rng.normal(0, np.sqrt(1.0 / n_h), (n_out, n_h)),
            "b": np.zeros(n_h), "c": np.zeros(n_out)}


def forward(p, X, h0=None):
    """X has shape (T, n_in).  Returns outputs (T, n_out) and the cache.

    a_t = U x_t + W h_{t-1} + b,   h_t = tanh(a_t),   yhat_t = V h_t + c
    """
    T = X.shape[0]
    n_h = p["W"].shape[0]
    H = np.zeros((T + 1, n_h))              # H[0] is h_{-1}
    if h0 is not None:
        H[0] = h0
    A = np.zeros((T, n_h))
    Y = np.zeros((T, p["V"].shape[0]))
    for t in range(T):
        A[t] = p["U"] @ X[t] + p["W"] @ H[t] + p["b"]
        H[t + 1] = np.tanh(A[t])
        Y[t] = p["V"] @ H[t + 1] + p["c"]
    return Y, (X, A, H)


def mse(Y, target):
    return 0.5 * np.mean(np.sum((Y - target) ** 2, axis=1))


# ---------------------------------------------------------------------------
# 2.  Backpropagation through time, Eqs. (11.bpttlast)-(11.bpttparams)
# ---------------------------------------------------------------------------

def bptt(p, cache, Y, target, return_norms=False):
    """Gradients of Eq. (11.cost) by the recursion (11.bptth)."""
    X, A, H = cache
    T = X.shape[0]
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dY = (Y - target) / T                    # dL/dyhat_t for the cost (11.cost)
    dh_next = np.zeros(p["W"].shape[0])
    norms = []
    for t in reversed(range(T)):
        g["V"] += np.outer(dY[t], H[t + 1])
        g["c"] += dY[t]
        dh = p["V"].T @ dY[t] + dh_next      # Eq. (11.bptth)
        da = (1.0 - H[t + 1] ** 2) * dh      # through tanh
        g["U"] += np.outer(da, X[t])
        g["W"] += np.outer(da, H[t])
        g["b"] += da
        dh_next = p["W"].T @ da
        if return_norms:
            norms.append(np.linalg.norm(dh))
    return (g, norms[::-1]) if return_norms else g


# ---------------------------------------------------------------------------
# 3.  Gradient clipping, Eq. (11.clip)
# ---------------------------------------------------------------------------

def clip(g, theta):
    """Rescale the whole gradient if its norm exceeds theta."""
    n = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))
    if n > theta:
        for k in g:
            g[k] *= theta / n
    return g, n


# ---------------------------------------------------------------------------
# 4.  Adam, as in Chapter 4
# ---------------------------------------------------------------------------

def train(p, seqs, targets, n_epoch=200, eta=5e-3, theta=None, rng=None,
          verbose=False, every=50):
    rng = np.random.default_rng(0) if rng is None else rng
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}
    b1, b2, eps, it = 0.9, 0.999, 1e-8, 0
    hist = []
    for ep in range(1, n_epoch + 1):
        tot = 0.0
        for i in rng.permutation(len(seqs)):
            Y, cache = forward(p, seqs[i])
            tot += mse(Y, targets[i])
            g = bptt(p, cache, Y, targets[i])
            if theta is not None:
                g, _ = clip(g, theta)
            it += 1
            for k in p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                p[k] -= eta * (m[k] / (1 - b1**it)) / (np.sqrt(v[k] / (1 - b2**it)) + eps)
        hist.append((ep, tot / len(seqs)))
        if verbose and (ep % every == 0 or ep == 1):
            print(f"  epoch {ep:4d}  loss {tot/len(seqs):.6f}")
    return p, hist


# ---------------------------------------------------------------------------
# 5.  An LSTM cell, Eqs. (11.lstmf)-(11.lstmh)
# ---------------------------------------------------------------------------

def sigmoid(z):
    return np.where(z >= 0, 1.0/(1.0+np.exp(-np.abs(z))),
                    np.exp(-np.abs(z))/(1.0+np.exp(-np.abs(z))))


def init_lstm(n_in, n_h, n_out, rng=None, forget_bias=1.0):
    """The forget-gate bias is initialised positive, Section 11.forgetbias."""
    rng = np.random.default_rng(0) if rng is None else rng
    s_in, s_h = np.sqrt(1.0/n_in), np.sqrt(1.0/n_h)
    P = {}
    for g in "fig o".replace(" ", ""):
        P[f"W{g}x"] = rng.normal(0, s_in, (n_h, n_in))
        P[f"W{g}h"] = rng.normal(0, s_h, (n_h, n_h))
        P[f"b{g}"] = np.zeros(n_h)
    P["bf"] += forget_bias
    P["V"] = rng.normal(0, s_h, (n_out, n_h))
    P["c_out"] = np.zeros(n_out)
    return P


def lstm_forward(P, X):
    """Returns outputs, the gate trace and the cell states."""
    T, n_h = X.shape[0], P["Wfh"].shape[0]
    h = np.zeros(n_h); c = np.zeros(n_h)
    F, C = [], []
    Y = np.zeros((T, P["V"].shape[0]))
    for t in range(T):
        f = sigmoid(P["Wfx"] @ X[t] + P["Wfh"] @ h + P["bf"])   # forget
        i = sigmoid(P["Wix"] @ X[t] + P["Wih"] @ h + P["bi"])   # input
        g = np.tanh(P["Wgx"] @ X[t] + P["Wgh"] @ h + P["bg"])   # candidate
        o = sigmoid(P["Wox"] @ X[t] + P["Woh"] @ h + P["bo"])   # output
        c = f * c + i * g                                       # Eq. (11.lstmc)
        h = o * np.tanh(c)                                      # Eq. (11.lstmh)
        Y[t] = P["V"] @ h + P["c_out"]
        F.append(f); C.append(c.copy())
    return Y, np.array(F), np.array(C)

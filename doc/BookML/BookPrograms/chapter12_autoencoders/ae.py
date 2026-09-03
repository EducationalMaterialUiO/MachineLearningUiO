"""Chapter 12: autoencoders from scratch, in the idiom of Chapters 8 to 11."""
import numpy as np

# ---------------------------------------------------------------------------
# activations, as in Chapter 8
# ---------------------------------------------------------------------------
def identity(z):  return z
def d_identity(z): return np.ones_like(z)
def tanh_(z):     return np.tanh(z)
def d_tanh(z):    return 1.0 - np.tanh(z) ** 2
def relu(z):      return np.maximum(0.0, z)
def d_relu(z):    return (z > 0).astype(z.dtype)
def sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-np.abs(z))), np.exp(-np.abs(z))/(1+np.exp(-np.abs(z))))
def d_sigmoid(z):
    s = sigmoid(z); return s * (1 - s)

ACT = {"identity": (identity, d_identity), "tanh": (tanh_, d_tanh),
       "relu": (relu, d_relu), "sigmoid": (sigmoid, d_sigmoid)}


# ---------------------------------------------------------------------------
# an autoencoder is a network whose target is its own input
# ---------------------------------------------------------------------------
def init_ae(sizes, acts, rng=None):
    """sizes = [d, ..., p, ..., d]; acts has one entry per weight matrix."""
    rng = np.random.default_rng(0) if rng is None else rng
    P = []
    for i in range(len(sizes) - 1):
        nin, nout = sizes[i], sizes[i + 1]
        s = np.sqrt(2.0 / nin) if acts[i] == "relu" else np.sqrt(1.0 / nin)
        P.append([rng.normal(0, s, (nin, nout)), np.zeros(nout)])
    return P


def ae_forward(P, X, acts):
    """Returns the reconstruction and the cache; X has shape (N, d)."""
    A = [X]; Z = []
    a = X
    for l, (W, b) in enumerate(P):
        z = a @ W + b
        Z.append(z)
        a = ACT[acts[l]][0](z)
        A.append(a)
    return a, (A, Z)


def ae_backward(P, cache, Xhat, X, acts):
    """Backpropagation with the target equal to the input, Eq. (12.cost)."""
    A, Z = cache
    n = X.shape[0]
    g = [[np.zeros_like(W), np.zeros_like(b)] for W, b in P]
    delta = (Xhat - X) / n * ACT[acts[-1]][1](Z[-1])
    for l in reversed(range(len(P))):
        g[l][0] = A[l].T @ delta
        g[l][1] = delta.sum(axis=0)
        if l > 0:
            delta = (delta @ P[l][0].T) * ACT[acts[l - 1]][1](Z[l - 1])
    return g


def encode(P, X, acts, layer):
    """Run the first `layer` weight matrices: the code z = f(x)."""
    a = X
    for l in range(layer):
        a = ACT[acts[l]][0](a @ P[l][0] + P[l][1])
    return a


def mse(Xhat, X):
    """Reconstruction error ||X - Xhat||_F^2 / N, the quantity in Thm 12.2."""
    return float(np.mean(np.sum((Xhat - X) ** 2, axis=1)))


def cost(Xhat, X):
    """What backpropagation minimises: half the reconstruction error, so that
    delta^L = (xhat - x)/n exactly as in Chapter 8."""
    return 0.5 * mse(Xhat, X)


def train_ae(P, X, acts, n_epoch=400, batch=32, eta=1e-2, rng=None,
             verbose=False, every=100, Xval=None):
    rng = np.random.default_rng(0) if rng is None else rng
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in P]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in P]
    b1, b2, eps, it = 0.9, 0.999, 1e-8, 0
    hist = []
    for ep in range(1, n_epoch + 1):
        order = rng.permutation(len(X))
        for s in range(0, len(order), batch):
            xb = X[order[s:s + batch]]
            Xhat, cache = ae_forward(P, xb, acts)
            g = ae_backward(P, cache, Xhat, xb, acts)
            it += 1
            for l in range(len(P)):
                for j in range(2):
                    m[l][j] = b1 * m[l][j] + (1 - b1) * g[l][j]
                    v[l][j] = b2 * v[l][j] + (1 - b2) * g[l][j] ** 2
                    P[l][j] -= eta * (m[l][j] / (1 - b1**it)) / \
                        (np.sqrt(v[l][j] / (1 - b2**it)) + eps)
        tr = mse(ae_forward(P, X, acts)[0], X)
        va = mse(ae_forward(P, Xval, acts)[0], Xval) if Xval is not None else np.nan
        hist.append((ep, tr, va))
        if verbose and (ep % every == 0 or ep == 1):
            print(f"  epoch {ep:4d}  train {tr:.6f}  val {va:.6f}")
    return P, hist


# ---------------------------------------------------------------------------
# PCA, for comparison
# ---------------------------------------------------------------------------
def pca(X, p):
    """Return the rank-p projector, the loadings U_p and the eigenvalues."""
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / len(Xc)
    lam, U = np.linalg.eigh(S)
    idx = np.argsort(lam)[::-1]
    lam, U = lam[idx], U[:, idx]
    Up = U[:, :p]
    return Up @ Up.T, Up, lam

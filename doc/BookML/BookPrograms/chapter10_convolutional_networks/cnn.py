"""Chapter 10: a convolutional network written from scratch with NumPy.

Same idiom as the network of Chapter 8: parameters are plain arrays held in a
list, the forward pass is a function, and the backward pass returns gradients
that are checked against finite differences.  Nothing is imported from a deep
learning library.
"""
import numpy as np

# ---------------------------------------------------------------------------
# 1.  The convolution itself
# ---------------------------------------------------------------------------

def pad2d(X, P):
    """Zero-pad the two spatial axes of X with shape (N, C, H, W)."""
    if P == 0:
        return X
    return np.pad(X, ((0, 0), (0, 0), (P, P), (P, P)), mode="constant")


def im2col(X, F, S, P):
    """Rearrange every F x F patch of X into a column, Eq. (10.im2col).

    X has shape (N, C, H, W); the result has shape (N, C*F*F, H2*W2) with
    H2 = (H - F + 2P)/S + 1, so that a convolution becomes one matrix product.
    """
    N, C, H, W = X.shape
    H2 = (H - F + 2 * P) // S + 1
    W2 = (W - F + 2 * P) // S + 1
    Xp = pad2d(X, P)
    cols = np.empty((N, C * F * F, H2 * W2))
    for i in range(F):
        for j in range(F):
            patch = Xp[:, :, i:i + S * H2:S, j:j + S * W2:S]      # (N,C,H2,W2)
            cols[:, (i * F + j)::F * F, :] = patch.reshape(N, C, -1)
    return cols


def col2im(cols, X_shape, F, S, P):
    """Adjoint of im2col: scatter columns back, accumulating overlaps."""
    N, C, H, W = X_shape
    H2 = (H - F + 2 * P) // S + 1
    W2 = (W - F + 2 * P) // S + 1
    Xp = np.zeros((N, C, H + 2 * P, W + 2 * P))
    for i in range(F):
        for j in range(F):
            patch = cols[:, (i * F + j)::F * F, :].reshape(N, C, H2, W2)
            np.add.at(Xp, (slice(None), slice(None),
                           slice(i, i + S * H2, S), slice(j, j + S * W2, S)), patch)
    return Xp if P == 0 else Xp[:, :, P:-P, P:-P]


def conv_forward(X, W, b, S=1, P=0):
    """Cross-correlation, Eq. (10.crosscorr2d).  W has shape (K, C, F, F)."""
    N, C, H, Wd = X.shape
    K, _, F, _ = W.shape
    H2 = (H - F + 2 * P) // S + 1
    W2 = (Wd - F + 2 * P) // S + 1
    cols = im2col(X, F, S, P)                       # (N, C*F*F, H2*W2)
    out = np.einsum("kd,ndp->nkp", W.reshape(K, -1), cols) + b[None, :, None]
    return out.reshape(N, K, H2, W2), cols


def conv_backward(dY, X, W, cols, S=1, P=0):
    """Gradients of the convolution, Eqs. (10.dconvW), (10.dconvb), (10.dconvX)."""
    N, K, H2, W2 = dY.shape
    _, C, F, _ = W.shape
    dYf = dY.reshape(N, K, -1)                                   # (N,K,H2W2)
    dW = np.einsum("nkp,ndp->kd", dYf, cols).reshape(W.shape)
    db = dYf.sum(axis=(0, 2))
    dcols = np.einsum("kd,nkp->ndp", W.reshape(K, -1), dYf)
    dX = col2im(dcols, X.shape, F, S, P)
    return dX, dW, db


# ---------------------------------------------------------------------------
# 2.  Pooling
# ---------------------------------------------------------------------------

def maxpool_forward(X, F=2, S=2):
    """Max pooling, Eq. (10.maxpool).  Returns the output and an argmax mask."""
    N, C, H, W = X.shape
    H2, W2 = (H - F) // S + 1, (W - F) // S + 1
    patches = np.empty((N, C, H2, W2, F * F))
    for i in range(F):
        for j in range(F):
            patches[..., i * F + j] = X[:, :, i:i + S * H2:S, j:j + S * W2:S]
    idx = patches.argmax(axis=-1)
    out = np.take_along_axis(patches, idx[..., None], axis=-1)[..., 0]
    return out, idx


def maxpool_backward(dY, X, idx, F=2, S=2):
    """Route each gradient to the argmax that produced it, Eq. (10.dmaxpool)."""
    N, C, H, W = X.shape
    H2, W2 = dY.shape[2], dY.shape[3]
    dX = np.zeros_like(X)
    for i in range(F):
        for j in range(F):
            mask = (idx == i * F + j)
            np.add.at(dX, (slice(None), slice(None),
                           slice(i, i + S * H2, S), slice(j, j + S * W2, S)),
                      dY * mask)
    return dX


# ---------------------------------------------------------------------------
# 3.  The rest of the network, as in Chapter 8
# ---------------------------------------------------------------------------

def relu(Z):        return np.maximum(0.0, Z)
def relu_prime(Z):  return (Z > 0).astype(Z.dtype)


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(probs, y_onehot):
    n = probs.shape[0]
    return -np.sum(y_onehot * np.log(probs + 1e-12)) / n


# ---------------------------------------------------------------------------
# 4.  A complete small CNN:  conv - relu - pool - conv - relu - pool - dense
# ---------------------------------------------------------------------------

def init_cnn(C_in=1, K1=8, K2=16, F=3, n_out=10, flat=None, rng=None):
    """He initialisation, Eq. (8.he), with fan-in C*F*F for a conv kernel."""
    rng = np.random.default_rng(0) if rng is None else rng
    def he(shape, fan_in):
        return rng.normal(0.0, np.sqrt(2.0 / fan_in), shape)
    return {
        "W1": he((K1, C_in, F, F), C_in * F * F), "b1": np.zeros(K1),
        "W2": he((K2, K1, F, F), K1 * F * F),     "b2": np.zeros(K2),
        "W3": he((flat, n_out), flat),            "b3": np.zeros(n_out),
    }


def forward(p, X):
    """Returns the class probabilities and everything the backward pass needs."""
    Z1, c1 = conv_forward(X, p["W1"], p["b1"], S=1, P=1)
    A1 = relu(Z1)
    P1, i1 = maxpool_forward(A1, 2, 2)
    Z2, c2 = conv_forward(P1, p["W2"], p["b2"], S=1, P=1)
    A2 = relu(Z2)
    P2, i2 = maxpool_forward(A2, 2, 2)
    flat = P2.reshape(P2.shape[0], -1)
    Z3 = flat @ p["W3"] + p["b3"]
    return softmax(Z3), (X, Z1, A1, i1, P1, c1, Z2, A2, i2, P2, flat, c2)


def backward(p, cache, probs, Y):
    """Backpropagation, Eqs. (10.dconvW)-(10.dmaxpool); delta^L = (a-y)/n."""
    X, Z1, A1, i1, P1, c1, Z2, A2, i2, P2, flat, c2 = cache
    n = X.shape[0]
    d3 = (probs - Y) / n                                    # softmax + CE
    g = {"W3": flat.T @ d3, "b3": d3.sum(axis=0)}
    dflat = d3 @ p["W3"].T
    dP2 = dflat.reshape(P2.shape)
    dA2 = maxpool_backward(dP2, A2, i2, 2, 2)
    dZ2 = dA2 * relu_prime(Z2)
    dP1, g["W2"], g["b2"] = conv_backward(dZ2, P1, p["W2"], c2, S=1, P=1)
    dA1 = maxpool_backward(dP1, A1, i1, 2, 2)
    dZ1 = dA1 * relu_prime(Z1)
    _, g["W1"], g["b1"] = conv_backward(dZ1, X, p["W1"], c1, S=1, P=1)
    return g


def train(p, Xtr, Ytr, Xte, yte, epochs=20, batch=32, eta=3e-3, rng=None,
          verbose=True):
    """Adam, Eq. (4.adam), on mini-batches."""
    rng = np.random.default_rng(0) if rng is None else rng
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    hist = []
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(Xtr))
        tot = 0.0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            probs, cache = forward(p, Xtr[idx])
            tot += cross_entropy(probs, Ytr[idx]) * len(idx)
            g = backward(p, cache, probs, Ytr[idx])
            t += 1
            for k in p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                p[k] -= eta * (m[k] / (1 - b1**t)) / (np.sqrt(v[k] / (1 - b2**t)) + eps)
        acc = accuracy(p, Xte, yte)
        hist.append((ep, tot / len(Xtr), acc))
        if verbose:
            print(f"  epoch {ep:3d}  train loss {tot/len(Xtr):.4f}   test acc {acc:.4f}")
    return p, hist


def accuracy(p, X, y):
    return float((forward(p, X)[0].argmax(axis=1) == y).mean())


def n_params(p):
    return sum(v.size for v in p.values())

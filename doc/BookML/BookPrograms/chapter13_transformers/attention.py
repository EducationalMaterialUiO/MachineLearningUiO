"""Chapter 13: self-attention and a transformer block, from scratch.

Written with autograd, as in Chapter 9: the forward pass is plain NumPy and the
gradients come from reverse-mode differentiation, because hand-differentiating a
softmax inside a matrix product is exactly the drudgery Section 4.autodiff
warned against.
"""
import autograd.numpy as np
from autograd import grad


def softmax_rows(S):
    """Row-wise softmax, shifted for stability."""
    S = S - np.max(S, axis=-1, keepdims=True)
    E = np.exp(S)
    return E / np.sum(E, axis=-1, keepdims=True)


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention, Eq. (13.attention).

    Q is (n, d_k), K is (m, d_k), V is (m, d_v); the output is (n, d_v).
    """
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)                 # (n, m) scores
    if mask is not None:
        S = S + mask                           # -inf where attention is banned
    A = softmax_rows(S)                        # (n, m), rows sum to one
    return A @ V, A


def self_attention(P, X, mask=None):
    """One head: Q = X W_Q, K = X W_K, V = X W_V, Eq. (13.qkv)."""
    return attention(X @ P["WQ"], X @ P["WK"], X @ P["WV"], mask)


# ---------------------------------------------------------------------------
# multi-head attention
# ---------------------------------------------------------------------------
def init_mha(d, H, d_k=None, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    d_k = d // H if d_k is None else d_k
    s = np.sqrt(1.0 / d)
    return {"WQ": rng.normal(0, s, (H, d, d_k)),
            "WK": rng.normal(0, s, (H, d, d_k)),
            "WV": rng.normal(0, s, (H, d, d_k)),
            "WO": rng.normal(0, np.sqrt(1.0 / (H * d_k)), (H * d_k, d))}


def multihead(P, X, mask=None):
    """MultiHead(X) = Concat(head_1, ..., head_H) W_O, Eq. (13.multihead)."""
    heads, As = [], []
    for h in range(P["WQ"].shape[0]):
        y, A = attention(X @ P["WQ"][h], X @ P["WK"][h], X @ P["WV"][h], mask)
        heads.append(y); As.append(A)
    return np.concatenate(heads, axis=-1) @ P["WO"], np.array(As)


# ---------------------------------------------------------------------------
# the transformer block
# ---------------------------------------------------------------------------
def layernorm(Z, gamma, beta, eps=1e-5):
    """Normalise each token across its features, Eq. (13.layernorm)."""
    mu = np.mean(Z, axis=-1, keepdims=True)
    var = np.mean((Z - mu) ** 2, axis=-1, keepdims=True)
    return gamma * (Z - mu) / np.sqrt(var + eps) + beta


def gelu(z):
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                                    (z + 0.044715 * z ** 3)))


def init_block(d, H, d_ff, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    P = init_mha(d, H, rng=rng)
    P.update({"W1": rng.normal(0, np.sqrt(1.0 / d), (d, d_ff)),
              "b1": np.zeros(d_ff),
              "W2": rng.normal(0, np.sqrt(1.0 / d_ff), (d_ff, d)),
              "b2": np.zeros(d),
              "g1": np.ones(d), "be1": np.zeros(d),
              "g2": np.ones(d), "be2": np.zeros(d)})
    return P


def block(P, X, mask=None):
    """Pre-norm transformer block, Eq. (13.block).

    X -> X + MHA(LN(X)) -> X + MLP(LN(X)).  The residual paths carry the
    identity, which is what keeps deep stacks trainable (cf. Section 11.lstmwhy).
    """
    Y, A = multihead(P, layernorm(X, P["g1"], P["be1"]), mask)
    X = X + Y
    Z = layernorm(X, P["g2"], P["be2"])
    X = X + gelu(Z @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]
    return X, A


def causal_mask(n):
    """0 on and below the diagonal, -inf above: position i sees only j <= i."""
    M = np.zeros((n, n))
    return M + np.triu(np.full((n, n), -np.inf), 1)


def positional_encoding(n, d):
    """Sinusoidal encoding, Eq. (13.posenc)."""
    pos = np.arange(n)[:, None]
    i = np.arange(d)[None, :]
    ang = pos / np.power(10000.0, 2 * (i // 2) / d)
    return np.where(i % 2 == 0, np.sin(ang), np.cos(ang))


# ---------------------------------------------------------------------------
# batched versions: X has shape (B, n, d).  Same equations, one extra axis.
# ---------------------------------------------------------------------------
def multihead_batched(P, X, mask=None):
    H = P["WQ"].shape[0]
    Q = np.einsum("bnd,hdk->bhnk", X, P["WQ"])
    K = np.einsum("bnd,hdk->bhnk", X, P["WK"])
    V = np.einsum("bnd,hdk->bhnk", X, P["WV"])
    S = np.einsum("bhnk,bhmk->bhnm", Q, K) / np.sqrt(Q.shape[-1])
    if mask is not None:
        S = S + mask
    A = softmax_rows(S)
    Yh = np.einsum("bhnm,bhmk->bhnk", A, V)
    B, H_, n, dk = Yh.shape
    Y = np.transpose(Yh, (0, 2, 1, 3)).reshape(B, n, H_ * dk)
    return Y @ P["WO"], A


def block_batched(P, X, mask=None):
    Y, A = multihead_batched(P, layernorm(X, P["g1"], P["be1"]), mask)
    X = X + Y
    Z = layernorm(X, P["g2"], P["be2"])
    X = X + gelu(Z @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]
    return X, A

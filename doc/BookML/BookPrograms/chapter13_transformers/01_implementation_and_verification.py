"""Chapter 13: listing 1, from the section on implementation and verification.

Extracted from doc/BookML/chapter13.tex.
"""

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

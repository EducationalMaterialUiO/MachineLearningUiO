"""Chapter 10: listing 3, from the section on the im2col transformation.

Extracted from doc/BookML/chapter10.tex.
"""

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

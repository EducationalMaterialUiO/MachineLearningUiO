"""Chapter 10: listing 1, from the section on the im2col transformation.

Extracted from doc/BookML/chapter10.tex.
"""

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

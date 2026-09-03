"""Chapter 10: listing 2, from the section on the im2col transformation.

Extracted from doc/BookML/chapter10.tex.
"""

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

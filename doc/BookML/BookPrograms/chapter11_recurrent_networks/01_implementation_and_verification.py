"""Chapter 11: listing 1, from the section on implementation and verification.

Extracted from doc/BookML/chapter11.tex.
"""

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

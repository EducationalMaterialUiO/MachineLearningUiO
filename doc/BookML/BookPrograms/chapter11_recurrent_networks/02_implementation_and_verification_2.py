"""Chapter 11: listing 2, from the section on implementation and verification.

Extracted from doc/BookML/chapter11.tex.
"""

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

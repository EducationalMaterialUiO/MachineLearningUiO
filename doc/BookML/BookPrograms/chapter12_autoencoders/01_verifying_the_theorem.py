"""Chapter 12: listing 1, from the section on verifying the theorem.

Extracted from doc/BookML/chapter12.tex.
"""

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

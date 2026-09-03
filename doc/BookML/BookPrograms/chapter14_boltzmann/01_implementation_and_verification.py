"""Chapter 14: listing 1, from the section on implementation and verification.

Extracted from doc/BookML/chapter14.tex.
"""

def free_energy(P, X):
    """F(x) = -a.x - sum_j log(1 + exp(b_j + (x^T W)_j)),  Eq. (14.freeenergy).

    The hidden units have been summed out exactly, which is what the
    restriction to a bipartite graph buys.
    """
    z = X @ P["W"] + P["b"]
    return -(X @ P["a"]) - np.sum(np.logaddexp(0.0, z), axis=-1)


def p_h_given_x(P, X):
    return sigmoid(X @ P["W"] + P["b"])


def p_x_given_h(P, H):
    return sigmoid(H @ P["W"].T + P["a"])


def gibbs_step(P, X, rng):
    """One sweep of block Gibbs: sample all h, then all x, Eq. (14.blockgibbs)."""
    ph = p_h_given_x(P, X)
    H = (rng.random(ph.shape) < ph).astype(float)
    px = p_x_given_h(P, H)
    Xn = (rng.random(px.shape) < px).astype(float)
    return Xn, H, ph, px

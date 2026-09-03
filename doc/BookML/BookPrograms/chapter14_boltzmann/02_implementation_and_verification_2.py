"""Chapter 14: listing 2, from the section on implementation and verification.

Extracted from doc/BookML/chapter14.tex.
"""

def exact_gradient(P, X):
    """The true gradient of the log-likelihood, Eq. (14.gradient).

    The negative phase is computed by enumerating all 2^M visible states and
    weighting by the exact model distribution.  Only feasible for small M, and
    that is exactly why it is useful: it is the ground truth against which
    contrastive divergence can be judged.
    """
    pos = positive_phase(P, X)
    V = all_states(len(P["a"]))
    logp = -free_energy(P, V)
    logp = logp - np.logaddexp.reduce(logp)
    w = np.exp(logp)                                  # exact p(v)
    ph = p_h_given_x(P, V)
    neg = {"W": (V * w[:, None]).T @ ph, "a": w @ V, "b": w @ ph}
    return {k: pos[k] - neg[k] for k in pos}

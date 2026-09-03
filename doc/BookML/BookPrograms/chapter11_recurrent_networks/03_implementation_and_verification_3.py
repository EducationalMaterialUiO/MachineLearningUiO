"""Chapter 11: listing 3, from the section on implementation and verification.

Extracted from doc/BookML/chapter11.tex.
"""

def clip(g, theta):
    """Rescale the whole gradient if its norm exceeds theta."""
    n = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))
    if n > theta:
        for k in g:
            g[k] *= theta / n
    return g, n

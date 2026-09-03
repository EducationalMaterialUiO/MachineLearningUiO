"""Chapter 9: listing 12, from the section on physics informed neural networks.

Extracted from doc/BookML/chapter9.tex.
"""

from autograd.misc import flatten

def adam_general(cost, params, n_iter=2000, eta=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    """Adam on any nested list of arrays -- here the weights and the scalar D."""
    flat, unflatten = flatten(params)
    gradient = grad(lambda f: cost(unflatten(f)))
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    for it in range(1, n_iter + 1):
        g = gradient(flat)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        flat = flat - eta * (m / (1 - b1**it)) / (np.sqrt(v / (1 - b2**it)) + eps)
    return unflatten(flat)

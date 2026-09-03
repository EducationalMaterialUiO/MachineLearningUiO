"""Chapter 9: listing 1, from the section on differentiating the network with respect.

Extracted from doc/BookML/chapter9.tex.
"""

import autograd.numpy as np
from autograd import grad, elementwise_grad

def network(P, X, activation="tanh"):
    """Feed-forward pass, Eq. (8.forwardbatch), with a linear output layer."""
    f = ACT[activation]
    a = X
    for l, (W, b) in enumerate(P):
        z = a @ W + b
        a = f(z) if l < len(P) - 1 else z
    return a[:, 0]


def network_derivs(P, X, activation="tanh", order=2, k=0):
    """N, dN/dx_k and d2N/dx_k^2 by the forward recursions (9.firstderiv)
    and (9.secondderiv).  The activations and their derivatives are those
    of Section 8.nncode."""
    f = ACT[activation]
    fp = elementwise_grad(f)             # f'
    fpp = elementwise_grad(fp)           # f''

    a = X
    da = np.zeros_like(X) + (np.arange(X.shape[1]) == k)   # da0/dxk = e_k
    d2a = np.zeros_like(X)

    for l, (W, b) in enumerate(P):
        z, dz, d2z = a @ W + b, da @ W, d2a @ W
        if l < len(P) - 1:
            a, da, d2a = f(z), fp(z) * dz, fpp(z) * dz**2 + fp(z) * d2z
        else:                                              # linear output
            a, da, d2a = z, dz, d2z
    if order == 1:
        return a[:, 0], da[:, 0]
    return a[:, 0], da[:, 0], d2a[:, 0]

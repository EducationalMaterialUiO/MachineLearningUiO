"""Chapter 10: listing 4, from the section on a complete network.

Extracted from doc/BookML/chapter10.tex.
"""

def forward(p, X):
    """Returns the class probabilities and everything the backward pass needs."""
    Z1, c1 = conv_forward(X, p["W1"], p["b1"], S=1, P=1)
    A1 = relu(Z1)
    P1, i1 = maxpool_forward(A1, 2, 2)
    Z2, c2 = conv_forward(P1, p["W2"], p["b2"], S=1, P=1)
    A2 = relu(Z2)
    P2, i2 = maxpool_forward(A2, 2, 2)
    flat = P2.reshape(P2.shape[0], -1)
    Z3 = flat @ p["W3"] + p["b3"]
    return softmax(Z3), (X, Z1, A1, i1, P1, c1, Z2, A2, i2, P2, flat, c2)


def backward(p, cache, probs, Y):
    """Backpropagation, Eqs. (10.dconvW)-(10.dmaxpool); delta^L = (a-y)/n."""
    X, Z1, A1, i1, P1, c1, Z2, A2, i2, P2, flat, c2 = cache
    n = X.shape[0]
    d3 = (probs - Y) / n                                    # softmax + CE
    g = {"W3": flat.T @ d3, "b3": d3.sum(axis=0)}
    dflat = d3 @ p["W3"].T
    dP2 = dflat.reshape(P2.shape)
    dA2 = maxpool_backward(dP2, A2, i2, 2, 2)
    dZ2 = dA2 * relu_prime(Z2)
    dP1, g["W2"], g["b2"] = conv_backward(dZ2, P1, p["W2"], c2, S=1, P=1)
    dA1 = maxpool_backward(dP1, A1, i1, 2, 2)
    dZ1 = dA1 * relu_prime(Z1)
    _, g["W1"], g["b1"] = conv_backward(dZ1, X, p["W1"], c1, S=1, P=1)
    return g

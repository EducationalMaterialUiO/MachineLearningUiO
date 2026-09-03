"""Chapter 8: listing 6, from the section on a neural network from scratch.

Extracted from doc/BookML/chapter8.tex.
"""

def gradient_check(net, X, Y, h=1e-6, n_samples=15):
    """Compare backpropagation with the central difference (4.gradcheck)."""
    rng = np.random.default_rng(0)
    gW, gb = net._backward(X, Y)
    worst = 0.0
    for l in range(len(net.W)):
        for _ in range(n_samples):
            i = rng.integers(net.W[l].shape[0]); j = rng.integers(net.W[l].shape[1])
            net.W[l][i, j] += h; c1 = net.cost(X, Y)
            net.W[l][i, j] -= 2 * h; c2 = net.cost(X, Y)
            net.W[l][i, j] += h
            num = (c1 - c2) / (2 * h)
            worst = max(worst, abs(num - gW[l][i, j])
                        / (abs(num) + abs(gW[l][i, j]) + 1e-12))
    return worst

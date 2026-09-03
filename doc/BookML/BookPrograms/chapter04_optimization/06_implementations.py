"""Chapter 4: listing 6, from the section on implementations.

Extracted from doc/BookML/chapter4.tex.
"""

import numpy as np

def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def sgd_adaptive(X, y, method="adam", n_epochs=100, batch_size=10,
                 eta=0.01, gamma=0.9, rho=0.99,
                 beta1=0.9, beta2=0.999, eps=1e-8, rng=None):
    """Stochastic gradient descent with the optimisers of this chapter.

    method is one of "plain", "momentum", "adagrad", "rmsprop", "adam".
    """
    rng = np.random.default_rng() if rng is None else rng
    n, p = X.shape
    theta = rng.normal(size=(p, 1))

    change = np.zeros((p, 1))          # momentum velocity, Eq. (4.momentum)
    r = np.zeros((p, 1))               # accumulated second moment
    m = np.zeros((p, 1))               # first moment, Eq. (4.adamfirst)
    t = 0

    for epoch in range(n_epochs):
        for batch in make_batches(n, batch_size, rng):
            t += 1
            Xb, yb = X[batch], y[batch]
            g = (2.0 / len(batch)) * Xb.T @ (Xb @ theta - yb)

            if method == "plain":
                update = eta * g
            elif method == "momentum":
                change = eta * g + gamma * change
                update = change
            elif method == "adagrad":
                r += g * g                                   # Eq. (4.adagradaccum)
                update = eta * g / (np.sqrt(r) + eps)        # Eq. (4.adagrad)
            elif method == "rmsprop":
                r = rho * r + (1 - rho) * g * g              # Eq. (4.rmspropaccum)
                update = eta * g / (np.sqrt(r) + eps)        # Eq. (4.rmsprop)
            elif method == "adam":
                m = beta1 * m + (1 - beta1) * g              # Eq. (4.adamfirst)
                r = beta2 * r + (1 - beta2) * g * g          # Eq. (4.adamsecond)
                m_hat = m / (1 - beta1**t)                   # Eq. (4.adambias)
                r_hat = r / (1 - beta2**t)
                update = eta * m_hat / (np.sqrt(r_hat) + eps)
            else:
                raise ValueError(f"unknown method {method}")

            theta -= update

    return theta

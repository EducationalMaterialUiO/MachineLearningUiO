"""Chapter 5: listing 6, from the section on an implementation.

Extracted from doc/BookML/chapter5.tex.
"""

import numpy as np

def generate_binary_data(n_samples=100, n_features=2, random_state=None):
    """Two Gaussian clusters, class 0 around -2 and class 1 around +2."""
    rng = np.random.default_rng(random_state)
    n0 = n_samples // 2
    n1 = n_samples - n0
    X0 = rng.normal(size=(n0, n_features)) - 2.0
    X1 = rng.normal(size=(n1, n_features)) + 2.0
    return np.vstack((X0, X1)), np.array([0] * n0 + [1] * n1)


def generate_multiclass_data(n_samples=150, n_features=2, n_classes=3,
                             random_state=None):
    """One Gaussian cluster per class, centred on a circle."""
    rng = np.random.default_rng(random_state)
    per = n_samples // n_classes
    Xs, ys = [], []
    for k in range(n_classes):
        angle = 2.0 * np.pi * k / n_classes
        centre = 4.0 * np.array([np.cos(angle), np.sin(angle)])
        centre = np.resize(centre, n_features)
        Xs.append(rng.normal(size=(per, n_features)) + centre)
        ys.append(np.full(per, k))
    return np.vstack(Xs), np.concatenate(ys)


X, y = generate_binary_data(200, random_state=2024)
model = LogisticRegression(lr=0.1, epochs=2000).fit(X, y)
print("binary training accuracy:", np.mean(model.predict(X) == y))

Xm, ym = generate_multiclass_data(300, n_classes=3, random_state=2024)
multi = LogisticRegression(lr=0.1, epochs=2000).fit(Xm, ym)
print("multiclass training accuracy:", np.mean(multi.predict(Xm) == ym))

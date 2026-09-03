"""Chapter 1: listing 12, from the section on principal component analysis.

Extracted from doc/BookML/chapter1.tex.
"""

import numpy as np

def pca(X, n_components=None):
    """Principal component analysis through the SVD of the centred data.

    Returns the scores T, the principal directions V, and the explained
    variance ratio of each component.
    """
    Xc = X - np.mean(X, axis=0)                  # centre each feature
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    explained = s**2 / np.sum(s**2)
    k = n_components if n_components is not None else len(s)
    T = U[:, :k] * s[:k]                         # scores = Xc @ V
    return T, Vt[:k].T, explained[:k]

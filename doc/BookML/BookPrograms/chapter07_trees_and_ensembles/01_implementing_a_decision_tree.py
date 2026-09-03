"""Chapter 7: listing 1, from the section on implementing a decision tree.

Extracted from doc/BookML/chapter7.tex.
"""

import numpy as np


def gini(y, classes):
    """Eq. (7.gini)."""
    p = np.array([np.mean(y == c) for c in classes])
    return 1.0 - np.sum(p**2)


def entropy(y, classes):
    """Eq. (7.entropy)."""
    p = np.array([np.mean(y == c) for c in classes])
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def mse_impurity(y):
    """Eq. (7.nodemse)."""
    return np.mean((y - y.mean())**2) if len(y) else 0.0


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "n")

    def __init__(self, value=None, n=0):
        self.feature = self.threshold = self.left = self.right = None
        self.value, self.n = value, n

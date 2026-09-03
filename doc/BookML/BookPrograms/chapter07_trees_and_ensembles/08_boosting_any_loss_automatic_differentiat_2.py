"""Chapter 7: listing 8, from the section on boosting any loss automatic differentiat.

Extracted from doc/BookML/chapter7.tex.
"""

class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self, value):
        self.feature = self.threshold = self.left = self.right = None
        self.value = value

class GradientTree:
    """A regression tree grown on the derivatives g_i, h_i of the loss.

    Split criterion: the gain of Eq. (7.xgbgain); leaf value: Eq. (7.xgbweight).
    With h_i = 1 and lmbda = 0 it is a least-squares tree fitted to the
    pseudo-residual -g_i, Eq. (7.gainmse); with the true h_i it is XGBoost's tree.
    """
    def __init__(self, max_depth=3, lmbda=1.0, gamma=0.0, min_child_weight=1.0):
        self.max_depth, self.lmbda, self.gamma = max_depth, lmbda, gamma
        self.min_child_weight = min_child_weight

    def _build(self, X, g, h, depth):
        G, H = g.sum(), h.sum()
        node = Node(-G / (H + self.lmbda))                    # Eq. (7.xgbweight)
        if depth >= self.max_depth or len(g) < 2:
            return node
        best_gain, best_j, best_t = 0.0, None, None
        parent = G**2 / (H + self.lmbda)
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j])
            xs, gs, hs = X[order, j], g[order], h[order]
            GL, HL = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1]   # sums left of each split
            GR, HR = G - GL, H - HL
            gain = 0.5 * (GL**2 / (HL + self.lmbda)
                          + GR**2 / (HR + self.lmbda) - parent) - self.gamma
            valid = ((xs[:-1] < xs[1:]) & (HL >= self.min_child_weight)
                     & (HR >= self.min_child_weight))
            gain = np.where(valid, gain, -np.inf)              # Eq. (7.xgbgain)
            k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain, best_j, best_t = gain[k], j, 0.5 * (xs[k] + xs[k + 1])
        if best_j is None:
            return node
        mask = X[:, best_j] <= best_t
        node.feature, node.threshold = best_j, best_t
        node.left = self._build(X[mask], g[mask], h[mask], depth + 1)
        node.right = self._build(X[~mask], g[~mask], h[~mask], depth + 1)
        return node

    def fit(self, X, g, h):
        self.root_ = self._build(np.asarray(X, float), np.asarray(g), np.asarray(h), 0)
        return self

    def predict(self, X):
        out = []
        for x in np.asarray(X, float):
            node = self.root_
            while node.feature is not None:
                node = node.left if x[node.feature] <= node.threshold else node.right
            out.append(node.value)
        return np.array(out)

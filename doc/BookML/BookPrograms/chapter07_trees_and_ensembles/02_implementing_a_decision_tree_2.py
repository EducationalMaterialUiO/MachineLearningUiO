"""Chapter 7: listing 2, from the section on implementing a decision tree.

Extracted from doc/BookML/chapter7.tex.
"""

class DecisionTree:
    """CART for regression and classification.

    task       : "classification" or "regression"
    criterion  : "gini" or "entropy" (classification only)
    max_features: if set, only a random subset of features is tried at each
                  split -- this is what turns bagging into a random forest.
    """

    def __init__(self, task="classification", criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features=None,
                 rng=None):
        self.task, self.criterion = task, criterion
        self.max_depth = np.inf if max_depth is None else max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng() if rng is None else rng

    def _impurity(self, y, w=None):
        if self.task == "regression":
            return mse_impurity(y)
        if w is None:
            return (gini(y, self.classes_) if self.criterion == "gini"
                    else entropy(y, self.classes_))
        p = np.array([w[y == c].sum() / w.sum() for c in self.classes_])
        if self.criterion == "gini":
            return 1.0 - np.sum(p**2)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def _leaf_value(self, y, w=None):
        if self.task == "regression":
            return y.mean()                       # the optimal constant
        counts = ([np.sum(y == c) for c in self.classes_] if w is None
                  else [w[y == c].sum() for c in self.classes_])
        return self.classes_[int(np.argmax(counts))]      # majority vote

    def _best_split(self, X, y, w):
        n, p = X.shape
        features = np.arange(p)
        if self.max_features is not None and self.max_features < p:
            features = self.rng.choice(p, self.max_features, replace=False)

        parent = self._impurity(y, w)
        W = w.sum() if w is not None else n
        best_gain, best_j, best_t = 0.0, None, None

        for j in features:
            xs = X[:, j]
            uniq = np.unique(xs)
            if len(uniq) < 2:
                continue
            for t in (uniq[:-1] + uniq[1:]) / 2.0:        # candidate midpoints
                mask = xs <= t
                nl, nr = mask.sum(), n - mask.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                if w is None:
                    wl, wr = nl, nr
                else:
                    wl, wr = w[mask].sum(), w[~mask].sum()
                    if wl <= 0 or wr <= 0:
                        continue
                child = ((wl / W) * self._impurity(y[mask],
                                                   None if w is None else w[mask])
                         + (wr / W) * self._impurity(y[~mask],
                                                     None if w is None else w[~mask]))
                gain = parent - child                     # Eq. (7.gain)
                if gain > best_gain + 1e-12:
                    best_gain, best_j, best_t = gain, j, t
        return best_gain, best_j, best_t

    def _build(self, X, y, w, depth):
        node = Node(self._leaf_value(y, w), len(y))
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            return node
        gain, j, t = self._best_split(X, y, w)
        if j is None or gain <= 0:
            return node
        mask = X[:, j] <= t
        node.feature, node.threshold = j, t
        node.left = self._build(X[mask], y[mask],
                                None if w is None else w[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask],
                                 None if w is None else w[~mask], depth + 1)
        return node

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if self.task == "classification":
            self.classes_ = np.unique(y)
        self.root_ = self._build(X, y, sample_weight, 0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = []
        for x in X:
            node = self.root_
            while node.feature is not None:               # walk to a leaf
                node = node.left if x[node.feature] <= node.threshold else node.right
            out.append(node.value)
        return np.array(out)

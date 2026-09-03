"""Chapter 7: listing 3, from the section on random forests.

Extracted from doc/BookML/chapter7.tex.
"""

class BaggingClassifier:
    """Bootstrap aggregation of unpruned trees, with out-of-bag scoring."""

    def __init__(self, n_estimators=100, max_depth=None, max_features=None,
                 rng=None):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.max_features = max_features
        self.rng = np.random.default_rng(0) if rng is None else rng

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        n = len(y)
        self.classes_ = np.unique(y)
        self.trees_, self.oob_ = [], []
        for _ in range(self.n_estimators):
            idx = self.rng.integers(0, n, n)            # bootstrap resample
            self.oob_.append(np.setdiff1d(np.arange(n), idx))
            tree = DecisionTree("classification", "gini", max_depth=self.max_depth,
                                max_features=self.max_features, rng=self.rng)
            self.trees_.append(tree.fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votes = np.stack([t.predict(X) for t in self.trees_])
        out = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i in range(X.shape[0]):                     # majority vote
            counts = [np.sum(votes[:, i] == c) for c in self.classes_]
            out[i] = self.classes_[int(np.argmax(counts))]
        return out

    def oob_score(self, X, y):
        """Eq. (7.oob): each point is scored by the trees that never saw it."""
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        tally = np.zeros((len(y), len(self.classes_)))
        for tree, oob in zip(self.trees_, self.oob_):
            if len(oob) == 0:
                continue
            pred = tree.predict(X[oob])
            for k, c in enumerate(self.classes_):
                tally[oob[pred == c], k] += 1
        seen = tally.sum(axis=1) > 0
        vote = self.classes_[np.argmax(tally[seen], axis=1)]
        return np.mean(vote == y[seen])


class RandomForestClassifier(BaggingClassifier):
    """Bagging plus a random subset of m features at every split."""

    def __init__(self, n_estimators=100, max_depth=None, max_features="sqrt",
                 rng=None):
        super().__init__(n_estimators, max_depth, None, rng)
        self._mf = max_features

    def fit(self, X, y):
        p = np.asarray(X).shape[1]
        self.max_features = int(np.sqrt(p)) if self._mf == "sqrt" else self._mf
        return super().fit(X, y)          # Eq. (7.mtry)

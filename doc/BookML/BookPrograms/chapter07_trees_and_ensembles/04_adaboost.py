"""Chapter 7: listing 4, from the section on adaboost.

Extracted from doc/BookML/chapter7.tex.
"""

class AdaBoost:
    """Discrete AdaBoost with decision stumps; labels must be -1 and +1."""

    def __init__(self, n_estimators=50, max_depth=1, rng=None):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.rng = np.random.default_rng(0) if rng is None else rng

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n = len(y)
        w = np.full(n, 1.0 / n)                     # step 1
        self.trees_, self.alphas_ = [], []

        for _ in range(self.n_estimators):
            tree = DecisionTree("classification", "gini",
                                max_depth=self.max_depth, rng=self.rng)
            tree.fit(X, y, sample_weight=w)         # weighted weak learner
            miss = (tree.predict(X) != y).astype(float)

            err = np.sum(w * miss) / np.sum(w)      # Eq. (7.weightederr)
            err = min(max(err, 1e-10), 1 - 1e-10)
            alpha = np.log((1 - err) / err)         # Eq. (7.alpha)

            w = w * np.exp(alpha * miss)            # Eq. (7.weightupdate)
            w /= w.sum()

            self.trees_.append(tree)
            self.alphas_.append(alpha)
            if err <= 1e-10:                        # a perfect learner: stop
                break
        return self

    def decision_function(self, X):
        return np.sum([a * t.predict(X)
                       for a, t in zip(self.alphas_, self.trees_)], axis=0)

    def predict(self, X):
        return np.sign(self.decision_function(X))   # Eq. (7.adacombine)

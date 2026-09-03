"""Chapter 7: listing 5, from the section on gradient boosting.

Extracted from doc/BookML/chapter7.tex.
"""

class GradientBoostingRegressor:
    """Gradient boosting with the squared-error loss, Eq. (7.gbresidual)."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 rng=None):
        self.n_estimators, self.lr = n_estimators, learning_rate
        self.max_depth = max_depth
        self.rng = np.random.default_rng(0) if rng is None else rng

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.f0_ = y.mean()                       # the optimal constant
        f = np.full(len(y), self.f0_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            residual = y - f                      # negative gradient
            tree = DecisionTree("regression", max_depth=self.max_depth,
                                rng=self.rng).fit(X, residual)
            f = f + self.lr * tree.predict(X)     # shrunken update
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        f = np.full(X.shape[0], self.f0_)
        for tree in self.trees_:
            f = f + self.lr * tree.predict(X)
        return f


class GradientBoostingClassifier:
    """Binary gradient boosting on the logistic loss, Eq. (7.gblogistic).

    Labels must be 0 and 1.  The model is additive in the log-odds.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 rng=None):
        self.n_estimators, self.lr = n_estimators, learning_rate
        self.max_depth = max_depth
        self.rng = np.random.default_rng(0) if rng is None else rng

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        pbar = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.f0_ = np.log(pbar / (1 - pbar))      # constant log-odds
        f = np.full(len(y), self.f0_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-f))
            residual = y - p                      # Eq. (7.gblogistic)
            tree = DecisionTree("regression", max_depth=self.max_depth,
                                rng=self.rng).fit(X, residual)
            f = f + self.lr * tree.predict(X)
            self.trees_.append(tree)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        f = np.full(X.shape[0], self.f0_)
        for tree in self.trees_:
            f = f + self.lr * tree.predict(X)
        return f

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

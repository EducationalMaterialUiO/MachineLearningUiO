"""Chapter 7: listing 9, from the section on boosting any loss automatic differentiat.

Extracted from doc/BookML/chapter7.tex.
"""

class Boosting:
    """Gradient boosting for any pointwise loss, with all derivatives from JAX.

    order=1: Friedman's gradient boosting -- the tree fits the negative
             gradient by least squares (h_i = 1, lmbda = 0), Section 7.gradientboosting;
    order=2: Newton boosting as in XGBoost -- the tree is grown on g_i, h_i
             with the regularised gain and leaf weights of Section 7.xgboost.
    """
    def __init__(self, loss, order=2, n_estimators=100, learning_rate=0.1,
                 max_depth=3, lmbda=1.0, gamma=0.0, min_child_weight=1.0, f0=None):
        self.loss, self.order = loss, order
        self.g, self.h = derivatives(loss)
        self.n_estimators, self.lr, self.f0 = n_estimators, learning_rate, f0
        self.tree_kw = dict(max_depth=max_depth, gamma=gamma,
                            lmbda=lmbda if order == 2 else 0.0,
                            min_child_weight=min_child_weight if order == 2 else 0.0)

    def _constant(self, y):
        """The constant minimising sum_i L(y_i, c): bisection on the monotone sum of g."""
        lo, hi = min(float(y.min()), -50.0) - 1.0, max(float(y.max()), 50.0) + 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if float(jnp.sum(self.g(y, jnp.full(len(y), mid)))) > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def fit(self, X, y):
        y = jnp.asarray(y, dtype=float)
        self.f0_ = self._constant(y) if self.f0 is None else self.f0
        f = jnp.full(len(y), self.f0_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            g = self.g(y, f)                                    # pseudo-residual is -g
            h = self.h(y, f) if self.order == 2 else jnp.ones_like(g)
            tree = GradientTree(**self.tree_kw).fit(X, g, h)
            f = f + self.lr * jnp.asarray(tree.predict(X))     # shrunken update
            self.trees_.append(tree)
        return self

    def decision_function(self, X):
        f = np.full(np.asarray(X).shape[0], self.f0_)
        for tree in self.trees_:
            f = f + self.lr * tree.predict(X)
        return f

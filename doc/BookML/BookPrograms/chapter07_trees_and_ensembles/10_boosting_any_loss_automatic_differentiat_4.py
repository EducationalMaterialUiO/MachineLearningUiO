"""Chapter 7: listing 10, from the section on boosting any loss automatic differentiat.

Extracted from doc/BookML/chapter7.tex.
"""

from sklearn.datasets import make_moons, make_regression
from sklearn.model_selection import train_test_split
import xgboost as xgb

g, h = derivatives(logistic)
y0, f0 = jnp.array([0., 1., 1., 0.]), jnp.array([-1., 0.3, 2., 0.1])
print("g - (p - y):", float(jnp.max(jnp.abs(g(y0, f0) - (sigmoid(f0) - y0)))))
print("h - p(1-p): ", float(jnp.max(jnp.abs(h(y0, f0) - sigmoid(f0) * (1 - sigmoid(f0))))))

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ours = Boosting(logistic, order=2, n_estimators=100, learning_rate=0.1,
                max_depth=3, lmbda=1.0, f0=0.0).fit(X_train, y_train)
lib = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                        reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,
                        base_score=0.5, tree_method="exact").fit(X_train, y_train)
p_ours = 1.0 / (1.0 + np.exp(-ours.decision_function(X_test)))
p_lib = lib.predict_proba(X_test)[:, 1]
print("max |p_ours - p_xgboost| =", np.max(np.abs(p_ours - p_lib)))
print("test accuracy: ours", np.mean((p_ours > 0.5) == y_test),
      " xgboost", np.mean((p_lib > 0.5) == y_test))

first = Boosting(logistic, order=1, n_estimators=200, learning_rate=0.1,
                 max_depth=2).fit(X_train, y_train)
print("first-order, 200 trees of depth 2: test accuracy",
      np.mean((first.decision_function(X_test) > 0) == y_test))

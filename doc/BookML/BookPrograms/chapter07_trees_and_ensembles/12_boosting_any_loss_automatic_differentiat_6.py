"""Chapter 7: listing 12, from the section on boosting any loss automatic differentiat.

Extracted from doc/BookML/chapter7.tex.
"""

def jax_objective(loss):
    """Turn a pointwise JAX loss into the (grad, hess) callback XGBoost accepts."""
    g, h = derivatives(loss)
    def objective(y_true, y_pred):
        y_true, y_pred = jnp.asarray(y_true, float), jnp.asarray(y_pred, float)
        return np.asarray(g(y_true, y_pred)), np.asarray(h(y_true, y_pred))
    return objective

custom = xgb.XGBRegressor(objective=jax_objective(logistic), n_estimators=100,
                          learning_rate=0.1, max_depth=3, reg_lambda=1.0,
                          base_score=0.0, tree_method="exact").fit(X_train, y_train)
p_custom = 1.0 / (1.0 + np.exp(-custom.predict(X_test)))
print("JAX objective vs built-in binary:logistic: max |dp| =",
      np.max(np.abs(p_custom - lib.predict_proba(X_test)[:, 1])))

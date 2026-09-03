"""Chapter 3: listing 15, from the section on choosing the penalty cross validation fo.

Extracted from doc/BookML/chapter3.tex.
"""

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

folds = np.array_split(np.random.default_rng(2024).permutation(100), 5)
cv = [(np.setdiff1d(np.arange(100), f), f) for f in folds]  # the same folds as above

model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
# scikit-learn's Ridge minimises ||y - X theta||^2 + alpha ||theta||^2, so
# alpha = n lambda, with n = 80 the size of the training folds
search = GridSearchCV(model, {"ridge__alpha": 80 * lambdas},
                      scoring="neg_mean_squared_error", cv=cv, refit=True)
search.fit(X_train, y_train)

lambda_star = search.best_params_["ridge__alpha"] / 80
cv_curve_sklearn = -search.cv_results_["mean_test_score"]
final_model = search.best_estimator_    # a new fit on all of X_train, not a fold fit
print(f"lambda* {lambda_star:.4f}, CV {-search.best_score_:.3f}, "
      f"test {np.mean((y_test - final_model.predict(X_test)) ** 2):.3f}")
cv_mean, _ = cv_curve(ridge_fit, X_train, y_train, lambdas, K=5,
                      rng=np.random.default_rng(2024))
print("largest difference from the hand-written curve:",
      np.max(np.abs(cv_curve_sklearn - cv_mean)))

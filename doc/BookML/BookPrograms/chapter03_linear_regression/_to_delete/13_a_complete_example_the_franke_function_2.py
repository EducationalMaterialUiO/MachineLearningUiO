"""Chapter 3: listing 13, from the section on a complete example the franke function.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score

rng = np.random.default_rng(2024)
# A small, noisy sample: with n = 100 the number of parameters (3.numfeatures)
# overtakes the 80 training points around degree eleven, which is where the
# difference between the three methods becomes visible.
x, y, z = make_franke_data(n=100, noise=0.2, rng=rng)

max_degree = 14
kfold = KFold(n_splits=5, shuffle=True, random_state=2024)
lambdas = np.logspace(-5, 1, 25)

results = {"OLS": [], "Ridge": [], "Lasso": []}
for degree in range(1, max_degree + 1):
    X = design_matrix_2d(x, y, degree)[:, 1:]      # drop the intercept column
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2,
                                                        random_state=2024)

    # OLS: the scaler is fitted on the training fold only, inside the pipeline
    ols_pipe = make_pipeline(StandardScaler(), LinearRegression())
    ols_pipe.fit(X_train, z_train)
    results["OLS"].append(mse(z_test, ols_pipe.predict(X_test)))

    # Ridge and Lasso: choose lambda by cross-validation on the training set
    for name, Model in (("Ridge", Ridge), ("Lasso", Lasso)):
        cv_error = [
            -cross_val_score(make_pipeline(StandardScaler(),
                                           Model(alpha=lmb, max_iter=5000)),
                             X_train, z_train, cv=kfold,
                             scoring="neg_mean_squared_error").mean()
            for lmb in lambdas
        ]
        best = make_pipeline(StandardScaler(),
                             Model(alpha=lambdas[int(np.argmin(cv_error))],
                                   max_iter=5000))
        best.fit(X_train, z_train)
        results[name].append(mse(z_test, best.predict(X_test)))
        if name == "Lasso":
            n_kept = int(np.sum(best[-1].coef_ != 0))

for name, errors in results.items():
    print(f"{name:>6}: best degree {int(np.argmin(errors)) + 1:2d}, "
          f"test MSE {min(errors):.4f}, "
          f"MSE at degree {max_degree} {errors[-1]:.4f}")

"""Chapter 2: listing 6, from the section on cross validation.

Extracted from doc/BookML/chapter2.tex.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

np.random.seed(3155)

x = np.random.randn(100)[:, np.newaxis]
y = 3 * x.ravel()**2 + np.random.randn(100)          # noise variance is 1

lambdas = np.logspace(-3, 5, 100)
kfold = KFold(n_splits=5, shuffle=True, random_state=3155)

mse = np.empty(len(lambdas))
for i, lmb in enumerate(lambdas):
    # The scaler is refitted on each training fold, never on the test fold
    pipe = make_pipeline(PolynomialFeatures(degree=6),
                         StandardScaler(),
                         Ridge(alpha=lmb))
    scores = -cross_val_score(pipe, x, y, cv=kfold,
                              scoring="neg_mean_squared_error")
    mse[i] = scores.mean()

best = np.argmin(mse)
print(f"best lambda {lambdas[best]:.4g}, cross-validated MSE {mse[best]:.4f}")

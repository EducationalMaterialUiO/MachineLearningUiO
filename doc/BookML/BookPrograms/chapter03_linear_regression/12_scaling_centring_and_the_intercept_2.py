"""Chapter 3: listing 12, from the section on scaling centring and the intercept.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np

def fit_with_intercept(X, y, fit_centred):
    """Fit a penalised model without penalising the intercept.

    fit_centred(Xc, yc) must return coefficients for centred data with no
    intercept column.  Returns (theta_0, theta) plus the training
    statistics needed to transform future data identically.
    """
    x_mean = np.mean(X, axis=0)
    y_mean = np.mean(y)
    x_std = np.std(X, axis=0)
    x_std[x_std == 0.0] = 1.0                  # leave constant columns alone

    Xc = (X - x_mean) / x_std
    theta = fit_centred(Xc, y - y_mean)

    # Undo the scaling so the coefficients apply to the raw features
    theta = theta / x_std
    theta_0 = y_mean - x_mean @ theta
    return theta_0, theta, (x_mean, x_std, y_mean)


def predict(X_new, theta_0, theta):
    return theta_0 + X_new @ theta

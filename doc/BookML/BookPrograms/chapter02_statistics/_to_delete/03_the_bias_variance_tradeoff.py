"""Chapter 2: listing 3, from the section on the bias variance tradeoff.

Extracted from doc/BookML/chapter2.tex.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

np.random.seed(2018)

n, n_bootstraps, degree = 500, 100, 18      # a deliberately high degree
x = np.linspace(-1, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x - 2)**2) + np.random.normal(0, 0.1, x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = make_pipeline(PolynomialFeatures(degree=degree),
                      LinearRegression(fit_intercept=False))

# Column i holds the predictions of the model fitted to bootstrap sample i
y_pred = np.empty((y_test.shape[0], n_bootstraps))
for i in range(n_bootstraps):
    x_, y_ = resample(x_train, y_train)
    y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

# Expectations over training sets are averages along axis 1; keepdims=True
# preserves the column shape and is essential for the bias to come out right.
error    = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
bias     = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
variance = np.mean(np.var(y_pred, axis=1, keepdims=True))

print(f"Error = {error:.5f}, Bias^2 = {bias:.5f}, Var = {variance:.5f}")
print(f"{error:.5f} >= {bias:.5f} + {variance:.5f} = {bias + variance:.5f}")

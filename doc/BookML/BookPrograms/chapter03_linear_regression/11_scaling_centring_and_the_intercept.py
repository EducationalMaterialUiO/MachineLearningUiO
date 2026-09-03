"""Chapter 3: listing 11, from the section on scaling centring and the intercept.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np
from sklearn.linear_model import Lasso

rng = np.random.default_rng(2026)
n = 50
x1 = rng.random(n)                     # a length, measured in metres
x2 = rng.random(n)                     # a time, measured in seconds
y = 2.0 * x1 - 3.0 * x2 + 0.1 * rng.standard_normal(n)

X = np.column_stack([x1, x2])
D = np.diag([1000.0, 1.0])             # metres -> millimetres in column 1
XD = X @ D                             # the same data, in the new units

Xc, yc = X - X.mean(axis=0), y - y.mean()      # centred: intercept handled
XDc = XD - XD.mean(axis=0)                     # separately and unpenalised

lmbda = 0.1
def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]
def ridge(X, y, P):                    # generalised penalty lambda theta^T P theta
    return np.linalg.solve(X.T @ X + lmbda * P, X.T @ y)

I = np.eye(2)
t, tD = ols(Xc, yc), ols(XDc, yc)
print("OLS   coefficients (m,s):", t, "  (mm,s):", tD)
print("OLS   D theta_mm = theta_m:", np.allclose(np.diag(D) * tD, t),
      "  max |prediction change|: %.2e" % np.max(np.abs(XDc @ tD - Xc @ t)))

r, rD = ridge(Xc, yc, I), ridge(XDc, yc, I)
print("Ridge coefficients (m,s):", r, "  (mm,s):", rD)
print("Ridge max |prediction change|: %.3f" % np.max(np.abs(XDc @ rD - Xc @ r)))
r_mod = ridge(Xc, yc, np.linalg.inv(D @ D))    # penalty lambda theta^T D^-2 theta
print("Ridge on XD = modified-penalty ridge on X:",
      np.allclose(np.diag(D) * rD, r_mod))

la = Lasso(alpha=lmbda / 2, fit_intercept=False, max_iter=100000)
lm, lmm = la.fit(Xc, yc).coef_.copy(), la.fit(XDc, yc).coef_.copy()
print("Lasso coefficients (m,s):", lm, "  (mm,s):", lmm)
print("Lasso max |prediction change|: %.3f" % np.max(np.abs(XDc @ lmm - Xc @ lm)))

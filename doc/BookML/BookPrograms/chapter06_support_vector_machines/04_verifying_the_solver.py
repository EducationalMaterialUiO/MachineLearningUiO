"""Chapter 6: listing 4, from the section on verifying the solver.

Extracted from doc/BookML/chapter6.tex.
"""

import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.svm import SVC as SklearnSVC

np.random.seed(0)

# --- a separable linear problem ---
X, y01 = make_blobs(n_samples=80, centers=2, cluster_std=0.8, random_state=3)
y = np.where(y01 == 0, -1.0, 1.0)

model = SVC(C=10.0, kernel=linear_kernel).fit(X, y)
w = (model.lam_sv * model.y_sv) @ model.X_sv            # Eq. (6.wfromlambda)

reference = SklearnSVC(C=10.0, kernel="linear").fit(X, y)
print("ours    w =", np.round(w, 5), " b =", round(model.b_, 5))
print("sklearn w =", np.round(reference.coef_.ravel(), 5),
      " b =", round(reference.intercept_[0], 5))

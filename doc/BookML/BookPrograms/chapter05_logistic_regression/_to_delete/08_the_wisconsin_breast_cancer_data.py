"""Chapter 5: listing 8, from the section on the wisconsin breast cancer data.

Extracted from doc/BookML/chapter5.tex.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
print(X_train.shape, X_test.shape)

# Scaling inside the pipeline, so it is refitted on each training fold
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
logreg.fit(X_train, y_train)

print(f"test accuracy: {logreg.score(X_test, y_test):.3f}")
scores = cross_validate(logreg, X_train, y_train, cv=10)["test_score"]
print(f"cross-validated accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

"""Chapter 8: listing 2, from the section on why one layer is not enough the xor prob.

Extracted from doc/BookML/chapter8.tex.
"""

from sklearn.linear_model import LogisticRegression

for name, y in [("XOR", [0, 1, 1, 0]), ("OR", [0, 1, 1, 1]), ("AND", [0, 0, 0, 1])]:
    y = np.array(y)
    logreg = LogisticRegression().fit(X, y)
    print(f"{name}: accuracy {logreg.score(X, y):.2f}")

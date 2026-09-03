"""Chapter 8: listing 8, from the section on examples.

Extracted from doc/BookML/chapter8.tex.
"""

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(50,), activation="relu",
                    alpha=1e-4, max_iter=500, random_state=1)
mlp.fit(X_train, y_train)
print(f"test accuracy {mlp.score(X_test, y_test):.4f}")

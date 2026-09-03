"""Chapter 8: listing 7, from the section on examples.

Extracted from doc/BookML/chapter8.tex.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
X_train, X_test, y_train, y_test = train_test_split(X, digits.target,
                                                    test_size=0.2, random_state=42)

for activation in ["sigmoid", "relu"]:
    net = NeuralNetwork([64, 50, 10], activation, "classification",
                        eta=0.1, lmbd=1e-4, epochs=60, batch_size=32,
                        rng=np.random.default_rng(2024)).fit(X_train, y_train)
    print(f"{activation:8s}: train {np.mean(net.predict(X_train) == y_train):.4f}  "
          f"test {np.mean(net.predict(X_test) == y_test):.4f}  "
          f"final loss {net.loss_[-1]:.4f}")

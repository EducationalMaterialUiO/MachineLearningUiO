"""Chapter 5: listing 5, from the section on an implementation.

Extracted from doc/BookML/chapter5.tex.
"""

import numpy as np

class LogisticRegression:
    """Logistic regression for binary and multiclass classification.

    Binary problems use the sigmoid (5.sigmoid) with the cross entropy
    (5.crossentropy); multiclass problems use the softmax (5.softmax)
    with the multiclass cross entropy (5.multicrossentropy).
    """

    def __init__(self, lr=0.01, epochs=1000, fit_intercept=True, lmbda=0.0):
        self.lr = lr                       # learning rate for gradient descent
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.lmbda = lmbda                 # l2 penalty, Eq. (5.penalised)
        self.weights = None
        self.multi_class = False

    @staticmethod
    def _add_intercept(X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    @staticmethod
    def _sigmoid(z):
        out = np.empty_like(z, dtype=float)
        pos, neg = z >= 0, z < 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    @staticmethod
    def _softmax(Z):
        """Softmax with the shift of Eq. (5.softmaxstable)."""
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if self.fit_intercept:
            X = self._add_intercept(X)
        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        self.multi_class = len(self.classes_) > 2

        if self.multi_class:
            n_classes = len(self.classes_)
            index = {c: k for k, c in enumerate(self.classes_)}
            Y = np.zeros((n_samples, n_classes))            # one-hot targets
            Y[np.arange(n_samples), [index[c] for c in y]] = 1
            self.weights = np.zeros((n_features, n_classes))

            for _ in range(self.epochs):
                probs = self._softmax(X @ self.weights)
                grad = X.T @ (probs - Y) / n_samples        # Eq. (5.softmaxgradient)
                grad += 2.0 * self.lmbda * self.weights
                self.weights -= self.lr * grad
        else:
            yb = (y == self.classes_[1]).astype(float)      # map labels to {0,1}
            self.weights = np.zeros(n_features)

            for _ in range(self.epochs):
                probs = self._sigmoid(X @ self.weights)
                grad = X.T @ (probs - yb) / n_samples       # Eq. (5.gradient)
                grad += 2.0 * self.lmbda * self.weights
                self.weights -= self.lr * grad
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = self._add_intercept(X)
        if self.multi_class:
            return self._softmax(X @ self.weights)
        p1 = self._sigmoid(X @ self.weights)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

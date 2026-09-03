"""Chapter 8: listing 5, from the section on a neural network from scratch.

Extracted from doc/BookML/chapter8.tex.
"""

class NeuralNetwork:
    """Fully connected feed-forward network trained by backpropagation.

    layer_sizes : e.g. [64, 50, 10] for 64 inputs, one hidden layer of 50
                  units and 10 outputs.
    task        : "classification" (softmax + cross entropy) or
                  "regression" (linear output + half-squared error).
    """

    def __init__(self, layer_sizes, hidden_activation="sigmoid",
                 task="classification", eta=0.1, lmbd=0.0, epochs=100,
                 batch_size=32, rng=None):
        self.sizes, self.task = layer_sizes, task
        self.f, self.fp = ACT[hidden_activation]
        self.eta, self.lmbd = eta, lmbd
        self.epochs, self.batch = epochs, batch_size
        self.rng = np.random.default_rng(0) if rng is None else rng
        self._init_parameters(hidden_activation)

    def _init_parameters(self, act):
        """Xavier (8.xavier) or He (8.he) initialisation, by activation."""
        self.W, self.b = [], []
        for i in range(len(self.sizes) - 1):
            nin, nout = self.sizes[i], self.sizes[i + 1]
            s = np.sqrt(2.0 / nin) if act in ("relu", "leaky_relu", "elu") \
                else np.sqrt(1.0 / nin)
            self.W.append(self.rng.normal(0, s, (nin, nout)))
            self.b.append(np.zeros(nout) + 0.01)

    def _forward(self, X):
        """Eq. (8.forwardbatch); keeps every z and a for the backward pass."""
        a, z = [X], []
        for l in range(len(self.W)):
            zl = a[-1] @ self.W[l] + self.b[l]
            z.append(zl)
            if l == len(self.W) - 1:
                a.append(softmax(zl) if self.task == "classification" else zl)
            else:
                a.append(self.f(zl))
        return a, z

    def _backward(self, X, Y):
        """The four equations (8.bp1)-(8.bp4)."""
        n = X.shape[0]
        a, z = self._forward(X)

        # With softmax + cross entropy, and with a linear output under the
        # half-squared error, the output error is the same expression:
        delta = (a[-1] - Y) / n                       # Eq. (8.deltaLsimple)

        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for l in range(len(self.W) - 1, -1, -1):
            gW[l] = a[l].T @ delta + self.lmbd * self.W[l]   # Eq. (8.bp4)
            gb[l] = np.sum(delta, axis=0)                    # Eq. (8.bp3)
            if l > 0:
                delta = (delta @ self.W[l].T) * self.fp(z[l - 1])  # Eq. (8.bp2)
        return gW, gb

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if self.task == "classification":
            self.classes_ = np.unique(y)
            idx = {c: i for i, c in enumerate(self.classes_)}
            Y = np.zeros((len(y), len(self.classes_)))        # one-hot
            Y[np.arange(len(y)), [idx[c] for c in y]] = 1
        else:
            Y = np.asarray(y, dtype=float).reshape(-1, 1)

        n = X.shape[0]
        self.loss_ = []
        for _ in range(self.epochs):
            order = self.rng.permutation(n)                   # shuffle, Sec. 4.practicaltips
            for s in range(0, n, self.batch):
                b = order[s:s + self.batch]
                gW, gb = self._backward(X[b], Y[b])
                for l in range(len(self.W)):                  # Eq. (8.update)
                    self.W[l] -= self.eta * gW[l]
                    self.b[l] -= self.eta * gb[l]
            self.loss_.append(self.cost(X, Y))
        return self

    def cost(self, X, Y):
        o = self._forward(X)[0][-1]
        if self.task == "classification":
            return float(-np.mean(np.sum(Y * np.log(np.clip(o, 1e-12, 1)), axis=1)))
        return float(0.5 * np.mean((o - Y)**2))               # Eq. (8.mse)

    def predict_proba(self, X):
        return self._forward(np.asarray(X, dtype=float))[0][-1]

    def predict(self, X):
        o = self.predict_proba(X)
        return (self.classes_[np.argmax(o, axis=1)]
                if self.task == "classification" else o.ravel())

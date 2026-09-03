"""Chapter 8: listing 3, from the section on why one layer is not enough the xor prob.

Extracted from doc/BookML/chapter8.tex.
"""

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

for name, y in [("XOR", [0, 1, 1, 0]), ("OR", [0, 1, 1, 1]), ("AND", [0, 0, 0, 1])]:
    y = np.array(y)
    net = NeuralNetwork([2, 2, 2], "sigmoid", "classification",
                        eta=1.0, epochs=4000, batch_size=4,
                        rng=np.random.default_rng(7)).fit(X, y)
    print(f"{name}: accuracy {np.mean(net.predict(X) == y):.2f}, "
          f"predictions {net.predict(X)}")

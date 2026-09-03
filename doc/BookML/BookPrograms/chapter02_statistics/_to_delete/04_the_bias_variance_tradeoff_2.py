"""Chapter 2: listing 4, from the section on the bias variance tradeoff.

Extracted from doc/BookML/chapter2.tex.
"""

import matplotlib.pyplot as plt

n, n_bootstraps, maxdegree = 40, 100, 14
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x - 2)**2) + np.random.normal(0, 0.1, x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

error    = np.zeros(maxdegree)
bias     = np.zeros(maxdegree)
variance = np.zeros(maxdegree)

for degree in range(maxdegree):
    model = make_pipeline(PolynomialFeatures(degree=degree),
                          LinearRegression(fit_intercept=False))
    y_pred = np.empty((y_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        x_, y_ = resample(x_train, y_train)
        y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

    error[degree]    = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
    bias[degree]     = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
    variance[degree] = np.mean(np.var(y_pred, axis=1, keepdims=True))

plt.plot(range(maxdegree), error,    label="Error")
plt.plot(range(maxdegree), bias,     label="Bias$^2$")
plt.plot(range(maxdegree), variance, label="Variance")
plt.xlabel("Polynomial degree"); plt.legend(); plt.show()

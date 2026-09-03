"""Chapter 5: listing 8, from the section on one program for every loss.

Extracted from doc/BookML/chapter5.tex.
"""

rng = np.random.default_rng(2024)
n = 400
X = jnp.asarray(np.c_[np.ones(n), rng.normal(size=(n, 2))])
theta_true = jnp.array([0.5, 2.0, -1.0])
p_true = sigmoid(X @ theta_true)                       # the truth we try to recover
y = jnp.asarray((rng.random(n) < np.asarray(p_true)).astype(float))

theta0 = jnp.array([0.1, -0.3, 0.2])
g_ad = grad(cost)(theta0, X, y, cross_entropy)
g_hand = -X.T @ (y - sigmoid(X @ theta0)) / n                 # Eq. (5.gradient)
print("gradient: max |AD - hand| =", float(jnp.max(jnp.abs(g_ad - g_hand))))

p0 = sigmoid(X @ theta0)
H_ad = hessian(cost)(theta0, X, y, cross_entropy)
H_hand = X.T @ ((p0 * (1 - p0))[:, None] * X) / n              # Eq. (5.hessian)
print("Hessian:  max |AD - hand| =", float(jnp.max(jnp.abs(H_ad - H_hand))))

print("Newton :", newton(cross_entropy, X, y))
print("GD     :", fit(cross_entropy, X, y))

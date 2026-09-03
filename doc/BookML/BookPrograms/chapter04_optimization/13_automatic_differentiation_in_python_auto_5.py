"""Chapter 4: listing 13, from the section on automatic differentiation in python auto.

Extracted from doc/BookML/chapter4.tex.
"""

def ridge_cost(theta, X, y, lmbda):        # Eq. (4.ridgecost)
    return jnp.sum((X @ theta - y)**2) / len(y) + lmbda * jnp.sum(theta**2)

lmbda = 0.001
theta = jnp.asarray(rng.normal(size=(2, 1)))
step = jit(lambda th: th - eta * grad(ridge_cost)(th, Xj, yj, lmbda))
for k in range(1000):
    theta = step(theta)

I = np.eye(2)
theta_closed = np.linalg.inv(X.T @ X + n * lmbda * I) @ X.T @ y
print("gradient descent:", np.asarray(theta).ravel())
print("closed form:     ", theta_closed.ravel())

# The derivative with respect to lambda, argument number 3
dC_dlambda = grad(ridge_cost, argnums=3)(theta, Xj, yj, lmbda)
print(dC_dlambda, "=", jnp.sum(theta**2))    # theta^T theta

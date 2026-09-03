"""Chapter 4: listing 14, from the section on automatic differentiation in python auto.

Extracted from doc/BookML/chapter4.tex.
"""

from jax import hessian, jvp

H = hessian(cost)(theta, Xj, yj).reshape(2, 2)     # forward-over-reverse
print(H)
print((2.0 / n) * X.T @ X)                         # Eq. (4.gdhessian)

# One Newton step from a random start, Eq. (4.newtonopt)
theta0 = jnp.asarray(rng.normal(size=(2, 1)))
theta_newton = theta0 - jnp.linalg.solve(H, grad(cost)(theta0, Xj, yj))
print("one Newton step:", np.asarray(theta_newton).ravel())

# Hessian-vector product without the Hessian, Eq. (4.hvp)
def hvp(f, theta, v):
    return jvp(grad(f), (theta,), (v,))[1]

v = jnp.array([[1.0], [0.0]])
print(hvp(lambda th: cost(th, Xj, yj), theta0, v).ravel(), (H @ v).ravel())

"""Chapter 4: listing 12, from the section on automatic differentiation in python auto.

Extracted from doc/BookML/chapter4.tex.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import grad, jit

# The data of Section 4.gdlinreg
n = 100
rng = np.random.default_rng(2024)
x = 2.0 * rng.random((n, 1))
y = 4.0 + 3.0 * x + rng.normal(size=(n, 1))
X = np.c_[np.ones((n, 1)), x]
Xj, yj = jnp.asarray(X), jnp.asarray(y)

def cost(theta, X, y):                     # Eq. (4.gdcost)
    return jnp.sum((X @ theta - y)**2) / len(y)

cost_grad = jit(grad(cost, argnums=0))     # d cost / d theta, compiled

# Check against the hand-derived gradient, Eq. (4.gdgradient)
theta = jnp.asarray(rng.normal(size=(2, 1)))
g_ad = cost_grad(theta, Xj, yj)
g_hand = (2.0 / n) * X.T @ (X @ np.asarray(theta) - y)
print("max |AD - hand| =", np.max(np.abs(np.asarray(g_ad) - g_hand)))

# Gradient descent, Eq. (4.gditeration), with the automatic gradient
eta = 0.1
@jit
def gd_step(theta):
    return theta - eta * grad(cost)(theta, Xj, yj)

for k in range(1000):
    theta = gd_step(theta)
print("gradient descent with AD:", np.asarray(theta).ravel())
theta_exact = np.linalg.pinv(X.T @ X) @ X.T @ y
print("analytical:              ", theta_exact.ravel())

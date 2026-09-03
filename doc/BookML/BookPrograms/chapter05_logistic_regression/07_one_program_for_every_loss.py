"""Chapter 5: listing 7, from the section on one program for every loss.

Extracted from doc/BookML/chapter5.tex.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, hessian
from jax.nn import sigmoid, softplus, log_sigmoid

# Every loss is a function of the linear predictor t = x^T theta and the label
# y in {0, 1}, evaluated pointwise; softplus(t) = log(1 + e^t) is the stable
# form of Eq. (5.crossentropycompact), log_sigmoid(t) = log sigma(t).
def cross_entropy(t, y):
    return softplus(t) - y * t                          # Eq. (5.logisticloss)

def squared_error(t, y):
    return (y - sigmoid(t))**2                          # the Brier score

def hinge(t, y):
    return jnp.maximum(0.0, 1.0 - (2*y - 1) * t)        # margin m = (2y-1) t

def squared_hinge(t, y):
    return jnp.maximum(0.0, 1.0 - (2*y - 1) * t)**2

def exponential(t, y):
    return jnp.exp(-(2*y - 1) * t)

def focal(t, y, gamma=2.0):                             # Eq. (5.focal)
    p = sigmoid(t)
    return -(y * (1 - p)**gamma * log_sigmoid(t)
             + (1 - y) * p**gamma * log_sigmoid(-t))

LOSSES = {"cross entropy": cross_entropy, "squared error": squared_error,
          "hinge": hinge, "squared hinge": squared_hinge,
          "exponential": exponential, "focal (gamma=2)": focal}

def cost(theta, X, y, loss, lmbda=0.0):
    """Mean loss over the data plus an l2 penalty that spares the intercept."""
    return jnp.mean(loss(X @ theta, y)) + lmbda * jnp.sum(theta[1:]**2)

def fit(loss, X, y, eta=0.5, epochs=3000, lmbda=0.0):
    """Gradient descent, Eq. (5.gd), with the gradient supplied by JAX."""
    c = lambda th: cost(th, X, y, loss, lmbda)
    step = jit(lambda th: th - eta * grad(c)(th))
    theta = jnp.zeros(X.shape[1])
    for _ in range(epochs):
        theta = step(theta)
    return theta

def newton(loss, X, y, iters=10, lmbda=1e-8):
    """Newton-Raphson, Eq. (5.newtonabstract), with the Hessian from JAX."""
    c = lambda th: cost(th, X, y, loss, lmbda)
    g, H = jit(grad(c)), jit(hessian(c))
    theta = jnp.zeros(X.shape[1])
    for _ in range(iters):
        theta = theta - jnp.linalg.solve(H(theta), g(theta))
    return theta

"""Chapter 4: listing 10, from the section on automatic differentiation.

Extracted from doc/BookML/chapter4.tex.
"""

import jax.numpy as jnp
from jax import grad, jit, vmap

def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.0)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))

fast_derivative = jit(derivative_fn)      # compiled on first call

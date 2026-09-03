"""Chapter 7: listing 7, from the section on boosting any loss automatic differentiat.

Extracted from doc/BookML/chapter7.tex.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.nn import softplus, sigmoid

# Pointwise losses L(y, f).  f is the additive model: the value itself for
# regression, the log-odds for classification, the log-rate for counts.
def squared(y, f):
    return 0.5 * (y - f)**2                                # Eq. (7.gbresidual)

def logistic(y, f):
    return softplus(f) - y * f                             # cross entropy, f = log-odds

def huber(y, f, delta=1.0):
    r = jnp.abs(y - f)
    return jnp.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))

def quantile(y, f, tau=0.9):
    r = y - f
    return jnp.maximum(tau * r, (tau - 1.0) * r)           # pinball loss

def poisson(y, f):
    return jnp.exp(f) - y * f                              # counts, f = log(rate)

def derivatives(loss):
    """g_i = dL/df and h_i = d^2L/df^2 at every data point, from JAX."""
    g = jit(vmap(grad(loss, argnums=1)))
    h = jit(vmap(grad(grad(loss, argnums=1), argnums=1)))
    return g, h

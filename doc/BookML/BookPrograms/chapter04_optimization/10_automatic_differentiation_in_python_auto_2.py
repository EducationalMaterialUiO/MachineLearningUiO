"""Chapter 4: listing 10, from the section on automatic differentiation in python auto.

Extracted from doc/BookML/chapter4.tex.
"""

import jax
jax.config.update("jax_enable_x64", True)   # double precision throughout
import jax.numpy as jnp
from jax import grad, jit, vmap

def f(x):
    return jnp.sin(2 * jnp.pi * x + x**2)

df = grad(f)                       # reverse-mode gradient of a scalar function
print(df(1.0))                     # 4.475424121402227, exact to all digits
print(grad(grad(f))(1.0))          # second derivative, by composing grad

xs = jnp.linspace(0.0, 1.0, 5)
print(vmap(df)(xs))                # the derivative at five points at once
fast_df = jit(df)                  # compiled on first call, then very fast
print(fast_df(1.0))

"""Chapter 4: listing 11, from the section on automatic differentiation in python auto.

Extracted from doc/BookML/chapter4.tex.
"""

from jax import jvp, vjp, jacfwd, jacrev

def F(x):                          # a map from R^2 to R^3
    return jnp.array([x[0] * x[1], jnp.sin(x[0]), jnp.exp(x[1])])

x0 = jnp.array([1.0, 2.0])
J = jacfwd(F)(x0)                  # 3 x 2 Jacobian, built from two forward sweeps
print(J)
print(jnp.allclose(J, jacrev(F)(x0)))       # ...and from three reverse sweeps

v = jnp.array([1.0, 0.0])          # a direction in the input space
_, Jv = jvp(F, (x0,), (v,))        # forward mode: J v, first column of J
print(Jv)

u = jnp.array([1.0, 1.0, 1.0])     # a covector in the output space
_, vjp_fn = vjp(F, x0)
print(vjp_fn(u)[0])                # reverse mode: u^T J, the column sums of J

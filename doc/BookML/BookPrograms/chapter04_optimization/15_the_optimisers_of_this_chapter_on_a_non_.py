"""Chapter 4: listing 15, from the section on the optimisers of this chapter on a non .

Extracted from doc/BookML/chapter4.tex.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, hessian

def rosenbrock(p):                                 # Eq. (4.rosenbrock)
    x, y = p
    return (1 - x)**2 + 100 * (y - x**2)**2

rgrad = jit(grad(rosenbrock))                      # nobody derived these
rhess = jit(hessian(rosenbrock))
p0 = jnp.array([-1.5, 2.0])

def run(stepper, tol=1e-10, max_iter=100000):
    """Iterate p, state = stepper(p, state, t) until f(p) < tol."""
    p, state = p0, None
    for t in range(1, max_iter + 1):
        p, state = stepper(p, state, t)
        if float(rosenbrock(p)) < tol:
            return t, np.asarray(p)
    return None, np.asarray(p)

def gd(p, state, t, eta=1e-3):                     # Eq. (4.gd)
    return p - eta * rgrad(p), state

def momentum(p, state, t, eta=1e-3, gamma=0.9):    # Eq. (4.momentum)
    v = jnp.zeros(2) if state is None else state
    v = gamma * v + eta * rgrad(p)
    return p - v, v

def adam(p, state, t, eta=0.02, b1=0.9, b2=0.999, eps=1e-8):   # Eq. (4.adam)
    m, r = (jnp.zeros(2), jnp.zeros(2)) if state is None else state
    g = rgrad(p)
    m = b1 * m + (1 - b1) * g
    r = b2 * r + (1 - b2) * g * g
    m_hat, r_hat = m / (1 - b1**t), r / (1 - b2**t)
    return p - eta * m_hat / (jnp.sqrt(r_hat) + eps), (m, r)

def newton(p, state, t):                           # Eq. (4.newtonopt)
    return p - jnp.linalg.solve(rhess(p), rgrad(p)), state

for name, stepper in [("gradient descent", gd), ("momentum", momentum),
                      ("Adam", adam), ("Newton", newton)]:
    steps, p = run(stepper)
    print(f"{name:17s} {steps:6d} iterations, p = {p}")

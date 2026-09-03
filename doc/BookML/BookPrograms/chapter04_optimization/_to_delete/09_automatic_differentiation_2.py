"""Chapter 4: listing 9, from the section on automatic differentiation.

Extracted from doc/BookML/chapter4.tex.
"""

import autograd.numpy as np
from autograd import grad

def cost(theta, X, y):
    return np.sum((X @ theta - y)**2) / len(y)

training_gradient = grad(cost, 0)      # differentiate w.r.t. argument 0

theta = np.random.randn(2, 1)
eta = 0.1
for k in range(1000):
    theta -= eta * training_gradient(theta, X, y)

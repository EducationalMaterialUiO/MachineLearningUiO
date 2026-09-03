"""Chapter 8: listing 4, from the section on a neural network from scratch.

Extracted from doc/BookML/chapter8.tex.
"""

import numpy as np


def sigmoid(z):
    """Numerically stable logistic function, Eq. (8.sigmoid)."""
    out = np.empty_like(z, dtype=float)
    p, n = z >= 0, z < 0
    out[p] = 1.0 / (1.0 + np.exp(-z[p]))
    e = np.exp(z[n]); out[n] = e / (1.0 + e)
    return out

def sigmoid_prime(z):  s = sigmoid(z); return s * (1 - s)
def relu(z):           return np.maximum(0.0, z)            # Eq. (8.relu)
def relu_prime(z):     return (z > 0).astype(float)
def leaky_relu(z, a=0.01):        return np.where(z > 0, z, a * z)
def leaky_relu_prime(z, a=0.01):  return np.where(z > 0, 1.0, a)
def elu(z, a=1.0):                                          # Eq. (8.elu)
    return np.where(z > 0, z, a * (np.exp(np.minimum(z, 0)) - 1))
def elu_prime(z, a=1.0):
    return np.where(z > 0, 1.0, a * np.exp(np.minimum(z, 0)))
def tanh_(z):          return np.tanh(z)
def tanh_prime(z):     return 1 - np.tanh(z)**2
def identity(z):       return z
def identity_prime(z): return np.ones_like(z)

def softplus(z):       return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
def softplus_prime(z): return sigmoid(z)

def gelu(z):
    """Exact GELU, Eq. (8.gelu); erf comes from scipy."""
    from scipy.special import erf
    return z * 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

def gelu_prime(z):
    from scipy.special import erf
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)      # the normal density
    return Phi + z * phi                                   # d/dz [z Phi(z)]

def gelu_tanh(z):
    """The approximation of Eq. (8.geluapprox) used by most frameworks."""
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

def swish(z):          return z * sigmoid(z)               # Eq. (8.swish)
def swish_prime(z):
    s = sigmoid(z); return s + z * s * (1 - s)

def mish(z):                                               # Eq. (8.mish)
    sp = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
    return z * np.tanh(sp)

def mish_prime(z, h=1e-6):
    return (mish(z + h) - mish(z - h)) / (2 * h)           # numerical is adequate

def softmax(z):
    """Eq. (8.softmax), with the shift of Eq. (5.softmaxstable)."""
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

ACT = {"sigmoid": (sigmoid, sigmoid_prime), "relu": (relu, relu_prime),
       "leaky_relu": (leaky_relu, leaky_relu_prime), "elu": (elu, elu_prime),
       "tanh": (tanh_, tanh_prime), "identity": (identity, identity_prime),
       "softplus": (softplus, softplus_prime), "gelu": (gelu, gelu_prime),
       "swish": (swish, swish_prime), "mish": (mish, mish_prime)}

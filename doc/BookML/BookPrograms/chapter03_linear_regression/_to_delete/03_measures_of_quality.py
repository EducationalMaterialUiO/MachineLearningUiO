"""Chapter 3: listing 3, from the section on measures of quality.

Extracted from doc/BookML/chapter3.tex.
"""

import numpy as np

def mse(y, y_tilde):
    return np.mean((y - y_tilde)**2)

def r2(y, y_tilde):
    return 1.0 - np.sum((y - y_tilde)**2) / np.sum((y - np.mean(y))**2)

def mae(y, y_tilde):
    return np.mean(np.abs(y - y_tilde))

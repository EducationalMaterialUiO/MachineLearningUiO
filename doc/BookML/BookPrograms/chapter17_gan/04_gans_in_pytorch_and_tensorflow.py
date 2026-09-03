"""Chapter 17: listing 4, from the section on gans in pytorch and tensorflow.

Extracted from doc/BookML/chapter17.tex.
"""

import torch
import torch.nn as nn


def make_generator_fc(k_z=100, n_out=784, widths=(256, 512, 1024)):
    """z in R^k -> image in [-1,1]^784, Eq. (17.generator)."""
    layers, n_in = [], k_z
    for w in widths:
        layers += [nn.Linear(n_in, w), nn.BatchNorm1d(w),
                   nn.LeakyReLU(0.2, inplace=True)]
        n_in = w
    layers += [nn.Linear(n_in, n_out), nn.Tanh()]
    return nn.Sequential(*layers)


def make_discriminator_fc(n_in=784, widths=(1024, 512, 256), p_drop=0.3):
    """image -> logit u(x); D(x) = sigmoid(u(x)).  No batch norm in D."""
    layers = []
    for w in widths:
        layers += [nn.Linear(n_in, w), nn.LeakyReLU(0.2, inplace=True),
                   nn.Dropout(p_drop)]
        n_in = w
    layers += [nn.Linear(n_in, 1)]
    return nn.Sequential(*layers)

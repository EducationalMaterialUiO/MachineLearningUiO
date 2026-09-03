"""Chapter 4: listing 4, from the section on stochastic gradient descent.

Extracted from doc/BookML/chapter4.tex.
"""

import numpy as np

n = 100            # data points
M = 5              # size of each minibatch
m = int(n / M)     # number of minibatches
n_epochs = 10

for epoch in range(1, n_epochs + 1):
    for i in range(m):
        k = np.random.randint(m)     # pick the k-th minibatch at random
        # compute the gradient using the data in minibatch B_k
        # update theta with Eq. (4.sgdupdate)
        pass

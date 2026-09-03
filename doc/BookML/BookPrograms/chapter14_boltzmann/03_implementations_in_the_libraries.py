"""Chapter 14: listing 3, from the section on implementations in the libraries.

Extracted from doc/BookML/chapter14.tex.
"""

import torch
import torch.nn as nn


class RBM(nn.Module):
    """Binary-binary RBM, Eq. (14.energyBB).  Note there is no forward()
    returning a loss: the gradient is Eq. (14.rbmgradient), assigned by hand."""

    def __init__(self, M=784, N=256, k=1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(M, N) * 0.01)
        self.a = nn.Parameter(torch.zeros(M))
        self.b = nn.Parameter(torch.zeros(N))
        self.k = k

    def free_energy(self, x):                       # Eq. (14.freeBB)
        return -(x @ self.a) - torch.nn.functional.softplus(
            x @ self.W + self.b).sum(1)

    def gibbs(self, x):                             # Eq. (14.blockgibbs)
        ph = torch.sigmoid(x @ self.W + self.b)
        h = torch.bernoulli(ph)
        px = torch.sigmoid(h @ self.W.t() + self.a)
        return torch.bernoulli(px)

    def cd_loss(self, x):
        """A surrogate whose gradient equals Eq. (14.cdk).

        The negative sample is detached so that no gradient flows through the
        sampler: the chain supplies configurations, not a differentiable path.
        """
        v = x
        for _ in range(self.k):
            v = self.gibbs(v)
        return self.free_energy(x).mean() - self.free_energy(v.detach()).mean()


model = RBM(784, 256, k=1)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(20):
    for xb, _ in train_loader:
        xb = torch.bernoulli(xb.view(-1, 784))      # binarise the pixels
        loss = model.cd_loss(xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

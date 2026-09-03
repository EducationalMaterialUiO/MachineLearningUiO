"""Chapter 17: listing 6, from the section on gans in pytorch and tensorflow.

Extracted from doc/BookML/chapter17.tex.
"""

def gradient_penalty(D, x_real, x_fake):
    eps = torch.rand(x_real.size(0), *([1] * (x_real.dim() - 1)))
    xhat = (eps * x_real + (1 - eps) * x_fake).requires_grad_(True)
    g = torch.autograd.grad(D(xhat).sum(), xhat, create_graph=True)[0]
    return ((g.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()

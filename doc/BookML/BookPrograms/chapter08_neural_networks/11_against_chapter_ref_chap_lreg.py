"""Chapter 8: listing 11, from the section on against chapter ref chap lreg.

Extracted from doc/BookML/chapter8.tex.
"""

import torch

model = torch.nn.Sequential(
    torch.nn.Linear(L, 200), torch.nn.ReLU(), torch.nn.Linear(200, 1)
).double()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
lossf = torch.nn.MSELoss()
for ep in range(400):
    perm = torch.randperm(ntr)
    for s in range(0, ntr, 40):
        b = perm[s:s + 40]
        opt.zero_grad()
        lossf(model(Xt[b]).squeeze(-1), yt[b]).backward()
        opt.step()

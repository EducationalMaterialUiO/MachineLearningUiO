"""Chapter 15: listing 3, from the section on implementations in the libraries.

Extracted from doc/BookML/chapter15.tex.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, d=784, d_h=16, hidden=256):
        super().__init__()
        self.fc = nn.Linear(d, hidden)
        self.fc_mu = nn.Linear(hidden, d_h)
        self.fc_logvar = nn.Linear(hidden, d_h)
        self.dec = nn.Sequential(nn.Linear(d_h, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d))

    def encode(self, x):                              # Eq. (15.encoder)
        e = F.relu(self.fc(x))
        return self.fc_mu(e), self.fc_logvar(e)

    def reparameterise(self, mu, logvar):             # Eq. (15.reparam)
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        h = self.reparameterise(mu, logvar)
        return self.dec(h), mu, logvar


def negative_elbo(logits, x, mu, logvar):
    """Eq. (15.objective), summed over pixels and averaged over the batch."""
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(1)
    return (rec + kl).mean()


model = VAE()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(30):
    for xb, _ in train_loader:
        xb = torch.bernoulli(xb.view(-1, 784))        # binarise
        logits, mu, logvar = model(xb)
        loss = negative_elbo(logits, xb, mu, logvar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

# generating new data: draw from the prior and decode, no Markov chain needed
with torch.no_grad():
    samples = torch.sigmoid(model.dec(torch.randn(64, 16)))

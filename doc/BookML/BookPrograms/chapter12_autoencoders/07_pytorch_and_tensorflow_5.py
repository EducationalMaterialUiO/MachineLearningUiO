"""Chapter 12: listing 7, from the section on pytorch and tensorflow.

Extracted from doc/BookML/chapter12.tex.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)


class Autoencoder(nn.Module):
    """784 -> 256 -> 128 -> 256 -> 784, an exact mirror; Eq. (12.factor)."""
    def __init__(self, x_dim=784, h_dim=256, z_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(),
            nn.Linear(h_dim, z_dim), nn.BatchNorm1d(z_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(),
            nn.Linear(h_dim, x_dim), nn.Sigmoid(),       # data in [0,1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1, 101):
    model.train()
    total = 0.0
    for data, _ in train_loader:
        inputs = data.view(-1, 784)
        loss = loss_fn(model(inputs), inputs)            # target == input
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item()
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  train {total/len(train_loader):.6f}")

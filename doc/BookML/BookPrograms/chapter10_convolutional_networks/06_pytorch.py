"""Chapter 10: listing 6, from the section on pytorch.

Extracted from doc/BookML/chapter10.tex.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),      # MNIST mean and std
])
train_set = datasets.MNIST(root="./data", train=True,  download=True,
                           transform=transform)
test_set  = datasets.MNIST(root="./data", train=False, download=True,
                           transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=64)


class CNN(nn.Module):
    """Eq. (10.arch) with two conv blocks; 28 -> 14 -> 7 by Eq. (10.outsize)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 'same': 28 x 28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 'same': 14 x 14
        self.pool = nn.MaxPool2d(2, 2)                # halves each time
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))          # (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))          # (N, 64,  7,  7)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)                            # logits, not probabilities


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()      # applies softmax internally
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    running = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()                # Eqs. (10.dconvW)-(10.dmaxpool), by autograd
        optimizer.step()
        running += loss.item()
    print(f"epoch {epoch+1}: loss {running/len(train_loader):.4f}")

model.eval()
correct = total = 0
with torch.no_grad():
    for data, target in test_loader:
        pred = model(data.to(device)).argmax(dim=1)
        correct += (pred == target.to(device)).sum().item()
        total += target.size(0)
print(f"test accuracy {100*correct/total:.2f}%")

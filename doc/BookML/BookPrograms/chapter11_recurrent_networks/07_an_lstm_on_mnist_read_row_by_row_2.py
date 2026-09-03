"""Chapter 11: listing 7, from the section on an lstm on mnist read row by row.

Extracted from doc/BookML/chapter11.tex.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST("./data", train=True, download=True,
                                         transform=transform),
                          batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST("./data", train=False,
                                        transform=transform), batch_size=64)


class LSTMModel(nn.Module):
    """Eqs. (11.lstmf)-(11.lstmh); nn.LSTM already biases b_f, see below."""
    def __init__(self, input_size=28, hidden_size=128, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 28, 28)              # image as a sequence of rows
        out, _ = self.lstm(x)                  # (batch, seq, hidden)
        return self.fc(out[:, -1, :])          # last state -> class scores


model = LSTMModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            pred = model(images.to(device)).argmax(dim=1)
            correct += (pred == labels.to(device)).sum().item()
            total += labels.size(0)
    print(f"epoch {epoch+1}: test accuracy {100*correct/total:.2f}%")

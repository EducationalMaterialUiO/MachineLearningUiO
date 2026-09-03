"""Chapter 11: listing 5, from the section on forecasting a sine wave.

Extracted from doc/BookML/chapter11.tex.
"""

import numpy as np
import torch
import torch.nn as nn

seq_length = 20
data = np.sin(np.linspace(0, 100, 500)).astype("float32")
X = np.stack([data[i:i+seq_length] for i in range(len(data)-seq_length)])
y = data[seq_length:].reshape(-1, 1)
X = torch.tensor(X).unsqueeze(-1)              # (N, T, 1), batch_first
y = torch.tensor(y)
split = int(0.8 * len(X))


class SineRNN(nn.Module):
    """nn.RNN implements Eq. (11.rnn) with tanh by default."""
    def __init__(self, hidden=16):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, h_T = self.rnn(x)                 # out: (N, T, hidden)
        return self.fc(out[:, -1, :])          # read out the last state only


model = SineRNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossfn = nn.MSELoss()
for epoch in range(50):
    model.train()
    opt.zero_grad()
    loss = lossfn(model(X[:split]), y[:split])
    loss.backward()                            # BPTT, Eqs. (11.bptth)-(11.bpttparams)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # Eq. (11.clip)
    opt.step()
    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1}: train loss {loss.item():.5f}")

model.eval()
with torch.no_grad():
    print(f"test loss {lossfn(model(X[split:]), y[split:]).item():.5f}")

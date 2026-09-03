"""Chapter 8: listing 9, from the section on examples.

Extracted from doc/BookML/chapter8.tex.
"""

import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(64, 50),   # Eq. (8.forwardbatch)
                      nn.ReLU(),           # Eq. (8.relu)
                      nn.Linear(50, 10))   # softmax folded into the loss
loss_fn = nn.CrossEntropyLoss()            # Eq. (5.multicrossentropy)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)   # Sec. 4.adam

Xt = torch.tensor(X_train, dtype=torch.float32)
yt = torch.tensor(y_train, dtype=torch.long)
for epoch in range(100):
    optimiser.zero_grad()
    loss = loss_fn(model(Xt), yt)
    loss.backward()                        # this is Eqs. (8.bp1)-(8.bp4)
    optimiser.step()                       # this is Eq. (8.update)

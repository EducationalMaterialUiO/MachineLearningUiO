"""The three recurrent experiments of Chapter 11, in PyTorch.

1.  Forecasting a sine wave one step ahead with ``nn.RNN``, the example of
    Section 11.rnnsine.
2.  Classifying MNIST read row by row with ``nn.LSTM``, the example of
    Section 11.lstmmnist, which is a sequence-to-label task and therefore the
    hardest case of Section 11.bpttvanishing.
3.  A delayed-recall task at increasing lag, comparing the simple RNN against
    the LSTM and the GRU.  Theorem 11.vanishing says the gradient over a lag
    decays exponentially and Eq. (11.lstmjac) says the gated architectures
    break the product; experiment 3 is where that stops being a statement
    about Jacobians and becomes a statement about what the network can learn.

Run ``rnn_tf.py`` for the TensorFlow counterpart.
"""
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

from mnist_data import load_mnist

SEED = 1
out = open("torch.txt", "w", buffering=1)


# ===========================================================================
# 1.  a sine wave, one step ahead
# ===========================================================================
def sine_data(seq_length=20, n=500):
    data = np.sin(np.linspace(0, 100, n)).astype("float32")
    X = np.stack([data[i:i + seq_length] for i in range(len(data) - seq_length)])
    y = data[seq_length:].reshape(-1, 1)
    return torch.tensor(X).unsqueeze(-1), torch.tensor(y)


class SineRNN(nn.Module):
    """nn.RNN implements Eq. (11.rnn) with tanh; we read out the last state."""

    def __init__(self, hidden=16, cell="rnn"):
        super().__init__()
        C = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = C(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        o = self.rnn(x)[0]
        return self.fc(o[:, -1, :])            # sequence-to-label, Eq. (11.tasks)


out.write("=== 1. forecasting a sine wave, Section 11.rnnsine ===\n")
torch.manual_seed(SEED)
X, y = sine_data()
split = int(0.8 * len(X))
model = SineRNN()
n_par = sum(q.numel() for q in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
lossfn = nn.MSELoss()
hist = []
for epoch in range(1, 201):
    model.train()
    perm = torch.randperm(split)
    tot, nb = 0.0, 0
    for i in range(0, split, 32):             # minibatches, not one full batch
        idx = perm[i:i + 32]
        opt.zero_grad(set_to_none=True)
        loss = lossfn(model(X[idx]), y[idx])
        loss.backward()                       # BPTT, Eqs. (11.bptth)-(11.bpttparams)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # Eq. (11.clip)
        opt.step()
        tot += loss.item()
        nb += 1
    model.eval()
    with torch.no_grad():
        te = lossfn(model(X[split:]), y[split:]).item()
    hist.append((epoch, tot / nb, te))
    if epoch % 50 == 0:
        out.write(f"  epoch {epoch:3d}: train mse {tot/nb:.6f}   "
                  f"test mse {te:.6f}\n")
out.write(f"  {n_par} parameters: 16*(1+16+1) = 288 recurrent, 16+1 = 17 read-out\n")
with torch.no_grad():
    pred = model(X[split:]).numpy().ravel()
np.save("sine_hist.npy", np.array(hist))
np.save("sine_pred.npy", pred)
np.save("sine_true.npy", y[split:].numpy().ravel())
out.write(f"  final test mse {hist[-1][2]:.6f}, "
          f"rms error {np.sqrt(hist[-1][2]*2):.4f} on a signal of amplitude 1\n\n")


# ===========================================================================
# 2.  MNIST read row by row
# ===========================================================================
out.write("=== 2. MNIST as 28 rows of 28, Section 11.lstmmnist ===\n")
EPOCHS = int(os.environ.get("CH11_EPOCHS", 3))
(xtr, ytr), (xte, yte) = load_mnist()
Xtr, Ytr = torch.tensor(xtr), torch.tensor(ytr)
Xte, Yte = torch.tensor(xte), torch.tensor(yte)


class RowLSTM(nn.Module):
    def __init__(self, hidden=128, cell="lstm"):
        super().__init__()
        C = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = C(28, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc(self.rnn(x)[0][:, -1, :])


mnist_hist = {}
for cell in ["lstm", "rnn"]:
    torch.manual_seed(SEED)
    net = RowLSTM(128, cell)
    n_par = sum(q.numel() for q in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    rows, t0 = [], time.time()
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(Xtr))
        tot, nb = 0.0, 0
        for i in range(0, len(Xtr) - 64 + 1, 64):
            idx = perm[i:i + 64]
            opt.zero_grad(set_to_none=True)
            loss = lf(net(Xtr[idx]), Ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(Xte[i:i + 1000]).argmax(1)
                              for i in range(0, len(Xte), 1000)])
            acc = (pred == Yte).double().mean().item()
        rows.append((ep, tot / nb, acc, time.time() - t0))
        out.write(f"  {cell:4s} epoch {ep}: loss {tot/nb:.4f}  "
                  f"test accuracy {acc:.4f}  ({time.time()-t0:.0f}s)\n")
    gates = 4 if cell == "lstm" else 1
    out.write(f"  {cell:4s}: {n_par} parameters, of which "
              f"{gates}*128*(28+128+2) = {gates*128*(28+128+2)} in the cell "
              f"(PyTorch carries two bias vectors)\n")
    mnist_hist[cell] = np.array(rows)
np.save("mnist_lstm.npy", mnist_hist["lstm"])
np.save("mnist_rnn.npy", mnist_hist["rnn"])
out.write("\n")


# ===========================================================================
# 3.  the adding problem: does the gating actually buy anything?
# ===========================================================================
out.write("=== 3. the adding problem at increasing sequence length ===\n")
out.write("  Each sequence carries a channel of uniform values and a channel of\n"
          "  markers, with exactly one marker in each half.  The target is half\n"
          "  the sum of the two marked values.  Solving it requires holding one\n"
          "  number across the whole sequence while ignoring everything else,\n"
          "  which is the task Theorem 11.vanishing says is hard.\n\n")


def adding_problem(n, T, rng):
    v = rng.uniform(0, 1, (n, T)).astype("float32")
    m = np.zeros((n, T), dtype="float32")
    i1 = rng.integers(0, T // 2, n)
    i2 = rng.integers(T // 2, T, n)
    m[np.arange(n), i1] = 1.0
    m[np.arange(n), i2] = 1.0
    y = (v[np.arange(n), i1] + v[np.arange(n), i2]).reshape(-1, 1) / 2.0
    return torch.tensor(np.stack([v, m], -1)), torch.tensor(y)


LAGS = [10, 30, 60, 120]
CELLS = ["rnn", "lstm", "gru"]
SEEDS = [0, 1]
table = np.zeros((len(CELLS), len(LAGS)))
base_row = np.zeros(len(LAGS))
out.write("   T   baseline" + "".join(f"{c:>12s}" for c in CELLS) + "\n")
for j, T in enumerate(LAGS):
    Xr, yr = adding_problem(3000, T, np.random.default_rng(3))
    Xv, yv = adding_problem(1000, T, np.random.default_rng(4))
    base = float(((yv - yv.mean()) ** 2).mean())    # predict the mean and stop
    base_row[j] = base
    line = f"  {T:4d} {base:9.4f}"
    for i, cell in enumerate(CELLS):
        runs = []
        for sd in SEEDS:
            torch.manual_seed(sd)
            C = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
            core = C(2, 32, batch_first=True)
            head = nn.Linear(32, 1)
            params = list(core.parameters()) + list(head.parameters())
            opt = torch.optim.Adam(params, lr=5e-3)
            lf = nn.MSELoss()
            for ep in range(40):
                perm = torch.randperm(len(Xr))
                for k in range(0, len(Xr), 64):
                    idx = perm[k:k + 64]
                    opt.zero_grad(set_to_none=True)
                    loss = lf(head(core(Xr[idx])[0][:, -1, :]), yr[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
            with torch.no_grad():
                runs.append(lf(head(core(Xv)[0][:, -1, :]), yv).item())
        table[i, j] = np.mean(runs)
        line += f"{np.mean(runs):12.4f}"
    out.write(line + "\n")
out.write("  mean squared error on held-out sequences, averaged over two seeds;\n"
          "  the baseline column is what predicting the mean and stopping gives\n")
np.save("adding_table.npy", table)
np.save("adding_lags.npy", np.array(LAGS))
np.save("adding_baseline.npy", base_row)
out.close()
print(open("torch.txt").read())

"""Train the network of Section 10.cnnlibraries on MNIST, in PyTorch.

Two runs: the architecture exactly as printed in the chapter -- two
convolutional blocks and a dense layer of 1024 units -- and the same
convolutional trunk with the dense head replaced by global average pooling,
so that the claim in Section 10.tensorflow about where the parameters live can
be tested rather than asserted.

Run ``cnn_tf.py`` for the TensorFlow counterpart; the two must be separate
processes, because importing both frameworks and then training a large model
is not reliable.
"""
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

from mnist_data import load_mnist

EPOCHS = int(os.environ.get("CH10_EPOCHS", 5))
BATCH, SEED = 64, 1

out = open("torch.txt", "w", buffering=1)
(xtr, ytr), (xte, yte) = load_mnist(normalise="standard")
out.write(f"MNIST: {len(xtr)} training and {len(xte)} test images, "
          f"{EPOCHS} epochs, batch {BATCH}, Adam 1e-3, seed {SEED}\n\n")


def model(head="dense"):
    """Eq. (10.arch) with two conv blocks; `head` selects dense or GAP."""
    trunk = [nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)]
    if head == "dense":
        tail = [nn.Flatten(), nn.Linear(64 * 7 * 7, 1024), nn.ReLU(),
                nn.Dropout(0.5), nn.Linear(1024, 10)]
    else:
        tail = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)]
    return nn.Sequential(*(trunk + tail))


def run(head):
    torch.manual_seed(SEED)
    net = model(head)
    n_par = sum(q.numel() for q in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr, Ytr = torch.tensor(xtr[:, None]), torch.tensor(ytr)
    Xte, Yte = torch.tensor(xte[:, None]), torch.tensor(yte)

    hist, t0 = [], time.time()
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(xtr))
        run_loss, nb = 0.0, 0
        for i in range(0, len(xtr) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Xtr[idx]), Ytr[idx])
            loss.backward()                 # Eqs. (10.dconvW)-(10.dmaxpool)
            opt.step()
            run_loss += loss.item()
            nb += 1
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(Xte[i:i + 1000]).argmax(1)
                              for i in range(0, len(Xte), 1000)])
            acc = (pred == Yte).double().mean().item()
        hist.append((ep, run_loss / nb, acc, time.time() - t0))
        out.write(f"  {head:5s} epoch {ep}: loss {hist[-1][1]:.4f}  "
                  f"test accuracy {acc:.4f}  ({hist[-1][3]:.0f}s)\n")
    return np.array(hist), n_par, net


out.write("=== the chapter architecture, dense head ===\n")
h_d, n_d, net_d = run("dense")
out.write("\n=== the same trunk, global average pooling instead ===\n")
h_g, n_g, _ = run("gap")

out.write("\n  head    parameters   final test accuracy   seconds/epoch\n")
for nm, h, n in [("dense", h_d, n_d), ("GAP", h_g, n_g)]:
    out.write(f"  {nm:6s} {n:11d}   {h[-1,2]:19.4f}   {h[-1,3]/EPOCHS:13.0f}\n")
out.write(f"\n  global average pooling reaches {h_g[-1,2]:.4f} against "
          f"{h_d[-1,2]:.4f} with {n_d/n_g:.0f} times fewer parameters\n")

np.save("hist_torch.npy", h_d)
np.save("hist_torch_gap.npy", h_g)
np.save("nparams_torch.npy", np.array([n_d, n_g]))
np.save("filters_torch.npy", net_d[0].weight.detach().numpy()[:, 0])
out.close()
print(open("torch.txt").read())

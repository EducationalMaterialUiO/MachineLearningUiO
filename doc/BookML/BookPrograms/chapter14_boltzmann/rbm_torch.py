"""A restricted Boltzmann machine on MNIST, in PyTorch.

Neither framework ships one, so this is the equations of Section 14.rbm written
in torch tensors: the free energy of Eq. (14.freeenergy), block Gibbs sampling
from Eqs. (14.condh) and (14.condx), and the contrastive-divergence update of
Eq. (14.cdk).  The optimiser and the tensor arithmetic are all the library
supplies, which is the point made in Section 14.rbmlibraries.

Two things are measured that the small-model experiments of Section 14.cdbias
cannot reach, because with 784 visible units the partition function is out of
range:

*  the \\emph{pseudo-likelihood}, which is computable without Z and is the
   standard stand-in for it;
*  the difference between CD-1, CD-10 and persistent CD at the same cost, so
   that the bias measured exactly on nine visible units can be seen to matter,
   or not, on seven hundred and eighty-four.

Run ``rbm_tf.py`` for the TensorFlow counterpart.
"""
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from mnist_data import load_mnist

out = open("torch.txt", "w", buffering=1)
SEED = 0
EPOCHS = int(os.environ.get("CH14_EPOCHS", 5))
M, N, BATCH = 784, 128, 64

(xtr, _), (xte, _) = load_mnist(flat=True)
gen = torch.Generator().manual_seed(123)
Xtr = torch.bernoulli(torch.tensor(xtr), generator=gen)      # binarise once
Xte = torch.bernoulli(torch.tensor(xte), generator=gen)
out.write(f"MNIST binarised: {len(Xtr)} train, {len(Xte)} test, "
          f"M = {M} visible, N = {N} hidden, {EPOCHS} epochs\n\n")


def free_energy(W, a, b, X):
    """Eq. (14.freeenergy), with the hidden units summed out exactly."""
    return -(X @ a) - torch.nn.functional.softplus(X @ W + b).sum(-1)


def pseudo_likelihood(W, a, b, X, g):
    """log PL(x) = M * log sigmoid(F(x_flipped) - F(x)) for one random pixel.

    The partition function cancels between the two free energies, which is why
    this is computable when Eq. (14.loglik) is not.
    """
    i = torch.randint(0, M, (len(X),), generator=g)
    Xf = X.clone()
    Xf[torch.arange(len(X)), i] = 1.0 - Xf[torch.arange(len(X)), i]
    return float((M * torch.nn.functional.logsigmoid(
        free_energy(W, a, b, Xf) - free_energy(W, a, b, X))).mean())


def gibbs(W, a, b, V, k, g):
    """k sweeps of block Gibbs, Eq. (14.blockgibbs)."""
    for _ in range(k):
        ph = torch.sigmoid(V @ W + b)
        H = torch.bernoulli(ph, generator=g)
        pv = torch.sigmoid(H @ W.T + a)
        V = torch.bernoulli(pv, generator=g)
    return V


def train(mode, k, eta=0.05, momentum=0.5):
    """`mode` is "cd" for CD-k started at the data, "pcd" for a persistent chain."""
    g = torch.Generator().manual_seed(SEED)
    W = 0.01 * torch.randn(M, N, generator=g)
    a = torch.zeros(M)
    b = torch.zeros(N)
    vW = torch.zeros_like(W)
    va = torch.zeros_like(a)
    vb = torch.zeros_like(b)
    chain = torch.bernoulli(0.5 * torch.ones(BATCH, M), generator=g)
    hist, t0 = [], time.time()
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr) - BATCH + 1, BATCH):
            X = Xtr[perm[i:i + BATCH]]
            ph_pos = torch.sigmoid(X @ W + b)                # positive phase
            start = X if mode == "cd" else chain
            V = gibbs(W, a, b, start, k, g)                  # negative phase
            if mode == "pcd":
                chain = V
            ph_neg = torch.sigmoid(V @ W + b)
            dW = X.T @ ph_pos / len(X) - V.T @ ph_neg / len(V)
            da = X.mean(0) - V.mean(0)
            db = ph_pos.mean(0) - ph_neg.mean(0)
            vW = momentum * vW + eta * dW                    # Eq. (14.cdk)
            va = momentum * va + eta * da
            vb = momentum * vb + eta * db
            W, a, b = W + vW, a + va, b + vb
        pl = pseudo_likelihood(W, a, b, Xte[:2000],
                               torch.Generator().manual_seed(99))
        rec = float(((torch.sigmoid(torch.sigmoid(Xte[:2000] @ W + b) @ W.T + a)
                      - Xte[:2000]) ** 2).mean())
        hist.append((ep, pl, rec, time.time() - t0))
        out.write(f"  {mode}-{k:<2d} epoch {ep}: pseudo-likelihood {pl:9.2f}   "
                  f"reconstruction mse {rec:.5f}   ({hist[-1][3]:.0f}s)\n")
    return np.array(hist), (W, a, b)


out.write("=== the three training signals of Section 14.cd, on real data ===\n")
runs = {}
for name, mode, k in [("CD-1", "cd", 1), ("CD-10", "cd", 10),
                      ("PCD-1", "pcd", 1)]:
    h, params = train(mode, k)
    runs[name] = h
    if name == "CD-1":
        W1, a1, b1 = params
    if name == "PCD-1":
        Wp, ap, bp = params
    out.write("\n")

out.write("  signal   final pseudo-likelihood   final reconstruction mse"
          "   seconds/epoch\n")
for name, h in runs.items():
    out.write(f"  {name:7s} {h[-1,1]:23.2f}   {h[-1,2]:24.5f}"
              f"   {h[-1,3]/EPOCHS:13.0f}\n")
out.write("  pseudo-likelihood is per image and needs no partition function;\n"
          "  larger is better and the scale is set by the 784 pixels\n\n")
for name, h in runs.items():
    np.save(f"hist_{name.replace('-','').lower()}.npy", h)

# ---------------------------------------------------------------------------
# what the machine has learned: filters and samples
# ---------------------------------------------------------------------------
out.write("=== samples from the trained machine ===\n")
g = torch.Generator().manual_seed(5)
V = torch.bernoulli(0.5 * torch.ones(8, M), generator=g)
snapshots = []
for s in range(1, 1001):
    V = gibbs(Wp, ap, bp, V, 1, g)
    if s in (1, 10, 100, 1000):
        pv = torch.sigmoid(torch.sigmoid(V @ Wp + bp) @ Wp.T + ap)
        snapshots.append(pv.numpy())
        out.write(f"  after {s:5d} Gibbs sweeps: mean free energy "
                  f"{float(free_energy(Wp, ap, bp, V).mean()):9.2f}\n")
out.write("  free energy falls as the chain moves from noise towards the modes\n"
          "  of the model, which is Eq. (14.freeenergy) doing its job\n")
np.save("samples.npy", np.array(snapshots))
np.save("filters.npy", Wp[:, :64].T.numpy())
out.close()
print(open("torch.txt").read())

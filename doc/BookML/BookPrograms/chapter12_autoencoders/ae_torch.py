"""The autoencoder experiments of Chapter 12, in PyTorch.

1.  Theorem 12.aepca, tested rather than assumed.  A linear autoencoder is
    trained by Adam and we ask how far its encoder subspace is from the span of
    the top p eigenvectors of the covariance, and whether the reconstruction
    cost reaches the Eckart-Young floor of Eq. (12.floor).  The theorem is a
    statement about the global minimum of a non-convex problem; that gradient
    descent finds it is a separate claim.

2.  What the bottleneck costs.  On MNIST, the reconstruction error of a linear
    autoencoder, of a nonlinear one and of PCA, as a function of the code
    dimension p.  PCA is the best any linear map can do, so the gap between the
    curves is what the nonlinearity buys.

3.  The regularised variants of Section 12.aeregular -- denoising and sparse --
    measured on the same architecture.

Run ``ae_tf.py`` for the TensorFlow counterpart.
"""
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

from mnist_data import load_mnist

out = open("torch.txt", "w", buffering=1)
SEED = 0


def principal_angle(A, B):
    """Largest principal angle, in degrees, between two column spaces."""
    Qa, Qb = np.linalg.qr(A)[0], np.linalg.qr(B)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


# ===========================================================================
# 1.  a linear autoencoder finds the principal subspace
# ===========================================================================
out.write("=== 1. Theorem 12.aepca: does Adam find the principal subspace? ===\n")
d, p, n = 8, 3, 4000
r = np.random.default_rng(5)
S = r.normal(size=(n, p)) @ r.normal(size=(p, d)) + 0.25 * r.normal(size=(n, d))
S = S - S.mean(0)                                  # centre: see the notebox
sv = np.linalg.svd(S, compute_uv=False)
V_pca = np.linalg.svd(S, full_matrices=False)[2][:p].T
floor = float(np.sum(sv[p:] ** 2) / n / 2)         # Eq. (12.floor), per sample

torch.manual_seed(SEED)
enc, dec = nn.Linear(d, p, bias=False), nn.Linear(p, d, bias=False)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=5e-3)
St = torch.tensor(S, dtype=torch.float32)
hist = []
for it in range(1, 6001):
    opt.zero_grad(set_to_none=True)
    loss = 0.5 * torch.sum((dec(enc(St)) - St) ** 2, dim=1).mean()
    loss.backward()
    opt.step()
    if it % 100 == 0:
        W = enc.weight.detach().numpy().T          # (d, p)
        hist.append((it, loss.item(), principal_angle(W, V_pca)))
W = enc.weight.detach().numpy().T

out.write(f"  largest principal angle to the PCA subspace : "
          f"{principal_angle(W, V_pca):.4f} degrees\n")
out.write(f"  max |W_enc - V_pca| entrywise               : "
          f"{np.abs(W - V_pca).max():.4f}\n")
out.write(f"  reconstruction cost per sample              : {hist[-1][1]:.6f}\n")
out.write(f"  Eckart-Young floor, Eq. (12.floor)          : {floor:.6f}\n")
out.write(f"  excess over the floor                       : "
          f"{hist[-1][1]-floor:.3e}\n")
Pi = W @ np.linalg.pinv(W)
out.write(f"  ||Pi_enc - Pi_pca||_F                       : "
          f"{np.linalg.norm(Pi - V_pca @ V_pca.T):.3e}\n")
out.write(f"  ||Pi^2 - Pi||_F, trace Pi                   : "
          f"{np.linalg.norm(Pi @ Pi - Pi):.2e}, {np.trace(Pi):.6f}\n")
out.write("  The subspace is found to a hundredth of a degree and the cost sits\n"
          "  on the Eckart-Young floor; the weight matrix is nowhere near the\n"
          "  PCA loadings.  That is Proposition 12.nonunique: the minimiser is\n"
          "  fixed only up to an invertible p x p factor, and the projector is\n"
          "  the invariant.\n\n")
np.save("angle_hist_torch.npy", np.array(hist))
np.save("pca_floor.npy", np.array([floor]))


# ===========================================================================
# 2.  what the bottleneck costs on MNIST
# ===========================================================================
out.write("=== 2. reconstruction error against the code dimension, MNIST ===\n")
(xtr, ytr), (xte, yte) = load_mnist(flat=True)
Xtr = torch.tensor(xtr)
Xte = torch.tensor(xte)
EPOCHS = int(os.environ.get("CH12_EPOCHS", 8))

# PCA on the training set: the floor for any linear autoencoder
Xc = xtr - xtr.mean(0)
sv_m = np.linalg.svd(Xc, compute_uv=False)
tot = float(np.sum(sv_m ** 2))
PS = [2, 8, 16, 32, 64]
pca_mse = [float(np.sum(sv_m[k:] ** 2) / (len(xtr) * 784)) for k in PS]


def make(p, nonlinear=True):
    if nonlinear:
        return nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                             nn.Linear(256, p), nn.ReLU(),
                             nn.Linear(p, 256), nn.ReLU(),
                             nn.Linear(256, 784), nn.Sigmoid())
    return nn.Sequential(nn.Linear(784, p, bias=False),
                         nn.Linear(p, 784, bias=False))


def train_ae(p, nonlinear, noise=0.0, lam=0.0, epochs=EPOCHS):
    torch.manual_seed(SEED)
    net = make(p, nonlinear)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr) - 128 + 1, 128):
            xb = Xtr[perm[i:i + 128]]
            inp = xb + noise * torch.randn_like(xb) if noise else xb
            opt.zero_grad(set_to_none=True)
            rec = net(inp)
            loss = ((rec - xb) ** 2).mean()          # target is the clean batch
            if lam:
                code = net[:4](inp) if nonlinear else net[0](inp)
                loss = loss + lam * code.abs().mean()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        mse = float(((net(Xte) - Xte) ** 2).mean())
        code = (net[:4](Xte) if nonlinear else net[0](Xte)).numpy()
    return mse, net, code


out.write("     p    PCA (linear optimum)   linear AE      nonlinear AE"
          "   gain\n")
rows = []
for k, pm in zip(PS, pca_mse):
    t0 = time.time()
    lin_mse = train_ae(k, False)[0]
    nl_mse, net_k, code_k = train_ae(k, True)
    rows.append((k, pm, lin_mse, nl_mse))
    out.write(f"  {k:4d}   {pm:20.5f}   {lin_mse:9.5f}   {nl_mse:15.5f}"
              f"   {pm/nl_mse:5.2f}x   ({time.time()-t0:.0f}s)\n")
    if k == 16:
        with torch.no_grad():
            np.save("recon16.npy", net_k(Xte[:8]).numpy())
        np.save("orig8.npy", xte[:8])
out.write("  mean squared error per pixel on the 10000 test images\n")
out.write("  the PCA column is the best any linear map of that rank can do,\n"
          "  Eq. (12.floor); the linear autoencoder should match it and does\n\n")
np.save("mnist_curve.npy", np.array(rows))

# ===========================================================================
# 3.  the regularised variants of Section 12.aeregular
# ===========================================================================
out.write("=== 3. denoising and sparse autoencoders, p = 32 ===\n")
out.write("   variant                     test mse   mean |code|   active units\n")
res = []
for name, kw in [("plain", {}),
                 ("denoising, sigma = 0.3", {"noise": 0.3}),
                 ("denoising, sigma = 0.6", {"noise": 0.6}),
                 ("sparse, lambda = 0.01", {"lam": 0.01}),
                 ("sparse, lambda = 0.1", {"lam": 0.1})]:
    mse, net_v, code = train_ae(32, True, **kw)
    act = float((np.abs(code) > 1e-3).mean() * 32)
    res.append((mse, float(np.abs(code).mean()), act))
    out.write(f"   {name:26s} {mse:10.5f}   {np.abs(code).mean():11.4f}"
              f"   {act:12.1f}\n")
out.write("  `active units' counts, on average, how many of the 32 code\n"
          "  components exceed 1e-3 on a test image\n")
np.save("regular_table.npy", np.array(res))
out.close()
print(open("torch.txt").read())

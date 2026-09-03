"""The variational autoencoder experiments of Chapter 15, in PyTorch.

1.  Training on binarised MNIST at several latent dimensions, reporting the
    ELBO of Eq. (15.objective) and the number of active latent coordinates.
2.  The gap of Theorem 15.elbo, measured.  The theorem says
    $\\log p(\\bm{x}) = \\mathrm{ELBO} + D_{KL}(q_{\\bm{\\phi}}\\|p(\\bm{h}\\mid\\bm{x}))$,
    and the divergence is not computable because the true posterior is not.
    But $\\log p$ can be estimated from above by importance sampling with the
    encoder as proposal -- the $K$-sample bound $\\mathcal{L}_K$ -- so the gap
    can be squeezed between $\\mathcal{L}_1=\\mathrm{ELBO}$ and $\\mathcal{L}_K$.
3.  Posterior collapse against the weight $\\beta$ on the KL term, extending the
    latent-dimension study of Section 15.vaepathologies.

Run ``vae_tf.py`` for the TensorFlow counterpart.
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
EPOCHS = int(os.environ.get("CH15_EPOCHS", 6))
D, HID, BATCH = 784, 256, 128

(xtr, _), (xte, _) = load_mnist(flat=True)
g0 = torch.Generator().manual_seed(123)
Xtr = torch.bernoulli(torch.tensor(xtr), generator=g0)
Xte = torch.bernoulli(torch.tensor(xte), generator=g0)
out.write(f"MNIST binarised: {len(Xtr)} train, {len(Xte)} test, d = {D}, "
          f"{EPOCHS} epochs, Adam 1e-3, batch {BATCH}\n\n")


class VAE(nn.Module):
    """Eqs. (15.encoder) and (15.reparam) with a Bernoulli decoder."""

    def __init__(self, dh):
        super().__init__()
        self.dh = dh
        self.enc = nn.Sequential(nn.Linear(D, HID), nn.ReLU(),
                                 nn.Linear(HID, 2 * dh))
        self.dec = nn.Sequential(nn.Linear(dh, HID), nn.ReLU(),
                                 nn.Linear(HID, D))

    def encode(self, x):
        o = self.enc(x)
        return o[:, :self.dh], o[:, self.dh:]

    def forward(self, x, beta=1.0):
        mu, logvar = self.encode(x)
        h = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        logits = self.dec(h)
        rec = -nn.functional.binary_cross_entropy_with_logits(
            logits, x, reduction="none").sum(-1)
        kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1.0)
        return rec, kl, mu, logvar


def iwae_bound(model, X, K, chunk=100, kblock=32):
    """L_K = E log (1/K) sum_k w_k, Eq. (15.iwae); L_1 is the ELBO.

    The K samples are drawn in blocks so that memory stays bounded however
    large K is; the log-weights are concatenated before the logsumexp.
    """
    tot = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            x = X[i:i + chunk]
            mu0, logvar0 = model.encode(x)
            lws = []
            done = 0
            while done < K:
                kb = min(kblock, K - done)
                mu = mu0.unsqueeze(0).expand(kb, -1, -1)
                logvar = logvar0.unsqueeze(0).expand(kb, -1, -1)
                e = torch.randn_like(mu)
                h = mu + torch.exp(0.5 * logvar) * e
                logits = model.dec(h)
                logpxh = -nn.functional.binary_cross_entropy_with_logits(
                    logits, x.unsqueeze(0).expand(kb, -1, -1),
                    reduction="none").sum(-1)
                logqh = (-0.5 * (e ** 2 + logvar + np.log(2 * np.pi))).sum(-1)
                logph = (-0.5 * (h ** 2 + np.log(2 * np.pi))).sum(-1)
                lws.append(logpxh + logph - logqh)
                done += kb
            lw = torch.cat(lws, 0)
            tot.append(torch.logsumexp(lw, 0) - np.log(K))
    return float(torch.cat(tot).mean())


def train(dh, beta=1.0, epochs=EPOCHS):
    torch.manual_seed(SEED)
    m = VAE(dh)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr) - BATCH + 1, BATCH):
            x = Xtr[perm[i:i + BATCH]]
            rec, kl, _, _ = m(x)
            loss = -(rec - beta * kl.sum(-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        rec, kl, mu, logvar = m(Xte)
        elbo = float((rec - kl.sum(-1)).mean())
        kl_j = kl.mean(0).numpy()
    return m, elbo, kl_j, time.time() - t0


# ===========================================================================
# 1.  latent dimension
# ===========================================================================
out.write("=== 1. the latent dimension, on the full MNIST set ===\n")
out.write("   d_h   test ELBO   active units (KL_j > 0.01)   mean KL per active"
          "   seconds\n")
rows, models = [], {}
for dh in [2, 5, 10, 20, 50]:
    m, elbo, kl_j, dt = train(dh)
    act = int((kl_j > 0.01).sum())
    rows.append((dh, elbo, act, float(kl_j[kl_j > 0.01].mean())))
    models[dh] = m
    out.write(f"  {dh:4d}  {elbo:10.4f}  {act:26d}   {rows[-1][3]:17.4f}"
              f"   {dt:7.0f}\n")
np.save("dh_table.npy", np.array(rows))
out.write("\n")

# ===========================================================================
# 2.  the gap of Theorem 15.elbo
# ===========================================================================
out.write("=== 2. the ELBO gap, Theorem 15.elbo, measured ===\n")
out.write("  L_K is the K-sample importance-weighted bound; L_1 is the ELBO and\n"
          "  L_K increases monotonically to log p(x) as K grows.  The difference\n"
          "  L_K - L_1 is a lower bound on the KL divergence between q and the\n"
          "  true posterior.\n\n")
out.write("   d_h      L_1 (ELBO)        L_16        L_128       L_1024"
          "    L_1024 - L_1\n")
gap_rows = []
Xg = Xte[:500]
for dh in [2, 10, 50]:
    m = models[dh]
    vals = [iwae_bound(m, Xg, K) for K in [1, 16, 128, 1024]]
    gap_rows.append([dh] + vals + [vals[-1] - vals[0]])
    out.write(f"  {dh:4d}  {vals[0]:12.4f} {vals[1]:12.4f} {vals[2]:12.4f}"
              f" {vals[3]:12.4f} {vals[-1]-vals[0]:14.4f}\n")
out.write("  the gap is the price of the approximate posterior, and it is\n"
          "  several nats: the ELBO is not a tight estimate of log p(x)\n\n")
np.save("gap_table.npy", np.array(gap_rows))

# ===========================================================================
# 3.  posterior collapse against beta
# ===========================================================================
out.write("=== 3. posterior collapse against the KL weight beta ===\n")
out.write("   beta   test ELBO   active units of 50   mean KL per active\n")
beta_rows = []
for beta in [0.25, 0.5, 1.0, 2.0, 4.0]:
    m, elbo, kl_j, _ = train(50, beta=beta, epochs=max(4, EPOCHS // 2))
    act = int((kl_j > 0.01).sum())
    mk = float(kl_j[kl_j > 0.01].mean()) if act else 0.0
    beta_rows.append((beta, elbo, act, mk))
    out.write(f"  {beta:5.2f}  {elbo:10.4f}  {act:18d}   {mk:18.4f}\n")
    if beta == 1.0:
        np.save("kl_per_unit.npy", np.sort(kl_j)[::-1].copy())
out.write("  beta > 1 buys a smaller code at the price of the ELBO, which is\n"
          "  the beta-VAE trade; beta < 1 keeps every unit alive and overfits\n"
          "  the reconstruction term\n")
np.save("beta_table.npy", np.array(beta_rows))

# reconstructions and prior samples for the figure
m = models[10]
with torch.no_grad():
    mu, _ = m.encode(Xte[:8])
    np.save("recon.npy", torch.sigmoid(m.dec(mu)).numpy())
    np.save("orig.npy", Xte[:8].numpy())
    torch.manual_seed(4)
    np.save("prior_samples.npy",
            torch.sigmoid(m.dec(torch.randn(8, 10))).numpy())
out.close()
print(open("torch.txt").read())

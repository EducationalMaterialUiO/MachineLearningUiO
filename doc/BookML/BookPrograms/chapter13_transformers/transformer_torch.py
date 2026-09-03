"""The transformer experiments of Chapter 13, in PyTorch.

1.  Proposition 13.variance and the $1/\\sqrt{d_k}$ of Eq. (13.attention),
    measured: the variance of the logits, the entropy of the attention rows and
    -- the quantities that actually matter -- the largest attention weight
    and the norm of the softmax Jacobian, with and without the scaling, as
    $d_k$ grows.
2.  Theorem 13.perm, tested: permuting the tokens must permute the output
    exactly, and adding the positional encoding of Eq. (13.posenc) must destroy
    that property.
3.  Associative recall, extending Table 13.recall to L = 8 and adding the
    LSTM of Chapter 11 and a learned positional encoding to the comparison.

Run ``transformer_tf.py`` for the TensorFlow counterpart of experiment 3.
"""
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

out = open("torch.txt", "w", buffering=1)
NSYM = 8
SEED = 0


# ===========================================================================
# 1.  why the 1/sqrt(d_k)
# ===========================================================================
out.write("=== 1. Proposition 13.variance and the 1/sqrt(d_k) scaling ===\n")
out.write("  Q and K have independent standard normal entries, so a logit is a\n"
          "  sum of d_k products and has variance d_k.  We measure that, the\n"
          "  entropy of the attention rows, the largest weight in a row, and the\n"
          "  norm of the softmax Jacobian diag(a) - a a^T, which is what decides\n"
          "  how much gradient survives the softmax.\n\n")
out.write("   d_k   var(logit)   entropy scaled   entropy unscaled"
          "   max weight sc/unsc   ||dA/dS|| sc/unsc\n")
rows = []
for d_k in [4, 16, 64, 256, 1024]:
    g = torch.Generator().manual_seed(SEED)
    n = 64
    Q = torch.randn(n, d_k, generator=g)
    K = torch.randn(n, d_k, generator=g)
    var = float((Q @ K.T).var())
    res = []
    for scaled in [True, False]:
        S = Q @ K.T
        if scaled:
            S = S / np.sqrt(d_k)                 # Eq. (13.attention)
        A = torch.softmax(S, dim=-1)
        ent = float(-(A * torch.log(A + 1e-30)).sum(-1).mean())
        # the softmax Jacobian of one row is diag(a) - a a^T; its size is what
        # decides how much gradient survives the softmax
        jac = float(torch.stack([
            (torch.diag(a) - torch.outer(a, a)).norm() for a in A]).mean())
        res.append((ent, float(A.max(-1).values.mean()), jac))
    rows.append((d_k, var, res[0][0], res[1][0], res[0][1], res[1][1],
                 res[0][2], res[1][2]))
    out.write(f"  {d_k:5d}   {var:10.2f}   {res[0][0]:14.4f}   {res[1][0]:16.4f}"
              f"   {res[0][1]:9.3f}/{res[1][1]:.3f}"
              f"   {res[0][2]:12.4f}/{res[1][2]:.4f}\n")
out.write(f"  the entropy of a uniform row over 64 keys is "
          f"log 64 = {np.log(64):.4f}\n")
out.write("  unscaled, the rows collapse onto a single key -- the mean largest\n"
          "  weight goes to one -- and the softmax Jacobian goes to zero with\n"
          "  them, so no gradient survives.  Scaled, both are almost independent\n"
          "  of d_k, which is exactly what dividing a variance-d_k logit by\n"
          "  sqrt(d_k) is for.\n\n")
np.save("scaling_table.npy", np.array(rows))


# ===========================================================================
# 2.  permutation equivariance, Theorem 13.perm
# ===========================================================================
out.write("=== 2. Theorem 13.perm, tested ===\n")
torch.manual_seed(SEED)
n, d, H = 9, 32, 4
mha = nn.MultiheadAttention(d, H, batch_first=True)
X = torch.randn(1, n, d)
perm = torch.randperm(n)
Y = mha(X, X, X)[0]
Yp = mha(X[:, perm], X[:, perm], X[:, perm])[0]
out.write(f"  max |MHA(PX) - P MHA(X)|                      : "
          f"{float((Yp - Y[:, perm]).abs().max()):.3e}\n")

pe = torch.tensor(np.stack([
    [np.sin(pos / 10000 ** (2 * (i // 2) / d)) if i % 2 == 0
     else np.cos(pos / 10000 ** (2 * (i // 2) / d)) for i in range(d)]
    for pos in range(n)]), dtype=torch.float32)[None]
Z, Zp = mha(X + pe, X + pe, X + pe)[0], mha(
    X[:, perm] + pe, X[:, perm] + pe, X[:, perm] + pe)[0]
out.write(f"  with positional encoding, Eq. (13.posenc)     : "
          f"{float((Zp - Z[:, perm]).abs().max()):.3e}\n")
out.write("  the first line is zero to rounding, the second is not: attention\n"
          "  is a set operation until the positions are written into the tokens\n\n")


# ===========================================================================
# 3.  associative recall
# ===========================================================================
out.write("=== 3. associative recall, extending Table 13.recall ===\n")
out.write("  the L keys must be distinct, so L is at most the number of\n"
          "  symbols; with NSYM = 8 the longest sequence is T = 17.\n\n")
LENGTHS = [2, 4, 8]


def make_batch(n_b, L, rng):
    T = 2 * L + 1
    X = np.zeros((n_b, T), dtype="int64")
    y = np.zeros(n_b, dtype="int64")
    for b in range(n_b):
        keys = rng.permutation(NSYM)[:L]
        vals = rng.integers(0, NSYM, L)
        X[b, 0:2 * L:2] = keys
        X[b, 1:2 * L:2] = vals
        q = rng.integers(0, L)
        X[b, 2 * L] = keys[q]
        y[b] = vals[q]
    return torch.tensor(X), torch.tensor(y)


def sinusoidal(n, d):
    """Eq. (13.posenc), as a tensor."""
    pe = np.zeros((n, d))
    for pos in range(n):
        for i in range(d):
            a = pos / 10000 ** (2 * (i // 2) / d)
            pe[pos, i] = np.sin(a) if i % 2 == 0 else np.cos(a)
    return torch.tensor(pe, dtype=torch.float32)


class TransformerRecall(nn.Module):
    """One or more blocks of Eq. (13.block), reading out the last position."""

    def __init__(self, d=32, H=2, d_ff=64, n_blocks=1, n_ctx=64,
                 learned_pos=False):
        super().__init__()
        self.emb = nn.Embedding(NSYM, d)
        if learned_pos:
            self.pos = nn.Parameter(0.02 * torch.randn(n_ctx, d))
        else:
            self.register_buffer("pos", sinusoidal(n_ctx, d))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, H, d_ff, dropout=0.0,
                                       activation="gelu", batch_first=True,
                                       norm_first=True)
            for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, NSYM)

    def forward(self, idx):
        x = self.emb(idx) + self.pos[:idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x[:, -1]))


class RecurrentRecall(nn.Module):
    def __init__(self, cell="lstm", h=48):
        super().__init__()
        self.emb = nn.Embedding(NSYM, 32)
        C = {"rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        self.rnn = C(32, h, batch_first=True)
        self.head = nn.Linear(h, NSYM)

    def forward(self, idx):
        return self.head(self.rnn(self.emb(idx))[0][:, -1])


def run(make_model, L, n_iter=3000, seeds=(0, 1)):
    accs, npar = [], 0
    for sd in seeds:
        torch.manual_seed(sd)
        rng = np.random.default_rng(sd)
        net = make_model()
        npar = sum(q.numel() for q in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        lf = nn.CrossEntropyLoss()
        for it in range(n_iter):
            Xb, yb = make_batch(64, L, rng)
            opt.zero_grad(set_to_none=True)
            lf(net(Xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        Xv, yv = make_batch(400, L, np.random.default_rng(99))
        with torch.no_grad():
            accs.append(float((net(Xv).argmax(1) == yv).double().mean()))
    return float(np.mean(accs)), npar, net


out.write("    L    T   model                                 params"
          "   test accuracy (2 seeds)\n")
table = {}
for L in LENGTHS:
    t0 = time.time()
    for name, mk in [
            ("transformer, 1 block",
             lambda: TransformerRecall(n_blocks=1)),
            ("transformer, 2 blocks",
             lambda: TransformerRecall(n_blocks=2)),
            ("transformer, 2 blocks, learned pos.",
             lambda: TransformerRecall(n_blocks=2, learned_pos=True)),
            ("LSTM (Chapter 11)", lambda: RecurrentRecall("lstm")),
            ("RNN (Chapter 11)", lambda: RecurrentRecall("rnn"))]:
        acc, npar, net = run(mk, L)
        table[(name, L)] = acc
        out.write(f"  {L:3d}  {2*L+1:3d}   {name:35s} {npar:7d}   {acc:.3f}\n")
        if name == "transformer, 2 blocks" and L == 4:
            best = net
    out.write(f"       ({time.time()-t0:.0f}s)\n")
out.write(f"  chance = {1/NSYM:.3f}\n")
np.save("recall_table.npy", np.array(
    [[table[(nm, L)] for L in LENGTHS]
     for nm in ["transformer, 1 block", "transformer, 2 blocks",
                "transformer, 2 blocks, learned pos.",
                "LSTM (Chapter 11)", "RNN (Chapter 11)"]]))
np.save("recall_lengths.npy", np.array(LENGTHS))
out.close()
print(open("torch.txt").read())

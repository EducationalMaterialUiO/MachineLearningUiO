"""The one-dimensional Ising chain as a test case for neural networks.

Chapter 3 fitted the Hamiltonian

    H(s) = - J sum_j s_j s_{j+1}

by ordinary least squares, Ridge and Lasso, using the L^2 outer-product
features X[i, jL+k] = s_j s_k.  Those features are what makes the problem
linear; they were put in by hand.

This program asks the harder question.  Given only the L raw spins, can a
model learn the energy?  No linear model can: the answer is exactly zero,
and the reason is a symmetry.  A neural network can, and there is a closed
form for the network that does it, with 2L hidden ReLU units and weights
+-1.

It also needs the coupling matrices written by ising_regression.py in
BookPrograms/chapter03_linear_regression, so that the two sets of estimators
are compared on identical training and test rows.  Output goes to standard
output; the figures are made by BookFigures/ch03_ch08_ising_figures.py from
the .npy files saved here.
"""
import numpy as np

rng = np.random.default_rng(2718)

L, n, Jtrue = 40, 10000, 1.0
spins = rng.choice([-1, 1], size=(n, L))
energies = -Jtrue * np.einsum("ij,ij->i", spins, np.roll(spins, 1, axis=1))

ntr = 400
# exactly the split of ising_regression.py: the same generator is drawn in the
# same order, so tr and te below are the same rows the linear estimators saw
idx = rng.permutation(n)
tr, te = idx[:ntr], idx[ntr:]
Str, Ste = spins[tr].astype(float), spins[te].astype(float)
ytr, yte = energies[tr].astype(float), energies[te].astype(float)


def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


line = "-" * 74

# ===========================================================================
print("=== 1. why no linear model can read the raw spins ===")
# ===========================================================================
print("""
  The spin configurations are drawn independently and uniformly, so the
  law of s is invariant under flipping any single spin s_j -> -s_j.  The
  energy H = -J sum_k s_k s_{k+1} contains s_j only through the two terms
  s_{j-1}s_j and s_j s_{j+1}, and flipping s_j changes the sign of both
  and of s_j itself.  Hence E[H s_j] = -E[H s_j] = 0 for every j: the
  energy is uncorrelated with every single spin, although it is a
  deterministic function of all of them.""")

corr = np.array([np.corrcoef(spins[:, j], energies)[0, 1] for j in range(L)])
print("\n  measured over %d configurations:" % n)
print("    max_j |corr(s_j, H)|                        : %.4f" % np.abs(corr).max())
print("    max_j |mean(H s_j)|                         : %.4f"
      % np.abs((energies[:, None] * spins).mean(axis=0)).max())

Sfull = spins.astype(float)
Xa = np.column_stack([np.ones(n), Sfull])
theta_raw = np.linalg.lstsq(Xa, energies, rcond=None)[0]
print("\n  ordinary least squares on the L raw spins plus an intercept,")
print("  fitted on all %d configurations:" % n)
print("    intercept                                   : %+.4f" % theta_raw[0])
print("    max_j |theta_j|                             : %.4f"
      % np.abs(theta_raw[1:]).max())
print("    R^2                                         : %.6f"
      % r2(energies, Xa @ theta_raw))
print("""
  The intercept reproduces the mean energy and nothing else is learned.
  Ridge and Lasso can only shrink these coefficients further, so they
  do no better.  The linear model on raw spins is not merely weak; it is
  exactly as good as predicting the mean.""")

# ===========================================================================
print("\n=== 2. the exact ReLU network, Proposition 8.isingexact ===")
# ===========================================================================
print("""
  For s_j, s_k in {-1,+1},
      s_j s_k = |s_j + s_k| - 1,
  because the sum is +-2 when the spins agree and 0 when they differ.
  Since |z| = ReLU(z) + ReLU(-z),
      H(s) = -J sum_j ( |s_j + s_{j+1}| - 1 )
           = J L - J sum_j [ ReLU(s_j + s_{j+1}) + ReLU(-s_j - s_{j+1}) ],
  which is a one-hidden-layer ReLU network with 2L hidden units, all
  first-layer weights equal to +-1, all first-layer biases zero, all
  output weights equal to -J and output bias J L.""")

W1 = np.zeros((L, 2 * L))
for j in range(L):
    W1[j, 2 * j] = 1.0
    W1[(j + 1) % L, 2 * j] = 1.0
    W1[j, 2 * j + 1] = -1.0
    W1[(j + 1) % L, 2 * j + 1] = -1.0
b1 = np.zeros(2 * L)
W2 = np.full(2 * L, -Jtrue)
b2 = Jtrue * L

pred_exact = np.maximum(Sfull @ W1 + b1, 0.0) @ W2 + b2
print("\n  the hand-built network evaluated on all %d configurations:" % n)
print("    hidden units                                : %d" % (2 * L))
print("    distinct first-layer weights                : %s"
      % np.unique(W1).tolist())
print("    max |network(s) - H(s)|                     : %.3e"
      % np.abs(pred_exact - energies).max())
print("    R^2                                         : %.10f"
      % r2(energies, pred_exact))
print("""
  Exactly zero error, not approximately zero.  The universal
  approximation theorem of Section 8.5 guarantees that some network comes
  arbitrarily close; here the construction is finite and exact, and the
  width 2L it needs is the same piecewise-linear counting that appears in
  the proof.""")

# ===========================================================================
print("\n=== 3. a network trained from scratch on the raw spins ===")
# ===========================================================================


class Net:
    """One hidden layer, ReLU, scalar output, trained by Adam."""

    def __init__(self, nin, nh, seed):
        g = np.random.default_rng(seed)
        self.W1 = g.normal(0, np.sqrt(2.0 / nin), (nin, nh))
        self.b1 = np.zeros(nh)
        self.W2 = g.normal(0, np.sqrt(2.0 / nh), nh)
        self.b2 = 0.0
        self.m = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2)] + [0.0]
        self.v = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2)] + [0.0]
        self.t = 0

    def forward(self, X):
        self.z = X @ self.W1 + self.b1
        self.a = np.maximum(self.z, 0.0)
        return self.a @ self.W2 + self.b2

    def predict(self, X):
        return np.maximum(X @ self.W1 + self.b1, 0.0) @ self.W2 + self.b2

    def step(self, X, y, lr):
        m = len(y)
        out = self.forward(X)
        d = 2.0 * (out - y) / m
        gW2 = self.a.T @ d
        gb2 = d.sum()
        da = np.outer(d, self.W2) * (self.z > 0)
        gW1 = X.T @ da
        gb1 = da.sum(axis=0)
        self.t += 1
        b1c, b2c = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        for i, (p, g) in enumerate(zip(("W1", "b1", "W2", "b2"),
                                       (gW1, gb1, gW2, gb2))):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            upd = lr * (self.m[i] / b1c) / (np.sqrt(self.v[i] / b2c) + 1e-8)
            setattr(self, p, getattr(self, p) - upd)
        return np.mean((out - y) ** 2)


def train(Xtr, ytr_, Xte, yte_, nh, epochs, lr, seed, batch=40, record=None):
    net = Net(Xtr.shape[1], nh, seed)
    g = np.random.default_rng(seed + 1)
    m = len(ytr_)
    for ep in range(epochs):
        idx = g.permutation(m)
        for s in range(0, m, batch):
            b = idx[s:s + batch]
            net.step(Xtr[b], ytr_[b], lr)
        if record is not None and (ep + 1) % 20 == 0:
            record.append((ep + 1,
                           r2(ytr_, net.predict(Xtr)),
                           r2(yte_, net.predict(Xte))))
    return net


print("""
  The same 400 training configurations that Chapter 3 used, but with the
  L = %d raw spins as input instead of the L^2 = %d outer products.  One
  hidden layer, ReLU, mean squared error, Adam.""" % (L, L * L))

curve = []
net_raw = train(Str, ytr, Ste, yte, 200, 400, 3e-3, 11, record=curve)
print("\n   epochs   R^2 train    R^2 test")
for ep, a, b in curve[::4]:
    print("   %6d   %9.6f   %9.6f" % (ep, a, b))
np.save("curve_raw.npy", np.array(curve))

print("\n  width study, 400 epochs each:")
print("\n   hidden units   R^2 train    R^2 test")
widths, wres = [10, 25, 50, 100, 200, 400], []
for nh in widths:
    nt = train(Str, ytr, Ste, yte, nh, 400, 3e-3, 11)
    a, b = r2(ytr, nt.predict(Str)), r2(yte, nt.predict(Ste))
    wres.append((nh, a, b))
    print("   %12d   %9.6f   %9.6f" % (nh, a, b))
np.save("width_raw.npy", np.array(wres))

print("\n  training-set-size study, 200 hidden units, 400 epochs:")
print("\n   training rows   R^2 test")
sizes, sres = [100, 200, 400, 800, 1600, 3200], []
hold = idx[3200:]
for m in sizes:
    Sm, ym = spins[idx[:m]].astype(float), energies[idx[:m]].astype(float)
    Se, ye = spins[hold].astype(float), energies[hold].astype(float)
    nt = train(Sm, ym, Se, ye, 200, 400, 3e-3, 11)
    b = r2(ye, nt.predict(Se))
    sres.append((m, b))
    print("   %13d   %9.6f" % (m, b))
np.save("size_raw.npy", np.array(sres))

print("""
  Representable is not the same as learnable.  Proposition 8.isingexact
  produces a network with 80 hidden units and zero error, yet 400
  configurations are nowhere near enough to find it: the training score
  is 1 and the test score is a quarter of that.  With more data the gap
  closes.  Given 8000 configurations, 400 hidden units and 1500 epochs:""")

Sb, yb = spins[idx[:8000]].astype(float), energies[idx[:8000]].astype(float)
Sh, yh = spins[idx[8000:]].astype(float), energies[idx[8000:]].astype(float)
net_big = train(Sb, yb, Sh, yh, 400, 1500, 3e-3, 11, batch=64)
print("\n    R^2 train                                   : %.6f"
      % r2(yb, net_big.predict(Sb)))
print("    R^2 test                                    : %.6f"
      % r2(yh, net_big.predict(Sh)))
print("    RMS error in units of the energy quantum 2J : %.4f"
      % (np.sqrt(np.mean((yh - net_big.predict(Sh)) ** 2)) / (2 * Jtrue)))
np.save("big_raw.npy", np.array([r2(yb, net_big.predict(Sb)),
                                 r2(yh, net_big.predict(Sh))]))

# ===========================================================================
print("\n=== 4. a network on the outer-product features of Chapter 3 ===")
# ===========================================================================
Xout = np.einsum("ij,ik->ijk", spins, spins).reshape(n, L * L).astype(float)
Xtr_o, Xte_o = Xout[tr], Xout[te]
print("""
  Given the L^2 features the problem is linear, and Chapter 3 showed that
  the Lasso solves it almost exactly.  A network given the same features
  has to discover that a linear function suffices.""")
curve_o = []
net_out = train(Xtr_o, ytr, Xte_o, yte, 200, 400, 3e-3, 11, record=curve_o)
print("\n   epochs   R^2 train    R^2 test")
for ep, a, b in curve_o[::4]:
    print("   %6d   %9.6f   %9.6f" % (ep, a, b))
np.save("curve_out.npy", np.array(curve_o))

# ===========================================================================
print("\n=== 5. everything on one test set ===")
# ===========================================================================
# the coupling matrices left behind by ising_regression.py, which lives with
# the Chapter 3 programs
import os
CH3 = os.environ.get("ISING_CH3", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "chapter03_linear_regression"))
load3 = lambda f: np.load(f if os.path.exists(f) else os.path.join(CH3, f))
J_ols = load3("J_ols.npy").reshape(-1)
J_ridge = load3("J_ridge.npy").reshape(-1)
J_lasso = load3("J_lasso.npy").reshape(-1)

rows = [
    ("OLS,   L^2 outer products", r2(yte, Xte_o @ J_ols)),
    ("Ridge, L^2 outer products", r2(yte, Xte_o @ J_ridge)),
    ("Lasso, L^2 outer products", r2(yte, Xte_o @ J_lasso)),
    ("network, L^2 outer products", r2(yte, net_out.predict(Xte_o))),
    ("OLS,   L raw spins", r2(yte, np.column_stack([np.ones(len(yte)), Ste])
                             @ np.linalg.lstsq(
                                 np.column_stack([np.ones(ntr), Str]),
                                 ytr, rcond=None)[0])),
    ("network, L raw spins", r2(yte, net_raw.predict(Ste))),
    ("exact network, L raw spins", r2(yte, np.maximum(Ste @ W1 + b1, 0) @ W2 + b2)),
]
print("\n   model                            R^2 on the %d test configurations" % len(yte))
for name, v in rows:
    print("   %-32s %14.6f" % (name, v))
np.save("summary_r2.npy", np.array([v for _, v in rows]))

# ===========================================================================
print("\n=== 6. the same network in PyTorch ===")
# ===========================================================================
try:
    import torch

    torch.manual_seed(11)
    dev = "cpu"
    Xt = torch.tensor(Str, dtype=torch.float64)
    yt = torch.tensor(ytr, dtype=torch.float64)
    Xv = torch.tensor(Ste, dtype=torch.float64)
    yv = torch.tensor(yte, dtype=torch.float64)
    model = torch.nn.Sequential(
        torch.nn.Linear(L, 200), torch.nn.ReLU(), torch.nn.Linear(200, 1)
    ).double()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(12)
    for ep in range(400):
        perm = torch.randperm(ntr, generator=g)
        for s in range(0, ntr, 40):
            b = perm[s:s + 40]
            opt.zero_grad()
            lossf(model(Xt[b]).squeeze(-1), yt[b]).backward()
            opt.step()
    with torch.no_grad():
        ptr = model(Xt).squeeze(-1).numpy()
        pte = model(Xv).squeeze(-1).numpy()
    print("\n   framework      R^2 train    R^2 test")
    print("   numpy          %9.6f   %9.6f"
          % (r2(ytr, net_raw.predict(Str)), r2(yte, net_raw.predict(Ste))))
    print("   PyTorch        %9.6f   %9.6f" % (r2(ytr, ptr), r2(yte, pte)))
    print("""
  The two implementations are independent -- different initialisations,
  different shuffles -- so they are not expected to agree digit by digit.
  They agree on the conclusion, which is what a cross-check is for.""")
except ImportError:
    print("  PyTorch not available")

np.save("meta_nn.npy", np.array([L, n, ntr, Jtrue]))
print("\ndone")

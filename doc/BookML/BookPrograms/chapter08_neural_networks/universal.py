"""The quantitative side of the universal approximation theorem, Section 8.4.

The classical theorems of Cybenko and Hornik are existence statements: they say
that a suitable network exists and say nothing about how wide it must be, nor
about whether gradient descent will find it.  This program supplies the three
things the classical statements leave out, and checks each of them numerically.

1.  *An explicit width.*  Theorem 8.relurate says that a single hidden layer of
    $N$ ReLU units approximates any $L$-Lipschitz function on $[a,b]$ to within
    $L(b-a)/2N$, by interpolation at $N+1$ equispaced knots.  We build that
    network by hand -- no training -- and measure the error against the bound,
    both for a smooth target, where the bound is loose, and for the worst-case
    target of Remark 8.tight, where it is attained exactly.

2.  *Representability is not learnability.*  The same architecture, the same
    width, but fitted by Adam from a random start.  The gap between the two
    columns is the whole practical difficulty of deep learning.

3.  *Depth beats width.*  The sawtooth $s_k$ of Theorem 8.depth is computed
    exactly by a ReLU network of depth $k$ with two units per layer.  Any
    network with a single hidden layer of $N$ units is piecewise linear with at
    most $N+1$ pieces, so it cannot come within $1/2$ of $s_k$ unless
    $N\\ge 2^{k}-1$.  We build the deep network, count its linear pieces, and
    then fit shallow networks of increasing width to see the wall.

4.  *Escaping the curse.*  Barron's theorem gives a rate $\\bigO(N^{-1/2})$ that
    does not degrade with the input dimension, for the restricted class of
    functions with a finite Barron norm.  We measure the rate in $d=1$, $5$ and
    $20$.

All numbers printed here are quoted verbatim in Section 8.4.
"""
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
out = open("universal.txt", "w", buffering=1)

# ===========================================================================
# 1.  the constructive bound of Theorem 8.relurate
# ===========================================================================
out.write("=== 1. the explicit construction of Theorem 8.relurate ===\n")
out.write("  A single hidden layer of N ReLU units, weights written down rather\n"
          "  than trained: interpolate F at the N+1 knots t_k = a + k(b-a)/N.\n"
          "  The bound is L(b-a)/2N.\n\n")


def relu_interpolant(F, a, b, N):
    """The exact CPWL interpolant of F at N+1 equispaced knots, as a
    one-hidden-layer ReLU network with N units.

    P(x) = c + sum_k v_k relu(x - t_k),  v_k = s_k - s_{k-1},
    with s_k the slope on [t_k, t_{k+1}] and s_{-1} = 0.  The hidden weights
    are all 1, the hidden biases are -t_k: only the output layer carries
    information about F.
    """
    t = a + (b - a) * np.arange(N + 1) / N
    y = F(t)
    s = np.diff(y) / np.diff(t)                    # N slopes
    v = np.diff(np.concatenate(([0.0], s)))        # N output weights
    c = y[0] - s[0] * (0.0)                        # value at x = t_0
    knots = t[:N]

    def P(x):
        x = np.atleast_1d(x)
        return c + (np.maximum(0.0, x[:, None] - knots[None, :]) * v).sum(1)

    return P, knots, v, c


def sup_error(F, P, a, b, m=200001):
    x = np.linspace(a, b, m)
    return float(np.abs(F(x) - P(x)).max())


a, b = 0.0, 1.0
Fs = lambda x: np.sin(2 * np.pi * x)               # smooth, L = 2 pi
Ls = 2 * np.pi

out.write("  target F(x) = sin(2 pi x) on [0,1], Lipschitz constant L = 2 pi\n")
out.write("     N    sup error   bound L/2N   ratio\n")
rows_smooth = []
for N in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    P, _, _, _ = relu_interpolant(Fs, a, b, N)
    e = sup_error(Fs, P, a, b)
    bd = Ls * (b - a) / (2 * N)
    rows_smooth.append((N, e, bd))
    out.write(f"  {N:4d}   {e:10.3e}   {bd:10.3e}   {e/bd:6.4f}\n")
out.write("  the bound holds with room to spare: a smooth target is much easier\n"
          "  than the worst Lipschitz function of the same constant\n\n")

# the worst case of Remark 8.tight: F(x) = L dist(x, hZ), h = (b-a)/N
out.write("  worst case, F(x) = L*dist(x, hZ) with h = 1/N: the interpolant is\n"
          "  identically zero at every knot, so the error equals the bound\n")
out.write("     N    sup error   bound L/2N   ratio\n")
rows_worst = []
for N in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    h = (b - a) / N
    Lw = 1.0
    Fw = lambda x, h=h: np.abs(x / h - np.round(x / h)) * h      # L = 1
    P, _, _, _ = relu_interpolant(Fw, a, b, N)
    e = sup_error(Fw, P, a, b)
    bd = Lw * (b - a) / (2 * N)
    rows_worst.append((N, e, bd))
    out.write(f"  {N:4d}   {e:10.3e}   {bd:10.3e}   {e/bd:6.4f}\n")
out.write("  the constant 1/2 in Theorem 8.relurate cannot be improved\n\n")
np.save("construct_smooth.npy", np.array(rows_smooth))
np.save("construct_worst.npy", np.array(rows_worst))
xplot = np.linspace(a, b, 2001)
np.save("plot_x.npy", xplot)
np.save("plot_target.npy", Fs(xplot))
for Np in (4, 16):
    Pp, _, _, _ = relu_interpolant(Fs, a, b, Np)
    np.save(f"plot_interp{Np}.npy", Pp(xplot))

# ===========================================================================
# 2.  the same width, trained instead of constructed
# ===========================================================================
out.write("=== 2. representability is not learnability ===\n")
out.write("  Same architecture, same width, but the weights are found by Adam\n"
          "  from a random start rather than written down.  Best of five seeds,\n"
          "  6000 full-batch steps on 2001 points, lr 0.01 with cosine decay.\n\n")

Xg = torch.tensor(np.linspace(a, b, 2001))[:, None]
Yg = torch.tensor(Fs(np.linspace(a, b, 2001)))[:, None]
Xf = np.linspace(a, b, 200001)


def make_net(N, sd, spread):
    """A width-$N$ ReLU network.  With ``spread`` the kinks $-b_i/w_i$ are
    placed uniformly in $[0,1]$, which is where the construction of
    Theorem 8.relurate puts them; without it PyTorch's default initialisation
    scatters most of them outside the interval, where their units are dead."""
    torch.manual_seed(sd)
    m = nn.Sequential(nn.Linear(1, N), nn.ReLU(), nn.Linear(N, 1))
    if spread:
        g = torch.Generator().manual_seed(sd)
        t = torch.rand(N, generator=g)
        s = torch.where(torch.rand(N, generator=g) < 0.5, -1.0, 1.0)
        with torch.no_grad():
            m[0].weight.copy_(s[:, None])
            m[0].bias.copy_(-s * t)
            m[2].weight.mul_(0.1)
            m[2].bias.zero_()
    return m


def fit_shallow(target_np, N, steps=6000, seeds=(0, 1, 2, 3, 4), lr=1e-2,
                X=None, Y=None, spread=True):
    """Fit a one-hidden-layer ReLU network of width N; return the best run."""
    if X is None:
        X, Y = Xg, Yg
    best, best_state = np.inf, None
    for sd in seeds:
        m = make_net(N, sd, spread)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            loss = ((m(X) - Y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
        with torch.no_grad():
            e = float(np.abs(m(torch.tensor(Xf)[:, None])[:, 0].numpy()
                             - target_np(Xf)).max())
        if e < best:
            best, best_state = e, [p.detach().clone() for p in m.parameters()]
    return best, best_state


out.write("     N   constructed   default init   kinks spread   ratio\n")
rows_train = []
t0 = time.time()
for N in [2, 4, 8, 16, 32, 64]:
    P, _, _, _ = relu_interpolant(Fs, a, b, N)
    ec = sup_error(Fs, P, a, b)
    ed, _ = fit_shallow(Fs, N, spread=False)
    es, _ = fit_shallow(Fs, N, spread=True)
    rows_train.append((N, ec, ed, es))
    out.write(f"  {N:4d}   {ec:11.3e}   {ed:12.3e}   {es:12.3e}"
              f"   {es/ec:5.1f}\n")
out.write("  the last column is the trained-with-spread-kinks error divided by\n"
          "  the constructed error; best of five seeds in each column\n")
out.write("  the default initialisation is the worse problem: most kinks start\n"
          "  outside [0,1] and never enter it.  Even with the kinks placed where\n"
          "  the construction puts them, gradient descent falls further behind\n"
          "  the constructed network as the width grows.  The existence proof\n"
          "  says nothing whatever about the optimisation problem.\n\n")
np.save("trained_vs_constructed.npy", np.array(rows_train))

# ===========================================================================
# 3.  depth separation: the sawtooth of Theorem 8.depth
# ===========================================================================
out.write("=== 3. depth against width: the sawtooth s_k ===\n")
out.write("  T(x) = 2 relu(x) - 4 relu(x - 1/2) is the tent map, computed by one\n"
          "  layer of two ReLU units.  s_k = T composed k times has 2^k linear\n"
          "  pieces, so a depth-k network with 2k units in total represents it\n"
          "  exactly, while one hidden layer needs at least 2^k - 1 units.\n\n")


def tent(x):
    return 2 * np.maximum(x, 0) - 4 * np.maximum(x - 0.5, 0)


def sawtooth(x, k):
    for _ in range(k):
        x = tent(x)
    return x


def count_pieces(f, m=2 ** 19 + 1, tol=1e-9):
    """Number of maximal intervals on which f is affine, counted from the
    changes in the numerical slope.  The grid is dyadic so that the breakpoints
    of the sawtooth fall exactly on grid points; otherwise a breakpoint inside
    a cell produces two slope changes instead of one and is counted twice."""
    x = np.linspace(0.0, 1.0, m)
    y = f(x)
    s = np.diff(y) / np.diff(x)
    return int(1 + (np.abs(np.diff(s)) > tol * max(1.0, np.abs(s).max())).sum())


def pieces_of_fit(W1, b1, W2, lo=0.0, hi=1.0, tol=1e-12):
    """The exact number of affine pieces of a fitted shallow network on
    [lo, hi], read off from Lemma 8.pieces: one piece per live kink that falls
    inside the interval, plus one."""
    w, v = W1[:, 0], W2[0]
    live = (np.abs(w) > tol) & (np.abs(v) > tol)
    t = -b1[live] / w[live]
    t = np.unique(np.round(t[(t > lo) & (t < hi)], 12))
    return int(len(t) + 1)


out.write("     k   units in the deep net   linear pieces measured   2^k\n")
for k in range(1, 9):
    out.write(f"  {k:4d}   {2*k:21d}   {count_pieces(lambda x, k=k: sawtooth(x, k)):22d}"
              f"   {2**k:4d}\n")
out.write("  the deep network is exact and its size grows linearly in k while\n"
          "  the number of pieces it produces grows exponentially\n\n")

out.write("  Now the shallow networks.  For each k we fit one hidden layer of\n"
          "  width N to s_k and measure the sup error and the number of linear\n"
          "  pieces the fitted network actually has.\n\n")
Xs = torch.tensor(np.linspace(0.0, 1.0, 4001))[:, None]
wall = []
for k in [2, 3, 4]:
    Ys = torch.tensor(sawtooth(np.linspace(0.0, 1.0, 4001), k))[:, None]
    tgt = lambda x, k=k: sawtooth(x, k)
    out.write(f"   k = {k}, s_k has {2**k} pieces, the theorem needs "
              f"N >= {2**k - 1}\n")
    out.write("      N   sup error   pieces of the fit   N+1\n")
    for N in [2, 4, 8, 16, 32, 64]:
        e, st = fit_shallow(tgt, N, steps=6000, X=Xs, Y=Ys)
        W1, b1, W2, b2 = [p.numpy() for p in st]
        g = lambda x: (np.maximum(0.0, x[:, None] * W1[:, 0][None, :]
                                  + b1[None, :]) @ W2[0] + b2[0])
        wall.append((k, N, e, pieces_of_fit(W1, b1, W2)))
        out.write(f"   {N:4d}   {e:9.4f}   {wall[-1][3]:17d}   {N+1:4d}\n")
        if k == 3 and N in (4, 8, 32):
            np.save(f"plot_saw_fit_N{N}.npy", g(xplot))
    out.write("\n")
np.save("depth_wall.npy", np.array(wall))
for kk in (1, 2, 3, 4):
    np.save(f"plot_saw{kk}.npy", sawtooth(xplot, kk))
out.write("  below the threshold the error sits at about 1/2, which is the\n"
          "  bound of Theorem 8.depth; above it the fit improves\n\n")

# ===========================================================================
# 4.  Barron's rate, and the constant that replaces the dimension
# ===========================================================================
out.write("=== 4. Barron's rate, Theorem 8.barron ===\n")
out.write("  Target F(x) = (1/M) sum_m c_m cos(kappa a_m . x), M = 12 unit ridge\n"
          "  directions a_m drawn at random.  Its Fourier transform is a sum of\n"
          "  point masses at +-kappa a_m, so the Barron norm of Eq. (8.barronnorm)\n"
          "  is known exactly:  C_F = kappa (1/M) sum_m |c_m|.  Crucially C_F does\n"
          "  not depend on d, and we draw x uniformly from the unit ball so that\n"
          "  the radius r = 1 in every dimension too.  Theorem 8.barron then\n"
          "  predicts the same bound (2 r C_F)^2 / N in d = 1, 5 and 20.\n\n")


def ball_sample(rng, n, d):
    """Uniform on the unit ball of R^d."""
    g = rng.normal(size=(n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g * rng.random((n, 1)) ** (1.0 / d)


def ridge_target(d, kappa, M=12, seed=7):
    r = np.random.default_rng(seed)
    A = r.normal(size=(M, d))
    A /= np.linalg.norm(A, axis=1, keepdims=True)      # unit directions
    cc = r.normal(size=M)
    CF = kappa * np.abs(cc).mean()                     # exact Barron norm
    return (lambda X: np.cos(kappa * (X @ A.T)) @ cc / M), CF


def fit_tanh(d, F, N, Xtr, Ytr, Xte, steps=4000, seeds=(0, 1)):
    best = np.inf
    Xt = torch.tensor(Xtr)
    Yt = torch.tensor(Ytr)[:, None]
    for sd in seeds:
        torch.manual_seed(sd)
        m = nn.Sequential(nn.Linear(d, N), nn.Tanh(), nn.Linear(N, 1))
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            loss = ((m(Xt) - Yt) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
        with torch.no_grad():
            p = m(torch.tensor(Xte))[:, 0].numpy()
        best = min(best, float(((p - F(Xte)) ** 2).mean()))
    return best


for kappa in [2.0, 8.0]:
    _, CF = ridge_target(1, kappa)
    out.write(f"  kappa = {kappa:.0f}:  C_F = {CF:.4f}\n")
    out.write("      N       d = 1       d = 5      d = 20"
              "   bound (2 C_F)^2/N\n")
    tab = []
    for N in [4, 8, 16, 32, 64, 128]:
        row = [N]
        for d in [1, 5, 20]:
            F, _ = ridge_target(d, kappa)
            r = np.random.default_rng(100 + d)
            Xtr = ball_sample(r, 4000, d)
            Xte = ball_sample(r, 20000, d)
            row.append(fit_tanh(d, F, N, Xtr, F(Xtr), Xte))
        bd = (2 * CF) ** 2 / N
        tab.append(row + [bd])
        out.write(f"  {N:5d}  {row[1]:10.3e}  {row[2]:10.3e}  {row[3]:10.3e}"
                  f"  {bd:18.3e}\n")
    tab = np.array(tab)
    sl = [float(np.polyfit(np.log(tab[:, 0]), np.log(tab[:, j]), 1)[0])
          for j in (1, 2, 3)]
    out.write(f"  measured slopes of log(mse) against log(N):"
              f"  {sl[0]:6.2f} {sl[1]:6.2f} {sl[2]:6.2f}"
              f"   (Theorem 8.barron: -1)\n")
    out.write(f"  ratio of the d = 20 error to the d = 1 error at N = 128:"
              f"  {tab[-1, 3]/tab[-1, 1]:.1f}\n\n")
    np.save(f"barron_k{int(kappa)}.npy", tab)
out.write("  the bound of Theorem 8.barron holds everywhere and with room to\n"
          "  spare; the error grows with kappa, that is with C_F, and only\n"
          "  weakly with d.  Compare Eq. (8.curse): for a general Lipschitz\n"
          "  target the width needed at fixed accuracy is raised to the power d,\n"
          "  and no experiment at d = 20 would be possible at all\n")
out.close()
print(open("universal.txt").read())

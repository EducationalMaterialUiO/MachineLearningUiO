"""Chapter 4, Section "Stochastic gradient descent": the experiments behind the
extended section.  Same least-squares problem as figure sgd_vs_gd in
ch04_figures.py (n = 100 000, p = 20, AR(1) correlation 0.9, kappa = 213, seed 2026).

Prints every number quoted in the text and writes
  sgd_three_axes   the same runs against updates, epochs and estimated wall time
  sgd_batchsize    time per update against M, and the constant-rate plateau against gamma/M
Wall-clock numbers are machine dependent; everything else is reproducible."""
from common import *
import time

# ---- 0. the running mean: SGD on f_i = (theta - a_i)^2 / 2 with gamma_t = 1/(t+1) ----
r0 = np.random.default_rng(2026)
a = 2.0 + 3.0 * r0.normal(size=1_000_000)
abar, s_a = a.mean(), a.std(ddof=1)
print(f"  running mean: n={len(a)}, abar={abar:.4f}, s_a={s_a:.4f}")
theta, errs = 0.0, {}
for t, i in enumerate(r0.integers(0, len(a), 10_000)):
    theta = (1 - 1 / (t + 1)) * theta + a[i] / (t + 1)
    if t + 1 in (100, 1000, 10_000): errs[t + 1] = theta - abar
for t, e in errs.items():
    print(f"     after {t:6d} samples: theta - abar = {e:+.4f}; s_a/sqrt(t) = {s_a/np.sqrt(t):.4f}")
reps = np.array([a[r0.integers(0, len(a), 1000)].mean() - abar for _ in range(2000)])
print(f"     2000 repetitions of t=1000: mean error {reps.mean():+.4f}, std {reps.std():.4f} (s_a/sqrt(1000) = {s_a/np.sqrt(1000):.4f})")

# ---- the large least-squares problem ----
n, p, rho, sigma = 100_000, 20, 0.9, 1.0
rng = np.random.default_rng(2026)
z = rng.normal(size=(n, p)); X = np.empty_like(z); X[:, 0] = z[:, 0]
for j in range(1, p): X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho**2) * z[:, j]
mu_x, sd_x = X.mean(0), X.std(0); X = (X - mu_x) / sd_x
theta_true = rng.normal(size=p); y = X @ theta_true + sigma * rng.normal(size=n)
theta_hat = np.linalg.solve(X.T @ X, X.T @ y); c_hat = np.mean((X @ theta_hat - y)**2)
excess = lambda th: np.mean((X @ th - y)**2) - c_hat
floor = sigma**2 * p / n
ev = np.linalg.eigvalsh(2 * X.T @ X / n); L, mu = ev.max(), ev.min(); gstar = 2 / (L + mu)
# a validation set from the same model, standardised with the training statistics
rv = np.random.default_rng(7); zv = rv.normal(size=(20_000, p)); Xv = np.empty_like(zv); Xv[:, 0] = zv[:, 0]
for j in range(1, p): Xv[:, j] = rho * Xv[:, j - 1] + np.sqrt(1 - rho**2) * zv[:, j]
Xv = (Xv - mu_x) / sd_x; yv = Xv @ theta_true + sigma * rv.normal(size=len(Xv))
val = lambda th: np.mean((Xv @ th - yv)**2)
G = 2 * X * (X @ theta_hat - y)[:, None]; trS = np.trace(np.cov(G.T))
print(f"  problem: L={L:.2f}, mu={mu:.4f}, kappa={L/mu:.0f}, gamma*={gstar:.4f}, floor={floor:.1e}; "
      f"tr S at the optimum = {trS:.1f} (4 sigma^2 p = {4*sigma**2*p:.0f}); validation MSE at theta_hat {val(theta_hat):.4f}")

# ---- 1. one step: expected progress against the bound ----
th0 = np.zeros(p); g0 = (2 / n) * X.T @ (X @ th0 - y); G0 = 2 * X * (X @ th0 - y)[:, None]
sig2 = np.mean(np.sum((G0 - g0)**2, axis=1))
print(f"  at theta=0: C-C*={excess(th0):.2f}, |grad|^2={g0@g0:.1f}, sigma_g^2={sig2:.0f}")
rb = np.random.default_rng(3)
for M, gam in ((32, 0.02), (1, 0.002)):
    dec = []
    for _ in range(4000):
        b = rb.integers(0, n, M); gb = (2 / M) * X[b].T @ (X[b] @ th0 - y[b]); dec.append(excess(th0 - gam * gb))
    bound = excess(th0) - gam * (1 - L * gam / 2) * (g0 @ g0) + L * gam**2 * sig2 / (2 * M)
    print(f"     M={M}, gamma={gam}: mean excess after one step {np.mean(dec):.3f} (bound {bound:.3f}); "
          f"fraction of steps that increase the cost {np.mean(np.array(dec) > excess(th0)):.3f}; "
          f"progress term {gam*(1-L*gam/2)*(g0@g0):.3f}, noise term {L*gam**2*sig2/(2*M):.3f}")

# ---- 2. time per update against the batch size (machine dependent) ----
def time_update(M, reps):
    """The SGD loop itself -- shuffled batches, schedule, update -- without any recording."""
    idx = np.random.default_rng(0).permutation(n); th = np.zeros(p); k = 0; t0 = time.perf_counter()
    for t in range(1, reps + 1):
        if k + M > n: k = 0
        b = idx[k:k + M]; k += M
        gam = 0.001 * 1000.0 / (t + 1000.0)
        Xb = X[b]; th = th - gam * (2.0 / M) * (Xb.T @ (Xb @ th - y[b]))
    return (time.perf_counter() - t0) / reps
Ms = [1, 8, 32, 128, 512, 2048, 8192, n]
t_upd = {M: min(time_update(M, max(20, min(20000, 2_000_000 // M))) for _ in range(3)) for M in Ms}
print("  time per update (s) and throughput (examples/s):")
for M in Ms: print(f"     M={M:6d}: {t_upd[M]:.2e} s, {M/t_upd[M]:.2e} examples/s, one epoch {t_upd[M]*n/M:.2f} s")

# ---- 3. the runs: gradient descent, and SGD at four batch sizes with tuned schedules ----
def sgd(M, g0, t0, epochs, seed=1, const=False, rec=20, average_from=None):
    r = np.random.default_rng(seed); th = np.zeros(p); t = 0; out = [(0, excess(th))]
    per = max(1, (n // M) // rec); avg = np.zeros(p); navg = 0
    for e in range(epochs):
        idx = r.permutation(n)
        for i, k in enumerate(range(0, n, M)):
            b = idx[k:k + M]; t += 1
            gam = g0 if const else g0 * t0 / (t + t0)
            Xb = X[b]; th = th - gam * (2.0 / len(b)) * (Xb.T @ (Xb @ th - y[b]))
            if average_from is not None and e >= average_from: avg += th; navg += 1
            if (i + 1) % per == 0: out.append((t, excess(th)))
    return np.array(out), th, (avg / navg if navg else None)

th = np.zeros(p); gd = [(0, excess(th))]
for k in range(1, 1001):
    th = th - gstar * (2.0 / n) * X.T @ (X @ th - y); gd.append((k, excess(th)))
    if len(gd) == 641: th_gd_floor = th.copy()
gd = np.array(gd); k_gd = int(np.argmax(gd[:, 1] < floor))
print(f"  gradient descent: below the floor after {k_gd} updates = {k_gd} epochs, {k_gd*4*n*p:.1e} flops, "
      f"est. wall time {k_gd*t_upd[n]:.2f} s; validation MSE there {val(th_gd_floor):.4f}")
schedules = {1: (0.002, 1e4), 8: (0.01, 1e3), 32: (0.02, 1e3), 128: (0.08, 300.0), 512: (0.08, 1e3)}
curves, rows = {}, []
for M, (g0_, t0_) in schedules.items():
    hits, levels = [], []
    for seed in (1, 2, 3, 4, 5):
        out, th_end, _ = sgd(M, g0_, t0_, 12, seed=seed)
        if seed == 1: curves[M] = out
        levels.append([out[(out[:, 0] * M / n > e - 0.5) & (out[:, 0] * M / n <= e), 1].mean() for e in (1, 3, 5, 12)])
        below = np.nonzero(out[:, 1] < floor)[0]
        hits.append(out[below[0], 0] if len(below) else np.nan)
    hits = np.array(hits); med = np.nanmedian(hits)
    t_M = t_upd[M]
    print(f"  SGD M={M:3d}, gamma_t={g0_}*{t0_:g}/(t+{t0_:g}): first below the floor after updates {hits} -> median {med:.0f} updates = "
          f"{med*M/n:.1f} epochs (range {np.nanmin(hits)*M/n:.1f}-{np.nanmax(hits)*M/n:.1f}), {med*4*M*p:.1e} flops, est. wall time {med*t_M:.2f} s; "
          f"excess after 12 epochs (seed 5) {out[-1,1]:.1e}, validation MSE {val(th_end):.4f}")
    lev = np.mean(levels, axis=0)
    quasi = [g0_ * t0_ / ((e - 0.25) * n / M + t0_) * trS / (4 * M) for e in (1, 3, 5, 12)]
    print("       mean excess over the last half of epochs 1, 3, 5, 12 (5 seeds): " + " ".join(f"{v:.1e}" for v in lev)
          + "; quasi-static gamma_t tr(S)/(4M): " + " ".join(f"{v:.1e}" for v in quasi))
    rows.append((M, med, med * M / n, med * t_M))

fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
cols = {1: "C1", 32: "crimson", 512: "C0"}
for ax, kind in zip(axes, ("updates", "epochs", "time")):
    sx = {"updates": 1.0, "epochs": 1.0, "time": t_upd[n]}[kind]
    ax.loglog(gd[1:, 0] * sx, gd[1:, 1], color="0.3", lw=1.8, label=r"gradient descent, $\gamma^*$")
    for M in (1, 32, 512):
        c = curves[M]; xs = c[1:, 0] * {"updates": 1.0, "epochs": M / n, "time": t_upd[M]}[kind]
        ax.loglog(xs, c[1:, 1], color=cols[M], lw=1.2, label=rf"SGD, $M={M}$")
    ax.axhline(floor, color="k", ls=":", lw=1)
    ax.set_xlabel({"updates": "parameter updates", "epochs": "epochs (passes over the data)",
                   "time": "estimated wall time (s)"}[kind])
axes[0].set_ylabel(r"$C(\boldsymbol{\theta})-C(\hat{\boldsymbol{\theta}})$"); axes[0].set_ylim(1e-5, 30)
axes[1].legend(fontsize=8, loc="upper right")
save(fig, 4, "sgd_three_axes")

# ---- 4. the constant-rate plateau: excess cost gamma tr(S)/(4M) ----
print("  plateau at constant gamma: measured mean excess over the last third of the run against gamma tr(S)/(4M)")
plat = []
for M, gam in ((8, 0.005), (32, 0.005), (32, 0.01), (32, 0.02), (128, 0.02), (512, 0.02), (512, 0.05)):
    E = max(6, int(np.ceil(15 / (gam * mu) / (n / M))))          # long enough for the slowest direction to equilibrate
    out, _, _ = sgd(M, gam, None, E, const=True)
    meas = out[out[:, 0] > (2 * E // 3) * n / M, 1].mean(); pred = gam * trS / (4 * M)
    plat.append((gam / M, meas, pred)); print(f"     M={M:3d}, gamma={gam}, {E} epochs: measured {meas:.2e}, predicted {pred:.2e}, ratio {meas/pred:.2f}")
print(f"     the plateau equals the floor when gamma/M = 1/n = {1/n:.0e}")
# ---- 5. iterate averaging at a constant learning rate ----
out, th_last, th_avg = sgd(32, 0.02, None, 5, const=True, average_from=2)
print(f"  averaging (M=32, gamma=0.02, 5 epochs, average over epochs 3-5): last iterate {excess(th_last):.2e}, averaged iterate {excess(th_avg):.2e}")

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
ax = axes[0]
ax.loglog(Ms, [t_upd[M] for M in Ms], "o-", color="C0", label="time per update (s)")
ax.set_xlabel("minibatch size $M$"); ax.set_ylabel("seconds per update", color="C0")
ax2 = ax.twinx(); ax2.loglog(Ms, [M / t_upd[M] for M in Ms], "s--", color="crimson"); ax2.set_ylabel("examples per second", color="crimson")
ax.set_title("cost of an update", fontsize=10)
ax = axes[1]; plat = np.array(plat)
ax.loglog(plat[:, 0], plat[:, 1], "o", color="crimson", mec="k", mew=0.4, label="measured")
gg = np.array([plat[:, 0].min() / 2, plat[:, 0].max() * 2]); ax.loglog(gg, gg * trS / 4, "-", color="0.3", lw=1.2, label=r"$\gamma\,\mathrm{tr}\,\boldsymbol{S}/(4M)$")
ax.axhline(floor, color="k", ls=":", lw=1); ax.text(gg[0], floor * 1.2, r"$\sigma^2p/n$", fontsize=8)
ax.set_xlabel(r"$\gamma/M$"); ax.set_ylabel("excess cost on the plateau"); ax.legend(fontsize=8, loc="upper left")
ax.set_title("the plateau of a constant learning rate", fontsize=10)
fig.tight_layout()
save(fig, 4, "sgd_batchsize")

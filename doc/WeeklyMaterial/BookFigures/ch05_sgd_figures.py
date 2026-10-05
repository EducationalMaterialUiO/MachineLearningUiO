"""Chapter 5, Section "Stochastic gradient descent for logistic regression":
gradient descent against SGD, and the adaptive methods of Chapter 4, on a
logistic regression with n = 50 000 and p = 20.  Prints every number quoted
in the text and writes logreg_sgd_vs_gd and logreg_adaptive."""
from common import *
import warnings; warnings.filterwarnings("ignore")

# ---- the data: AR(1)-correlated standardised features, Bernoulli labels ----
n, p = 50_000, 20
rng = np.random.default_rng(2026)
z = rng.normal(size=(n, p - 1)); rho = 0.9
xc = np.empty_like(z); xc[:, 0] = z[:, 0]
for j in range(1, p - 1):
    xc[:, j] = rho * xc[:, j - 1] + np.sqrt(1 - rho**2) * z[:, j]
xc = (xc - xc.mean(0)) / xc.std(0)
X = np.column_stack([np.ones(n), xc])
theta_true = 0.5 * rng.normal(size=p)

def sigmoid(t):
    a = np.exp(-np.abs(t))
    return np.where(t >= 0, 1.0 / (1.0 + a), a / (1.0 + a))

y = (rng.random(n) < sigmoid(X @ theta_true)).astype(float)
scales = np.concatenate([[1.0], 10.0**rng.uniform(-1, 1, p - 1)])   # the "raw" units
print(f"  n={n}, p={p}, class balance {y.mean():.3f}, accuracy of theta_true {np.mean((X@theta_true>0)==y):.4f}")

def cost(Xd, th):
    zz = Xd @ th
    return np.mean(np.logaddexp(0.0, zz) - y * zz)

def grad(Xd, th, idx):
    Xb = Xd[idx]
    return Xb.T @ (sigmoid(Xb @ th) - y[idx]) / Xb.shape[0]

def newton(Xd):
    th = np.zeros(p)
    for it in range(50):
        pr = sigmoid(Xd @ th); W = pr * (1 - pr)
        H = Xd.T @ (W[:, None] * Xd) / n
        step = np.linalg.solve(H, Xd.T @ (pr - y) / n); th -= step
        if np.linalg.norm(step) < 1e-12: break
    return th, H, it + 1

def optimiser_step(method, theta, g, state, t, gamma, rho=0.99,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    if method == "plain":
        return theta - gamma * g
    if method == "adagrad":
        state["r"] = r = state.get("r", 0.0) + g * g
        return theta - gamma * g / (np.sqrt(r) + eps)
    if method == "rmsprop":
        state["r"] = r = rho * state.get("r", 0.0) + (1 - rho) * g * g
        return theta - gamma * g / (np.sqrt(r) + eps)
    if method == "adam":
        state["m"] = m = beta1 * state.get("m", 0.0) + (1 - beta1) * g
        state["r"] = r = beta2 * state.get("r", 0.0) + (1 - beta2) * g * g
        return theta - gamma * (m / (1 - beta1**t)) / (np.sqrt(r / (1 - beta2**t)) + eps)

def run_gd(Xd, c_hat, gamma, epochs):
    th = np.zeros(p); ex = [cost(Xd, th) - c_hat]
    for k in range(epochs):
        th = th - gamma * grad(Xd, th, slice(None)); ex.append(cost(Xd, th) - c_hat)
    return np.array(ex), th

REC = 10                                   # cost recorded this many times per epoch
def run_sgd(Xd, c_hat, method, gamma, M=32, epochs=20, t0=None, seed=1):
    """Returns the excess cost at 0, 1/REC, 2/REC, ... epochs; index REC*k is the end of epoch k."""
    r = np.random.default_rng(seed); th = np.zeros(p); st = {}; t = 0
    ex = [cost(Xd, th) - c_hat]
    batches = [(k, min(k + M, n)) for k in range(0, n, M)]
    record = {round(len(batches) * j / REC) for j in range(1, REC + 1)}   # includes the epoch end
    for e in range(epochs):
        idx = r.permutation(n)
        for i, (a, b) in enumerate(batches):
            t += 1
            gam = gamma if t0 is None else gamma * t0 / (t + t0)
            th = optimiser_step(method, th, grad(Xd, th, idx[a:b]), st, t, gam)
            if i + 1 in record:
                ex.append(cost(Xd, th) - c_hat)
    return np.array(ex), th

# ---- part 1: standardised features, gradient descent against SGD ----
th_hat, H, it = newton(X); c_hat = cost(X, th_hat)
ev = np.linalg.eigvalsh(H); evx = np.linalg.eigvalsh(X.T @ X / n)
gamma_safe = 8.0 / evx.max()
floor = cost(X, theta_true) - c_hat
print(f"  Newton: {it} iterations, C*={c_hat:.5f}, accuracy {np.mean((X@th_hat>0)==y):.4f}")
print(f"  Hessian at the optimum: lambda_max={ev.max():.3f}, lambda_min={ev.min():.4f}, kappa={ev.max()/ev.min():.0f}")
print(f"  lambda_max(X^T X/n)={evx.max():.2f} -> safe gamma 8/lambda_max={gamma_safe:.3f}; 2/lambda_max(H*)={2/ev.max():.3f}")
print(f"  excess cost of theta_true {floor:.2e}, p/(2n)={p/(2*n):.1e}")
gd_ex, th_gd = run_gd(X, c_hat, gamma_safe, 200)
print("  GD, safe gamma, excess after 1,5,20,100,200 epochs: " + " ".join(f"{gd_ex[k]:.2e}" for k in (1, 5, 20, 100, 200)))
for k in (1, 2, 5, 20, 100, 200):
    print(f"     GD accuracy after {k:3d}: {np.mean((X@run_gd(X,c_hat,gamma_safe,k)[1]>0)==y):.4f}", end="")
print()
curves = {}
for lab, kw in [("SGD, $M=32$, $\\gamma=0.1$", dict(gamma=0.1)),
                ("SGD, $M=32$, $\\gamma=0.03$", dict(gamma=0.03)),
                ("SGD, $M=32$, $\\gamma=0.01$", dict(gamma=0.01)),
                ("SGD, $M=32$, $\\gamma_t=0.1\\cdot 10^3/(t+10^3)$", dict(gamma=0.1, t0=1000.0))]:
    curves[lab], th_s = run_sgd(X, c_hat, "plain", epochs=20, **kw)
    print(f"  {lab}: excess after 1,2,5,10,20 epochs: " + " ".join(f"{curves[lab][REC*k]:.2e}" for k in (1, 2, 5, 10, 20))
          + f"; mean over epochs 15-20: {curves[lab][15*REC:].mean():.2e}"
          + f"; accuracy {np.mean((X@th_s>0)==y):.4f}")
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ep = np.arange(0, 201)
ax.semilogy(ep[1:], gd_ex[1:], color="0.3", lw=1.8, label=rf"gradient descent, $\gamma={gamma_safe:.2f}$")
for (lab, ex), col in zip(curves.items(), ["C0", "C1", "C2", "crimson"]):
    ax.semilogy(np.arange(1, len(ex)) / REC, ex[1:], color=col, lw=1.1, label=lab)
ax.axhline(floor, color="k", ls=":", lw=1); ax.text(0.11, floor * 1.3, r"excess cost of $\boldsymbol{\theta}_{\rm true}$", fontsize=8)
ax.set_xscale("log"); ax.set_xlim(0.09, 220); ax.set_ylim(1e-5, 3)
ax.set_xlabel("epochs (passes over the data)"); ax.set_ylabel(r"$C(\boldsymbol{\theta})-C(\hat{\boldsymbol{\theta}})$")
ax.legend(fontsize=8, loc="upper right")
save(fig, 5, "logreg_sgd_vs_gd")

# ---- part 2: the adaptive methods, standardised and raw features ----
grids = {"plain": [0.003, 0.01, 0.03, 0.1, 0.3], "adagrad": [0.01, 0.03, 0.1, 0.3, 1.0],
         "rmsprop": [0.0003, 0.001, 0.003, 0.01, 0.03], "adam": [0.0003, 0.001, 0.003, 0.01, 0.03]}
names = {"plain": "plain SGD", "adagrad": "AdaGrad", "rmsprop": "RMSProp", "adam": "Adam"}
cols = {"plain": "C0", "adagrad": "C1", "rmsprop": "C2", "adam": "crimson"}
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
for ax, (title, Xd) in zip(axes, [("standardised features", X), ("raw features, columns scaled by $10^{\\pm1}$", X * scales)]):
    th_hat, H, it = newton(Xd); c_hat = cost(Xd, th_hat); ev = np.linalg.eigvalsh(H)
    evx = np.linalg.eigvalsh(Xd.T @ Xd / n); gs = 8.0 / evx.max()
    print(f"  == {title}: kappa(H*)={ev.max()/ev.min():.3g}, safe gamma {gs:.3f}")
    gd_ex, _ = run_gd(Xd, c_hat, gs, 200)
    print("     GD excess after 20, 200 epochs: " + " ".join(f"{gd_ex[k]:.2e}" for k in (20, 200)))
    ax.semilogy(np.arange(1, 201), gd_ex[1:], color="0.3", lw=1.8, label=rf"gradient descent, $\gamma={gs:.2f}$")
    for m, grid in grids.items():
        best = None
        for gam in grid:
            ex, th_s = run_sgd(Xd, c_hat, m, gam, epochs=20)
            tail = ex[15 * REC:].mean()
            print(f"     {names[m]:9s} gamma={gam:<7}: excess after 1,5,10,20: " + " ".join(f"{ex[REC*k]:.1e}" for k in (1, 5, 10, 20))
                  + f"; mean over epochs 15-20 {tail:.1e}; accuracy {np.mean((Xd@th_s>0)==y):.4f}")
            if best is None or tail < best[2]: best = (gam, ex, tail)
        ax.semilogy(np.arange(1, len(best[1])) / REC, best[1][1:], color=cols[m], lw=1.1,
                    label=rf"{names[m]}, $\gamma={best[0]}$")
        print(f"     -> best {names[m]} gamma={best[0]}")
    ax.set_xscale("log"); ax.set_xlim(0.09, 220); ax.set_title(title, fontsize=10)
    ax.set_xlabel("epochs"); ax.legend(fontsize=8, loc="upper right")
axes[0].set_ylabel(r"$C(\boldsymbol{\theta})-C(\hat{\boldsymbol{\theta}})$"); axes[0].set_ylim(1e-5, 3)
save(fig, 5, "logreg_adaptive")

# ---- part 3: what Adam does, and does not do, about the scales ----
# Diagonal preconditioning removes the column scales from the curvature exactly;
# Adam followed for 60 epochs, parameter by parameter, on both versions of the data.
def adam_history(Xd, gamma, epochs=60, M=32, seed=1):
    r = np.random.default_rng(seed); th = np.zeros(p); st = {}; t = 0; hist = []
    for e in range(epochs):
        idx = r.permutation(n)
        for k in range(0, n, M):
            t += 1
            th = optimiser_step("adam", th, grad(Xd, th, idx[k:k + M]), st, t, gamma)
        hist.append(th.copy())
    return np.array(hist)

print("  == Adam and the scales")
res = {}
for lab, Xd, gam, s in (("standardised", X, 0.001, np.ones(p)), ("raw", X * scales, 0.003, scales)):
    th_hat, H, it = newton(Xd); c_hat = cost(Xd, th_hat); d = np.sqrt(np.diag(H))
    print(f"     {lab}: kappa(H*)={np.linalg.cond(H):.4g}, after diagonal preconditioning "
          f"kappa(D^-1/2 H* D^-1/2)={np.linalg.cond(H / np.outer(d, d)):.1f}")
    hist = adam_history(Xd, gam)
    res[lab] = dict(ex=np.array([cost(Xd, th) - c_hat for th in hist]), err=np.abs(hist - th_hat) * s, th_hat=th_hat,
                    rel=np.abs(hist - th_hat) / np.abs(th_hat))
    print(f"     {lab}: Adam gamma={gam}: excess after 1,5,10,20,40,60 epochs: "
          + " ".join(f"{res[lab]['ex'][k-1]:.1e}" for k in (1, 5, 10, 20, 40, 60))
          + f"; mean 15-20 {res[lab]['ex'][14:20].mean():.1e}, mean 40-60 {res[lab]['ex'][39:].mean():.1e}")
js, jl = 1 + np.argmin(scales[1:]), 1 + np.argmax(scales[1:])
print(f"     smallest column scale {scales[js]:.3f}: theta_hat'={res['raw']['th_hat'][js]:.2f}, relative error after 1,5,10,20,40 epochs: "
      + " ".join(f"{res['raw']['rel'][k-1, js]:.2f}" for k in (1, 5, 10, 20, 40)))
print(f"     largest column scale {scales[jl]:.2f}: theta_hat'={res['raw']['th_hat'][jl]:.4f}")
rms = lambda E, a, b: np.sqrt((E[a - 1:b, 1:]**2).mean(0))
small, large = scales[1:] < 0.3, scales[1:] > 3.0
for a, b in ((1, 5), (15, 20), (50, 60)):
    rr = rms(res["raw"]["err"], a, b); qq = rms(res["standardised"]["err"], a, b)
    print(f"     rms error (standardised units), epochs {a}-{b}: raw, s<0.3 ({small.sum()} columns): {rr[small].mean():.3f}, "
          f"s>3 ({large.sum()} columns): {rr[large].mean():.3f}; standardised run: {qq.mean():.3f}")
th_hat, H, it = newton(X * scales); c_hat = cost(X * scales, th_hat)
for gam in (0.001, 0.01):                  # no single gamma is right for both groups
    ex = np.array([cost(X * scales, th) - c_hat for th in adam_history(X * scales, gam)])
    print(f"     raw: Adam gamma={gam}: excess after 5,20,40,60 epochs: "
          + " ".join(f"{ex[k-1]:.1e}" for k in (5, 20, 40, 60)) + f"; mean 40-60 {ex[39:].mean():.1e}")
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))
ax = axes[0]; ep = np.arange(1, 61)
ax.semilogy(ep, res["standardised"]["ex"], color="C0", lw=1.4, label=r"Adam, standardised, $\gamma=10^{-3}$")
ax.semilogy(ep, res["raw"]["ex"], color="crimson", lw=1.4, label=r"Adam, raw features, $\gamma=3\times10^{-3}$")
ax.axhline(p / (2 * n), color="k", ls=":", lw=1); ax.text(1, p / (2 * n) * 1.15, "$p/(2n)$", fontsize=8)
ax.axvline(20, color="0.5", ls="--", lw=0.8)
ax.set_xlabel("epochs"); ax.set_ylabel(r"$C(\boldsymbol{\theta})-C(\hat{\boldsymbol{\theta}})$")
ax.legend(fontsize=8, loc="upper right"); ax.set_title("excess cost", fontsize=10)
ax = axes[1]
for (a, b), col in (((1, 5), "C1"), ((15, 20), "crimson"), ((50, 60), "C2")):
    ax.loglog(scales[1:], rms(res["raw"]["err"], a, b), "o", color=col, ms=5, mec="k", mew=0.4, label=f"raw, epochs {a}-{b}")
lo = rms(res["standardised"]["err"], 15, 20)
ax.axhspan(lo.min(), lo.max(), color="C0", alpha=0.15, label="standardised, epochs 15-20")
g = np.array([0.1, 7.0]); ax.loglog(g, 3e-3 * g, ":", color="k", lw=1, label=r"one step, $\gamma s_j$")
ax.set_ylim(2e-4, 6.0); ax.set_xlabel(r"scale $s_j$ of feature column $j$")
ax.set_ylabel(r"rms error of $\theta_j$, standardised units")
ax.legend(fontsize=7.5, loc="upper center", ncol=2); ax.set_title("error of each parameter", fontsize=10)
save(fig, 5, "logreg_adam_scales")

"""Chapter 14: every verification quoted in Sections 14.6 and 14.6.1."""
import numpy as np
import rbm

# --- 1. the gradient identity, Theorem 14.1 ---------------------------------
print("--- exact gradient vs finite differences of the log-likelihood ---")
rng = np.random.default_rng(0)
M, N = 6, 4
P = rbm.init_rbm(M, N, rng, scale=0.5)
X = (rng.random((40, M)) < 0.5).astype(float)
g = rbm.exact_gradient(P, X)


def ll():
    return rbm.log_likelihood(P, X)


for k in ["W", "a", "b"]:
    A = P[k]; err = 0.0
    it = np.nditer(A, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index; old = A[i]; h = 1e-6
        A[i] = old + h; fp = ll()
        A[i] = old - h; fm = ll()
        A[i] = old
        num = (fp - fm) / (2 * h)
        err = max(err, abs(num - g[k][i]) / max(1e-12, abs(num) + abs(g[k][i])))
        it.iternext()
    print(f"  {k}: max relative error over all {A.size} entries = {err:.2e}")

# --- 2. block Gibbs samples the model, Proposition 14.5 ---------------------
print("\n=== does block Gibbs sample the model distribution? ===")
M2 = 6
P2 = rbm.init_rbm(M2, 3, np.random.default_rng(1), scale=1.2)
V = rbm.all_states(M2)
logp = -rbm.free_energy(P2, V); logp -= np.logaddexp.reduce(logp)
pex = np.exp(logp)
idx = {tuple(v): i for i, v in enumerate(V.astype(int))}
rng = np.random.default_rng(0)
for nsweep, burn in [(200, 50), (2000, 200), (20000, 500)]:
    counts = np.zeros(len(V))
    Xs = (rng.random((500, M2)) < 0.5).astype(float)
    for t in range(nsweep):
        Xs, _, _, _ = rbm.gibbs_step(P2, Xs, rng)
        if t >= burn:
            for row in Xs.astype(int):
                counts[idx[tuple(row)]] += 1
    tv = 0.5 * np.abs(counts / counts.sum() - pex).sum()
    print(f"  {nsweep:6d} sweeps: total-variation distance to the exact p(v) = {tv:.4f}")
print(f"  (a uniform guess would give "
      f"{0.5*np.abs(np.ones(len(V))/len(V) - pex).sum():.4f})")

# --- 3. the bias of CD-k, Section 14.6.1 ------------------------------------
print("\n=== CD-k is a BIASED estimator of the gradient ===")
M3, N3 = 8, 4
P3 = rbm.init_rbm(M3, N3, np.random.default_rng(1), scale=1.0)
base = (np.random.default_rng(2).random((4, M3)) < 0.5).astype(float)
X3 = base[np.random.default_rng(3).integers(0, 4, 200)]
flat = lambda d: np.concatenate([np.ravel(d[k]) for k in ("W", "a", "b")])
ge = flat(rbm.exact_gradient(P3, X3))
print("  k     cos(CD-k, exact)   ||CD-k - exact|| / ||exact||")
for k in [1, 2, 5, 10, 25, 100]:
    acc = None
    for r in range(60):                    # average away the sampling noise
        gk = flat(rbm.cd_gradient(P3, X3, k=k, rng=np.random.default_rng(1000 + r)))
        acc = gk if acc is None else acc + gk
    gk = acc / 60
    cos = float(gk @ ge / (np.linalg.norm(gk) * np.linalg.norm(ge)))
    rel = float(np.linalg.norm(gk - ge) / np.linalg.norm(ge))
    print(f"  {k:3d}     {cos:.6f}          {rel:.4f}")

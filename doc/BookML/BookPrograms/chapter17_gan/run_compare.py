"""The three numerical experiments of Chapter 17.

1.  The game at equilibrium.  A balanced GAN on the eight-Gaussian ring:
    does $D(\\bm{x})$ really go to $1/2$ and $V$ to $-2\\log 2$?
2.  Gradient starvation.  Train a discriminator against a fixed, untrained
    generator until it is nearly perfect, then measure the generator gradient
    that the saturating and non-saturating losses deliver.
3.  Mode collapse.  The 5x5 grid of narrow modes, with three objectives.

Everything written here is quoted in the chapter; nothing in the chapter is
quoted from anywhere else.
"""
import time

import autograd.numpy as np
from autograd import grad

import gan

out = open("compare.txt", "w", buffering=1)

# ===========================================================================
# 1.  the game at equilibrium
# ===========================================================================
rng = np.random.default_rng(0)
X8, C8 = gan.eight_gaussians(6000, rng)
np.save("data8.npy", X8)
np.save("centres8.npy", C8)

t0 = time.time()
P8, Q8, h8 = gan.train(X8, mode="nonsat", n_iter=4000, n_critic=1, eta=2e-3,
                       rng=np.random.default_rng(1))
dt8 = time.time() - t0
h8 = np.array(h8)
S8 = gan.sample(P8, 4000, rng=np.random.default_rng(2))
np.save("samples8.npy", S8)
np.save("hist8.npy", h8)

nm8, cnt8, near8 = gan.modes_covered(S8, C8)
out.write("=== 1. the game at equilibrium: eight Gaussians, non-saturating ===\n")
out.write(f"  trained in {dt8:.1f}s, {nm8}/8 modes covered, "
          f"{near8*100:.1f}% of samples within 0.5 of a mode\n")
out.write(f"  energy distance to the data: {gan.energy_distance(S8, X8):.5f}\n\n")
out.write("     iteration        V        D(x)     D(G(z))\n")
for row in h8[[0, 4, 9, 19, 29, 39]]:
    out.write(f"  {int(row[0]):10d}  {row[1]:9.4f}  {row[2]:7.4f}  {row[3]:7.4f}\n")
tail = h8[-10:]
out.write(f"  mean over the last 1000 iterations: V = {tail[:,1].mean():.4f}, "
          f"D(x) = {tail[:,2].mean():.4f}, D(G(z)) = {tail[:,3].mean():.4f}\n")
out.write(f"  the theoretical equilibrium is V = -2log2 = {-2*np.log(2):.4f}, "
          f"D = 1/2\n\n")

# ===========================================================================
# 2.  gradient starvation: what a strong discriminator does to each loss
# ===========================================================================
out.write("=== 2. the generator gradient under a strong discriminator ===\n")
rng = np.random.default_rng(10)
P0 = gan.init_mlp((2, 64, 64, 2), rng)          # a fixed, untrained generator
Q0 = gan.init_mlp((2, 64, 64, 1), rng)
M, V_, t_ = gan.adam_state(Q0)
dg = grad(gan.d_loss, 0)
g_sat = grad(gan.g_loss_sat, 0)
g_non = grad(gan.g_loss_nonsat, 0)


def gnorm(G):
    return float(np.sqrt(sum(np.sum(gW ** 2) + np.sum(gb ** 2) for gW, gb in G)))


out.write("   D steps    D(G(z))    ||dL_sat/dtheta||   ||dL_nonsat/dtheta||"
          "   ratio  1/D(G(z))\n")
erng = np.random.default_rng(99)          # evaluation only: never trains
z_eval = erng.normal(size=(2048, 2))
starve = []
for step in range(2001):
    if step % 25 == 0:
        dgz = float(np.mean(gan.sigmoid(gan.discriminator(
            Q0, gan.generator(P0, z_eval)))))
        a = gnorm(g_sat(P0, Q0, z_eval))
        b = gnorm(g_non(P0, Q0, z_eval))
        starve.append((step, dgz, a, b))
        if step in (0, 50, 200, 500, 1000, 2000):
            out.write(f"  {step:7d}   {dgz:8.2e}   {a:17.3e}   {b:18.3e} "
                      f"{b/a:8.1f} {1/dgz:10.1f}\n")
    xb = X8[rng.integers(0, len(X8), 256)]
    zb = rng.normal(size=(256, 2))
    Q0 = gan.adam_step(Q0, dg(Q0, P0, xb, zb, 1.0), M, V_, t_, 2e-3)
np.save("starve.npy", np.array(starve))
out.write("  the generator has not moved: only the discriminator was trained.\n"
          "  The measured ratio grows like 1/D(G(z)) but stays below it, because\n"
          "  a batch gradient is dominated by the few samples the discriminator\n"
          "  still rates highest; Eq. (17.gradratio) is the pointwise statement.\n\n")

# the same, with one-sided label smoothing, from the same starting point
rng = np.random.default_rng(10)
P0s = gan.init_mlp((2, 64, 64, 2), rng)
Q0s = gan.init_mlp((2, 64, 64, 1), rng)
M, V_, t_ = gan.adam_state(Q0s)
for step in range(2001):
    xb = X8[rng.integers(0, len(X8), 256)]
    zb = rng.normal(size=(256, 2))
    Q0s = gan.adam_step(Q0s, dg(Q0s, P0s, xb, zb, 0.9), M, V_, t_, 2e-3)
zb = rng.normal(size=(2048, 2))
out.write(f"  with one-sided label smoothing s = 0.9, after 2000 steps:\n")
out.write(f"     D(x)    = "
          f"{float(np.mean(gan.sigmoid(gan.discriminator(Q0s, X8[:2048])))):.4f} "
          f"(unsmoothed run: "
          f"{float(np.mean(gan.sigmoid(gan.discriminator(Q0, X8[:2048])))):.4f})\n")
out.write(f"     D(G(z)) = "
          f"{float(np.mean(gan.sigmoid(gan.discriminator(Q0s, gan.generator(P0s, zb))))):.2e}"
          f"  ||dL_sat/dtheta|| = {gnorm(g_sat(P0s, Q0s, zb)):.3e}\n\n")

# ===========================================================================
# 3.  mode collapse on the 5x5 grid
# ===========================================================================
rng = np.random.default_rng(0)
X25, C25 = gan.grid_gaussians(8000, rng)
np.save("data25.npy", X25)
np.save("centres25.npy", C25)

runs = [("non-saturating", "nonsat",
         dict(mode="nonsat", n_iter=8000, n_critic=1, eta=2e-3)),
        ("strong discriminator", "strongd",
         dict(mode="nonsat", n_iter=2000, n_critic=3, eta=2e-3, eta_d=4e-3,
              hidden_d=(128, 128))),
        ("WGAN-GP", "wgangp",
         dict(mode="wgan", n_iter=6000, n_critic=5, eta=1e-3))]

out.write("=== 3. mode collapse: a 5x5 grid of narrow modes ===\n")
out.write("  objective              time   modes  in-mode   energy dist  "
          "min/max per mode\n")
rows = {}
for name, tag, kw in runs:
    t0 = time.time()
    P, Q, h = gan.train(X25, rng=np.random.default_rng(1), **kw)
    dt = time.time() - t0
    S = gan.sample(P, 5000, rng=np.random.default_rng(2))
    nm, cnt, near = gan.modes_covered(S, C25, tol=0.25, frac=0.005)
    ed = gan.energy_distance(S, X25)
    np.save(f"samples25_{tag}.npy", S)
    np.save(f"hist25_{tag}.npy", np.array(h))
    np.save(f"counts25_{tag}.npy", cnt)
    rows[tag] = cnt
    out.write(f"  {name:21s} {dt:5.0f}s  {nm:3d}/25  {near*100:6.1f}%  "
              f"{ed:11.5f}   {cnt.min():4d} /{cnt.max():5d}\n")
out.write("  a uniform generator would put 200 of the 5000 samples on each mode\n\n")
for name, tag, _ in runs:
    out.write(f"  {name} -- samples per mode\n")
    for r in rows[tag].reshape(5, 5):
        out.write("      " + " ".join(f"{c:5d}" for c in r) + "\n")
    out.write("\n")
out.close()
print(open("compare.txt").read())

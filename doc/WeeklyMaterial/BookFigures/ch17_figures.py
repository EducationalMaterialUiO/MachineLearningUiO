"""Figures for Chapter 17, generative adversarial networks.

Run ``verify_gan.py`` and ``run_compare.py`` in
``BookPrograms/chapter17_gan`` first: this script only plots the arrays they
write, so that every number in a figure is the number quoted in the text.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "BookPrograms", "chapter17_gan"))
from common import plt, np, save                                # noqa: E402

RUN = os.environ.get("CH17_RUN", ".")
L = lambda name: np.load(os.path.join(RUN, name))


# ---------------------------------------------------------------------------
# figure 1: the game at equilibrium
# ---------------------------------------------------------------------------
X = L("data8.npy")
S = L("samples8.npy")
h = L("hist8.npy")

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
axs[0].plot(X[:2000, 0], X[:2000, 1], ".", ms=2, color="0.7", label="data")
axs[0].plot(S[:2000, 0], S[:2000, 1], ".", ms=2, color="C0",
            label="generator")
axs[0].set_xlabel("$x_1$")
axs[0].set_ylabel("$x_2$")
axs[0].set_title("(a) eight Gaussians")
axs[0].legend(fontsize=8, markerscale=4)
axs[0].set_aspect("equal")

axs[1].plot(h[:, 0], h[:, 2], "C0", lw=1.4, label=r"$D(\mathbf{x})$")
axs[1].plot(h[:, 0], h[:, 3], "C3", lw=1.4, label=r"$D(G(\mathbf{z}))$")
axs[1].axhline(0.5, color="k", ls=":", lw=1.0)
axs[1].set_ylim(0, 1)
axs[1].set_xlabel("iteration")
axs[1].set_ylabel("discriminator output")
axs[1].set_title("(b) the discriminator stays near $1/2$")
axs[1].legend(fontsize=9)

axs[2].plot(h[:, 0], h[:, 1], "C0", lw=1.4)
axs[2].axhline(-2 * np.log(2), color="k", ls=":", lw=1.0)
axs[2].annotate(r"$-2\log 2$", (h[2, 0], -2 * np.log(2)),
                textcoords="offset points", xytext=(6, -14), fontsize=9)
axs[2].set_xlabel("iteration")
axs[2].set_ylabel("$V(G,D)$")
axs[2].set_title("(c) the value function")
save(fig, 17, "gan_equilibrium")


# ---------------------------------------------------------------------------
# figure 2: the discriminator the theorem predicts, and the two gradients
# ---------------------------------------------------------------------------
g = L("dstar_grid.npy")
pr = L("dstar_pr.npy")
pg = L("dstar_pg.npy")
Dn = L("dstar_net.npy")
Ds = L("dstar_closed.npy")
st = L("starve.npy")                    # step, D(G(z)), |dL_sat|, |dL_nonsat|

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
axs[0].plot(g, pr, "C0", lw=1.6, label=r"$p_r=\mathcal{N}(0,1)$")
axs[0].plot(g, pg, "C3", lw=1.6, label=r"$p_g=\mathcal{N}(1.5,0.6^2)$")
axs[0].fill_between(g, 0, np.minimum(pr, pg), color="0.85")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("density")
axs[0].set_title("(a) the two distributions")
axs[0].legend(fontsize=8)

axs[1].plot(g, Ds, "k", lw=2.0, label=r"$p_r/(p_r+p_g)$")
axs[1].plot(g, Dn, "C0--", lw=1.6, label="trained network")
mix = 0.5 * (pr + pg)
ok = mix > 0.01 * mix.max()
axs[1].axvspan(g[ok].min(), g[ok].max(), color="C2", alpha=0.10)
axs[1].set_xlabel("$x$")
axs[1].set_ylabel(r"$D(x)$")
axs[1].set_title("(b) the optimal discriminator")
axs[1].legend(fontsize=8, loc="center left")
axs[1].text(0.5 * (g[ok].min() + g[ok].max()), 0.04,
            "where the data are", ha="center", fontsize=8, color="C2")

d = st[:, 1]
axs[2].loglog(d, st[:, 3] / st[:, 2], "C0o", ms=3,
              label="measured ratio of the two")
dd = np.logspace(np.log10(d.min()), np.log10(d.max()), 50)
axs[2].loglog(dd, (1 - dd) / dd, "k--", lw=1.2,
              label=r"pointwise $(1-D)/D$")
axs[2].set_xlabel(r"$D(G(\mathbf{z}))$ after training $D$ alone")
axs[2].set_ylabel(r"$\|\nabla\mathcal{L}_G^{\rm non-sat}\|\,/\,"
                  r"\|\nabla\mathcal{L}_G^{\rm sat}\|$")
axs[2].set_title("(c) gradient starvation")
axs[2].legend(fontsize=8)
save(fig, 17, "gan_discriminator")


# ---------------------------------------------------------------------------
# figure 3: where the Jensen-Shannon divergence gives up
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
th = np.linspace(0, 3, 601)
js_u = np.minimum(th, 1.0) * np.log(2.0)
axs[0].plot(th, js_u, "C0", lw=1.8, label=r"$D_{JS}$")
axs[0].plot(th, th, "C3", lw=1.8, label=r"$W$")
axs[0].axhline(np.log(2), color="k", ls=":", lw=1.0)
axs[0].annotate(r"$\log 2$", (2.4, np.log(2)), textcoords="offset points",
                xytext=(0, 6), fontsize=9)
axs[0].set_xlabel(r"$\theta$")
axs[0].set_ylabel("divergence / distance")
axs[0].set_title(r"(a) $U[0,1]$ against $U[\theta,\theta+1]$")
axs[0].legend(fontsize=9)
axs[0].set_ylim(0, 3)

mu = np.linspace(0, 8, 401)
x = np.linspace(-14, 22, 40001)
dx = x[1] - x[0]
gs = lambda z, m: np.exp(-0.5 * (z - m) ** 2) / np.sqrt(2 * np.pi)
p = gs(x, 0.0)
js_g = []
for m in mu:
    q = gs(x, m)
    mm = 0.5 * (p + q)
    ok = mm > 1e-300
    js_g.append(0.5 * np.sum(p[ok] * np.log(p[ok] / mm[ok] + 1e-300) * dx)
                + 0.5 * np.sum(q[ok] * np.log(q[ok] / mm[ok] + 1e-300) * dx))
axs[1].plot(mu, js_g, "C0", lw=1.8, label=r"$D_{JS}$")
axs[1].plot(mu, mu, "C3", lw=1.8, label=r"$W$")
axs[1].axhline(np.log(2), color="k", ls=":", lw=1.0)
axs[1].set_xlabel(r"$\mu$")
axs[1].set_ylabel("divergence / distance")
axs[1].set_title(r"(b) $\mathcal{N}(0,1)$ against $\mathcal{N}(\mu,1)$")
axs[1].legend(fontsize=9)
axs[1].set_ylim(0, 8)
save(fig, 17, "gan_divergences")


# ---------------------------------------------------------------------------
# figure 4: mode collapse
# ---------------------------------------------------------------------------
X25 = L("data25.npy")
tags = [("nonsat", "non-saturating"), ("strongd", "strong discriminator"),
        ("wgangp", "WGAN-GP")]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
axs[0].plot(X25[:3000, 0], X25[:3000, 1], ".", ms=2, color="0.6")
axs[0].set_title("(a) the target: 25 modes")
axs[0].set_xlabel("$x_1$")
axs[0].set_ylabel("$x_2$")
for k, (tag, name) in enumerate(tags):
    S = L(f"samples25_{tag}.npy")
    axs[k + 1].plot(X25[:2000, 0], X25[:2000, 1], ".", ms=2, color="0.85")
    axs[k + 1].plot(S[:3000, 0], S[:3000, 1], ".", ms=2, color=f"C{k}")
    axs[k + 1].set_title(f"({chr(98+k)}) {name}")
    axs[k + 1].set_xlabel("$x_1$")
for a in axs:
    a.set_aspect("equal")
    a.set_xlim(-3.2, 3.2)
    a.set_ylim(-3.2, 3.2)
save(fig, 17, "gan_collapse")

fig, ax = plt.subplots(figsize=(7.5, 3.4))
w = 0.27
idx = np.arange(25)
for k, (tag, name) in enumerate(tags):
    c = np.sort(L(f"counts25_{tag}.npy"))[::-1]
    ax.bar(idx + (k - 1) * w, c, w, label=name, color=f"C{k}")
ax.axhline(200, color="k", ls=":", lw=1.0)
ax.annotate("uniform coverage", (18, 200), textcoords="offset points",
            xytext=(0, 6), fontsize=9)
ax.set_xlabel("mode, ordered by how often it is generated")
ax.set_ylabel("samples out of 5000")
ax.legend(fontsize=9)
save(fig, 17, "gan_modecounts")

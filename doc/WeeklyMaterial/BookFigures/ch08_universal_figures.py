"""Figures for the quantitative part of Section 8.4.

Run ``universal.py`` in ``BookPrograms/chapter08_neural_networks`` first: it
writes the ``.npy`` files this script reads.  Kept separate from
``ch08_figures.py`` so that it has no dependency on the from-scratch network
implementation.

    cd BookFigures && python3 ch08_universal_figures.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import plt, np, save

# --- appended: the quantitative approximation results of Section 8.4 --------
# Needs universal.py to have been run in BookPrograms/chapter08_neural_networks.
import os as _os
RUN = _os.environ.get("CH8_RUN", _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "..", "BookPrograms", "chapter08_neural_networks"))
L = lambda n: np.load(_os.path.join(RUN, n))

x = L("plot_x.npy")
tgt = L("plot_target.npy")

fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.6))

# (a) the network of Eq. (8.interpolant), written down rather than trained
axs[0].plot(x, tgt, "k-", lw=1.6, label=r"$F(x)=\sin 2\pi x$")
for Np, c, ls in [(4, "C3", "--"), (16, "C0", "-")]:
    axs[0].plot(x, L(f"plot_interp{Np}.npy"), c, ls=ls, lw=1.3,
                label=f"$N={Np}$ ReLU units")
    t = np.arange(Np + 1) / Np
    axs[0].plot(t, np.sin(2 * np.pi * t), "o", color=c, ms=3.5)
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$g(x)$")
axs[0].legend(fontsize=8)
axs[0].set_title("(a) the construction of Thm 8.relurate", fontsize=10)

# (b) the bound, and what training actually achieves
cs, cw = L("construct_smooth.npy"), L("construct_worst.npy")
tr = L("trained_vs_constructed.npy")               # N, constructed, default, spread
axs[1].loglog(cs[:, 0], cs[:, 2], "k:", lw=1.4, label=r"bound $L/2N$")
axs[1].loglog(cw[:, 0], cw[:, 1], "C2s-", ms=4,
              label="constructed, worst case")
axs[1].loglog(cs[:, 0], cs[:, 1], "C0o-", ms=4, label=r"constructed, $\sin$")
axs[1].loglog(tr[:, 0], tr[:, 3], "C3^--", ms=5, label="trained, kinks spread")
axs[1].loglog(tr[:, 0], tr[:, 2], "C1v--", ms=5, label="trained, default init")
axs[1].set_xlabel("hidden units $N$")
axs[1].set_ylabel(r"$\sup|F-g|$")
axs[1].legend(fontsize=7.5, loc="lower left")
axs[1].set_title("(b) representable versus reachable", fontsize=10)

# (c) Barron's rate in three dimensions
for kap, ls, al in [(2, "--", 0.45), (8, "-", 1.0)]:
    t = L(f"barron_k{kap}.npy")                    # N, d1, d5, d20, bound
    for j, (d, c) in enumerate(zip([1, 5, 20], ["C0", "C1", "C3"]), start=1):
        axs[2].loglog(t[:, 0], t[:, j], ls, marker="o", color=c, ms=4,
                      alpha=al, label=f"$d={d}$" if kap == 8 else None)
axs[2].loglog(t[:, 0], 3e-2 / t[:, 0], "k:", lw=1.3, label=r"slope $-1$")
axs[2].set_xlabel("hidden units $N$")
axs[2].set_ylabel("mean squared error")
axs[2].legend(fontsize=8, ncol=2)
axs[2].set_title(r"(c) Thm 8.barron: solid $\kappa=8$, faint $\kappa=2$",
                 fontsize=10)
save(fig, 8, "universal_construction")

# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.6))

# (a) the sawtooth: two units per layer, exponentially many pieces
for k, c in zip([1, 2, 3, 4], ["C0", "C1", "C2", "C3"]):
    axs[0].plot(x, L(f"plot_saw{k}.npy") + 1.25 * (4 - k), color=c, lw=1.1)
    axs[0].text(1.01, 1.25 * (4 - k) + 0.35, f"$s_{k}$", color=c, fontsize=9)
axs[0].set_xlabel("$x$")
axs[0].set_yticks([])
axs[0].set_xlim(0, 1.06)
axs[0].set_title(r"(a) $s_k=T^{\circ k}$, depth $k$, $2k$ units", fontsize=10)

# (b) what a single hidden layer manages on s_3
axs[1].plot(x, L("plot_saw3.npy"), "-", color="0.6", lw=3.2, label="$s_3$")
for N, c, ls in [(4, "C1", "-"), (8, "C3", "-"), (32, "C0", "--")]:
    axs[1].plot(x, L(f"plot_saw_fit_N{N}.npy"), color=c, lw=1.2, ls=ls,
                label=f"one hidden layer, $N={N}$")
axs[1].set_xlabel("$x$")
axs[1].legend(fontsize=8, loc="upper center", ncol=1)
axs[1].set_ylim(-0.35, 1.75)
axs[1].set_title("(b) the shallow fits of $s_3$", fontsize=10)

# (c) the wall of Theorem 8.depth
w = L("depth_wall.npy")                            # k, N, sup error, pieces
for k, c in zip([2, 3, 4], ["C0", "C1", "C3"]):
    m = w[:, 0] == k
    axs[2].semilogx(w[m, 1], w[m, 2], "o-", color=c, ms=5, label=f"$k={k}$")
    axs[2].axvline(2 ** k - 1, color=c, ls=":", lw=1.1)
axs[2].axhline(0.5, color="k", ls="--", lw=1.0)
axs[2].text(2.2, 0.52, r"bound of Thm 8.depth: $1/2$", fontsize=8)
axs[2].set_xlabel("width $N$ of the single hidden layer")
axs[2].set_ylabel(r"$\sup|g-s_k|$")
axs[2].set_xticks([2, 4, 8, 16, 32, 64])
axs[2].set_xticklabels(["2", "4", "8", "16", "32", "64"])
axs[2].legend(fontsize=8)
axs[2].set_title(r"(c) dotted lines: $N=2^k-1$", fontsize=10)
save(fig, 8, "depth_separation")

"""Figures for the Ising example, Section 3.ising and Section 8.ising.

Run ising_regression.py in BookPrograms/chapter03_linear_regression and
ising_network.py in BookPrograms/chapter08_neural_networks first; this script
reads the .npy files they leave behind and writes into
BookFigures/chapter03_linear_regression and
BookFigures/chapter08_neural_networks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import plt, np, save

# the .npy files are written next to the programs
R = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("ISING_RUN", os.path.join(R, "..", "BookPrograms",
                                             "chapter03_linear_regression"))
D8 = os.environ.get("ISING_RUN8", os.path.join(R, "..", "BookPrograms",
                                               "chapter08_neural_networks"))
def L_(name):
    for d in (D, D8):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(name)

L, n, ntr, lam = L_("meta.npy")
L = int(L)

# ===========================================================================
# Chapter 3, figure 1: the three coupling matrices
# ===========================================================================
Js = [("ordinary least squares", L_("J_ols.npy")),
      ("Ridge, $\\lambda=0.01$", L_("J_ridge.npy")),
      ("Lasso, $\\lambda=0.01$", L_("J_lasso.npy"))]

with plt.rc_context({"figure.autolayout": False}):
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for ax, (t, J) in zip(axs, Js):
        im = ax.imshow(J, cmap="seismic", vmin=-1.0, vmax=1.0)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("$k$")
        ax.grid(False)
    axs[0].set_ylabel("$j$")
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.12, top=0.92, wspace=0.20)
    cax = fig.add_axes([0.925, 0.12, 0.016, 0.80])
    fig.colorbar(im, cax=cax, label="$J_{jk}$")
    save(fig, 3, "ising_couplings")

# ===========================================================================
# Chapter 3, figure 2: the singular value cliff
# ===========================================================================
if os.path.exists(os.path.join(D, "svals.npy")):
    s = L_("svals.npy")
    pred = L * (L - 1) // 2 + 1
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(np.arange(len(s)), np.maximum(s, 1e-16), "-", color="C0", lw=1.4)
    ax.axvline(pred - 0.5, color="C3", ls="--", lw=1.2)
    ax.text(pred + 30, s[0] / 3,
            "$L(L-1)/2+1=%d$" % pred, color="C3", fontsize=9)
    ax.set_xlabel("index $i$")
    ax.set_ylabel("$\\sigma_i$")
    ax.set_title("singular values of the $%d\\times%d$ design matrix"
                 % (int(n), L * L), fontsize=10)
    save(fig, 3, "ising_spectrum")

# ===========================================================================
# Chapter 3, figure 3: R^2 against the penalty
# ===========================================================================
pr = L_("path_ridge.npy")          # lambda, train, test
pl = L_("path_lasso.npy")
nz = L_("path_lasso_nnz.npy")
ols_tr, ols_te = L_("ols_r2.npy")

fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.0))
axs[0].semilogx(pr[:, 0], pr[:, 1], "o-", ms=3.5, color="C0", label="train")
axs[0].semilogx(pr[:, 0], pr[:, 2], "s-", ms=3.5, color="C3", label="test")
axs[0].axhline(ols_te, color="k", ls=":", lw=1.2, label="OLS, test")
axs[0].set_xlabel("$\\lambda$")
axs[0].set_ylabel("$R^{2}$")
axs[0].set_ylim(-0.08, 1.05)
axs[0].legend(fontsize=8, loc="lower left")
axs[0].set_title("(a) Ridge", fontsize=10)

axs[1].semilogx(pl[:, 0], pl[:, 1], "o-", ms=3.5, color="C0", label="train")
axs[1].semilogx(pl[:, 0], pl[:, 2], "s-", ms=3.5, color="C3", label="test")
axs[1].axhline(ols_te, color="k", ls=":", lw=1.2, label="OLS, test")
axs[1].set_xlabel("$\\lambda$")
axs[1].set_ylabel("$R^{2}$")
axs[1].set_ylim(-0.08, 1.05)
axs[1].legend(fontsize=8, loc="lower left")
axs[1].set_title("(b) Lasso", fontsize=10)
ax2 = axs[1].twinx()
ax2.semilogx(pl[:, 0], nz, "^--", ms=3.5, color="C2", lw=1.0)
ax2.set_ylabel("non-zero coefficients", color="C2")
ax2.tick_params(axis="y", labelcolor="C2")
ax2.grid(False)
save(fig, 3, "ising_penalty")

# ===========================================================================
# Chapter 8
# ===========================================================================
if os.path.exists(os.path.join(D8, "curve_raw.npy")):
    cr = L_("curve_raw.npy")        # epoch, train, test
    co = L_("curve_out.npy")
    wr = L_("width_raw.npy")        # width, train, test
    sz = L_("size_raw.npy")         # rows, test

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))

    axs[0].plot(cr[:, 0], cr[:, 2], "-", color="C3", lw=1.5,
                label="$L=%d$ raw spins" % L)
    axs[0].plot(co[:, 0], co[:, 2], "-", color="C0", lw=1.5,
                label="$L^{2}=%d$ products" % (L * L))
    axs[0].axhline(1.0, color="k", ls=":", lw=1.0)
    axs[0].axhline(0.522863, color="C7", ls="--", lw=1.0)
    axs[0].text(210, 0.545, "OLS on the products, Ch. 3", color="C7", fontsize=7.5)
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("$R^{2}$ on the test set")
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].legend(fontsize=8, loc="lower right")
    axs[0].set_title("(a) which inputs?", fontsize=10)

    axs[1].semilogx(wr[:, 0], wr[:, 1], "o-", ms=4, color="C0", label="train")
    axs[1].semilogx(wr[:, 0], wr[:, 2], "s-", ms=4, color="C3", label="test")
    axs[1].axvline(2 * L, color="C2", ls="--", lw=1.2)
    axs[1].text(2 * L * 1.08, -0.55, "$2L$", color="C2", fontsize=9)
    axs[1].set_xlabel("hidden units")
    axs[1].set_ylabel("$R^{2}$")
    axs[1].legend(fontsize=8, loc="lower right")
    axs[1].set_title("(b) width, raw spins", fontsize=10)

    axs[2].semilogx(sz[:, 0], sz[:, 1], "o-", ms=4, color="C3")
    axs[2].set_xlabel("training configurations")
    axs[2].set_ylabel("$R^{2}$ on the test set")
    axs[2].set_title("(c) how much data?", fontsize=10)
    save(fig, 8, "ising_network")

    names = ["OLS, $L^2$ products", "Ridge, $L^2$ products",
             "Lasso, $L^2$ products", "network, $L^2$ products",
             "OLS, raw spins", "network, raw spins",
             "exact network, raw spins",
             "network, raw spins, $8000$ train"]
    vals = np.concatenate([L_("summary_r2.npy"), L_("big_raw.npy")[1:]])
    cols = ["C0", "C0", "C0", "C1", "C0", "C1", "C2", "C1"]
    hat = [""] * 7 + ["//"]
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    yp = np.arange(len(vals))[::-1]
    ax.barh(yp, np.clip(vals, 0, None), color=cols, alpha=0.85, hatch=hat)
    for y, v in zip(yp, vals):
        ax.text(max(v, 0) + 0.015, y, "%.4f" % v, va="center", fontsize=8)
    ax.set_yticks(yp)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.22)
    ax.set_xlabel("$R^{2}$ on the test set")
    ax.grid(axis="y", alpha=0)
    save(fig, 8, "ising_comparison")

print("figures done")

"""The one-dimensional Ising model as a linear regression problem, Section 3.ising.

The energy of a spin configuration under the general pairwise Hamiltonian
H = -sum_{jk} J_{jk} s_j s_k is linear in the products s_j s_k, so recovering
the coupling matrix J from (configuration, energy) pairs is exactly the linear
model of this chapter with p = L^2 features.  Everything the chapter proves
about OLS, Ridge and the Lasso is visible in the answer.

1.  The design matrix and its rank, which is deficient by an exact and
    predictable amount rather than by numerical accident.
2.  The three estimators, and the coupling matrices they return.
3.  Proposition 3.duplicate, checked: with two identical columns the 2-norm
    methods split the coefficient equally and the 1-norm does not.
4.  R^2 against the penalty for all three, and the value the chapter quotes.

Every number printed here is quoted in the text.
"""
import numpy as np
from numpy.linalg import matrix_rank, svd, lstsq

out = open("ising_regression.txt", "w", buffering=1)
rng = np.random.default_rng(2718)

L, n, Jtrue = 40, 10000, 1.0

# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------
spins = rng.choice([-1, 1], size=(n, L))
energies = -Jtrue * np.einsum("ij,ij->i", spins, np.roll(spins, 1, axis=1))
X = np.einsum("ij,ik->ijk", spins, spins).reshape(n, L * L)
y = energies

out.write("=== 1. the design matrix ===\n")
out.write(f"  L = {L} spins, n = {n} configurations, J = {Jtrue}\n")
out.write(f"  design matrix X with X[i, j*L+k] = s_j s_k : shape {X.shape}\n")
out.write(f"  energies: min {y.min():.0f}, max {y.max():.0f}, "
          f"mean {y.mean():.4f}, std {y.std():.4f}\n\n")

out.write("  Two exact facts about this design matrix, both structural:\n")
out.write("    (a) the L diagonal columns are s_j^2 = 1 identically, so they\n"
          "        are all equal to the constant column and to each other;\n")
out.write("    (b) column (j,k) and column (k,j) are identical, since\n"
          "        s_j s_k = s_k s_j.\n\n")
diag_cols = [j * L + j for j in range(L)]
out.write(f"  max |X[:, diagonal] - 1|                    : "
          f"{np.abs(X[:, diag_cols] - 1.0).max():.3e}\n")
sym = max(np.abs(X[:, j * L + k] - X[:, k * L + j]).max()
          for j in range(L) for k in range(j))
out.write(f"  max |X[:, (j,k)] - X[:, (k,j)]| over all j<k: {sym:.3e}\n\n")

pred_rank = L * (L - 1) // 2 + 1
out.write(f"  distinct columns therefore number L(L-1)/2 + 1 = {pred_rank},\n"
          f"  so rank(X) <= {pred_rank} however many configurations we draw.\n")
r = matrix_rank(X)
out.write(f"  measured rank(X) with n = {n} rows          : {r}\n")
out.write(f"  number of columns p = L^2                   : {L*L}\n")
s = svd(X, compute_uv=False)
np.save("svals.npy", s)                    # for the figure of Fig. 3.isingspectrum
out.write(f"  singular values: sigma_0 = {s[0]:.4f}, "
          f"sigma_{pred_rank-1} = {s[pred_rank-1]:.4e}, "
          f"sigma_{pred_rank} = {s[pred_rank]:.4e}\n")
out.write(f"  ratio sigma_{pred_rank-1} / sigma_{pred_rank} : "
          f"{s[pred_rank-1]/max(s[pred_rank], 1e-300):.3e}\n")
out.write("  the spectrum falls off a cliff exactly where the counting says it\n"
          "  must: X^T X is singular by construction, not by rounding\n\n")

# ---------------------------------------------------------------------------
# the split.  Only 4% of the data is used for training, so p >> n_train and
# the problem is underdetermined even before the exact rank deficiency.
# ---------------------------------------------------------------------------
ntr = 400
idx = rng.permutation(n)
tr, te = idx[:ntr], idx[ntr:]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
out.write(f"=== 2. the three estimators, {ntr} training and {len(te)} test "
          "configurations ===\n")
out.write(f"  p = {L*L} features against n = {ntr} training rows, so p > n and\n"
          "  the least-squares problem is underdetermined twice over\n\n")


def r2(yt, yp):
    return 1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)


def ols_pinv(Xa, ya):
    """Minimum-norm least squares through the SVD, Eq. (3.olssvd)."""
    return lstsq(Xa, ya, rcond=None)[0]


def ridge(Xa, ya, lam):
    """Eq. (3.ridgesolution), solved in the dual by Theorem 3.ridgedual since
    here n < p: alpha = (X X^T + lambda I)^{-1} y and theta = X^T alpha."""
    K = Xa @ Xa.T
    al = np.linalg.solve(K + lam * np.eye(len(ya)), ya)
    return Xa.T @ al


def soft(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def lasso_cd(Xa, ya, lam, n_iter=3000, tol=1e-10):
    """Cyclic coordinate descent, Eq. (3.coorddescent)."""
    m, p = Xa.shape
    th = np.zeros(p)
    cn = (Xa ** 2).sum(0)
    cn[cn == 0] = 1.0
    r = ya - Xa @ th
    for _ in range(n_iter):
        old = th.copy()
        for j in range(p):
            if cn[j] == 0:
                continue
            r += Xa[:, j] * th[j]
            th[j] = soft(Xa[:, j] @ r, lam * m / 2.0) / cn[j]
            r -= Xa[:, j] * th[j]
        if np.abs(th - old).max() < tol:
            break
    return th


lam = 1e-2
th_ols = ols_pinv(Xtr, ytr)
th_rid = ridge(Xtr, ytr, lam)
th_las = lasso_cd(Xtr, ytr, lam)

names = ["OLS (pseudoinverse)", "Ridge, lambda = 0.01", "Lasso, lambda = 0.01"]
thetas = [th_ols, th_rid, th_las]
out.write("   estimator            R^2 train  R^2 test  ||theta||_2"
          "  ||theta||_1  non-zero\n")
for nm, th in zip(names, thetas):
    out.write(f"   {nm:20s} {r2(ytr, Xtr@th):10.6f} {r2(yte, Xte@th):9.6f}"
              f" {np.linalg.norm(th):12.4f} {np.abs(th).sum():12.4f}"
              f" {int((np.abs(th)>1e-8).sum()):9d}\n")
out.write("\n")

# ---------------------------------------------------------------------------
# what the coupling matrices look like
# ---------------------------------------------------------------------------
out.write("=== 3. the recovered coupling matrices ===\n")
out.write("  The true Hamiltonian has J_{j,j+1} = 1 and nothing else, but the\n"
          "  energy only ever sees the sum J_{j,j+1} + J_{j+1,j}, so what the\n"
          "  data determine is that sum and not the two terms separately.\n\n")
out.write("   estimator              J[0,1]    J[1,0]       sum    J[0,0]"
          "    J[0,5]\n")
rows = []
for nm, th in zip(names, thetas):
    Jm = th.reshape(L, L)
    rows.append((nm, Jm[0, 1], Jm[1, 0], Jm[0, 1] + Jm[1, 0], Jm[0, 0],
                 Jm[0, 5]))
    out.write(f"   {nm:20s} {Jm[0,1]:9.5f} {Jm[1,0]:9.5f} "
              f"{Jm[0,1]+Jm[1,0]:9.5f} {Jm[0,0]:9.5f} {Jm[0,5]:9.5f}\n")
out.write("\n  averaged over all forty nearest-neighbour pairs:\n")
out.write("   estimator            mean J[j,j+1]  mean J[j+1,j]  mean sum"
          "  mean |J| off-band\n")
band = np.zeros((L, L), dtype=bool)
for j in range(L):
    band[j, (j + 1) % L] = band[(j + 1) % L, j] = True
offband = ~band & ~np.eye(L, dtype=bool)
for nm, th in zip(names, thetas):
    Jm = th.reshape(L, L)
    a = np.array([Jm[j, (j + 1) % L] for j in range(L)])
    b = np.array([Jm[(j + 1) % L, j] for j in range(L)])
    out.write(f"   {nm:20s} {a.mean():13.5f}  {b.mean():13.5f}  "
              f"{(a+b).mean():8.5f}  {np.abs(Jm[offband]).mean():17.5f}\n")
np.save("J_ols.npy", th_ols.reshape(L, L))
np.save("J_ridge.npy", th_rid.reshape(L, L))
np.save("J_lasso.npy", th_las.reshape(L, L))
out.write("\n")

# ---------------------------------------------------------------------------
# Proposition 3.duplicate, checked
# ---------------------------------------------------------------------------
out.write("=== 4. Proposition 3.duplicate, checked on the (j,k)/(k,j) pairs ===\n")
out.write("  With two identical columns the 2-norm estimators must give the two\n"
          "  coefficients the same value, and the 1-norm need not.\n\n")
out.write("   estimator            max |J[j,k] - J[k,j]| over all j<k\n")
for nm, th in zip(names, thetas):
    Jm = th.reshape(L, L)
    out.write(f"   {nm:20s} {np.abs(Jm - Jm.T).max():34.3e}\n")
out.write("  OLS and Ridge are symmetric to machine precision, as the\n"
          "  proposition requires; the Lasso is not, and puts the whole coupling\n"
          "  on one side of the pair.\n\n")
out.write("  and the objective is genuinely indifferent to the split: taking the\n"
          "  Lasso solution and symmetrising each (j,k)/(k,j) pair,\n")
Jl = th_las.reshape(L, L)
Jsym = 0.5 * (Jl + Jl.T)
th_sym = Jsym.ravel()
Ctr = lambda th: (np.mean((ytr - Xtr @ th) ** 2) + lam * np.abs(th).sum())
out.write(f"    Lasso objective at the solver's answer : {Ctr(th_las):.10f}\n")
out.write(f"    Lasso objective at the symmetrised one : {Ctr(th_sym):.10f}\n")
out.write(f"    max |fit difference| on the test set   : "
          f"{np.abs(Xte@th_las - Xte@th_sym).max():.3e}\n")
out.write("  identical to ten decimals and identical in the fit, which is\n"
          "  Proposition 3.lassofit: the fit is unique, the coefficients are not\n\n")

# ---------------------------------------------------------------------------
# the penalty path
# ---------------------------------------------------------------------------
out.write("=== 5. R^2 against the penalty ===\n")
res = {"ridge": [], "lasso": []}

sv = np.linalg.svd(Xtr, compute_uv=False)
out.write("  The Ridge penalty competes with the squared singular values of the\n"
          "  design matrix, Eq. (3.ridgesvdsol), and here those run from\n")
out.write(f"    sigma_max^2 = {sv[0]**2:.4g}  down to  "
          f"sigma_min^2 = {sv[sv>1e-8][-1]**2:.4g}\n")
out.write("  on the training rows, so a penalty below about 10^3 changes nothing\n"
          "  at all and the table must be read over a very wide range.\n\n")

out.write("     lambda     Ridge train   Ridge test\n")
for lm in np.logspace(-4, 8, 25):
    tr_r = ridge(Xtr, ytr, lm)
    res["ridge"].append((lm, r2(ytr, Xtr @ tr_r), r2(yte, Xte @ tr_r)))
    out.write(f"  {lm:10.4g}   {res['ridge'][-1][1]:11.6f}   "
              f"{res['ridge'][-1][2]:10.6f}\n")

out.write("\n  The Lasso is a different story and needs no such range:\n\n")
out.write("     lambda     Lasso train   Lasso test   Lasso non-zeros\n")
for lm in np.logspace(-4, 1, 11):
    tl = lasso_cd(Xtr, ytr, lm)
    res["lasso"].append((lm, r2(ytr, Xtr @ tl), r2(yte, Xte @ tl),
                         int((np.abs(tl) > 1e-8).sum())))
    out.write(f"  {lm:10.4g}   {res['lasso'][-1][1]:11.6f}   "
              f"{res['lasso'][-1][2]:10.6f}   {res['lasso'][-1][3]:15d}\n")
np.save("path_ridge.npy", np.array(res["ridge"]))
np.save("path_lasso.npy", np.array([r[:3] for r in res["lasso"]]))
np.save("path_lasso_nnz.npy", np.array([r[3] for r in res["lasso"]]))
ols_te = r2(yte, Xte @ th_ols)
np.save("ols_r2.npy", np.array([r2(ytr, Xtr @ th_ols), ols_te]))
out.write(f"  ordinary least squares, for comparison: train "
          f"{r2(ytr, Xtr@th_ols):.6f}, test {ols_te:.6f}\n")
best_l = max(res["lasso"], key=lambda t: t[2])
best_r = max(res["ridge"], key=lambda t: t[2])
out.write(f"  best Lasso: lambda = {best_l[0]:.4g}, test R^2 = {best_l[2]:.6f}, "
          f"{best_l[3]} non-zero coefficients\n")
out.write(f"  best Ridge: lambda = {best_r[0]:.4g}, test R^2 = {best_r[2]:.6f}\n")
np.save("meta.npy", np.array([L, n, ntr, lam]))
out.close()
print("done")

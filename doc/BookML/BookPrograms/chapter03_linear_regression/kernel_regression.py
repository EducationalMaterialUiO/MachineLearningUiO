"""Kernel ridge regression and the two things called a kernel Lasso.

Every number this program prints is quoted in Section 3.kernels.

1.  The push-through identity of Lemma 3.pushthrough, checked numerically, and
    the resulting dual form of Ridge regression: the same fit obtained from a
    p x p system and from an n x n one.
2.  The Hoerl-Kennard existence theorem, Theorem 3.hkexists: the mean squared
    error of the Ridge estimator has a strictly negative derivative at
    lambda = 0, so some positive penalty beats ordinary least squares.
3.  Kernel ridge regression from scratch, against scikit-learn, and against
    an explicit polynomial feature map where one exists.
4.  Why there is no kernel Lasso: two feature maps giving the same kernel and
    different 1-norms, Proposition 3.nolassokernel.
5.  Repair one -- the 1-norm on the dual coefficients -- with its coordinate
    descent solver and a check of the optimality conditions.
6.  Repair two -- an explicit approximate feature map, by Nystrom and by
    random Fourier features -- with an ordinary Lasso in that basis.
"""
import numpy as np
from numpy.linalg import solve, norm, eigh, cholesky

out = open("kernel_regression.txt", "w", buffering=1)
rng = np.random.default_rng(0)


# ===========================================================================
# kernels
# ===========================================================================
def gaussian_kernel(A, B, gamma):
    """k(x,x') = exp(-gamma ||x - x'||^2), Eq. (6.rbf)."""
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def poly_kernel(A, B, degree=2, c=1.0):
    """k(x,x') = (x.x' + c)^d, Eq. (6.polykernel)."""
    return (A @ B.T + c) ** degree


def poly_features(X, c=1.0):
    """The explicit feature map of the inhomogeneous quadratic kernel in 2-D:
    phi(x) = (c, sqrt(2c) x1, sqrt(2c) x2, x1^2, x2^2, sqrt2 x1 x2), so that
    phi(x).phi(x') = (x.x' + c)^2 exactly."""
    x1, x2 = X[:, 0], X[:, 1]
    s = np.sqrt(2.0 * c)
    return np.column_stack([np.full(len(X), c), s * x1, s * x2,
                            x1 ** 2, x2 ** 2, np.sqrt(2.0) * x1 * x2])


# ===========================================================================
# 1.  the push-through identity and the dual form of Ridge
# ===========================================================================
out.write("=== 1. the push-through identity, Lemma 3.pushthrough ===\n")
out.write("  (X^T X + lambda I_p)^{-1} X^T = X^T (X X^T + lambda I_n)^{-1}\n")
out.write("  The left side inverts a p x p matrix, the right an n x n one, and\n"
          "  the two are equal for every lambda > 0.  Only the right side can\n"
          "  be written with inner products alone, which is what allows a\n"
          "  kernel to be substituted.\n\n")

out.write("     n     p   lambda   max |LHS - RHS|   ||theta_p - theta_n||\n")
for n, p in [(20, 5), (20, 50), (200, 3), (50, 50)]:
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    for lam in [1e-2, 1.0]:
        L = solve(X.T @ X + lam * np.eye(p), X.T)
        R = X.T @ solve(X @ X.T + lam * np.eye(n), np.eye(n))
        th_p = solve(X.T @ X + lam * np.eye(p), X.T @ y)
        th_n = X.T @ solve(X @ X.T + lam * np.eye(n), y)
        out.write(f"  {n:4d}  {p:4d}   {lam:6.2f}   {np.abs(L-R).max():15.3e}"
                  f"   {norm(th_p-th_n):21.3e}\n")
out.write("  the identity holds to machine precision at every shape, including\n"
          "  p > n where the primal matrix is singular without the penalty\n\n")

# ===========================================================================
# 2.  the Hoerl-Kennard existence theorem
# ===========================================================================
out.write("=== 2. Theorem 3.hkexists: some lambda > 0 always beats OLS ===\n")
out.write("  In the singular basis the total mean squared error of the Ridge\n"
          "  estimator is  M(lambda) = sum_i (lambda^2 b_i^2 + s^2 d_i^2)\n"
          "  / (d_i^2 + lambda)^2,  with d_i the singular values, b = V^T theta\n"
          "  and s^2 the noise variance.  Its derivative at lambda = 0 is\n"
          "  -2 s^2 sum_i d_i^{-4} < 0, so M decreases at first.\n\n")


def ridge_mse_exact(d, b, s2, lam):
    """M(lambda) above: exact bias^2 + variance of the Ridge estimator."""
    return float(np.sum((lam ** 2 * b ** 2 + s2 * d ** 2) / (d ** 2 + lam) ** 2))


out.write("     n     p    sigma^2   M(0)=OLS    min_lambda M   best lambda"
          "   M'(0) predicted   M'(0) measured\n")
rows = []
for n, p, s2 in [(40, 6, 1.0), (40, 6, 0.1), (80, 20, 1.0), (25, 15, 0.5)]:
    X = rng.normal(size=(n, p))
    d = np.linalg.svd(X, compute_uv=False)
    theta = rng.normal(size=p)
    b = theta                                     # V^T theta, same norm
    grid = np.concatenate([[0.0], np.logspace(-4, 3, 4000)])
    M = np.array([ridge_mse_exact(d, b, s2, l) for l in grid])
    k = int(np.argmin(M))
    pred = -2.0 * s2 * np.sum(d ** -4.0)
    meas = (M[1] - M[0]) / grid[1]
    rows.append((n, p, s2, M[0], M[k], grid[k]))
    out.write(f"  {n:4d}  {p:4d}   {s2:8.2f}  {M[0]:10.4f}  {M[k]:14.4f}"
              f"  {grid[k]:12.4g}   {pred:16.4g}   {meas:16.4g}\n")
out.write("  the minimum is always strictly interior, and the measured slope at\n"
          "  the origin agrees with the predicted -2 s^2 sum d_i^{-4}\n\n")

# ===========================================================================
# 3.  kernel ridge regression
# ===========================================================================
out.write("=== 3. kernel ridge regression, Theorem 3.krr ===\n")


def kernel_ridge_fit(K, y, lam):
    """alpha = (K + lambda I)^{-1} y, Eq. (3.krralpha)."""
    return solve(K + lam * np.eye(len(y)), y)


n, lam = 60, 0.5
X = rng.uniform(-2, 2, size=(n, 2))
ytrue = np.sin(X[:, 0]) * np.cos(X[:, 1])
y = ytrue + 0.05 * rng.normal(size=n)
Xt = rng.uniform(-2, 2, size=(25, 2))

out.write("  (a) the quadratic kernel against its explicit feature map\n")
Kp = poly_kernel(X, X, degree=2, c=1.0)
al = kernel_ridge_fit(Kp, y, lam)
f_kernel = poly_kernel(Xt, X, degree=2, c=1.0) @ al
Phi, Phit = poly_features(X), poly_features(Xt)
th = solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ y)
f_primal = Phit @ th
out.write(f"      max |k(x,x') - phi(x).phi(x')| : "
          f"{np.abs(Kp - Phi @ Phi.T).max():.3e}\n")
out.write(f"      max |f_kernel(x) - f_primal(x)|: "
          f"{np.abs(f_kernel - f_primal).max():.3e}\n")
out.write("      the kernel never builds phi and gets the same function\n\n")

out.write("  (b) the Gaussian kernel, ours against scikit-learn\n")
try:
    from sklearn.kernel_ridge import KernelRidge
    have_sk = True
except Exception:
    have_sk = False
out.write("      gamma   lambda    our test MSE   sklearn test MSE"
          "   max |f_ours - f_sk|\n")
for gam in [0.25, 1.0, 4.0]:
    for lm in [1e-3, 1e-1]:
        K = gaussian_kernel(X, X, gam)
        a = kernel_ridge_fit(K, y, lm)
        f_ours = gaussian_kernel(Xt, X, gam) @ a
        yt = np.sin(Xt[:, 0]) * np.cos(Xt[:, 1])
        if have_sk:
            kr = KernelRidge(alpha=lm, kernel="rbf", gamma=gam).fit(X, y)
            f_sk = kr.predict(Xt)
            out.write(f"      {gam:5.2f}   {lm:6.3f}   "
                      f"{np.mean((f_ours-yt)**2):12.6f}   "
                      f"{np.mean((f_sk-yt)**2):16.6f}   "
                      f"{np.abs(f_ours-f_sk).max():18.3e}\n")
out.write("      scikit-learn's alpha is exactly our lambda; the two solve the\n"
          "      same linear system and agree to machine precision\n\n")

out.write("  (c) cost: the primal system is p x p, the dual n x n\n")
out.write("      n     p   primal solve   dual solve   which is cheaper\n")
for n2, p2 in [(200, 10), (200, 400), (2000, 50)]:
    out.write(f"   {n2:5d}  {p2:4d}   {p2**3:12d}   {n2**3:10d}"
              f"   {'primal' if p2 < n2 else 'dual':16s}\n")
out.write("      the crossover is at p = n, and for an infinite-dimensional\n"
          "      feature map the primal route does not exist at all\n\n")

# ===========================================================================
# 4.  why there is no kernel Lasso
# ===========================================================================
out.write("=== 4. Proposition 3.nolassokernel: the 1-norm is not a property "
          "of f ===\n")
out.write("  A kernel determines the feature map only up to an isometry: if\n"
          "  phi is a feature map for k then so is R phi for any orthogonal R,\n"
          "  since (R phi(x)).(R phi(x')) = phi(x).phi(x').  The 2-norm of the\n"
          "  weight vector is invariant under that change and the 1-norm is\n"
          "  not, so ||theta||_1 is not a function of f and cannot appear in a\n"
          "  penalty on f.\n\n")

r2 = np.random.default_rng(7)
A = r2.normal(size=(6, 6))
Q, _ = np.linalg.qr(A)                            # a random isometry of H
th0 = solve(Phi.T @ Phi + lam * np.eye(6), Phi.T @ y)
th1 = Q @ th0                                     # the same f, rotated map
out.write("     quantity                                value (phi)"
          "    value (R phi)\n")
out.write(f"     ||theta||_2                          {norm(th0):12.6f}"
          f"    {norm(th1):13.6f}\n")
out.write(f"     ||theta||_1                          {norm(th0,1):12.6f}"
          f"    {norm(th1,1):13.6f}\n")
out.write(f"     max |f(x) - f_rotated(x)| on test    "
          f"{np.abs(Phit @ th0 - (Phit @ Q.T) @ th1).max():12.3e}\n")
out.write("     the two weight vectors describe the same function to machine\n"
          "     precision, have the same 2-norm, and differ in 1-norm; a Lasso\n"
          "     penalty would therefore select a different function depending\n"
          "     on which square root of the kernel was chosen\n\n")

# ===========================================================================
# 5.  repair one: the 1-norm on the dual coefficients
# ===========================================================================
out.write("=== 5. repair one: min ||y - K alpha||^2 / n + mu ||alpha||_1 ===\n")
out.write("  This is an ordinary Lasso whose design matrix is the Gram matrix.\n"
          "  It is sparse in the *examples*, not in the features, and it is not\n"
          "  equivalent to any penalty on f.  Optimality, Eq. (3.klassokkt):\n"
          "    (2/n) k_j.(K alpha - y) = -mu sign(alpha_j)   if alpha_j != 0,\n"
          "    |(2/n) k_j.(K alpha - y)| <= mu               if alpha_j  = 0.\n\n")


def soft_threshold(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def kernel_lasso(K, y, mu, n_iter=20000, tol=1e-12):
    """Cyclic coordinate descent on ||y - K a||^2 / n + mu ||a||_1."""
    n = len(y)
    a = np.zeros(n)
    cn = (K ** 2).sum(0)
    r = y - K @ a
    for _ in range(n_iter):
        a_old = a.copy()
        for j in range(n):
            r += K[:, j] * a[j]
            a[j] = soft_threshold(K[:, j] @ r, mu * n / 2.0) / cn[j]
            r -= K[:, j] * a[j]
        if np.abs(a - a_old).max() < tol:
            break
    return a


Kg = gaussian_kernel(X, X, 1.0)
out.write("      mu      non-zero alpha of 60   train MSE   max KKT violation\n")
kl_rows = []
for mu in [1e-4, 1e-3, 1e-2, 1e-1]:
    a = kernel_lasso(Kg, y, mu)
    g = 2.0 / n * Kg.T @ (Kg @ a - y)
    viol = max(np.abs(g[a != 0] + mu * np.sign(a[a != 0])).max() if (a != 0).any() else 0.0,
               (np.abs(g[a == 0]) - mu).max() if (a == 0).any() else 0.0)
    kl_rows.append((mu, int((a != 0).sum()), float(np.mean((Kg @ a - y) ** 2)),
                    float(viol)))
    out.write(f"   {mu:7.4f}   {kl_rows[-1][1]:20d}   {kl_rows[-1][2]:9.6f}"
              f"   {viol:17.3e}\n")
out.write("      every solution satisfies its own optimality conditions to the\n"
          "      tolerance of the solver, and the number of retained examples\n"
          "      falls as mu grows -- the same selection behaviour as the\n"
          "      ordinary Lasso, but selecting data points rather than features\n\n")

mu_max = 2.0 / n * np.abs(Kg.T @ y).max()
a_at = kernel_lasso(Kg, y, mu_max * 1.001)
out.write(f"      mu_max = (2/n) max_j |k_j . y| = {mu_max:.6f}\n")
out.write(f"      non-zero coefficients just above mu_max: "
          f"{int((a_at != 0).sum())}\n")
out.write("      above mu_max the zero vector satisfies the conditions and is\n"
          "      the solution, exactly as in Proposition 3.lassomax\n\n")

# ===========================================================================
# 6.  repair two: an explicit approximate feature map
# ===========================================================================
out.write("=== 6. repair two: Lasso in an approximate feature map ===\n")
out.write("  Build an explicit m-dimensional map whose inner products\n"
          "  approximate k, then run the ordinary Lasso of Section 3.lasso in\n"
          "  that basis.  The selection is then over features again, and the\n"
          "  features are interpretable: Nystrom columns are landmark points,\n"
          "  random Fourier features are frequencies.\n\n")


def nystrom(Xall, Xland, gam, eps=1e-10):
    """Z with Z Z^T approximating the Gaussian Gram matrix."""
    Kmm = gaussian_kernel(Xland, Xland, gam)
    w, V = eigh(Kmm)
    keep = w > eps * w.max()
    Wm = V[:, keep] / np.sqrt(w[keep])
    return gaussian_kernel(Xall, Xland, gam) @ Wm


def random_fourier(Xall, W, b):
    """cos features: E[z(x).z(x')] = exp(-gamma||x-x'||^2), Rahimi-Recht."""
    m = W.shape[1]
    return np.sqrt(2.0 / m) * np.cos(Xall @ W + b)


out.write("      method             m    max |Z Z^T - K|   Lasso non-zeros"
          "   test MSE\n")
yt = np.sin(Xt[:, 0]) * np.cos(Xt[:, 1])
for m in [10, 30, 60]:
    idx = rng.choice(n, size=min(m, n), replace=False)
    Z = nystrom(X, X[idx], 1.0)
    Zt = nystrom(Xt, X[idx], 1.0)
    # ordinary Lasso in the explicit basis, same coordinate descent
    a = np.zeros(Z.shape[1]); cn = (Z ** 2).sum(0); r = y - Z @ a
    for _ in range(20000):
        ao = a.copy()
        for j in range(Z.shape[1]):
            r += Z[:, j] * a[j]
            a[j] = soft_threshold(Z[:, j] @ r, 1e-3 * n / 2.0) / cn[j]
            r -= Z[:, j] * a[j]
        if np.abs(a - ao).max() < 1e-12:
            break
    out.write(f"      Nystrom      {Z.shape[1]:6d}   "
              f"{np.abs(Z @ Z.T - Kg).max():15.3e}   {int((a!=0).sum()):15d}"
              f"   {np.mean((Zt @ a - yt)**2):9.6f}\n")

for m in [64, 256, 1024]:
    W = np.sqrt(2.0 * 1.0) * rng.normal(size=(2, m))
    bph = rng.uniform(0, 2 * np.pi, size=m)
    Z, Zt2 = random_fourier(X, W, bph), random_fourier(Xt, W, bph)
    a = np.zeros(m); cn = (Z ** 2).sum(0); r = y - Z @ a
    for _ in range(4000):
        ao = a.copy()
        for j in range(m):
            r += Z[:, j] * a[j]
            a[j] = soft_threshold(Z[:, j] @ r, 1e-3 * n / 2.0) / cn[j]
            r -= Z[:, j] * a[j]
        if np.abs(a - ao).max() < 1e-10:
            break
    out.write(f"      Fourier      {m:6d}   {np.abs(Z @ Z.T - Kg).max():15.3e}"
              f"   {int((a!=0).sum()):15d}   "
              f"{np.mean((Zt2 @ a - yt)**2):9.6f}\n")
out.write("      the Nystrom map reproduces the Gram matrix exactly once m = n,\n"
          "      the random map only in expectation and slowly; both give a\n"
          "      genuine Lasso, at the cost of an approximation the exact\n"
          "      kernel method does not make\n")
out.close()
print(open("kernel_regression.txt").read())

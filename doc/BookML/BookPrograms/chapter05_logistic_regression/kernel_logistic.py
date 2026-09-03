"""Kernel logistic regression, Section 5.kernellogreg.

Every number this program prints is quoted in the text.

1.  Kernel IRLS, Eq. (5.kirls), against a plain gradient method on the same
    objective: two different algorithms, one minimiser.
2.  The kernel solution against the primal solution with an explicit feature
    map, and against scikit-learn's logistic regression in that map.
3.  The stationarity identity alpha = (y - p) / lambda, Proposition
    5.kalpharesidual, which says the dual coefficients *are* the residuals and
    are therefore dense.
4.  Newton's quadratic convergence, and what the penalty does to the fit.
"""
import numpy as np
from numpy.linalg import solve, norm

out = open("kernel_logistic.txt", "w", buffering=1)
rng = np.random.default_rng(3)


def sigmoid(z):
    """Numerically stable logistic function, Eq. (5.sigmoid)."""
    o = np.empty_like(z, dtype=float)
    pos, neg = z >= 0, z < 0
    o[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    o[neg] = e / (1.0 + e)
    return o


def gaussian_kernel(A, B, gamma):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def poly_features(X, c=1.0):
    """Explicit map of the quadratic kernel (x.x' + c)^2 in two dimensions."""
    x1, x2 = X[:, 0], X[:, 1]
    s = np.sqrt(2.0 * c)
    return np.column_stack([np.full(len(X), c), s * x1, s * x2,
                            x1 ** 2, x2 ** 2, np.sqrt(2.0) * x1 * x2])


def poly_kernel(A, B, c=1.0):
    return (A @ B.T + c) ** 2


def cost(K, y, a, lam):
    """C(alpha) of Eq. (5.kcost): cross entropy plus (lambda/2) a^T K a."""
    f = K @ a
    # -sum[ y f - log(1+e^f) ] written stably
    ll = np.sum(y * f - np.logaddexp(0.0, f))
    return float(-ll + 0.5 * lam * a @ (K @ a))


def kernel_irls(K, y, lam, n_iter=100, tol=1e-13, ridge=1e-12):
    """Kernel logistic regression by Newton's method, Eq. (5.kirls):

        alpha <- (K + lambda W^{-1})^{-1} z,   z = K alpha + W^{-1}(y - p),

    which is kernel ridge regression on the working response z with a
    per-observation penalty lambda / W_ii.  Returns alpha and the history of
    the cost and of the Newton decrement.
    """
    n = len(y)
    a = np.zeros(n)
    hist = []
    for it in range(n_iter):
        f = K @ a
        p = sigmoid(f)
        w = np.maximum(p * (1.0 - p), 1e-10)
        z = f + (y - p) / w
        a_new = solve(K + lam * np.diag(1.0 / w) + ridge * np.eye(n), z)
        step = norm(a_new - a)
        a = a_new
        hist.append((it + 1, cost(K, y, a, lam), step))
        if step < tol:
            break
    return a, hist


def kernel_gradient(K, y, lam, eta=0.5, n_iter=200000, tol=1e-12):
    """The same objective by plain gradient descent, for comparison.
    grad C = K[(p - y) + lambda alpha], Eq. (5.kgradient)."""
    a = np.zeros(len(y))
    for _ in range(n_iter):
        g = K @ (sigmoid(K @ a) - y + lam * a)
        a_new = a - eta * g / len(y)
        if norm(a_new - a) < tol:
            a = a_new
            break
        a = a_new
    return a


# ===========================================================================
# data: a problem no straight line can solve
# ===========================================================================
n = 120
X = rng.uniform(-2.5, 2.5, size=(n, 2))
lab = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 3.0).astype(float)   # an annulus
flip = rng.random(n) < 0.05
y = np.where(flip, 1.0 - lab, lab)                          # 5% label noise
Xt = rng.uniform(-2.5, 2.5, size=(400, 2))
yt = ((Xt[:, 0] ** 2 + Xt[:, 1] ** 2) < 3.0).astype(float)

out.write("=== 0. the problem ===\n")
out.write(f"  {n} points in the plane, class 1 inside a disc of radius "
          f"sqrt(3), 5% of the labels flipped;\n"
          f"  {int(y.sum())} positives, {n - int(y.sum())} negatives.  No "
          "linear boundary can do better than chance here.\n\n")

# ===========================================================================
# 1.  two algorithms, one minimiser
# ===========================================================================
out.write("=== 1. kernel IRLS against gradient descent, Eq. (5.kirls) ===\n")
lam = 1.0
Kg = gaussian_kernel(X, X, 0.5)
a_irls, hist = kernel_irls(Kg, y, lam)
a_gd = kernel_gradient(Kg, y, lam)
out.write(f"  Newton iterations to ||delta alpha|| < 1e-13 : {len(hist)}\n")
out.write(f"  cost at the IRLS solution                    : "
          f"{cost(Kg, y, a_irls, lam):.12f}\n")
out.write(f"  cost at the gradient solution                : "
          f"{cost(Kg, y, a_gd, lam):.12f}\n")
out.write(f"  max |f_irls(x) - f_grad(x)| on the training set: "
          f"{np.abs(Kg @ a_irls - Kg @ a_gd).max():.3e}\n\n")
out.write("  the Newton iterates, showing the quadratic rate:\n")
out.write("    iteration        cost          ||delta alpha||\n")
for it, c, s in hist:
    out.write(f"    {it:9d}   {c:14.10f}   {s:16.3e}\n")
out.write("  the step length squares at each iteration once close to the\n"
          "  minimum, which is the signature of Newton's method\n\n")

# ===========================================================================
# 2.  against the primal, and against scikit-learn
# ===========================================================================
out.write("=== 2. the kernel form against an explicit feature map ===\n")
out.write("  With the quadratic kernel (x.x' + 1)^2 the feature map is only\n"
          "  six-dimensional, so the primal problem can be solved directly and\n"
          "  the two answers compared.\n\n")
Kp = poly_kernel(X, X)
Phi, Phit = poly_features(X), poly_features(Xt)
out.write(f"  max |k(x,x') - phi(x).phi(x')|: {np.abs(Kp - Phi @ Phi.T).max():.3e}\n")

a_k, _ = kernel_irls(Kp, y, lam)
f_k = poly_kernel(Xt, X) @ a_k


def primal_newton(Phi, y, lam, n_iter=200):
    th = np.zeros(Phi.shape[1])
    for _ in range(n_iter):
        p = sigmoid(Phi @ th)
        w = np.maximum(p * (1 - p), 1e-10)
        H = Phi.T @ (w[:, None] * Phi) + lam * np.eye(Phi.shape[1])
        g = -Phi.T @ (y - p) + lam * th
        d = solve(H, g)
        th -= d
        if norm(d) < 1e-13:
            break
    return th


th = primal_newton(Phi, y, lam)
f_p = Phit @ th
out.write(f"  max |f_kernel(x) - f_primal(x)| on 400 test points: "
          f"{np.abs(f_k - f_p).max():.3e}\n")
out.write(f"  ||theta_primal||^2 = {th @ th:.10f},   "
          f"alpha^T K alpha = {a_k @ (Kp @ a_k):.10f}\n")
out.write("  the two penalties are the same number because both are the "
          "squared\n  norm of the same function in the same space\n\n")

try:
    from sklearn.linear_model import LogisticRegression
    sk = LogisticRegression(C=1.0 / lam, fit_intercept=False, tol=1e-12,
                            max_iter=10000).fit(Phi, y)
    out.write(f"  scikit-learn in the same map, C = 1/lambda:\n")
    out.write(f"    max |theta_ours - theta_sklearn| : "
              f"{np.abs(th - sk.coef_.ravel()).max():.3e}\n")
    out.write(f"    max |f_ours - f_sklearn| on test : "
              f"{np.abs(f_p - Phit @ sk.coef_.ravel()).max():.3e}\n\n")
except Exception as e:                                        # pragma: no cover
    out.write(f"  scikit-learn unavailable: {e}\n\n")

# ===========================================================================
# 3.  the coefficients are the residuals
# ===========================================================================
out.write("=== 3. Proposition 5.kalpharesidual: alpha = (y - p) / lambda ===\n")
out.write("  Setting the gradient K[(p - y) + lambda alpha] to zero and using\n"
          "  that K is invertible gives alpha_i = (y_i - p_i) / lambda: the\n"
          "  dual coefficient of a point is its residual, divided by the\n"
          "  penalty.  Residuals are never exactly zero, so alpha is dense.\n\n")
out.write("   lambda   max |alpha - (y-p)/lambda|   non-zero alpha of "
          f"{n}   |alpha|_min\n")
for lm in [0.1, 1.0, 10.0]:
    a_l, _ = kernel_irls(Kg, y, lm)
    p_l = sigmoid(Kg @ a_l)
    out.write(f"  {lm:7.2f}   {np.abs(a_l - (y - p_l) / lm).max():25.3e}"
              f"   {int((a_l != 0).sum()):17d}   {np.abs(a_l).min():.3e}\n")
out.write("  every one of the coefficients is non-zero at every penalty; the\n"
          "  contrast with the support vectors of Chapter 6 is the whole point\n"
          "  of Theorem 6.threelosses\n\n")

# ===========================================================================
# 4.  what the penalty buys
# ===========================================================================
out.write("=== 4. the penalty, measured ===\n")
out.write("   lambda    train accuracy   test accuracy   alpha^T K alpha\n")
for lm in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
    a_l, _ = kernel_irls(Kg, y, lm)
    tr = np.mean((sigmoid(Kg @ a_l) > 0.5) == (y > 0.5))
    te = np.mean((sigmoid(gaussian_kernel(Xt, X, 0.5) @ a_l) > 0.5) == (yt > 0.5))
    out.write(f"  {lm:8.3f}   {tr:14.4f}   {te:13.4f}   "
              f"{a_l @ (Kg @ a_l):15.4f}\n")
out.write("  a small penalty reproduces the 5% of flipped labels and pays for\n"
          "  it on the test set; a large one flattens the function towards a\n"
          "  constant.  The best test accuracy is in between, which is\n"
          "  Eq. (2.biasvariance) once more.\n")
out.close()
print(open("kernel_logistic.txt").read())

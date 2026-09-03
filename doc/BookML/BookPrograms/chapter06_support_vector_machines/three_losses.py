"""One penalty, three losses: Theorem 6.threelosses, measured.

All three of kernel ridge regression (Chapter 3), kernel logistic regression
(Chapter 5) and the support vector machine (this chapter) solve

    min_{f in H}  sum_i L(y_i, f(x_i))  +  (lambda/2) ||f||_H^2 ,

differing only in L.  The representer theorem gives f = sum_j alpha_j k(x_j,.)
for all three, and stationarity gives the single formula

    alpha_i = - L'(y_i, f(x_i)) / lambda ,

which predicts the coefficients of each method from the derivative of its loss.
This program checks that formula for all three, and shows that it is what makes
the support vector machine sparse and the other two dense.
"""
import numpy as np
from numpy.linalg import solve, norm

out = open("three_losses.txt", "w", buffering=1)
rng = np.random.default_rng(11)


def gaussian_kernel(A, B, gamma):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def sigmoid(z):
    o = np.empty_like(z, dtype=float)
    p, m = z >= 0, z < 0
    o[p] = 1.0 / (1.0 + np.exp(-z[p]))
    e = np.exp(z[m])
    o[m] = e / (1.0 + e)
    return o


# ---------------------------------------------------------------------------
# data, with labels in {-1, +1} as in Chapter 6
# ---------------------------------------------------------------------------
n, lam, gam = 80, 1.0, 0.5
X = rng.uniform(-2.5, 2.5, size=(n, 2))
y = np.where((X[:, 0] ** 2 + X[:, 1] ** 2) < 3.0, 1.0, -1.0)
flip = rng.random(n) < 0.05
y = np.where(flip, -y, y)
K = gaussian_kernel(X, X, gam)

out.write("=== the common set-up ===\n")
out.write(f"  {n} points, labels in {{-1,+1}}, Gaussian kernel with gamma = "
          f"{gam}, penalty lambda = {lam}\n")
out.write("  every method below minimises  sum_i L(y_i, f(x_i)) + "
          "(lambda/2)||f||^2\n  over the same space, with f = K alpha.\n\n")


# ===========================================================================
# 1.  squared loss  ->  kernel ridge regression
# ===========================================================================
# L(y,f) = (1/2)(y - f)^2,  L' = f - y,  so alpha = (y - f)/lambda, which
# rearranges to (K + lambda I) alpha = y.
a_ridge = solve(K + lam * np.eye(n), y)
f_ridge = K @ a_ridge


# ===========================================================================
# 2.  logistic loss  ->  kernel logistic regression
# ===========================================================================
# L(y,f) = log(1 + exp(-y f)),  L' = -y sigma(-y f),
# so alpha = y sigma(-y f) / lambda.
def kernel_logistic(K, y, lam, n_iter=100, tol=1e-13):
    a = np.zeros(len(y))
    for _ in range(n_iter):
        f = K @ a
        s = sigmoid(-y * f)                       # = 1 - p for the +1 class
        w = np.maximum(s * (1.0 - s), 1e-10)
        z = f + (y * s) / w
        a_new = solve(K + lam * np.diag(1.0 / w) + 1e-12 * np.eye(len(y)), z)
        if norm(a_new - a) < tol:
            a = a_new
            break
        a = a_new
    return a


a_logit = kernel_logistic(K, y, lam)
f_logit = K @ a_logit


# ===========================================================================
# 3.  hinge loss  ->  the support vector machine
# ===========================================================================
# L(y,f) = max(0, 1 - y f).  Writing alpha_i = y_i beta_i / lambda, the dual is
#   max_beta  sum_i beta_i - (1/(2 lambda)) beta^T (Y K Y) beta,  0 <= beta <= 1,
# a box-constrained quadratic programme with no equality constraint, because
# this formulation carries no unpenalised intercept.
def svm_no_bias(K, y, lam, n_iter=200000, tol=1e-13):
    n = len(y)
    Q = (y[:, None] * y[None, :]) * K
    beta = np.zeros(n)
    for _ in range(n_iter):
        b_old = beta.copy()
        for i in range(n):
            s = Q[i] @ beta - Q[i, i] * beta[i]
            beta[i] = min(1.0, max(0.0, (lam - s) / Q[i, i]))
        if np.abs(beta - b_old).max() < tol:
            break
    return beta


beta = svm_no_bias(K, y, lam)
a_hinge = y * beta / lam
f_hinge = K @ a_hinge

primal = float(np.sum(np.maximum(0.0, 1.0 - y * f_hinge))
               + 0.5 * lam * a_hinge @ (K @ a_hinge))
dual = float(beta.sum() - 0.5 / lam * beta @ (((y[:, None] * y[None, :]) * K) @ beta))
out.write("=== 0. the hinge solver certifies itself ===\n")
out.write(f"  primal value  {primal:.10f}\n  dual value    {dual:.10f}\n"
          f"  duality gap   {abs(primal-dual):.3e}\n")
out.write("  a zero gap proves the box quadratic programme was solved exactly,\n"
          "  by Section 6.dual; no external solver is needed to check it\n\n")


# ===========================================================================
# the theorem
# ===========================================================================
out.write("=== 1. Theorem 6.threelosses: alpha_i = -L'(y_i, f_i)/lambda ===\n")
out.write("   loss          L'(y,f)                      predicted alpha"
          "        max |alpha - predicted|\n")

pred_ridge = (y - f_ridge) / lam
pred_logit = y * sigmoid(-y * f_logit) / lam
Lp_hinge = np.where(y * f_hinge < 1.0, -y, 0.0)
pred_hinge = -Lp_hinge / lam

out.write(f"   squared       f - y                        (y - f)/lambda"
          f"        {np.abs(a_ridge - pred_ridge).max():.3e}\n")
out.write(f"   logistic      -y sigma(-y f)               y sigma(-y f)/lambda"
          f"  {np.abs(a_logit - pred_logit).max():.3e}\n")
gap = np.abs(y * f_hinge - 1.0)
strict = gap > 1e-8
out.write(f"   hinge         -y if yf<1, 0 if yf>1        y/lambda or 0"
          f"         {np.abs(a_hinge - pred_hinge)[strict].max():.3e}\n")
out.write("   (the hinge row excludes the points sitting exactly on the margin,\n"
          "    where L is not differentiable and alpha may take any value\n"
          "    between 0 and y/lambda -- there are "
          f"{int((~strict).sum())} of them here)\n\n")

# ===========================================================================
# the consequence
# ===========================================================================
out.write("=== 2. the consequence: which coefficients vanish ===\n")
out.write("   method                    non-zero alpha   |alpha|_max"
          "   sum L(y_i,f_i)   (lambda/2)||f||^2   objective\n")
for name, a, f, L in [
        ("kernel ridge", a_ridge, f_ridge, 0.5 * (y - f_ridge) ** 2),
        ("kernel logistic", a_logit, f_logit, np.logaddexp(0.0, -y * f_logit)),
        ("support vector machine", a_hinge, f_hinge,
         np.maximum(0.0, 1.0 - y * f_hinge))]:
    pen = 0.5 * lam * a @ (K @ a)
    out.write(f"   {name:24s} {int((np.abs(a) > 1e-9).sum()):14d}"
              f"   {np.abs(a).max():11.4f}   {L.sum():14.6f}"
              f"   {pen:17.6f}   {L.sum()+pen:9.6f}\n")
out.write("\n  Only the hinge loss is exactly flat on an open set -- it is zero\n"
          "  for every point with margin above one -- so only for the hinge is\n"
          "  L' exactly zero there, and only the support vector machine has\n"
          "  zero coefficients.  The squared and logistic losses have nowhere\n"
          "  vanishing derivatives, so every point contributes.\n\n")

out.write("   how the sparsity moves with the penalty:\n")
out.write("     lambda    SVM non-zeros   ridge non-zeros   logistic non-zeros\n")
for lm in [0.1, 1.0, 10.0, 100.0]:
    b = svm_no_bias(K, y, lm)
    ah = y * b / lm
    ar = solve(K + lm * np.eye(n), y)
    al = kernel_logistic(K, y, lm)
    out.write(f"   {lm:8.2f}   {int((np.abs(ah)>1e-9).sum()):13d}"
              f"   {int((np.abs(ar)>1e-9).sum()):17d}"
              f"   {int((np.abs(al)>1e-9).sum()):20d}\n")
out.write("   the support vectors grow in number as the penalty rises, because\n"
          "   a flatter f leaves more points inside the margin; the other two\n"
          "   are dense at every penalty\n\n")

# ===========================================================================
# and yet the three functions are similar
# ===========================================================================
out.write("=== 3. three different representations, one similar function ===\n")
Xt = rng.uniform(-2.5, 2.5, size=(2000, 2))
yt = np.where((Xt[:, 0] ** 2 + Xt[:, 1] ** 2) < 3.0, 1.0, -1.0)
Kt = gaussian_kernel(Xt, X, gam)
out.write("   method                    test accuracy   agreement with SVM\n")
s_h = np.sign(Kt @ a_hinge)
for name, a in [("kernel ridge", a_ridge), ("kernel logistic", a_logit),
                ("support vector machine", a_hinge)]:
    s = np.sign(Kt @ a)
    out.write(f"   {name:24s}   {np.mean(s == yt):13.4f}"
              f"   {np.mean(s == s_h):18.4f}\n")
out.write("   the three decision boundaries agree on the large majority of the\n"
          "   plane while being built from very different numbers of points.\n"
          "   The choice among them is a choice of what to pay for: a closed\n"
          "   form (squared), a calibrated probability (logistic), or a sparse\n"
          "   representation (hinge).\n")
out.close()
print(open("three_losses.txt").read())

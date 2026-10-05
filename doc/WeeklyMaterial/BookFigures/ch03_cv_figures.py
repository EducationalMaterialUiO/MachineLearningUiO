"""Chapter 3, section on choosing the penalty by cross-validation.

Figure cv_ridge_lasso: the 5-fold cross-validation curves of Ridge (left) and
the Lasso (right) on the sparse, correlated problem of the section, with the
one-standard-error band, the training error, the true expected test error and
the number of non-zero Lasso coefficients.

Figure cv_versus_bootstrap: the same true error against four estimates of it --
training error, 5-fold cross-validation, the leave-one-out bootstrap and the
.632 estimator -- for both estimators.

The data, the seeds and the functions are identical to the listings of the
section, so the figures reproduce the numbers quoted in the text."""
from common import *
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import Lasso

def make_data(n, p, rho, sigma, rng):
    cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    theta = np.zeros(p)
    theta[:8] = [3.0, -2.0, 1.5, -1.0, 1.0, -0.8, 0.6, -0.5]
    return X, 2.0 + X @ theta + sigma * rng.normal(size=n), theta

def ridge_fit(X, y, lmbda):
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lmbda * np.eye(p), X.T @ y)

def lasso_fit(X, y, lmbda):
    return Lasso(alpha=lmbda / 2.0, fit_intercept=False, max_iter=100000, tol=1e-10).fit(X, y).coef_

def cv_curve(fit, X, y, lambdas, K=5, rng=None):
    n = X.shape[0]; folds = np.array_split(rng.permutation(n), K)
    errors = np.empty((K, len(lambdas)))
    for k, held_out in enumerate(folds):
        train = np.setdiff1d(np.arange(n), held_out)
        mu, sd, y_mean = X[train].mean(0), X[train].std(0), y[train].mean()
        X_tr, X_va = (X[train] - mu) / sd, (X[held_out] - mu) / sd
        for j, lmbda in enumerate(lambdas):
            theta = fit(X_tr, y[train] - y_mean, lmbda)
            errors[k, j] = np.mean((y[held_out] - y_mean - X_va @ theta) ** 2)
    return errors.mean(0), errors.std(0, ddof=1) / np.sqrt(K)

def select(lambdas, cv_mean, cv_se):
    i = np.argmin(cv_mean)
    return lambdas[i], lambdas[cv_mean <= cv_mean[i] + cv_se[i]].max()

def refit(fit, X, y, lmbda):
    mu, sd, y_mean = X.mean(0), X.std(0), y.mean()
    theta = fit((X - mu) / sd, y - y_mean, lmbda)
    return y_mean - (mu / sd) @ theta, theta / sd

def bootstrap_errors(fit, X, y, lmbda, B=200, rng=None):
    n = X.shape[0]
    t0, t = refit(fit, X, y, lmbda); apparent = np.mean((y - t0 - X @ t) ** 2)
    err_sum, err_count = np.zeros(n), np.zeros(n)
    for _ in range(B):
        idx = rng.integers(0, n, n); out = np.setdiff1d(np.arange(n), idx)
        t0, t = refit(fit, X[idx], y[idx], lmbda)
        err_sum[out] += (y[out] - t0 - X[out] @ t) ** 2; err_count[out] += 1
    seen = err_count > 0; loo_boot = np.mean(err_sum[seen] / err_count[seen])
    return apparent, loo_boot, 0.368 * apparent + 0.632 * loo_boot

rng = np.random.default_rng(3155)
X, y, theta_true = make_data(n=150, p=30, rho=0.7, sigma=1.0, rng=rng)
X_train, y_train = X[:100], y[:100]
X_big, y_big, _ = make_data(n=20000, p=30, rho=0.7, sigma=1.0, rng=np.random.default_rng(1))
lambdas = np.logspace(-4, 1, 60)

results = {}
for name, fit in (("Ridge", ridge_fit), ("Lasso", lasso_fit)):
    cv_mean, cv_se = cv_curve(fit, X_train, y_train, lambdas, K=5, rng=np.random.default_rng(2024))
    fits = [refit(fit, X_train, y_train, l) for l in lambdas]
    train = np.array([np.mean((y_train - t0 - X_train @ t) ** 2) for t0, t in fits])
    true = np.array([np.mean((y_big - t0 - X_big @ t) ** 2) for t0, t in fits])
    nnz = np.array([np.sum(t != 0) for t0, t in fits])
    boot = np.array([bootstrap_errors(fit, X_train, y_train, l, B=100, rng=np.random.default_rng(2024)) for l in lambdas])
    results[name] = (cv_mean, cv_se, train, true, nnz, boot, select(lambdas, cv_mean, cv_se))
    l_min, l_1se = results[name][-1]
    print(f"{name}: lambda_min {l_min:.4f}  lambda_1se {l_1se:.4f}  argmin true {lambdas[np.argmin(true)]:.4f}"
          f"  argmin .632 {lambdas[np.argmin(boot[:,2])]:.4f}  argmin LOO-boot {lambdas[np.argmin(boot[:,1])]:.4f}")

# Figure 1: the cross-validation curves
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for a, name in zip(ax, ("Ridge", "Lasso")):
    cv_mean, cv_se, train, true, nnz, boot, (l_min, l_1se) = results[name]
    a.fill_between(lambdas, cv_mean - cv_se, cv_mean + cv_se, color="C0", alpha=0.2, label=r"$\pm$ one standard error")
    a.semilogx(lambdas, cv_mean, "C0", lw=1.8, label="5-fold CV error")
    a.semilogx(lambdas, true, "C3", lw=1.4, ls="--", label="true test error")
    a.semilogx(lambdas, train, "C2", lw=1.2, ls=":", label="training error")
    a.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    a.axvline(l_min, color="C0", lw=0.9, ls="-."); a.axvline(l_1se, color="C0", lw=0.9, ls="-.")
    a.text(l_min, 0.12, r"$\lambda_{\min}$", ha="right", va="bottom", fontsize=9, color="C0")
    a.text(l_1se, 0.12, r"$\lambda_{1\mathrm{SE}}$", ha="left", va="bottom", fontsize=9, color="C0")
    a.set_xlabel(r"$\lambda$"); a.set_title(name, fontsize=10); a.set_ylim(0, 3.2)
    if name == "Lasso":
        b = a.twinx(); b.semilogx(lambdas, nnz, "C1", lw=1.2, label="non-zero coefficients")
        b.set_ylabel("non-zero coefficients", color="C1"); b.set_ylim(0, 32); b.grid(False)
        h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    else:
        a.set_ylabel("mean squared error"); a.legend(loc="upper left", fontsize=8)
save(fig, 3, "cv_ridge_lasso")

# Figure 2: cross-validation against the bootstrap estimators
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for a, name in zip(ax, ("Ridge", "Lasso")):
    cv_mean, cv_se, train, true, nnz, boot, (l_min, l_1se) = results[name]
    a.semilogx(lambdas, true, "C3", lw=1.6, ls="--", label="true test error")
    a.semilogx(lambdas, cv_mean, "C0", lw=1.8, label="5-fold CV")
    a.semilogx(lambdas, boot[:, 1], "C4", lw=1.4, label="leave-one-out bootstrap")
    a.semilogx(lambdas, boot[:, 2], "C5", lw=1.4, label=".632 estimator")
    a.semilogx(lambdas, train, "C2", lw=1.2, ls=":", label="training error")
    for v, c in ((true, "C3"), (cv_mean, "C0"), (boot[:, 1], "C4"), (boot[:, 2], "C5")):
        i = np.argmin(v); a.plot(lambdas[i], v[i], "o", color=c, ms=5)
    a.set_xlabel(r"$\lambda$"); a.set_title(name, fontsize=10); a.set_ylim(0, 3.2)
ax[0].set_ylabel("mean squared error"); ax[0].legend(loc="upper left", fontsize=8)
save(fig, 3, "cv_versus_bootstrap")

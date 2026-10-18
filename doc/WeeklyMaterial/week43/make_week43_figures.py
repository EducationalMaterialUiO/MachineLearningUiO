"""Code and figures for the week 43 slides (Monday lecture and Tuesday session).

Part 1 repeats the differential-equation method of week 42 on one equation,
the one-dimensional Poisson equation -g'' = f of Section 9.5, solved three
ways: with the trial solution g_t = x(1-x) N(x) that carries the boundary
conditions exactly (hard constraints, Eq. 9.18), with a physics-informed
neural network whose boundary conditions are penalty terms in the cost (soft
constraints, Section 9.10), and with the classical three-point finite
difference scheme (Eq. 9.21), whose tridiagonal system (Eq. 9.22) is solved
by Thomas' algorithm in O(n) operations.  The network code is the one of
weeks 41 and 42, imported unchanged.

Part 2 is about reading the output of a binary classifier: the confusion
matrix, accuracy, precision, recall, F1, the ROC curve and the area under it,
the precision-recall curve, and the decision threshold (Section 5.12 of the
lecture notes).  The classifiers are the ones of Project 2 -- logistic
regression (the week-41 network with no hidden layer) and a network with one
hidden layer -- on the full MNIST set with "is the digit a five?" as the
question: a deliberately imbalanced problem (one image in eleven is a five)
where the accuracy misleads.  Every metric is implemented by hand and checked
against scikit-learn.  The ten-class confusion matrix of the week-41 softmax
network closes the part, since that matrix is required in Project 2.

Part 3 opens the convolutional networks of Chapter 10 (Sections 10.1-10.3)
with the pieces that can be computed in a few lines: convolution as
polynomial multiplication and as a Toeplitz matrix, the two-dimensional
cross-correlation of a digit with three fixed filters, and the parameter
count of Eq. (10.18).  Max pooling (Section 10.5) and the equivariance of
Theorem 10.7 (Section 10.4) are implemented and checked here too, for the
week-44 lecture.

Produces, in this directory:
  week43_poisson.pdf      the three solutions of the Poisson equation, their
                          errors, and the training costs of the two networks.
  week43_metrics.pdf      the output probabilities of the two classifiers by
                          class, the ROC curves, the precision-recall curves,
                          and precision/recall/F1 against the threshold.
  week43_confusion.pdf    the 2x2 confusion matrices of the two classifiers
                          at threshold 0.5, and of the network at the
                          threshold that maximises F1.
  week43_confusion10.pdf  the 10x10 confusion matrix of the softmax network.
  week43_convolution.pdf  a digit and its three feature maps for three fixed filters.

All functions appear verbatim in week43.ipynb so that the students reproduce
exactly the lecture figures and numbers; seed 2026 for the classifiers, seed 1
for the Poisson networks (as in week 42).  Run `python3 make_week43_figures.py`
to produce the figures and print every number quoted on the slides (about
three minutes; the Adam-trained Poisson numbers depend on the platform, see
the week-42 remark -- the classifier numbers do not).
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from make_week41_figures import (sigmoid, relu, ACTIVATIONS, create_layers, feed_forward, predict, cost,
                                 backpropagation, accuracy, train, one_hot, load_mnist)
from make_week42_figures import (network, d_dxk, to_jax, adam_minimise, solve_de, pinn_solve,
                                 poisson_f, trial_poisson, residual_poisson, exact_poisson)

BLUE, RED, YELLOW, GREY = "#004488", "#BB5566", "#DDAA33", "#777777"
GREEN = "#228833"


# ======================================================================
# Part 1: the Poisson equation three ways (Sections 9.5 and 9.10)
# ======================================================================
def thomas(a, b, c, d):
    """Solve the tridiagonal system with sub-diagonal a, diagonal b, super-diagonal c and right-hand side d.

    a[0] and c[-1] are not used.  Forward sweep: eliminate the sub-diagonal,
    then back substitution -- 8n flops, against 2n^3/3 for Gaussian
    elimination on the full matrix (Section 1.14).  This is the LU
    factorisation of Section 1.14 specialised to a matrix with three
    diagonals; nothing outside them is ever touched or stored.
    """
    n = len(d)
    cp, dp = np.empty(n), np.empty(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):                       # forward sweep
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    u = np.empty(n)
    u[-1] = dp[-1]
    for i in range(n - 2, -1, -1):              # back substitution
        u[i] = dp[i] - cp[i] * u[i + 1]
    return u


def fd_poisson_thomas(n_intervals):
    """Eq. (9.22): -u_{i-1} + 2u_i - u_{i+1} = dx^2 f_i on the interior points, solved with Thomas' algorithm."""
    n = n_intervals
    dx = 1.0 / n
    x = np.linspace(0, 1, n + 1)
    xi = x[1:-1]
    a = -np.ones(n - 1) / dx ** 2
    b = 2 * np.ones(n - 1) / dx ** 2
    c = -np.ones(n - 1) / dx ** 2
    f = (3 * xi + xi ** 2) * np.exp(xi)
    u = np.zeros(n + 1)
    u[1:-1] = thomas(a, b, c, f)
    return x, u


def fd_poisson_dense(n_intervals):
    """The same system solved with numpy's LU on the full (n-1) x (n-1) matrix: the O(n^3) way, for the timing."""
    n = n_intervals
    dx = 1.0 / n
    x = np.linspace(0, 1, n + 1)
    xi = x[1:-1]
    A = (np.diag(2 * np.ones(n - 1)) - np.diag(np.ones(n - 2), 1) - np.diag(np.ones(n - 2), -1)) / dx ** 2
    u = np.zeros(n + 1)
    u[1:-1] = np.linalg.solve(A, (3 * xi + xi ** 2) * np.exp(xi))
    return x, u


def fd_table(ns=(10, 20, 50, 100, 1000, 10000), dense_max=2000, repeats=5):
    """Maximum error and wall time of Thomas' algorithm and of the dense solve, for a range of grids."""
    rows = []
    for n in ns:
        t0 = time.perf_counter()
        for _ in range(repeats):
            x, u = fd_poisson_thomas(n)
        t_thomas = (time.perf_counter() - t0) / repeats
        err = np.max(np.abs(u - exact_poisson(x)))
        t_dense = np.nan
        if n <= dense_max:
            t0 = time.perf_counter()
            for _ in range(repeats):
                fd_poisson_dense(n)
            t_dense = (time.perf_counter() - t0) / repeats
        rows.append((n, err, t_thomas, t_dense))
    return rows


def solve_poisson_hard(n_iter=4000, seed=1, n_col=60, verbose=False):
    """Week 42 once more: g_t = x(1-x) N(x), Eq. (9.18); the cost is the mean squared residual on 60 collocation points."""
    X = np.linspace(0, 1, n_col)[:, None]
    t0 = time.time()
    layers, hist = solve_de(residual_poisson, [1, 30, 30, 1], X, "tanh", n_iter=n_iter, gamma=1e-2,
                            rng=np.random.default_rng(seed), every=10, verbose=verbose)
    xf = np.linspace(0, 1, 201)
    g = np.asarray(trial_poisson(layers, jnp.asarray(xf[:, None])))
    return {"layers": layers, "history": np.array(hist), "time": time.time() - t0, "x": xf, "g": g,
            "err": np.abs(g - exact_poisson(xf)), "max": float(np.max(np.abs(g - exact_poisson(xf)))),
            "n_iter": n_iter}


def u_1d(layers, X):
    """The PINN ansatz, Eq. (9.34): the network itself is the solution, nothing carries the boundary conditions."""
    return network(layers, X, "tanh")


def r_poisson_pinn(layers, X):
    """The equation residual -u'' - f, Eq. (9.35), with u'' from two nested derivatives."""
    d2u = d_dxk(d_dxk(u_1d, 0), 0)(layers, X)
    return -d2u - poisson_f(X[:, 0])


def solve_poisson_soft(lam=10.0, n_iter=4000, seed=1, n_col=60, verbose=False):
    """Eq. (9.38) for the Poisson equation: mean squared residual + lambda [u(0)^2 + u(1)^2]."""
    X_col = jnp.asarray(np.linspace(0, 1, n_col)[1:-1, None])
    X_bc = jnp.asarray(np.array([[0.0], [1.0]]))
    terms = [("pde", 1.0, r_poisson_pinn, X_col), ("bc", lam, u_1d, X_bc)]
    t0 = time.time()
    layers, hist = pinn_solve(terms, [1, 30, 30, 1], "tanh", n_iter=n_iter, gamma=1e-2,
                              rng=np.random.default_rng(seed), every=10, verbose=verbose)
    xf = np.linspace(0, 1, 201)
    u = np.asarray(u_1d(layers, jnp.asarray(xf[:, None])))
    hist = np.array(hist)
    return {"layers": layers, "history": hist, "time": time.time() - t0, "x": xf, "g": u,
            "err": np.abs(u - exact_poisson(xf)), "max": float(np.max(np.abs(u - exact_poisson(xf)))),
            "bc": (float(u[0]), float(u[-1])), "lam": lam, "n_iter": n_iter,
            "final_terms": hist[-1, 2:]}


def fig_poisson(hard, softs, fd_rows, fname="week43_poisson.pdf"):
    """Three panels: the solutions, the pointwise errors, and the training costs."""
    soft = softs[10.0]
    x10, u10 = fd_poisson_thomas(10)
    x100, u100 = fd_poisson_thomas(100)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
    xf = hard["x"]
    ax[0].plot(xf, exact_poisson(xf), color=GREY, lw=3, label="exact $x(1-x)e^x$")
    ax[0].plot(x10, u10, "o", color=GREEN, ms=5, label="finite differences, $n=10$ (Thomas)")
    ax[0].plot(xf, hard["g"], "--", color=BLUE, lw=1.6, label=f"hard: $x(1-x)N(x)$, {hard['n_iter']} it.")
    ax[0].plot(xf, soft["g"], ":", color=RED, lw=2, label=rf"soft (PINN), $\lambda={soft['lam']:g}$")
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$g(x)$")
    ax[0].legend(fontsize=7)
    ax[0].set_title("(a) $-g''=(3x+x^2)e^x$, $g(0)=g(1)=0$", fontsize=10)
    ax[1].semilogy(x10[1:-1], np.abs(u10 - exact_poisson(x10))[1:-1], "o-", color=GREEN, ms=4, lw=1,
                   label=f"FD $n=10$: max {np.max(np.abs(u10 - exact_poisson(x10))):.1e}")
    ax[1].semilogy(x100[1:-1], np.abs(u100 - exact_poisson(x100))[1:-1], "-", color=GREEN, lw=1, alpha=0.6,
                   label=f"FD $n=100$: max {np.max(np.abs(u100 - exact_poisson(x100))):.1e}")
    ax[1].semilogy(xf, np.maximum(hard["err"], 1e-16), "--", color=BLUE, lw=1.6, label=f"hard: max {hard['max']:.1e}")
    for lam, col in ((1.0, YELLOW), (10.0, RED), (100.0, "#882255")):
        s = softs[lam]
        ax[1].semilogy(xf, np.maximum(s["err"], 1e-16), ":", color=col, lw=1.8,
                       label=rf"soft $\lambda={lam:g}$: max {s['max']:.1e}")
    ax[1].set_xlabel("$x$")
    ax[1].set_ylabel("$|g_{\\mathrm{approx}}(x)-g(x)|$")
    ax[1].set_ylim(1e-8, 1)
    ax[1].legend(fontsize=6.5, loc="lower center", ncol=2)
    ax[1].set_title("(b) the pointwise errors", fontsize=10)
    h = hard["history"]
    ax[2].semilogy(h[:, 0], h[:, 1], color=BLUE, lw=1.4, label="hard: mean squared residual")
    tr = soft["history"]
    ax[2].semilogy(tr[:, 0], tr[:, 2], color=RED, lw=1.4, label=rf"soft $\lambda={soft['lam']:g}$: $L_{{\mathrm{{PDE}}}}$")
    ax[2].semilogy(tr[:, 0], np.maximum(tr[:, 3], 1e-16), color=RED, lw=1.4, ls=":",
                   label=r"soft: $L_{\mathrm{BC}}=\frac{1}{2}[u(0)^2+u(1)^2]$")
    ax[2].set_xlabel("Adam iteration")
    ax[2].set_ylabel("cost terms")
    ax[2].legend(fontsize=7)
    ax[2].set_title("(c) training: what the optimiser sees", fontsize=10)
    fig.tight_layout()
    fig.savefig(fname)
    fig.savefig(fname.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)


# ======================================================================
# Part 2: reading a binary classifier (Section 5.12), by hand
# ======================================================================
def confusion_matrix(y_true, y_pred):
    """Eq. (5.67): the four counts TP, FN, FP, TN for labels in {0, 1}; the positive class is 1."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fn, fp, tn


def classification_report(y_true, y_pred):
    """Every scalar summary is a function of the four counts: Eqs. (5.68)-(5.70) and a few more."""
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred)
    n = tp + fn + fp + tn
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if tp + fp else 0.0          # of those flagged positive, how many were?
    recall = tp / (tp + fn) if tp + fn else 0.0             # of the positives, how many did we find? (= TPR)
    specificity = tn / (tn + fp) if tn + fp else 0.0        # of the negatives, how many did we leave alone? (= 1 - FPR)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced = 0.5 * (recall + specificity)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0  # Matthews correlation coefficient: correlation of y and y_hat
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, accuracy=accuracy, precision=precision, recall=recall,
                specificity=specificity, f1=f1, balanced_accuracy=balanced, mcc=mcc)


def roc_curve(y_true, scores):
    """False and true positive rates as the threshold sweeps down through every distinct score.

    Sorting the scores once gives every point of the curve in O(n log n):
    walking down the sorted list, each sample flagged positive adds one to
    TP (if it is a positive) or to FP (if it is a negative).
    """
    y_true, scores = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y, s = y_true[order], scores[order]
    tp, fp = np.cumsum(y == 1), np.cumsum(y == 0)
    last = np.r_[np.diff(s) != 0, True]            # one point per distinct threshold
    tpr = np.r_[0.0, tp[last] / max(tp[-1], 1)]
    fpr = np.r_[0.0, fp[last] / max(fp[-1], 1)]
    thresholds = np.r_[np.inf, s[last]]
    return fpr, tpr, thresholds


def auc_trapezoid(fpr, tpr):
    """The area under the ROC curve by the trapezoidal rule."""
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def roc_auc(y_true, scores):
    """The AUC as the probability that a random positive outranks a random negative (Section 5.12), from the ranks.

    Tied scores get the average of the ranks they span, so that a tie between
    a positive and a negative counts half a pair -- which is also what the
    trapezoidal rule does along a tie segment of the ROC curve.
    """
    y_true, scores = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    first = np.cumsum(cnt) - cnt + 1                  # rank of the first member of each tie group
    ranks = (first + (cnt - 1) / 2)[inv]              # average rank within the group
    n_pos = int(np.sum(y_true == 1))
    n_neg = len(y_true) - n_pos
    S = np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2
    return float(S / (n_pos * n_neg))


def precision_recall_curve(y_true, scores):
    """Precision and recall at every distinct threshold, from the same sorted sweep as the ROC curve."""
    y_true, scores = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y, s = y_true[order], scores[order]
    tp, fp = np.cumsum(y == 1), np.cumsum(y == 0)
    last = np.r_[np.diff(s) != 0, True]
    precision = tp[last] / (tp[last] + fp[last])
    recall = tp[last] / max(tp[-1], 1)
    return np.r_[1.0, precision], np.r_[0.0, recall], s[last]


def threshold_sweep(y_true, scores, thresholds=np.linspace(0.01, 0.99, 99)):
    """Precision, recall and F1 of the decision 'score > t' for a grid of thresholds t."""
    rows = []
    for t in thresholds:
        r = classification_report(y_true, scores > t)
        rows.append((t, r["precision"], r["recall"], r["f1"], r["accuracy"]))
    return np.array(rows)


def confusion_matrix_multiclass(y_true, y_pred, n_classes):
    """C[i, j] = number of samples of true class i predicted as class j."""
    C = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(C, (np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)), 1)
    return C


def per_class_from_confusion(C):
    """Precision and recall of every class from the K x K matrix: each class against the rest."""
    tp = np.diag(C).astype(float)
    precision = tp / np.maximum(C.sum(axis=0), 1)
    recall = tp / np.maximum(C.sum(axis=1), 1)
    return precision, recall


# ----------------------------------------------------------------------
# The classifiers of Project 2 on "is it a five?": no hidden layer (logistic
# regression) and one hidden layer, the week-41 code unchanged
# ----------------------------------------------------------------------
POSITIVE_DIGIT = 5


def five_classifiers(epochs=10, batch_size=100, hidden=32, verbose=False):
    """Two binary classifiers on the full MNIST set with y = 1 for the fives; returns scores on the test set."""
    X_train, y_train, X_test, y_test = load_mnist()
    yb, ybt = (y_train == POSITIVE_DIGIT).astype(float), (y_test == POSITIVE_DIGIT).astype(float)
    out = {"y_test": ybt, "y_train": yb, "digits_test": y_test, "X_test": X_test,
           "prevalence_train": float(yb.mean()), "prevalence_test": float(ybt.mean())}
    for name, sizes, act in (("logistic regression", [784, 1], "sigmoid"),
                             (f"one hidden layer of {hidden}", [784, hidden, 1], "relu")):
        t0 = time.time()
        layers = create_layers(sizes, act, np.random.default_rng(2026))
        layers, hist = train(X_train, yb[:, None], layers, act, "binary", gamma=0.1, epochs=epochs,
                             batch_size=batch_size, X_test=X_test, y_test=ybt, verbose=verbose)
        p = predict(X_test, layers, act, "binary")[:, 0]
        out[name] = {"layers": layers, "history": hist, "time": time.time() - t0, "p_test": p,
                     "n_params": sum(W.size + b.size for W, b in layers),
                     "train_accuracy": accuracy(X_train, yb, layers, act, "binary"),
                     "test_accuracy": hist["test_accuracy"][-1]}
    return out


def softmax_network(epochs=20, hidden=100, batch_size=100, verbose=False):
    """The 784-100-10 ReLU softmax network of weeks 41 and 42, for the ten-class confusion matrix."""
    X_train, y_train, X_test, y_test = load_mnist()
    t0 = time.time()
    layers = create_layers([784, hidden, 10], "relu", np.random.default_rng(2026))
    layers, hist = train(X_train, one_hot(y_train, 10), layers, "relu", "softmax", gamma=0.1, epochs=epochs,
                         batch_size=batch_size, X_test=X_test, y_test=y_test, verbose=verbose)
    P = predict(X_test, layers, "relu", "softmax")
    yhat = np.argmax(P, axis=1)
    C = confusion_matrix_multiclass(y_test, yhat, 10)
    return {"layers": layers, "history": hist, "time": time.time() - t0, "P_test": P, "yhat": yhat, "y_test": y_test,
            "C": C, "test_accuracy": float(np.mean(yhat == y_test))}


def analyse_binary(out, names=("logistic regression", "one hidden layer of 32")):
    """Every number of Part 2: reports at 0.5, ROC/AUC (hand and sklearn), F1-optimal threshold, cost-weighted threshold."""
    from sklearn.metrics import roc_auc_score, confusion_matrix as sk_confusion, f1_score
    y = out["y_test"]
    res = {"majority_accuracy": float(1 - y.mean())}
    for name in names:
        p = out[name]["p_test"]
        r = {"report_05": classification_report(y, p > 0.5)}
        fpr, tpr, thr = roc_curve(y, p)
        r["roc"] = (fpr, tpr, thr)
        r["auc_trapezoid"] = auc_trapezoid(fpr, tpr)
        r["auc_rank"] = roc_auc(y, p)
        r["auc_sklearn"] = float(roc_auc_score(y, p))
        r["pr"] = precision_recall_curve(y, p)
        sweep = threshold_sweep(y, p)
        r["sweep"] = sweep
        k = int(np.argmax(sweep[:, 3]))
        r["t_f1"], r["f1_max"] = float(sweep[k, 0]), float(sweep[k, 3])
        r["report_f1"] = classification_report(y, p > r["t_f1"])
        # a screening-type choice: a missed five costs five times a false alarm
        costs = [(t, 5 * classification_report(y, p > t)["FN"] + classification_report(y, p > t)["FP"]) for t in sweep[:, 0]]
        r["t_cost"] = float(min(costs, key=lambda c: c[1])[0])
        r["report_cost"] = classification_report(y, p > r["t_cost"])
        r["sk_confusion"] = sk_confusion(y, p > 0.5)          # rows true 0/1, columns predicted 0/1
        r["sk_f1"] = float(f1_score(y, p > 0.5))
        res[name] = r
    return res


def fig_metrics(out, res, names=("logistic regression", "one hidden layer of 32"), fname="week43_metrics.pdf"):
    y = out["y_test"]
    cols = {names[0]: GREY, names[1]: BLUE}
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    p = out[names[1]]["p_test"]
    bins = np.linspace(0, 1, 41)
    ax[0].hist(p[y == 0], bins=bins, color=GREY, alpha=0.8, label=f"not a five ({int(np.sum(y == 0))})")
    ax[0].hist(p[y == 1], bins=bins, color=RED, alpha=0.8, label=f"a five ({int(np.sum(y == 1))})")
    ax[0].axvline(0.5, color="k", lw=1, ls="--")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("output probability $p$ (one hidden layer)")
    ax[0].set_ylabel("test images")
    ax[0].legend(fontsize=7)
    ax[0].set_title("(a) what the classifier outputs", fontsize=10)
    ax[1].plot([0, 1], [0, 1], color="k", lw=0.8, ls="--", label="random ranking: AUC 0.5")
    for name in names:
        fpr, tpr, thr = res[name]["roc"]
        ax[1].plot(fpr, tpr, color=cols[name], lw=1.6, label=f"{name}: AUC {res[name]['auc_rank']:.4f}")
        k = np.searchsorted(-thr, -0.5)
        ax[1].plot(fpr[min(k, len(fpr) - 1)], tpr[min(k, len(tpr) - 1)], "o", color=cols[name], ms=5)
    ax[1].set_xlabel("false positive rate FP/(FP+TN)")
    ax[1].set_ylabel("true positive rate TP/(TP+FN) = recall")
    ax[1].set_xlim(-0.01, 0.3)
    ax[1].set_ylim(0.6, 1.005)
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_title("(b) ROC curves (zoomed); dots: threshold 0.5", fontsize=10)
    for name in names:
        prec, rec, _ = res[name]["pr"]
        ax[2].plot(rec, prec, color=cols[name], lw=1.6, label=name)
    ax[2].axhline(out["prevalence_test"], color=RED, lw=1, ls=":", label=f"always 'five': precision {out['prevalence_test']:.3f}")
    ax[2].set_xlabel("recall")
    ax[2].set_ylabel("precision")
    ax[2].set_ylim(0, 1.02)
    ax[2].legend(fontsize=7, loc="lower left")
    ax[2].set_title("(c) precision-recall curves", fontsize=10)
    sw = res[names[1]]["sweep"]
    ax[3].plot(sw[:, 0], sw[:, 1], color=BLUE, lw=1.6, label="precision")
    ax[3].plot(sw[:, 0], sw[:, 2], color=RED, lw=1.6, label="recall")
    ax[3].plot(sw[:, 0], sw[:, 3], color=GREEN, lw=1.6, label="$F_1$")
    ax[3].plot(sw[:, 0], sw[:, 4], color=GREY, lw=1.2, ls="--", label="accuracy")
    ax[3].axvline(res[names[1]]["t_f1"], color=GREEN, lw=0.8, ls=":")
    ax[3].axvline(0.5, color="k", lw=0.8, ls="--")
    ax[3].set_xlabel("decision threshold $t$ (one hidden layer)")
    ax[3].set_ylim(0.6, 1.01)
    ax[3].legend(fontsize=7, loc="lower center", ncol=2)
    ax[3].set_title(f"(d) the threshold is a choice ($F_1$ peaks at $t={res[names[1]]['t_f1']:.2f}$)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(fname)
    fig.savefig(fname.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)


def draw_confusion(ax, tp, fn, fp, tn, title):
    """A 2x2 confusion matrix in the layout of Eq. (5.67): rows true, columns predicted, positive class first."""
    M = np.array([[tp, fn], [fp, tn]])
    ax.imshow(np.log10(M + 1), cmap="Blues", vmin=0, vmax=np.log10(M.sum()))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]}", ha="center", va="center", fontsize=13,
                    color="white" if M[i, j] > 0.3 * M.sum() else "black")
    ax.set_xticks([0, 1], ["pred. five", "pred. not"])
    ax.set_yticks([0, 1], ["true five", "true not"])
    for (i, j), lab in zip(((0, 0), (0, 1), (1, 0), (1, 1)), ("TP", "FN", "FP", "TN")):
        ax.text(j - 0.45, i - 0.42, lab, ha="left", va="top", fontsize=8,
                color="white" if M[i, j] > 0.3 * M.sum() else GREY)
    ax.set_title(title, fontsize=9)


def fig_confusion(out, res, names=("logistic regression", "one hidden layer of 32"), fname="week43_confusion.pdf"):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, name in zip(ax[:2], names):
        r = res[name]["report_05"]
        draw_confusion(a, r["TP"], r["FN"], r["FP"], r["TN"],
                       f"{name}, $t=0.5$\nacc. {r['accuracy']:.4f}, prec. {r['precision']:.3f}, rec. {r['recall']:.3f}, $F_1$ {r['f1']:.3f}")
    r = res[names[1]]["report_f1"]
    draw_confusion(ax[2], r["TP"], r["FN"], r["FP"], r["TN"],
                   f"{names[1]}, $t={res[names[1]]['t_f1']:.2f}$ (max $F_1$)\nacc. {r['accuracy']:.4f}, prec. {r['precision']:.3f}, rec. {r['recall']:.3f}, $F_1$ {r['f1']:.3f}")
    fig.suptitle(f"MNIST test set, 'is it a five?': {int(out['y_test'].sum())} fives among 10 000 images; "
                 f"'never a five' scores accuracy {res['majority_accuracy']:.4f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(fname)
    fig.savefig(fname.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)


def fig_confusion10(sm, fname="week43_confusion10.pdf"):
    C = sm["C"]
    prec, rec = per_class_from_confusion(C)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    off = C.astype(float).copy()
    np.fill_diagonal(off, np.nan)
    im = ax[0].imshow(off, cmap="Reds", vmin=0, vmax=np.nanmax(off))
    for i in range(10):
        for j in range(10):
            if i == j:
                ax[0].text(j, i, f"{C[i, j]}", ha="center", va="center", fontsize=7, color=BLUE, fontweight="bold")
            elif C[i, j] > 0:
                ax[0].text(j, i, f"{C[i, j]}", ha="center", va="center", fontsize=7,
                           color="white" if C[i, j] > 0.6 * np.nanmax(off) else "black")
    ax[0].set_xticks(range(10))
    ax[0].set_yticks(range(10))
    ax[0].set_xlabel("predicted digit")
    ax[0].set_ylabel("true digit")
    ax[0].set_title(f"(a) 784-100-10 softmax network, test accuracy {sm['test_accuracy']:.4f}\n"
                    f"diagonal in blue; colour = off-diagonal count", fontsize=9)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    w = 0.38
    ax[1].bar(np.arange(10) - w / 2, prec, w, color=BLUE, label="precision (column)")
    ax[1].bar(np.arange(10) + w / 2, rec, w, color=RED, label="recall (row)")
    ax[1].set_ylim(0.9, 1.0)
    ax[1].set_xticks(range(10))
    ax[1].set_xlabel("digit")
    ax[1].legend(fontsize=8, loc="lower left")
    ax[1].set_title("(b) each class against the rest, from the same matrix", fontsize=9)
    fig.tight_layout()
    fig.savefig(fname)
    fig.savefig(fname.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    return prec, rec


# ======================================================================
# Part 3: the first pieces of a convolutional network (Chapter 10)
# ======================================================================
def toeplitz_matrix(w, n):
    """Eq. (10.8): the (n+m-1) x n matrix T_w with (T_w)_{ij} = w_{i-j}, so that T_w x = w * x (full convolution)."""
    m = len(w)
    T = np.zeros((n + m - 1, n))
    for i in range(n + m - 1):
        for j in range(n):
            if 0 <= i - j < m:
                T[i, j] = w[i - j]
    return T


def convolution_checks():
    """Polynomial multiplication = convolution = Toeplitz matrix (Eqs. 10.5-10.8), on the example of Section 10.2.1."""
    alpha = np.array([1.0, 2.0, 3.0])          # p(t) = 1 + 2t + 3t^2
    beta = np.array([4.0, -1.0, 0.5, 2.0])     # s(t) = 4 - t + 0.5 t^2 + 2 t^3
    delta_poly = np.polynomial.polynomial.polymul(alpha, beta)
    delta_conv = np.convolve(alpha, beta)
    T = toeplitz_matrix(alpha, len(beta))
    return {"alpha": alpha, "beta": beta, "polymul": delta_poly, "convolve": delta_conv, "toeplitz": T @ beta,
            "T": T, "agree": bool(np.allclose(delta_poly, delta_conv) and np.allclose(delta_conv, T @ beta))}


def cross_correlate2d(X, W, stride=1, padding=0):
    """Eq. (10.15)/(10.16) for one channel and one filter: Y(i,j) = sum_{m,n} X(Si+m-P, Sj+n-P) W(m,n)."""
    if padding:
        X = np.pad(X, padding)
    F = W.shape[0]
    H2 = (X.shape[0] - F) // stride + 1
    W2 = (X.shape[1] - F) // stride + 1
    Y = np.zeros((H2, W2))
    for i in range(H2):
        for j in range(W2):
            Y[i, j] = np.sum(X[stride * i:stride * i + F, stride * j:stride * j + F] * W)
    return Y


def max_pool2d(X, F=2, S=2):
    """Eq. (10.21): the largest value in each F x F window, windows S apart; no parameters."""
    H2, W2 = (X.shape[0] - F) // S + 1, (X.shape[1] - F) // S + 1
    Y = np.zeros((H2, W2))
    for i in range(H2):
        for j in range(W2):
            Y[i, j] = np.max(X[S * i:S * i + F, S * j:S * j + F])
    return Y


FILTERS = {"vertical edges": np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]),
           "horizontal edges": np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]),
           "blur (mean)": np.ones((3, 3)) / 9.0}


def conv_arithmetic(H1, W1, C, K, F, S, P):
    """Proposition 10.1: output size, Eq. (10.17), and parameter count, Eq. (10.18)."""
    H2 = (H1 - F + 2 * P) // S + 1
    W2 = (W1 - F + 2 * P) // S + 1
    return H2, W2, K, K * (C * F * F + 1)


def equivariance_check(X, W, shift=(2, 3)):
    """Theorem 10.3 numerically: shifting the input shifts the feature map, exactly, away from the border."""
    Y = cross_correlate2d(X, W)
    Xs = np.roll(np.roll(X, shift[0], axis=0), shift[1], axis=1)
    Ys = cross_correlate2d(Xs, W)
    Ys_back = np.roll(np.roll(Ys, -shift[0], axis=0), -shift[1], axis=1)
    m = max(shift) + 1
    interior = np.max(np.abs(Y[m:-m, m:-m] - Ys_back[m:-m, m:-m]))
    return float(interior), float(np.max(np.abs(Y - Ys_back)))


def convolution_demo(index=None, fname="week43_convolution.pdf"):
    """A test digit (the first five), its three feature maps, and the max-pooled vertical-edge map; checks against scipy."""
    from scipy.signal import correlate2d
    X_train, y_train, X_test, y_test = load_mnist()
    if index is None:
        index = int(np.argmax(y_test == POSITIVE_DIGIT))
    img = X_test[index].reshape(28, 28)
    maps = {name: cross_correlate2d(img, W) for name, W in FILTERS.items()}
    checks = {name: float(np.max(np.abs(maps[name] - correlate2d(img, W, mode="valid")))) for name, W in FILTERS.items()}
    pooled = max_pool2d(np.maximum(maps["vertical edges"], 0.0))     # not shown this week: pooling is Section 10.5
    equi = equivariance_check(img, FILTERS["vertical edges"])         # not shown this week: equivariance is Section 10.4
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 3.6))
    ax[0].imshow(img, cmap="gray_r")
    ax[0].set_title(f"input $28\\times28$, label {y_test[index]}", fontsize=9)
    for a, (name, Y) in zip(ax[1:4], maps.items()):
        a.imshow(Y, cmap="RdBu_r", vmin=-np.max(np.abs(Y)), vmax=np.max(np.abs(Y)))
        a.set_title(f"{name}\n$3\\times3$ filter, valid: ${Y.shape[0]}\\times{Y.shape[1]}$", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(fname)
    fig.savefig(fname.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    return {"index": index, "label": int(y_test[index]), "scipy_checks": checks, "equivariance": equi,
            "shapes": {name: Y.shape for name, Y in maps.items()}, "pooled_shape": pooled.shape}


# ======================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    t_start = time.time()

    print("=== Part 1: the Poisson equation three ways (week43_poisson.pdf)")
    rows = fd_table()
    print("finite differences, Thomas' algorithm against the dense solve:")
    for n, err, t_th, t_de in rows:
        print(f"  n = {n:6d}  max err {err:.2e}   Thomas {1e3 * t_th:8.3f} ms   dense {'-' if np.isnan(t_de) else f'{1e3 * t_de:8.3f} ms'}")
    hard = solve_poisson_hard()
    print(f"hard constraints, x(1-x)N(x), [1,30,30,1], 4000 Adam iterations, seed 1: max err {hard['max']:.2e}, "
          f"final cost {hard['history'][-1, 1]:.2e}, {hard['time']:.1f} s")
    softs = {}
    for lam in (1.0, 10.0, 100.0):
        softs[lam] = solve_poisson_soft(lam=lam)
        s = softs[lam]
        print(f"soft constraints, lambda = {lam:5g}: max err {s['max']:.2e}, u(0) = {s['bc'][0]:+.2e}, u(1) = {s['bc'][1]:+.2e}, "
              f"L_PDE {s['final_terms'][0]:.2e}, L_BC {s['final_terms'][1]:.2e}, {s['time']:.1f} s")
    fig_poisson(hard, softs, rows)
    print(f"Part 1 done in {time.time() - t_start:.0f} s\n")

    print("=== Part 2: 'is it a five?' -- confusion matrix, ROC, thresholds (week43_metrics.pdf, week43_confusion.pdf)")
    t0 = time.time()
    out = five_classifiers()
    print(f"prevalence of fives: train {out['prevalence_train']:.4f}, test {out['prevalence_test']:.4f}")
    res = analyse_binary(out)
    print(f"the classifier that never says 'five': accuracy {res['majority_accuracy']:.4f}")
    for name in ("logistic regression", "one hidden layer of 32"):
        o, r = out[name], res[name]
        print(f"--- {name}: {o['n_params']} parameters, {o['time']:.0f} s, train acc {o['train_accuracy']:.4f}, test acc {o['test_accuracy']:.4f}")
        rep = r["report_05"]
        print(f"  t = 0.5: TP {rep['TP']} FN {rep['FN']} FP {rep['FP']} TN {rep['TN']}  acc {rep['accuracy']:.4f} prec {rep['precision']:.4f} "
              f"rec {rep['recall']:.4f} spec {rep['specificity']:.4f} F1 {rep['f1']:.4f} bal.acc {rep['balanced_accuracy']:.4f} MCC {rep['mcc']:.4f}")
        print(f"  sklearn confusion_matrix (rows true 0/1, cols pred 0/1): {r['sk_confusion'].tolist()}, sklearn F1 {r['sk_f1']:.4f}")
        print(f"  AUC: trapezoid {r['auc_trapezoid']:.5f}, ranks {r['auc_rank']:.5f}, sklearn {r['auc_sklearn']:.5f}")
        rep = r["report_f1"]
        print(f"  F1-optimal threshold t = {r['t_f1']:.2f}: TP {rep['TP']} FN {rep['FN']} FP {rep['FP']} TN {rep['TN']}  acc {rep['accuracy']:.4f} "
              f"prec {rep['precision']:.4f} rec {rep['recall']:.4f} F1 {rep['f1']:.4f}")
        rep = r["report_cost"]
        print(f"  cost-weighted threshold (a missed five = 5 false alarms) t = {r['t_cost']:.2f}: TP {rep['TP']} FN {rep['FN']} FP {rep['FP']} TN {rep['TN']}  "
              f"acc {rep['accuracy']:.4f} prec {rep['precision']:.4f} rec {rep['recall']:.4f} F1 {rep['f1']:.4f}")
    fig_metrics(out, res)
    fig_confusion(out, res)
    print(f"Part 2 binary done in {time.time() - t0:.0f} s")
    t0 = time.time()
    sm = softmax_network()
    prec, rec = fig_confusion10(sm)
    print(f"--- ten classes, 784-100-10 ReLU softmax, 20 epochs: test accuracy {sm['test_accuracy']:.4f}, {sm['time']:.0f} s")
    print("confusion matrix (rows true, columns predicted):")
    print(sm["C"])
    print(f"per-class precision: {prec}")
    print(f"per-class recall:    {rec}")
    C = sm["C"].copy()
    np.fill_diagonal(C, 0)
    worst = np.dstack(np.unravel_index(np.argsort(-C.ravel())[:6], C.shape))[0]
    print("largest confusions (true -> predicted: count): " + ", ".join(f"{i} -> {j}: {C[i, j]}" for i, j in worst))
    print(f"Part 2 ten classes done in {time.time() - t0:.0f} s\n")

    print("=== Part 3: convolution (week43_convolution.pdf)")
    ck = convolution_checks()
    print(f"polymul   {ck['polymul']}\nconvolve  {ck['convolve']}\nToeplitz  {ck['toeplitz']}   all agree: {ck['agree']}")
    print(f"T_alpha =\n{ck['T']}")
    print(f"Example 10.1: 32x32x3, K=10, F=5, S=1, P=0 -> {conv_arithmetic(32, 32, 3, 10, 5, 1, 0)} (H2, W2, K, parameters); "
          f"dense layer with the same outputs: {32 * 32 * 3 * 28 * 28 * 10:.2e}")
    print(f"MNIST 28x28x1, K=8, F=3, S=1, P=1 -> {conv_arithmetic(28, 28, 1, 8, 3, 1, 1)};  dense 784 -> 100: {784 * 100 + 100}")
    demo = convolution_demo()
    print(f"digit: test image {demo['index']} (label {demo['label']}); feature maps {demo['shapes']}, pooled {demo['pooled_shape']}")
    print(f"agreement with scipy.signal.correlate2d: {demo['scipy_checks']}")
    print(f"equivariance check, shift (2,3): interior discrepancy {demo['equivariance'][0]:.1e}, including the border {demo['equivariance'][1]:.2f}")
    print(f"\nall done in {time.time() - t_start:.0f} s")

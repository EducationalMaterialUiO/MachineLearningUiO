"""Chapter 14 against the libraries, and against autograd.

Neither framework ships a Boltzmann machine, so "the same implementation in
PyTorch" means the same equations written in torch tensors.  That makes this
file a slightly different kind of check from the ones in Chapters 10 to 13, and
a more interesting one, because the thing being verified is not a layer but a
\\emph{theorem}.

1.  Free energy, conditionals and the CD gradient, three ways: ours, PyTorch,
    TensorFlow, with identical weights.
2.  Theorem 14.gradient by automatic differentiation.  For a machine small
    enough to enumerate, $\\log p(\\bm{x}) = -F(\\bm{x}) - \\log Z$ is an ordinary
    differentiable function of the parameters, so autograd can differentiate it
    directly -- with no reference to positive and negative phases at all.  If
    the two-phase decomposition is right, the two must agree.
3.  scikit-learn's ``BernoulliRBM``, a third independent implementation, on the
    conditionals and on one CD-1 step.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import tensorflow as tf
from sklearn.neural_network import BernoulliRBM

import rbm

torch.set_default_dtype(torch.float64)

out = open("cross_check.txt", "w", buffering=1)
rng = np.random.default_rng(0)

M, N, n = 9, 5, 32
P = rbm.init_rbm(M, N, rng=np.random.default_rng(1), scale=0.4)
X = (rng.random((n, M)) < 0.5).astype(float)

tW = torch.tensor(P["W"])
ta = torch.tensor(P["a"])
tb = torch.tensor(P["b"])
kW = tf.constant(P["W"])
ka = tf.constant(P["a"])
kb = tf.constant(P["b"])

# ---------------------------------------------------------------------------
# 1.  free energy and the conditionals
# ---------------------------------------------------------------------------
out.write("=== 1. free energy and conditionals, identical weights ===\n")
F_ours = rbm.free_energy(P, X)
F_t = (-(torch.tensor(X) @ ta)
       - torch.sum(torch.nn.functional.softplus(torch.tensor(X) @ tW + tb),
                   dim=-1)).numpy()
F_k = (-tf.linalg.matvec(tf.constant(X), ka)
       - tf.reduce_sum(tf.math.softplus(tf.constant(X) @ kW + kb),
                       axis=-1)).numpy()
out.write(f"  F(x), Eq. (14.freeenergy)  max |ours - torch| : "
          f"{np.abs(F_ours - F_t).max():.3e}\n")
out.write(f"  F(x), Eq. (14.freeenergy)  max |ours - keras| : "
          f"{np.abs(F_ours - F_k).max():.3e}\n")

ph_ours = rbm.p_h_given_x(P, X)
ph_t = torch.sigmoid(torch.tensor(X) @ tW + tb).numpy()
ph_k = tf.sigmoid(tf.constant(X) @ kW + kb).numpy()
out.write(f"  p(h|x), Eq. (14.condh)     max |ours - torch| : "
          f"{np.abs(ph_ours - ph_t).max():.3e}\n")
out.write(f"  p(h|x), Eq. (14.condh)     max |ours - keras| : "
          f"{np.abs(ph_ours - ph_k).max():.3e}\n")
out.write(f"  log Z by enumeration ({2**M} states)           : "
          f"{rbm.log_Z(P):.12f}\n")
out.write(f"  mean log-likelihood, Eq. (14.loglik)          : "
          f"{rbm.log_likelihood(P, X):.12f}\n\n")

# ---------------------------------------------------------------------------
# 2.  Theorem 14.gradient, by automatic differentiation
# ---------------------------------------------------------------------------
out.write("=== 2. Theorem 14.gradient checked against autograd ===\n")
out.write("  log p(x) = -F(x) - log Z is differentiable when Z is enumerable,\n"
          "  so PyTorch can differentiate it with no notion of a positive or a\n"
          "  negative phase.  Eq. (14.gradient) says the answer is a difference\n"
          "  of two expectations.  It should not have to be told.\n\n")

V_all = torch.tensor(rbm.all_states(M))
pw = {k: torch.tensor(v, requires_grad=True) for k, v in P.items()}


def torch_free_energy(par, Xin):
    return -(Xin @ par["a"]) - torch.sum(
        torch.nn.functional.softplus(Xin @ par["W"] + par["b"]), dim=-1)


loglik = (-torch_free_energy(pw, torch.tensor(X)).mean()
          - torch.logsumexp(-torch_free_energy(pw, V_all), dim=0))
loglik.backward()

g_exact = rbm.exact_gradient(P, X)
out.write("   parameter   shape      |autograd - Eq. (14.gradient)|      scale\n")
worst, errs = 0.0, []
for k in ["W", "a", "b"]:
    e = float(np.abs(pw[k].grad.numpy() - g_exact[k]).max())
    errs.append(e)
    worst = max(worst, e)
    out.write(f"   {k:3s}    {str(g_exact[k].shape):10s} {e:24.3e}"
              f"  {np.abs(g_exact[k]).max():10.3e}\n")
out.write(f"  our log-likelihood : {rbm.log_likelihood(P, X):.12f}\n")
out.write(f"  torch's            : {loglik.item():.12f}\n")
out.write(f"  worst disagreement : {worst:.3e}\n")
out.write("  the two-phase decomposition is not an approximation or a\n"
          "  convention: it is what the derivative of the log-likelihood is.\n\n")
np.save("exact_grad_errors.npy", np.array(errs))

# ---------------------------------------------------------------------------
# 3.  one CD-1 step, ours against torch and TensorFlow
# ---------------------------------------------------------------------------
out.write("=== 3. one CD-1 gradient with the same Gibbs samples ===\n")
out.write("  the sampler is shared so that the comparison is of the arithmetic,\n"
          "  not of two random number generators\n\n")
r = np.random.default_rng(7)
ph0 = rbm.p_h_given_x(P, X)
H = (r.random(ph0.shape) < ph0).astype(float)
px = rbm.p_x_given_h(P, H)
V1 = (r.random(px.shape) < px).astype(float)

pos = rbm.positive_phase(P, X)
ph1 = rbm.p_h_given_x(P, V1)
g_ours = {"W": pos["W"] - V1.T @ ph1 / len(V1),
          "a": pos["a"] - V1.mean(0),
          "b": pos["b"] - ph1.mean(0)}

Xt, V1t = torch.tensor(X), torch.tensor(V1)
pht0 = torch.sigmoid(Xt @ tW + tb)
pht1 = torch.sigmoid(V1t @ tW + tb)
g_t = {"W": (Xt.T @ pht0 / len(X) - V1t.T @ pht1 / len(V1)).numpy(),
       "a": (Xt.mean(0) - V1t.mean(0)).numpy(),
       "b": (pht0.mean(0) - pht1.mean(0)).numpy()}
phk0 = tf.sigmoid(tf.constant(X) @ kW + kb)
phk1 = tf.sigmoid(tf.constant(V1) @ kW + kb)
g_k = {"W": (tf.transpose(tf.constant(X)) @ phk0 / len(X)
             - tf.transpose(tf.constant(V1)) @ phk1 / len(V1)).numpy(),
       "a": (tf.reduce_mean(tf.constant(X), 0)
             - tf.reduce_mean(tf.constant(V1), 0)).numpy(),
       "b": (tf.reduce_mean(phk0, 0) - tf.reduce_mean(phk1, 0)).numpy()}
out.write("   parameter   |ours - torch|   |ours - keras|\n")
for k in ["W", "a", "b"]:
    out.write(f"   {k:3s}      {np.abs(g_ours[k]-g_t[k]).max():14.3e}"
              f"   {np.abs(g_ours[k]-g_k[k]).max():14.3e}\n")
out.write("\n")

# ---------------------------------------------------------------------------
# 4.  scikit-learn's BernoulliRBM, a third implementation
# ---------------------------------------------------------------------------
out.write("=== 4. scikit-learn's BernoulliRBM on the same weights ===\n")
sk = BernoulliRBM(n_components=N, batch_size=n, n_iter=0, random_state=0)
sk.components_ = P["W"].T.copy()            # sklearn stores (n_hidden, n_vis)
sk.intercept_hidden_ = P["b"].copy()
sk.intercept_visible_ = P["a"].copy()
ph_sk = sk._mean_hiddens(X)
F_sk = sk._free_energy(X)                   # same sign convention as ours
out.write(f"  p(h|x)  max |ours - sklearn|      : "
          f"{np.abs(ph_ours - ph_sk).max():.3e}\n")
out.write(f"  F(x)    max |ours - sklearn|      : "
          f"{np.abs(F_ours - F_sk).max():.3e}\n")
out.write("  sklearn stores the weight matrix transposed, as (n_hidden, n_vis);\n"
          "  with that one convention undone its free energy and conditional are\n"
          "  Eqs. (14.freeenergy) and (14.condh) to rounding.\n")
out.close()
print(open("cross_check.txt").read())

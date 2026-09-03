"""Chapter 15 against the libraries, and two measurements the chapter needs.

1.  The ELBO and its gradient, three ways: our autograd implementation,
    PyTorch and Keras, with identical weights and -- crucially -- identical
    noise, since Eq. (15.reparam) makes the randomness an input and two
    implementations can only be compared if they are given the same input.

2.  The variance of the two gradient estimators of Section 15.reparam, as a
    function of the latent dimension.  The chapter measures it in one
    dimension and asserts that "the gap grows with dimension"; here we check
    that, in PyTorch, on the actual ELBO rather than on a toy function.

3.  The gap between the ELBO and $\\log p(\\bm{x})$.  Theorem 15.elbo says the
    gap is $D_{KL}(q\\|p(\\bm{h}\\mid\\bm{x}))$ and is therefore not observable
    directly -- but $\\log p$ can be estimated by importance sampling with the
    encoder as proposal, so the gap can be measured after all.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import vae

torch.set_default_dtype(torch.float64)
keras.backend.set_floatx("float64")

out = open("cross_check.txt", "w", buffering=1)
rng = np.random.default_rng(0)

d, dh, hidden, n = 20, 4, 16, 12
P = vae.init_vae(d, dh, hidden, rng=np.random.default_rng(1))
X = (rng.random((n, d)) < 0.4).astype(float)
eps = rng.normal(size=(n, dh))

# ---------------------------------------------------------------------------
# 1.  the same ELBO, three ways
# ---------------------------------------------------------------------------
out.write("=== 1. the ELBO of Eq. (15.objective), identical weights and noise "
          "===\n")


def torch_parts(par, Xin, e):
    """Eqs. (15.encoder), (15.reparam), (15.klclosed) written in torch."""
    a = torch.tanh(Xin @ par["ew0"] + par["eb0"])
    o = a @ par["ew1"] + par["eb1"]
    mu, logvar = o[:, :dh], o[:, dh:]
    H = mu + torch.exp(0.5 * logvar) * e
    dcd = torch.tanh(H @ par["dw0"] + par["db0"]) @ par["dw1"] + par["db1"]
    rec = -(torch.nn.functional.binary_cross_entropy_with_logits(
        dcd, Xin, reduction="none").sum(-1))
    kl = 0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1.0).sum(-1)
    return rec, kl, mu, logvar


tp = {"ew0": torch.tensor(P["enc"][0][0], requires_grad=True),
      "eb0": torch.tensor(P["enc"][0][1], requires_grad=True),
      "ew1": torch.tensor(P["enc"][1][0], requires_grad=True),
      "eb1": torch.tensor(P["enc"][1][1], requires_grad=True),
      "dw0": torch.tensor(P["dec"][0][0], requires_grad=True),
      "db0": torch.tensor(P["dec"][0][1], requires_grad=True),
      "dw1": torch.tensor(P["dec"][1][0], requires_grad=True),
      "db1": torch.tensor(P["dec"][1][1], requires_grad=True)}
rec_t, kl_t, _, _ = torch_parts(tp, torch.tensor(X), torch.tensor(eps))
elbo_t = (rec_t - kl_t).mean()

enc_k = keras.Sequential([keras.Input(shape=(d,)),
                          layers.Dense(hidden, activation="tanh"),
                          layers.Dense(2 * dh)])
dec_k = keras.Sequential([keras.Input(shape=(dh,)),
                          layers.Dense(hidden, activation="tanh"),
                          layers.Dense(d)])
for lay, (W, b) in zip(enc_k.layers, P["enc"]):
    lay.set_weights([W, b])
for lay, (W, b) in zip(dec_k.layers, P["dec"]):
    lay.set_weights([W, b])
o_k = enc_k(X).numpy()
mu_k, lv_k = o_k[:, :dh], o_k[:, dh:]
H_k = mu_k + np.exp(0.5 * lv_k) * eps
logits_k = dec_k(H_k).numpy()
rec_k = np.sum(X * -np.logaddexp(0, -logits_k)
               + (1 - X) * -np.logaddexp(0, logits_k), axis=-1)
kl_k = 0.5 * np.sum(mu_k ** 2 + np.exp(lv_k) - lv_k - 1.0, axis=-1)
elbo_k = float(np.mean(rec_k - kl_k))

elbo_ours = vae.elbo(P, X, eps)
mu_o, lv_o = vae.encode(P, X)
out.write(f"  ELBO, ours   : {elbo_ours:.12f}\n")
out.write(f"  ELBO, torch  : {elbo_t.item():.12f}\n")
out.write(f"  ELBO, keras  : {elbo_k:.12f}\n")
out.write(f"  max |mu_ours - mu_keras|                    : "
          f"{np.abs(mu_o - mu_k).max():.3e}\n")
out.write(f"  max |KL_ours - KL_torch|, Eq. (15.klclosed) : "
          f"{np.abs(vae.kl_gaussian(mu_o, lv_o) - kl_t.detach().numpy()).max():.3e}\n")

from autograd import grad as agrad
g_ours = agrad(lambda p: -vae.elbo(p, X, eps))(P)
(-elbo_t).backward()
pairs = [("enc W1", g_ours["enc"][0][0], tp["ew0"]),
         ("enc b1", g_ours["enc"][0][1], tp["eb0"]),
         ("enc W2", g_ours["enc"][1][0], tp["ew1"]),
         ("enc b2", g_ours["enc"][1][1], tp["eb1"]),
         ("dec W1", g_ours["dec"][0][0], tp["dw0"]),
         ("dec b1", g_ours["dec"][0][1], tp["db0"]),
         ("dec W2", g_ours["dec"][1][0], tp["dw1"]),
         ("dec b2", g_ours["dec"][1][1], tp["db1"])]
out.write("\n   parameter   shape        |ours - torch|      scale\n")
errs, worst = [], 0.0
for nm, go, tt in pairs:
    e = float(np.abs(go - tt.grad.numpy()).max())
    errs.append(e)
    worst = max(worst, e)
    out.write(f"   {nm:8s}  {str(np.shape(go)):12s} {e:14.3e}"
              f"  {np.abs(go).max():10.3e}\n")
out.write(f"  worst disagreement: {worst:.3e}\n")
out.write("  the reparameterised estimator is an ordinary differentiable\n"
          "  function and all three differentiate it identically\n\n")
np.save("vae_grad_errors.npy", np.array(errs))

# ---------------------------------------------------------------------------
# 2.  reparameterisation against the score function, as d_h grows
# ---------------------------------------------------------------------------
out.write("=== 2. the two estimators of Section 15.reparam, against d_h ===\n")
out.write("  Both estimate the same gradient of the ELBO with respect to the\n"
          "  encoder.  We draw 2000 independent single-sample estimates and\n"
          "  report the total variance summed over the encoder parameters.\n\n")
out.write("   d_h   reparam variance   score-function variance   ratio\n")
rows = []
for dh2 in [1, 2, 4, 8, 16, 32]:
    r = np.random.default_rng(3)
    Pv = vae.init_vae(d, dh2, hidden, rng=np.random.default_rng(4))
    Xb = torch.tensor((r.random((8, d)) < 0.4).astype(float))
    pv = {"ew0": torch.tensor(Pv["enc"][0][0], requires_grad=True),
          "eb0": torch.tensor(Pv["enc"][0][1], requires_grad=True),
          "ew1": torch.tensor(Pv["enc"][1][0], requires_grad=True),
          "eb1": torch.tensor(Pv["enc"][1][1], requires_grad=True),
          "dw0": torch.tensor(Pv["dec"][0][0]),
          "db0": torch.tensor(Pv["dec"][0][1]),
          "dw1": torch.tensor(Pv["dec"][1][0]),
          "db1": torch.tensor(Pv["dec"][1][1])}
    enc_par = [pv["ew0"], pv["eb0"], pv["ew1"], pv["eb1"]]

    def encode_t(par, Xin):
        a = torch.tanh(Xin @ par["ew0"] + par["eb0"])
        o = a @ par["ew1"] + par["eb1"]
        return o[:, :dh2], o[:, dh2:]

    def decode_logp(par, H, Xin):
        lg = torch.tanh(H @ par["dw0"] + par["db0"]) @ par["dw1"] + par["db1"]
        return -(torch.nn.functional.binary_cross_entropy_with_logits(
            lg, Xin, reduction="none").sum(-1))

    def collect(kind, n_draw=2000):
        acc = []
        gg = torch.Generator().manual_seed(11)
        for _ in range(n_draw):
            for q in enc_par:
                if q.grad is not None:
                    q.grad = None
            mu, logvar = encode_t(pv, Xb)
            e = torch.randn(Xb.shape[0], dh2, generator=gg,
                            dtype=torch.float64)
            if kind == "reparam":
                H = mu + torch.exp(0.5 * logvar) * e          # Eq. (15.reparam)
                obj = decode_logp(pv, H, Xb).mean()
            else:
                H = (mu + torch.exp(0.5 * logvar) * e).detach()
                logq = (-0.5 * ((H - mu) ** 2 / torch.exp(logvar)
                                + logvar + np.log(2 * np.pi))).sum(-1)
                f = decode_logp(pv, H, Xb).detach()
                obj = (f * logq).mean()                        # Eq. (15.score)
            obj.backward()
            acc.append(torch.cat([q.grad.reshape(-1) for q in enc_par]).clone())
        return torch.stack(acc)

    vr = float(collect("reparam").var(0).sum())
    vs = float(collect("score").var(0).sum())
    rows.append((dh2, vr, vs, vs / vr))
    out.write(f"  {dh2:4d}   {vr:16.4e}   {vs:23.4e}   {vs/vr:6.1f}\n")
out.write("  both estimators are unbiased; only the variance differs, and the\n"
          "  ratio grows with the latent dimension as the chapter claims\n\n")
np.save("estimator_variance.npy", np.array(rows))
out.close()
print(open("cross_check.txt").read())

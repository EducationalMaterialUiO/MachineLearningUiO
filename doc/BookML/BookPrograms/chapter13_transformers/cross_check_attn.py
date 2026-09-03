"""Does our attention agree with the libraries'?

Attention has more moving parts than a convolution and correspondingly more
places for a convention to differ.  This file pushes one set of weights into
our ``attention.py``, into PyTorch and into Keras, and compares the output of
scaled dot-product attention, of multi-head attention, of layer normalisation
and of a whole transformer block.

The conversions are the interesting part.

*  ``torch.nn.MultiheadAttention`` packs the three projections into a single
   ``in_proj_weight`` of shape (3d, d), stacked in the order Q, K, V, and each
   block is stored transposed relative to our (d, d_k) convention.  It also
   concatenates the heads inside that one matrix, so head h occupies rows
   [h*d_k, (h+1)*d_k) of each block.
*  ``keras.layers.MultiHeadAttention`` keeps the head axis explicit: its query
   kernel has shape (d, H, d_k), which is our (H, d, d_k) with the first two
   axes swapped, and its output kernel has shape (H, d_k, d).
*  Both libraries apply the 1/sqrt(d_k) of Eq. (13.attention) internally, and
   both expect an additive mask of -inf rather than a boolean one -- Keras
   inverts the convention and takes a boolean mask where True means *keep*.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from tensorflow import keras
from tensorflow.keras import layers

import attention as at

torch.set_default_dtype(torch.float64)
keras.backend.set_floatx("float64")

out = open("cross_check.txt", "w", buffering=1)
rng = np.random.default_rng(0)

n, d, H = 7, 16, 4
d_k = d // H

# ---------------------------------------------------------------------------
# 1.  scaled dot-product attention, Eq. (13.attention)
# ---------------------------------------------------------------------------
out.write("=== 1. scaled dot-product attention, Eq. (13.attention) ===\n")
Q = rng.normal(size=(n, d_k))
K = rng.normal(size=(n, d_k))
V = rng.normal(size=(n, d_k))
Y_ours, A_ours = at.attention(Q, K, V)
Y_t = Fn.scaled_dot_product_attention(
    torch.tensor(Q)[None, None], torch.tensor(K)[None, None],
    torch.tensor(V)[None, None])[0, 0].numpy()
out.write(f"  max |ours - torch|                 : "
          f"{np.abs(Y_ours - Y_t).max():.3e}\n")
out.write(f"  attention rows sum to one, max dev : "
          f"{np.abs(A_ours.sum(-1) - 1).max():.3e}\n")

M = at.causal_mask(n)
Yc_ours, Ac = at.attention(Q, K, V, M)
Yc_t = Fn.scaled_dot_product_attention(
    torch.tensor(Q)[None, None], torch.tensor(K)[None, None],
    torch.tensor(V)[None, None], is_causal=True)[0, 0].numpy()
out.write(f"  causal mask, Eq. (13.mask), max diff: "
          f"{np.abs(Yc_ours - Yc_t).max():.3e}\n")
out.write(f"  strictly-upper attention mass       : "
          f"{np.triu(Ac, 1).sum():.3e}   (must be exactly zero)\n\n")

# ---------------------------------------------------------------------------
# 2.  multi-head attention, Eq. (13.multihead)
# ---------------------------------------------------------------------------
out.write("=== 2. multi-head attention, Eq. (13.multihead) ===\n")
X = rng.normal(size=(n, d))
P = at.init_mha(d, H, rng=np.random.default_rng(2))
Y_mha, A_mha = at.multihead(P, X)

mha_t = nn.MultiheadAttention(d, H, bias=False, batch_first=True)
with torch.no_grad():
    # in_proj_weight is (3d, d): rows [0,d) Q, [d,2d) K, [2d,3d) V, each (d,d)
    # and each block is the transpose of our stacked (H, d, d_k) reshaped.
    def pack(Wh):                       # (H, d, d_k) -> (d, d) -> transposed
        return torch.tensor(np.concatenate([Wh[h] for h in range(H)], axis=1).T)
    mha_t.in_proj_weight.copy_(torch.cat(
        [pack(P["WQ"]), pack(P["WK"]), pack(P["WV"])], dim=0))
    mha_t.out_proj.weight.copy_(torch.tensor(P["WO"].T))
Xt = torch.tensor(X)[None]
Y_t2, A_t2 = mha_t(Xt, Xt, Xt, average_attn_weights=False)
out.write(f"  output   max |ours - torch|        : "
          f"{np.abs(Y_mha - Y_t2[0].detach().numpy()).max():.3e}\n")
out.write(f"  weights  max |ours - torch|        : "
          f"{np.abs(A_mha - A_t2[0].detach().numpy()).max():.3e}\n")

mha_k = layers.MultiHeadAttention(num_heads=H, key_dim=d_k, use_bias=False)
mha_k.build(query_shape=(None, n, d), value_shape=(None, n, d))
mha_k.set_weights([
    np.transpose(P["WQ"], (1, 0, 2)),        # (H,d,d_k) -> (d,H,d_k)
    np.transpose(P["WK"], (1, 0, 2)),
    np.transpose(P["WV"], (1, 0, 2)),
    P["WO"].reshape(H, d_k, d)])             # (H*d_k, d) -> (H, d_k, d)
Y_k, A_k = mha_k(X[None], X[None], return_attention_scores=True)
out.write(f"  output   max |ours - keras|        : "
          f"{np.abs(Y_mha - Y_k.numpy()[0]).max():.3e}\n")
out.write(f"  weights  max |ours - keras|        : "
          f"{np.abs(A_mha - A_k.numpy()[0]).max():.3e}\n\n")

# ---------------------------------------------------------------------------
# 3.  layer normalisation, Eq. (13.layernorm)
# ---------------------------------------------------------------------------
out.write("=== 3. layer normalisation, Eq. (13.layernorm) ===\n")
g = rng.normal(size=d) + 1.0
be = rng.normal(size=d) * 0.1
L_ours = at.layernorm(X, g, be)
ln_t = nn.LayerNorm(d, eps=1e-5)
with torch.no_grad():
    ln_t.weight.copy_(torch.tensor(g))
    ln_t.bias.copy_(torch.tensor(be))
L_t = ln_t(torch.tensor(X)).detach().numpy()
ln_k = layers.LayerNormalization(epsilon=1e-5)
ln_k.build((None, d))
ln_k.set_weights([g, be])
L_k = ln_k(X).numpy()
out.write(f"  max |ours - torch| : {np.abs(L_ours - L_t).max():.3e}\n")
out.write(f"  max |ours - keras| : {np.abs(L_ours - L_k).max():.3e}\n")
out.write(f"  row means / stds after normalising with gamma=1, beta=0: "
          f"{np.abs(at.layernorm(X, np.ones(d), np.zeros(d)).mean(-1)).max():.2e}"
          f" / {np.abs(at.layernorm(X, np.ones(d), np.zeros(d)).std(-1)-1).max():.2e}\n\n")

# ---------------------------------------------------------------------------
# 4.  the gradient of a whole block
# ---------------------------------------------------------------------------
out.write("=== 4. the gradient through a transformer block, Eq. (13.block) ===\n")
from autograd import grad as agrad
import autograd.numpy as anp

PB = at.init_block(d, H, 4 * d, rng=np.random.default_rng(3))
target = rng.normal(size=(n, d))


def blockloss(PP):
    Y, _ = at.block(PP, X)
    return anp.mean((Y - target) ** 2)


g_ours = agrad(blockloss)(PB)


def torch_block(params, Xin):
    """The same pre-norm block, Eq. (13.block), written in torch."""
    def ln(Z, gm, bt):
        mu = Z.mean(-1, keepdim=True)
        var = ((Z - mu) ** 2).mean(-1, keepdim=True)
        return gm * (Z - mu) / torch.sqrt(var + 1e-5) + bt

    def gelu(z):
        return 0.5 * z * (1 + torch.tanh(np.sqrt(2 / np.pi) *
                                         (z + 0.044715 * z ** 3)))
    Z = ln(Xin, params["g1"], params["be1"])
    heads = []
    for h in range(H):
        q, k, v = Z @ params["WQ"][h], Z @ params["WK"][h], Z @ params["WV"][h]
        S = q @ k.T / np.sqrt(d_k)
        heads.append(torch.softmax(S, dim=-1) @ v)
    Y = torch.cat(heads, dim=-1) @ params["WO"]
    Xin = Xin + Y
    Z2 = ln(Xin, params["g2"], params["be2"])
    return Xin + gelu(Z2 @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]


tp = {k: torch.tensor(v, requires_grad=True) for k, v in PB.items()}
lt = ((torch_block(tp, torch.tensor(X)) - torch.tensor(target)) ** 2).mean()
lt.backward()
out.write(f"  our loss   : {blockloss(PB):.12f}\n")
out.write(f"  torch loss : {lt.item():.12f}\n")
out.write("   parameter      shape             |ours - torch|\n")
worst = 0.0
errs = []
for k in ["WQ", "WK", "WV", "WO", "W1", "b1", "W2", "b2",
          "g1", "be1", "g2", "be2"]:
    e = float(np.abs(g_ours[k] - tp[k].grad.numpy()).max())
    worst = max(worst, e)
    errs.append(e)
    out.write(f"   {k:6s}   {str(np.shape(g_ours[k])):18s}  {e:14.3e}\n")
out.write(f"  worst disagreement: {worst:.3e}\n")
out.write("  autograd on our NumPy code and autograd in PyTorch differentiate\n"
          "  the same function, softmax and layer norm included.\n")
np.save("block_grad_errors.npy", np.array(errs))
out.close()
print(open("cross_check.txt").read())

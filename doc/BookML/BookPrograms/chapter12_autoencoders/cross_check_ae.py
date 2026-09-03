"""Do our autoencoder and the two libraries compute the same thing?

One set of weights is pushed into the network of Section 12.aeown, into a
PyTorch ``nn.Sequential`` and into a Keras ``Sequential``, and the
reconstruction and every gradient are compared.  The only conversion needed is
that PyTorch stores a dense weight as (out, in) while Keras and our code store
it as (in, out).

The experiments that test Theorem 12.aepca itself are in ``ae_torch.py`` and
``ae_tf.py``; they must be separate processes, because importing both
frameworks and then training in one of them is not reliable.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn
from tensorflow import keras
from tensorflow.keras import layers

import ae

torch.set_default_dtype(torch.float64)
keras.backend.set_floatx("float64")

out = open("cross_check.txt", "w", buffering=1)
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1.  the same network, three ways
# ---------------------------------------------------------------------------
out.write("=== 1. one autoencoder, identical weights ===\n")
d, h, p, N = 12, 8, 3, 16
sizes, acts = [d, h, p, h, d], ["tanh", "tanh", "tanh", "identity"]
X = rng.normal(size=(N, d))
P = ae.init_ae(sizes, acts, np.random.default_rng(1))

net_t = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, p), nn.Tanh(),
                      nn.Linear(p, h), nn.Tanh(), nn.Linear(h, d))
lin_t = [m for m in net_t if isinstance(m, nn.Linear)]
with torch.no_grad():
    for m, (W, b) in zip(lin_t, P):
        m.weight.copy_(torch.tensor(W.T))     # torch stores (out, in)
        m.bias.copy_(torch.tensor(b))

net_k = keras.Sequential([keras.Input(shape=(d,)),
                          layers.Dense(h, activation="tanh"),
                          layers.Dense(p, activation="tanh"),
                          layers.Dense(h, activation="tanh"),
                          layers.Dense(d)])
for lay, (W, b) in zip(net_k.layers, P):
    lay.set_weights([W, b])                   # keras stores (in, out), as we do

Xhat, cache = ae.ae_forward(P, X, acts)
Xt = net_t(torch.tensor(X))
Xk = net_k(X).numpy()
out.write(f"  reconstruction max |ours - torch| : "
          f"{np.abs(Xhat - Xt.detach().numpy()).max():.3e}\n")
out.write(f"  reconstruction max |ours - keras| : {np.abs(Xhat - Xk).max():.3e}\n")
out.write(f"  our cost, Eq. (12.cost)           : {ae.cost(Xhat, X):.12f}\n")

loss_t = 0.5 * torch.sum((Xt - torch.tensor(X)) ** 2, dim=1).mean()
net_t.zero_grad()
loss_t.backward()
out.write(f"  torch cost                        : {loss_t.item():.12f}\n")

g_ours = ae.ae_backward(P, cache, Xhat, X, acts)
errs = []
out.write("   layer      shape          |ours - torch| W    |ours - torch| b\n")
for i, (m, (gW, gb)) in enumerate(zip(lin_t, g_ours)):
    eW = np.abs(gW - m.weight.grad.numpy().T).max()
    eb = np.abs(gb - m.bias.grad.numpy()).max()
    errs.append(max(eW, eb))
    out.write(f"   {i}   {str(gW.shape):14s}  {eW:18.3e}  {eb:18.3e}\n")
out.write(f"  worst disagreement: {max(errs):.3e}\n\n")
np.save("ae_grad_errors.npy", np.array(errs))


out.close()
print(open("cross_check.txt").read())

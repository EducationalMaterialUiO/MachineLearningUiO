"""The autoencoder experiments of Chapter 12, in TensorFlow.

The counterpart of ``ae_torch.py``: Theorem 12.aepca tested on the same data
with the same optimiser, and the same MNIST bottleneck sweep.  Keras stores a
dense weight as (in, out), which is our convention, so no transposition is
needed anywhere in this file -- unlike the PyTorch version.
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from mnist_data import load_mnist

out = open("keras.txt", "w", buffering=1)
SEED = 0


def principal_angle(A, B):
    """Largest principal angle, in degrees, between two column spaces."""
    Qa, Qb = np.linalg.qr(A)[0], np.linalg.qr(B)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


# ===========================================================================
# 1.  a linear autoencoder finds the principal subspace
# ===========================================================================
out.write("=== 1. Theorem 12.aepca: does Adam find the principal subspace? ===\n")
d, p, n = 8, 3, 4000
r = np.random.default_rng(5)
S = r.normal(size=(n, p)) @ r.normal(size=(p, d)) + 0.25 * r.normal(size=(n, d))
S = S - S.mean(0)
sv = np.linalg.svd(S, compute_uv=False)
V_pca = np.linalg.svd(S, full_matrices=False)[2][:p].T
floor = float(np.sum(sv[p:] ** 2) / n / 2)

keras.utils.set_random_seed(SEED)
m = keras.Sequential([keras.Input(shape=(d,)),
                      layers.Dense(p, use_bias=False),
                      layers.Dense(d, use_bias=False)])
m.compile(optimizer=keras.optimizers.Adam(5e-3), loss="mse")
hist = []
for blk in range(1, 61):
    m.fit(S, S, epochs=100, batch_size=len(S), verbose=0)
    W = m.layers[0].get_weights()[0]
    hist.append((blk * 100, m.evaluate(S, S, verbose=0) * d / 2,
                 principal_angle(W, V_pca)))
W = m.layers[0].get_weights()[0]

out.write(f"  largest principal angle to the PCA subspace : "
          f"{principal_angle(W, V_pca):.4f} degrees\n")
out.write(f"  max |W_enc - V_pca| entrywise               : "
          f"{np.abs(W - V_pca).max():.4f}\n")
out.write(f"  reconstruction cost per sample              : {hist[-1][1]:.6f}\n")
out.write(f"  Eckart-Young floor, Eq. (12.floor)          : {floor:.6f}\n")
Pi = W @ np.linalg.pinv(W)
out.write(f"  ||Pi_enc - Pi_pca||_F                       : "
          f"{np.linalg.norm(Pi - V_pca @ V_pca.T):.3e}\n\n")
np.save("angle_hist_keras.npy", np.array(hist))


# ===========================================================================
# 2.  what the bottleneck costs on MNIST
# ===========================================================================
out.write("=== 2. reconstruction error against the code dimension, MNIST ===\n")
(xtr, ytr), (xte, yte) = load_mnist(flat=True)
EPOCHS = int(os.environ.get("CH12_EPOCHS", 8))
Xc = xtr - xtr.mean(0)
sv_m = np.linalg.svd(Xc, compute_uv=False)
PS = [2, 8, 16, 32, 64]
pca_mse = [float(np.sum(sv_m[k:] ** 2) / (len(xtr) * 784)) for k in PS]


def build(p, nonlinear=True):
    if nonlinear:
        return keras.Sequential([keras.Input(shape=(784,)),
                                 layers.Dense(256, activation="relu"),
                                 layers.Dense(p, activation="relu"),
                                 layers.Dense(256, activation="relu"),
                                 layers.Dense(784, activation="sigmoid")])
    return keras.Sequential([keras.Input(shape=(784,)),
                             layers.Dense(p, use_bias=False),
                             layers.Dense(784, use_bias=False)])


out.write("     p    PCA (linear optimum)   linear AE      nonlinear AE   gain\n")
rows = []
for k_, pm in zip(PS, pca_mse):
    t0 = time.time()
    res = []
    for nl in [False, True]:
        keras.utils.set_random_seed(SEED)
        mm = build(k_, nl)
        mm.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        mm.fit(xtr, xtr, epochs=EPOCHS, batch_size=128, verbose=0)
        res.append(float(mm.evaluate(xte, xte, verbose=0)))
    rows.append((k_, pm, res[0], res[1]))
    out.write(f"  {k_:4d}   {pm:20.5f}   {res[0]:9.5f}   {res[1]:15.5f}"
              f"   {pm/res[1]:5.2f}x   ({time.time()-t0:.0f}s)\n")
np.save("mnist_curve_keras.npy", np.array(rows))
out.close()
print(open("keras.txt").read())

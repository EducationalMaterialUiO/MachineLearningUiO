"""A restricted Boltzmann machine on MNIST, in TensorFlow.

The exact counterpart of ``rbm_torch.py``: the free energy of
Eq. (14.freeenergy), block Gibbs sampling from Eqs. (14.condh) and (14.condx),
and the contrastive-divergence update of Eq. (14.cdk), written in TensorFlow
tensors with the same learning rate, momentum, batch size, binarisation and
number of epochs.  Neither framework ships a Boltzmann machine, so the two
files differ only in which tensor library spells the same equations.
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf

from mnist_data import load_mnist

out = open("keras.txt", "w", buffering=1)
SEED = 0
EPOCHS = int(os.environ.get("CH14_EPOCHS", 5))
M, N, BATCH = 784, 128, 64

(xtr, _), (xte, _) = load_mnist(flat=True)
rng = np.random.default_rng(123)
Xtr = tf.constant((rng.random(xtr.shape) < xtr).astype("float32"))
Xte = tf.constant((rng.random(xte.shape) < xte).astype("float32"))
out.write(f"MNIST binarised: {Xtr.shape[0]} train, {Xte.shape[0]} test, "
          f"M = {M} visible, N = {N} hidden, {EPOCHS} epochs\n\n")


def free_energy(W, a, b, X):
    """Eq. (14.freeenergy)."""
    return -tf.linalg.matvec(X, a) - tf.reduce_sum(
        tf.math.softplus(X @ W + b), axis=-1)


def pseudo_likelihood(W, a, b, X, seed):
    r = np.random.default_rng(seed)
    i = r.integers(0, M, X.shape[0])
    Xn = X.numpy().copy()
    Xn[np.arange(len(Xn)), i] = 1.0 - Xn[np.arange(len(Xn)), i]
    return float(tf.reduce_mean(M * tf.math.log_sigmoid(
        free_energy(W, a, b, tf.constant(Xn)) - free_energy(W, a, b, X))))


def gibbs(W, a, b, V, k, gen):
    for _ in range(k):
        ph = tf.sigmoid(V @ W + b)
        H = tf.cast(gen.uniform(tf.shape(ph)) < ph, tf.float32)
        pv = tf.sigmoid(H @ tf.transpose(W) + a)
        V = tf.cast(gen.uniform(tf.shape(pv)) < pv, tf.float32)
    return V


def train(mode, k, eta=0.05, momentum=0.5):
    gen = tf.random.Generator.from_seed(SEED)
    W = tf.Variable(0.01 * gen.normal((M, N)))
    a = tf.Variable(tf.zeros(M))
    b = tf.Variable(tf.zeros(N))
    vW = tf.Variable(tf.zeros((M, N)))
    va = tf.Variable(tf.zeros(M))
    vb = tf.Variable(tf.zeros(N))
    chain = tf.cast(gen.uniform((BATCH, M)) < 0.5, tf.float32)
    hist, t0 = [], time.time()
    n = Xtr.shape[0]
    for ep in range(1, EPOCHS + 1):
        perm = np.random.default_rng(SEED + ep).permutation(n)
        for i in range(0, n - BATCH + 1, BATCH):
            X = tf.gather(Xtr, perm[i:i + BATCH])
            ph_pos = tf.sigmoid(X @ W + b)
            start = X if mode == "cd" else chain
            V = gibbs(W, a, b, start, k, gen)
            if mode == "pcd":
                chain = V
            ph_neg = tf.sigmoid(V @ W + b)
            dW = (tf.transpose(X) @ ph_pos / BATCH
                  - tf.transpose(V) @ ph_neg / BATCH)
            da = tf.reduce_mean(X, 0) - tf.reduce_mean(V, 0)
            db = tf.reduce_mean(ph_pos, 0) - tf.reduce_mean(ph_neg, 0)
            vW.assign(momentum * vW + eta * dW)
            va.assign(momentum * va + eta * da)
            vb.assign(momentum * vb + eta * db)
            W.assign_add(vW)
            a.assign_add(va)
            b.assign_add(vb)
        Xs = Xte[:2000]
        pl = pseudo_likelihood(W, a, b, Xs, 99)
        rec = float(tf.reduce_mean((tf.sigmoid(
            tf.sigmoid(Xs @ W + b) @ tf.transpose(W) + a) - Xs) ** 2))
        hist.append((ep, pl, rec, time.time() - t0))
        out.write(f"  {mode}-{k:<2d} epoch {ep}: pseudo-likelihood {pl:9.2f}   "
                  f"reconstruction mse {rec:.5f}   ({hist[-1][3]:.0f}s)\n")
    return np.array(hist)


out.write("=== the three training signals of Section 14.cd, on real data ===\n")
runs = {}
for name, mode, k in [("CD-1", "cd", 1), ("CD-10", "cd", 10),
                      ("PCD-1", "pcd", 1)]:
    runs[name] = train(mode, k)
    out.write("\n")
out.write("  signal   final pseudo-likelihood   final reconstruction mse"
          "   seconds/epoch\n")
for name, h in runs.items():
    out.write(f"  {name:7s} {h[-1,1]:23.2f}   {h[-1,2]:24.5f}"
              f"   {h[-1,3]/EPOCHS:13.0f}\n")
for name, h in runs.items():
    np.save(f"hist_{name.replace('-','').lower()}_keras.npy", h)
out.close()
print(open("keras.txt").read())

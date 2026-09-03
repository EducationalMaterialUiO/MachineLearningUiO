"""The variational autoencoder experiments of Chapter 15, in TensorFlow.

The counterpart of ``vae_torch.py``: the same architecture, the same
binarisation, the same optimiser and the same number of epochs, with the
reparameterisation of Eq. (15.reparam) written out as a layer so that it can be
read against the equation.  The latent-dimension sweep and the gap of
Theorem 15.elbo are computed the same way.
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from mnist_data import load_mnist

out = open("keras.txt", "w", buffering=1)
SEED = 0
EPOCHS = int(os.environ.get("CH15_EPOCHS", 6))
D, HID, BATCH = 784, 256, 128

(xtr, _), (xte, _) = load_mnist(flat=True)
r = np.random.default_rng(123)
Xtr = (r.random(xtr.shape) < xtr).astype("float32")
Xte = (r.random(xte.shape) < xte).astype("float32")
out.write(f"MNIST binarised: {len(Xtr)} train, {len(Xte)} test, d = {D}, "
          f"{EPOCHS} epochs, Adam 1e-3, batch {BATCH}\n\n")


class VAE(keras.Model):
    """Eqs. (15.encoder), (15.reparam) and (15.objective)."""

    def __init__(self, dh):
        super().__init__()
        self.dh = dh
        self.enc = keras.Sequential([keras.Input(shape=(D,)),
                                     layers.Dense(HID, activation="relu"),
                                     layers.Dense(2 * dh)])
        self.dec = keras.Sequential([keras.Input(shape=(dh,)),
                                     layers.Dense(HID, activation="relu"),
                                     layers.Dense(D)])

    def encode(self, x):
        o = self.enc(x)
        return o[:, :self.dh], o[:, self.dh:]

    def parts(self, x):
        mu, logvar = self.encode(x)
        eps = tf.random.normal(tf.shape(mu))
        h = mu + tf.exp(0.5 * logvar) * eps          # Eq. (15.reparam)
        logits = self.dec(h)
        rec = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=x, logits=logits), axis=-1)
        kl = 0.5 * (tf.square(mu) + tf.exp(logvar) - logvar - 1.0)
        return rec, kl


def iwae_bound(m, X, K, chunk=100, kblock=32):
    """Eq. (15.iwae), in blocks so that memory stays bounded."""
    tot = []
    for i in range(0, len(X), chunk):
        x = tf.constant(X[i:i + chunk])
        mu0, lv0 = m.encode(x)
        lws = []
        done = 0
        while done < K:
            kb = min(kblock, K - done)
            mu = tf.tile(mu0[None], [kb, 1, 1])
            lv = tf.tile(lv0[None], [kb, 1, 1])
            e = tf.random.normal(tf.shape(mu))
            h = mu + tf.exp(0.5 * lv) * e
            logits = m.dec(tf.reshape(h, [-1, m.dh]))
            logits = tf.reshape(logits, [kb, -1, D])
            xx = tf.tile(x[None], [kb, 1, 1])
            logpxh = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=xx, logits=logits), axis=-1)
            logq = tf.reduce_sum(-0.5 * (e ** 2 + lv + np.log(2 * np.pi)), -1)
            logp = tf.reduce_sum(-0.5 * (h ** 2 + np.log(2 * np.pi)), -1)
            lws.append(logpxh + logp - logq)
            done += kb
        lw = tf.concat(lws, 0)
        tot.append(tf.reduce_logsumexp(lw, 0) - np.log(K))
    return float(tf.reduce_mean(tf.concat(tot, 0)))


def train(dh, beta=1.0, epochs=EPOCHS):
    keras.utils.set_random_seed(SEED)
    m = VAE(dh)
    opt = keras.optimizers.Adam(1e-3)

    @tf.function
    def step(x):
        with tf.GradientTape() as tape:
            rec, kl = m.parts(x)
            loss = -tf.reduce_mean(rec - beta * tf.reduce_sum(kl, -1))
        opt.apply_gradients(zip(tape.gradient(loss, m.trainable_variables),
                                m.trainable_variables))
        return loss

    t0 = time.time()
    for ep in range(epochs):
        perm = np.random.default_rng(SEED + ep).permutation(len(Xtr))
        for i in range(0, len(Xtr) - BATCH + 1, BATCH):
            step(tf.constant(Xtr[perm[i:i + BATCH]]))
    rec, kl = m.parts(tf.constant(Xte))
    elbo = float(tf.reduce_mean(rec - tf.reduce_sum(kl, -1)))
    return m, elbo, tf.reduce_mean(kl, 0).numpy(), time.time() - t0


out.write("=== 1. the latent dimension ===\n")
out.write("   d_h   test ELBO   active units (KL_j > 0.01)   mean KL per active"
          "   seconds\n")
rows, models = [], {}
for dh in [2, 5, 10, 20, 50]:
    m, elbo, kl_j, dt = train(dh)
    act = int((kl_j > 0.01).sum())
    rows.append((dh, elbo, act, float(kl_j[kl_j > 0.01].mean())))
    models[dh] = m
    out.write(f"  {dh:4d}  {elbo:10.4f}  {act:26d}   {rows[-1][3]:17.4f}"
              f"   {dt:7.0f}\n")
np.save("dh_table_keras.npy", np.array(rows))

out.write("\n=== 2. the ELBO gap, Theorem 15.elbo ===\n")
out.write("   d_h      L_1 (ELBO)        L_16        L_128       L_1024"
          "    L_1024 - L_1\n")
gap = []
Xg = Xte[:500]
for dh in [2, 10, 50]:
    vals = [iwae_bound(models[dh], Xg, K) for K in [1, 16, 128, 1024]]
    gap.append([dh] + vals + [vals[-1] - vals[0]])
    out.write(f"  {dh:4d}  {vals[0]:12.4f} {vals[1]:12.4f} {vals[2]:12.4f}"
              f" {vals[3]:12.4f} {vals[-1]-vals[0]:14.4f}\n")
np.save("gap_table_keras.npy", np.array(gap))
out.close()
print(open("keras.txt").read())

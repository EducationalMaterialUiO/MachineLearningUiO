"""The associative-recall experiment of Chapter 13, in TensorFlow.

The counterpart of experiment 3 in ``transformer_torch.py``: the same task, the
same widths, the same optimiser and the same number of updates, built from
``keras.layers.MultiHeadAttention`` and ``keras.layers.LayerNormalization``
rather than from ``nn.TransformerEncoderLayer``.  The block is written out
rather than assembled, so that it can be compared line by line with
Eq. (13.block).
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

out = open("keras.txt", "w", buffering=1)
NSYM = 8


def make_batch(n_b, L, rng):
    T = 2 * L + 1
    X = np.zeros((n_b, T), dtype="int32")
    y = np.zeros(n_b, dtype="int32")
    for b in range(n_b):
        keys = rng.permutation(NSYM)[:L]
        vals = rng.integers(0, NSYM, L)
        X[b, 0:2 * L:2] = keys
        X[b, 1:2 * L:2] = vals
        q = rng.integers(0, L)
        X[b, 2 * L] = keys[q]
        y[b] = vals[q]
    return X, y


def sinusoidal(n, d):
    """Eq. (13.posenc)."""
    pe = np.zeros((n, d), dtype="float32")
    for pos in range(n):
        for i in range(d):
            a = pos / 10000 ** (2 * (i // 2) / d)
            pe[pos, i] = np.sin(a) if i % 2 == 0 else np.cos(a)
    return pe


class Block(layers.Layer):
    """Eq. (13.block), pre-norm, written out rather than assembled."""

    def __init__(self, d=32, H=2, d_ff=64, **kw):
        super().__init__(**kw)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.att = layers.MultiHeadAttention(num_heads=H, key_dim=d // H)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential([layers.Dense(d_ff, activation="gelu"),
                                     layers.Dense(d)])

    def call(self, x):
        h = self.ln1(x)
        x = x + self.att(h, h)                 # residual, Section 13.block
        return x + self.mlp(self.ln2(x))


def build(L, n_blocks=1, d=32, H=2, d_ff=64):
    T = 2 * L + 1
    inp = keras.Input(shape=(T,), dtype="int32")
    x = layers.Embedding(NSYM, d)(inp)
    x = x + tf.constant(sinusoidal(T, d))
    for _ in range(n_blocks):
        x = Block(d, H, d_ff)(x)
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    return keras.Model(inp, layers.Dense(NSYM)(x[:, -1]))


out.write("=== associative recall in TensorFlow, Table 13.recall ===\n")
out.write("    L    T   model                  params   test accuracy (2 seeds)\n")
rows = {}
for L in [2, 4, 8]:
    for n_blocks in [1, 2]:
        accs, npar, t0 = [], 0, time.time()
        for sd in [0, 1]:
            keras.utils.set_random_seed(sd)
            rng = np.random.default_rng(sd)
            m = build(L, n_blocks)
            npar = int(sum(np.prod(w.shape) for w in m.trainable_weights))
            opt = keras.optimizers.Adam(3e-3)
            lf = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            @tf.function
            def step(xb, yb):
                with tf.GradientTape() as tape:
                    loss = lf(yb, m(xb, training=True))
                g = tape.gradient(loss, m.trainable_variables)
                g, _ = tf.clip_by_global_norm(g, 1.0)     # Eq. (11.clip)
                opt.apply_gradients(zip(g, m.trainable_variables))
                return loss

            for it in range(3000):
                Xb, yb = make_batch(64, L, rng)
                step(tf.constant(Xb), tf.constant(yb))
            Xv, yv = make_batch(400, L, np.random.default_rng(99))
            accs.append(float(np.mean(
                np.argmax(m.predict(Xv, verbose=0), 1) == yv)))
        rows[(L, n_blocks)] = float(np.mean(accs))
        out.write(f"  {L:3d}  {2*L+1:3d}   transformer, {n_blocks} block(s)"
                  f"   {npar:7d}   {np.mean(accs):.3f}"
                  f"   ({time.time()-t0:.0f}s)\n")
out.write(f"  chance = {1/NSYM:.3f}\n")
np.save("recall_keras.npy",
        np.array([[rows[(L, nb)] for L in [2, 4, 8]] for nb in [1, 2]]))
out.close()
print(open("keras.txt").read())

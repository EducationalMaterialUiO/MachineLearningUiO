"""Train the network of Section 10.cnnlibraries on MNIST, in TensorFlow.

The exact counterpart of ``cnn_torch.py``: same architecture, same optimiser,
same batch size, same number of epochs and the same seed.  Keras uses the
opposite channel convention, (N, H, W, C), which is the only structural
difference between the two files.
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from mnist_data import load_mnist

EPOCHS = int(os.environ.get("CH10_EPOCHS", 5))
BATCH, SEED = 64, 1

out = open("keras.txt", "w", buffering=1)
(xtr, ytr), (xte, yte) = load_mnist(normalise="standard")
out.write(f"MNIST: {len(xtr)} training and {len(xte)} test images, "
          f"{EPOCHS} epochs, batch {BATCH}, Adam 1e-3, seed {SEED}\n\n")


def build(head="dense"):
    trunk = [keras.Input(shape=(28, 28, 1)),
             layers.Conv2D(32, 3, padding="same", activation="relu"),
             layers.MaxPooling2D((2, 2)),
             layers.Conv2D(64, 3, padding="same", activation="relu"),
             layers.MaxPooling2D((2, 2))]
    if head == "dense":
        tail = [layers.Flatten(), layers.Dense(1024, activation="relu"),
                layers.Dropout(0.5), layers.Dense(10)]
    else:
        tail = [layers.GlobalAveragePooling2D(), layers.Dense(10)]
    m = keras.Sequential(trunk + tail)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    return m


def run(head):
    keras.utils.set_random_seed(SEED)
    m = build(head)
    n_par = int(sum(np.prod(w.shape) for w in m.trainable_weights))
    hist, t0 = [], time.time()
    for ep in range(1, EPOCHS + 1):
        h = m.fit(xtr[..., None], ytr, epochs=1, batch_size=BATCH, verbose=0)
        _, acc = m.evaluate(xte[..., None], yte, verbose=0)
        hist.append((ep, h.history["loss"][0], acc, time.time() - t0))
        out.write(f"  {head:5s} epoch {ep}: loss {hist[-1][1]:.4f}  "
                  f"test accuracy {acc:.4f}  ({hist[-1][3]:.0f}s)\n")
    return np.array(hist), n_par, m


out.write("=== the chapter architecture, dense head ===\n")
h_d, n_d, m_d = run("dense")
out.write("\n=== the same trunk, global average pooling instead ===\n")
h_g, n_g, _ = run("gap")

out.write("\n  head    parameters   final test accuracy   seconds/epoch\n")
for nm, h, n in [("dense", h_d, n_d), ("GAP", h_g, n_g)]:
    out.write(f"  {nm:6s} {n:11d}   {h[-1,2]:19.4f}   {h[-1,3]/EPOCHS:13.0f}\n")

np.save("hist_keras.npy", h_d)
np.save("hist_keras_gap.npy", h_g)
np.save("nparams_keras.npy", np.array([n_d, n_g]))
out.close()
print(open("keras.txt").read())

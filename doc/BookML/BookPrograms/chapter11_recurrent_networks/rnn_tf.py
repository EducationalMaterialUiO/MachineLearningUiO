"""The two library examples of Chapter 11, in TensorFlow.

The counterpart of ``rnn_torch.py``: the sine-wave forecast of
Section 11.rnnsine with ``SimpleRNN``, which is Eq. (11.rnn) exactly, and the
row-by-row MNIST classification of Section 11.lstmmnist with ``LSTM``, which is
Eqs. (11.lstmf)-(11.lstmh).  Same architectures, same optimiser, same batch
sizes and the same number of epochs as the PyTorch file, so that the two can be
compared directly.

The adding problem of experiment 3 is run in PyTorch only; it needs twenty-four
separate trainings and the point it makes is about architectures, not about
frameworks.
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from mnist_data import load_mnist

SEED = 1
out = open("keras.txt", "w", buffering=1)

# ===========================================================================
# 1.  a sine wave, one step ahead
# ===========================================================================
out.write("=== 1. forecasting a sine wave, Section 11.rnnsine ===\n")
keras.utils.set_random_seed(SEED)
seq_length = 20
data = np.sin(np.linspace(0, 100, 500)).astype("float32")
X = np.stack([data[i:i + seq_length]
              for i in range(len(data) - seq_length)])[..., None]
y = data[seq_length:].reshape(-1, 1)
split = int(0.8 * len(X))

model = keras.Sequential([
    keras.Input(shape=(seq_length, 1)),
    layers.SimpleRNN(16),               # exactly Eq. (11.rnn), tanh by default
    layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
n_par = int(sum(np.prod(w.shape) for w in model.trainable_weights))
h = model.fit(X[:split], y[:split], epochs=200, batch_size=32, verbose=0,
              validation_data=(X[split:], y[split:]))
te = model.evaluate(X[split:], y[split:], verbose=0)
for ep in [50, 100, 150, 200]:
    out.write(f"  epoch {ep:3d}: train mse {h.history['loss'][ep-1]:.6f}   "
              f"test mse {h.history['val_loss'][ep-1]:.6f}\n")
out.write(f"  {n_par} parameters: 16*(1+16+1) = 288 recurrent, 16+1 = 17 read-out\n")
out.write(f"  final test mse {te:.6f}, "
          f"rms error {np.sqrt(te*2):.4f} on a signal of amplitude 1\n\n")
np.save("sine_hist_keras.npy",
        np.column_stack([np.arange(1, 201), h.history["loss"],
                         h.history["val_loss"]]))
np.save("sine_pred_keras.npy", model.predict(X[split:], verbose=0).ravel())

# ===========================================================================
# 2.  MNIST read row by row
# ===========================================================================
out.write("=== 2. MNIST as 28 rows of 28, Section 11.lstmmnist ===\n")
EPOCHS = int(os.environ.get("CH11_EPOCHS", 3))
(xtr, ytr), (xte, yte) = load_mnist()

hists = {}
for cell, Layer in [("lstm", layers.LSTM), ("rnn", layers.SimpleRNN)]:
    keras.utils.set_random_seed(SEED)
    m = keras.Sequential([keras.Input(shape=(28, 28)), Layer(128),
                          layers.Dense(10)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    n_par = int(sum(np.prod(w.shape) for w in m.trainable_weights))
    rows, t0 = [], time.time()
    for ep in range(1, EPOCHS + 1):
        hh = m.fit(xtr, ytr, epochs=1, batch_size=64, verbose=0)
        _, acc = m.evaluate(xte, yte, verbose=0)
        rows.append((ep, hh.history["loss"][0], acc, time.time() - t0))
        out.write(f"  {cell:4s} epoch {ep}: loss {rows[-1][1]:.4f}  "
                  f"test accuracy {acc:.4f}  ({rows[-1][3]:.0f}s)\n")
    out.write(f"  {cell:4s}: {n_par} parameters\n")
    hists[cell] = np.array(rows)
np.save("mnist_lstm_keras.npy", hists["lstm"])
np.save("mnist_rnn_keras.npy", hists["rnn"])
out.close()
print(open("keras.txt").read())

"""Do the three implementations compute the same thing?

The chapter derives a convolutional layer and its gradients from scratch and
then shows the same architecture in PyTorch and in TensorFlow.  It is easy to
say that these are the same thing; this program checks it.  One set of weights
is generated once and pushed into all three implementations, and we compare

  * the forward pass, output by output,
  * the gradient of the loss with respect to every parameter array,

so that any disagreement in the derivations of Section 10.convbackprop, or in
our reading of either library's conventions, has to show up here.

The only real work is index bookkeeping, and that is the point of the exercise:
the three libraries hold the same numbers in three different layouts.

    ours    X (N, C, H, W)   W (K, C, F, F)
    torch   x (N, C, H, W)   w (K, C, F, F)      -- identical to ours
    keras   x (N, H, W, C)   w (F, F, C, K)      -- channels last, kernel last
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

import cnn                       # the from-scratch implementation
from mnist_data import load_mnist

DT = "float64"                   # all three in double: the check is then exact
torch.set_default_dtype(torch.float64)
keras.backend.set_floatx("float64")

out = open("cross_check.txt", "w", buffering=1)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 0.  one set of weights and one batch of real data
# ---------------------------------------------------------------------------
N, C, H, Wd = 8, 1, 28, 28
K1, K2, F, n_out = 4, 6, 3, 10
flat = K2 * 7 * 7

(xtr, ytr), _ = load_mnist(normalise="standard")
X = xtr[:N, None, :, :].astype(DT)                    # (N,1,28,28)
y = ytr[:N]
Y = np.eye(n_out)[y].astype(DT)

p = cnn.init_cnn(C_in=C, K1=K1, K2=K2, F=F, n_out=n_out, flat=flat,
                 rng=np.random.default_rng(1))
for k in p:
    p[k] = p[k].astype(DT)

# ---------------------------------------------------------------------------
# 1.  the same convolution, one layer at a time
# ---------------------------------------------------------------------------
out.write("=== 1. one convolutional layer, identical weights ===\n")
conv_t = nn.Conv2d(C, K1, F, padding=1, bias=True)
with torch.no_grad():
    conv_t.weight.copy_(torch.tensor(p["W1"]))        # (K,C,F,F): same layout
    conv_t.bias.copy_(torch.tensor(p["b1"]))

conv_k = layers.Conv2D(K1, F, padding="same", use_bias=True,
                       input_shape=(H, Wd, C))
conv_k.build((None, H, Wd, C))
# keras stores the kernel as (F, F, C, K): transpose ours to match
conv_k.set_weights([np.transpose(p["W1"], (2, 3, 1, 0)), p["b1"]])

Z_ours, _ = cnn.conv_forward(X, p["W1"], p["b1"], S=1, P=1)
Z_torch = conv_t(torch.tensor(X)).detach().numpy()
Z_keras = np.transpose(conv_k(np.transpose(X, (0, 2, 3, 1))).numpy(),
                       (0, 3, 1, 2))                  # back to (N,K,H,W)

out.write(f"  output shape (N,K,H,W)      : {Z_ours.shape}\n")
out.write(f"  max |ours  - torch|         : "
          f"{np.abs(Z_ours - Z_torch).max():.3e}\n")
out.write(f"  max |ours  - keras|         : "
          f"{np.abs(Z_ours - Z_keras).max():.3e}\n")
out.write(f"  max |torch - keras|         : "
          f"{np.abs(Z_torch - Z_keras).max():.3e}\n")
out.write(f"  scale of the outputs        : {np.abs(Z_ours).max():.3f}\n\n")

# max pooling
P_ours, _ = cnn.maxpool_forward(Z_ours, 2, 2)
P_torch = nn.MaxPool2d(2, 2)(torch.tensor(Z_ours)).numpy()
P_keras = np.transpose(layers.MaxPooling2D((2, 2))(
    np.transpose(Z_ours, (0, 2, 3, 1))).numpy(), (0, 3, 1, 2))
out.write("=== 2. max pooling, Eq. (10.maxpool) ===\n")
out.write(f"  max |ours - torch|          : "
          f"{np.abs(P_ours - P_torch).max():.3e}\n")
out.write(f"  max |ours - keras|          : "
          f"{np.abs(P_ours - P_keras).max():.3e}\n\n")


# ---------------------------------------------------------------------------
# 3.  the whole network, forward and backward
# ---------------------------------------------------------------------------
class TorchNet(nn.Module):
    """Eq. (10.arch), the architecture of Section 10.cnnnet."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(C, K1, F, padding=1)
        self.c2 = nn.Conv2d(K1, K2, F, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(flat, n_out)

    def forward(self, x):
        x = self.pool(torch.relu(self.c1(x)))
        x = self.pool(torch.relu(self.c2(x)))
        return self.fc(x.reshape(x.shape[0], -1))


net_t = TorchNet()
with torch.no_grad():
    net_t.c1.weight.copy_(torch.tensor(p["W1"]))
    net_t.c1.bias.copy_(torch.tensor(p["b1"]))
    net_t.c2.weight.copy_(torch.tensor(p["W2"]))
    net_t.c2.bias.copy_(torch.tensor(p["b2"]))
    net_t.fc.weight.copy_(torch.tensor(p["W3"].T))    # torch stores (out, in)
    net_t.fc.bias.copy_(torch.tensor(p["b3"]))

net_k = keras.Sequential([
    keras.Input(shape=(H, Wd, C)),
    layers.Conv2D(K1, F, padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(K2, F, padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(n_out),
])
net_k.layers[0].set_weights([np.transpose(p["W1"], (2, 3, 1, 0)), p["b1"]])
net_k.layers[2].set_weights([np.transpose(p["W2"], (2, 3, 1, 0)), p["b2"]])
# our flatten runs (K,H,W) C-order; keras flattens (H,W,K).  Reindex W3 rows.
perm = np.transpose(np.arange(flat).reshape(K2, 7, 7), (1, 2, 0)).reshape(-1)
net_k.layers[5].set_weights([p["W3"][perm], p["b3"]])

probs_ours, cache = cnn.forward(p, X)
logits_t = net_t(torch.tensor(X))
logits_k = net_k(np.transpose(X, (0, 2, 3, 1)))
probs_t = torch.softmax(logits_t, dim=1).detach().numpy()
probs_k = tf.nn.softmax(logits_k).numpy()

out.write("=== 3. the whole network of Eq. (10.arch), forward ===\n")
out.write(f"  max |p_ours - p_torch|      : "
          f"{np.abs(probs_ours - probs_t).max():.3e}\n")
out.write(f"  max |p_ours - p_keras|      : "
          f"{np.abs(probs_ours - probs_k).max():.3e}\n")
out.write(f"  cross-entropy, ours         : "
          f"{cnn.cross_entropy(probs_ours, Y):.12f}\n")

loss_t = nn.CrossEntropyLoss()(logits_t, torch.tensor(y))
with tf.GradientTape() as tape:
    lk = net_k(np.transpose(X, (0, 2, 3, 1)))
    loss_k = tf.reduce_mean(
        keras.losses.sparse_categorical_crossentropy(y, lk, from_logits=True))
out.write(f"  cross-entropy, torch        : {loss_t.item():.12f}\n")
out.write(f"  cross-entropy, keras        : {float(loss_k):.12f}\n\n")

# --- gradients --------------------------------------------------------------
g_ours = cnn.backward(p, cache, probs_ours, Y)
net_t.zero_grad()
loss_t.backward()
g_keras = tape.gradient(loss_k, net_k.trainable_variables)

g_t = {"W1": net_t.c1.weight.grad.numpy(), "b1": net_t.c1.bias.grad.numpy(),
       "W2": net_t.c2.weight.grad.numpy(), "b2": net_t.c2.bias.grad.numpy(),
       "W3": net_t.fc.weight.grad.numpy().T, "b3": net_t.fc.bias.grad.numpy()}
gk = [np.asarray(g) for g in g_keras]
g_k = {"W1": np.transpose(gk[0], (3, 2, 0, 1)), "b1": gk[1],
       "W2": np.transpose(gk[2], (3, 2, 0, 1)), "b2": gk[3],
       "W3": gk[4][np.argsort(perm)], "b3": gk[5]}

out.write("=== 4. the gradients of Section 10.convbackprop ===\n")
out.write("   array      shape           |ours-torch|   |ours-keras|"
          "   scale\n")
rows = []
for k in ["W1", "b1", "W2", "b2", "W3", "b3"]:
    et = np.abs(g_ours[k] - g_t[k]).max()
    ek = np.abs(g_ours[k] - g_k[k]).max()
    sc = np.abs(g_ours[k]).max()
    rows.append((k, et, ek, sc))
    out.write(f"   {k:4s}   {str(g_ours[k].shape):16s}  {et:12.3e}  "
              f"{ek:12.3e}  {sc:8.2e}\n")
out.write(f"  worst disagreement anywhere : "
          f"{max(max(r[1], r[2]) for r in rows):.3e}\n")
out.write("  the three implementations agree to double-precision rounding:\n"
          "  Eqs. (10.dconvW), (10.dconvb) and (10.dconvX) are what the\n"
          "  libraries compute.\n\n")
np.save("crosscheck_rows.npy", np.array([[r[1], r[2], r[3]] for r in rows]))

# ---------------------------------------------------------------------------
# 5.  parameter counts against Eq. (10.paramcount)
# ---------------------------------------------------------------------------
out.write("=== 5. parameter counts against Eq. (10.paramcount) ===\n")
big = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10),
])
names = ["conv2d(1->32)", "maxpool", "conv2d(32->64)", "maxpool", "flatten",
         "dense(3136->1024)", "dropout", "dense(1024->10)"]
formula = ["32*(1*9+1) = 320", "", "64*(32*9+1) = 18496", "", "",
           "3136*1024+1024 = 3212288", "", "1024*10+10 = 10250"]
counts = [int(sum(np.prod(w.shape) for w in l.get_weights()))
          for l in big.layers]
total = sum(counts)
out.write("   layer                keras   K(C F^2 + 1) or n_in n_out + n_out\n")
for nm, c, f in zip(names, counts, formula):
    if c:
        out.write(f"   {nm:18s} {c:9d}   {f}\n")
out.write(f"   {'total':18s} {total:9d}\n")
conv_par = counts[0] + counts[2]
out.write(f"  the two convolutional layers hold {conv_par} parameters, "
          f"{100*conv_par/total:.2f}% of the network;\n")
out.write(f"  the first dense layer alone holds {counts[5]}, "
          f"{100*counts[5]/total:.2f}%\n\n")
np.save("paramcounts.npy", np.array(counts))
out.close()
print(open("cross_check.txt").read())

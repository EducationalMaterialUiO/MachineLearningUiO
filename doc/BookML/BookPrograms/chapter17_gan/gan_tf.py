"""Chapter 17: the same generative adversarial network on MNIST, in TensorFlow.

This is a line-by-line counterpart of ``gan_torch.py``.  The only real
difference is bookkeeping: PyTorch accumulates gradients on the parameters and
we zero them, whereas TensorFlow records a tape and we ask it for the gradients
of one loss with respect to one list of variables.  The two tapes below are the
two players.

Usage
-----
    python gan_tf.py --epochs 50                  # non-saturating GAN
    python gan_tf.py --epochs 50 --smooth 0.9     # one-sided label smoothing
    python gan_tf.py --epochs 50 --loss wgangp    # Wasserstein critic
    python gan_tf.py --epochs 50 --arch dcgan     # convolutional
"""
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# 1.  architectures
# ---------------------------------------------------------------------------
def make_generator_fc(k_z=100, n_out=784, widths=(256, 512, 1024)):
    m = keras.Sequential(name="generator")
    m.add(keras.Input(shape=(k_z,)))
    for w in widths:
        m.add(layers.Dense(w, use_bias=False))
        m.add(layers.BatchNormalization())
        m.add(layers.LeakyReLU(negative_slope=0.2))
    m.add(layers.Dense(n_out, activation="tanh"))
    return m


def make_discriminator_fc(n_in=784, widths=(1024, 512, 256), p_drop=0.3):
    """Returns a logit: the sigmoid is folded into the loss."""
    m = keras.Sequential(name="discriminator")
    m.add(keras.Input(shape=(n_in,)))
    for w in widths:
        m.add(layers.Dense(w))
        m.add(layers.LeakyReLU(negative_slope=0.2))
        m.add(layers.Dropout(p_drop))
    m.add(layers.Dense(1))
    return m


def make_generator_dc(k_z=100, ngf=64):
    init = keras.initializers.RandomNormal(0.0, 0.02)
    m = keras.Sequential(name="generator_dc")
    m.add(keras.Input(shape=(k_z,)))
    m.add(layers.Dense(7 * 7 * ngf * 4, use_bias=False,
                       kernel_initializer=init))
    m.add(layers.Reshape((7, 7, ngf * 4)))
    m.add(layers.BatchNormalization())
    m.add(layers.ReLU())
    m.add(layers.Conv2DTranspose(ngf * 2, 4, 2, "same", use_bias=False,
                                 kernel_initializer=init))      # 14x14
    m.add(layers.BatchNormalization())
    m.add(layers.ReLU())
    m.add(layers.Conv2DTranspose(1, 4, 2, "same", use_bias=False,
                                 activation="tanh",
                                 kernel_initializer=init))      # 28x28
    return m


def make_discriminator_dc(ndf=64):
    init = keras.initializers.RandomNormal(0.0, 0.02)
    m = keras.Sequential(name="discriminator_dc")
    m.add(keras.Input(shape=(28, 28, 1)))
    m.add(layers.Conv2D(ndf, 4, 2, "same", kernel_initializer=init))   # 14x14
    m.add(layers.LeakyReLU(negative_slope=0.2))
    m.add(layers.Conv2D(ndf * 2, 4, 2, "same", kernel_initializer=init))  # 7x7
    m.add(layers.LeakyReLU(negative_slope=0.2))
    m.add(layers.Flatten())
    m.add(layers.Dense(1, kernel_initializer=init))
    return m


# ---------------------------------------------------------------------------
# 2.  losses
# ---------------------------------------------------------------------------
bce = keras.losses.BinaryCrossentropy(from_logits=True)


def gradient_penalty(D, x_real, x_fake):
    """E[(||grad f_w(xhat)||_2 - 1)^2],  Eq. (17.wgangp)."""
    shape = [tf.shape(x_real)[0]] + [1] * (len(x_real.shape) - 1)
    eps = tf.random.uniform(shape, 0.0, 1.0)
    xhat = eps * x_real + (1.0 - eps) * x_fake
    with tf.GradientTape() as tape:
        tape.watch(xhat)
        u = D(xhat, training=True)
    g = tape.gradient(u, xhat)
    norm = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(g, [tf.shape(g)[0], -1])),
                                 axis=1) + 1e-12)
    return tf.reduce_mean(tf.square(norm - 1.0))


# ---------------------------------------------------------------------------
# 3.  one training step: two tapes, two optimisers
# ---------------------------------------------------------------------------
def make_steps(G, D, opt_g, opt_d, args):

    @tf.function
    def d_step(x):
        z = tf.random.normal([tf.shape(x)[0], args.k_z])
        x_fake = G(z, training=True)            # detached: outside the tape
        with tf.GradientTape() as tape:
            u_real = D(x, training=True)
            u_fake = D(x_fake, training=True)
            if args.loss == "wgangp":
                loss = (tf.reduce_mean(u_fake) - tf.reduce_mean(u_real)
                        + args.lam * gradient_penalty(D, x, x_fake))
            else:
                loss = (bce(tf.fill(tf.shape(u_real), args.smooth), u_real)
                        + bce(tf.zeros_like(u_fake), u_fake))
        opt_d.apply_gradients(zip(tape.gradient(loss, D.trainable_variables),
                                  D.trainable_variables))
        return loss, tf.sigmoid(u_real), tf.sigmoid(u_fake)

    @tf.function
    def g_step(n):
        z = tf.random.normal([n, args.k_z])
        with tf.GradientTape() as tape:
            u_fake = D(G(z, training=True), training=True)
            if args.loss == "wgangp":
                loss = -tf.reduce_mean(u_fake)
            elif args.loss == "sat":
                loss = -bce(tf.zeros_like(u_fake), u_fake)
            else:                                # non-saturating
                loss = bce(tf.ones_like(u_fake), u_fake)
        opt_g.apply_gradients(zip(tape.gradient(loss, G.trainable_variables),
                                  G.trainable_variables))
        return loss

    return d_step, g_step


def train(args):
    tf.random.set_seed(args.seed)
    (x, _), _ = keras.datasets.mnist.load_data()
    x = (x.astype("float32") - 127.5) / 127.5          # into [-1,1]
    conv = args.arch == "dcgan"
    x = x.reshape(-1, 28, 28, 1) if conv else x.reshape(-1, 784)
    ds = (tf.data.Dataset.from_tensor_slices(x)
          .shuffle(60000).batch(args.batch, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE))

    G = make_generator_dc(args.k_z) if conv else make_generator_fc(args.k_z)
    D = make_discriminator_dc() if conv else make_discriminator_fc()
    opt_g = keras.optimizers.Adam(args.lr, beta_1=0.5)
    opt_d = keras.optimizers.Adam(args.lr, beta_1=0.5)
    d_step, g_step = make_steps(G, D, opt_g, opt_d, args)
    fixed_z = tf.random.normal([64, args.k_z])

    for epoch in range(1, args.epochs + 1):
        s = np.zeros(4)
        m = 0
        for xb in ds:
            for _ in range(args.n_critic):
                ld, dr, df = d_step(xb)
            lg = g_step(tf.shape(xb)[0])
            s += [ld.numpy(), lg.numpy(),
                  tf.reduce_mean(dr).numpy(), tf.reduce_mean(df).numpy()]
            m += 1
        print(f"epoch {epoch:03d}  L_D {s[0]/m:7.4f}  L_G {s[1]/m:7.4f}"
              f"  D(x) {s[2]/m:.3f}  D(G(z)) {s[3]/m:.3f}", flush=True)
        img = (G(fixed_z, training=False).numpy().reshape(-1, 28, 28) + 1) / 2
        np.save(f"samples_epoch{epoch:03d}.npy", img)

    G.save("generator_mnist.keras")
    D.save("discriminator_mnist.keras")
    return G, D


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--k_z", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--smooth", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--n_critic", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--arch", choices=["fc", "dcgan"], default="fc")
    p.add_argument("--loss", choices=["nonsat", "sat", "wgangp"],
                   default="nonsat")
    args = p.parse_args()
    if args.loss == "wgangp" and args.n_critic == 1:
        args.n_critic = 5
    train(args)

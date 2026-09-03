"""Chapter 17: listing 7, from the section on gans in pytorch and tensorflow.

Extracted from doc/BookML/chapter17.tex.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

bce = keras.losses.BinaryCrossentropy(from_logits=True)


def make_generator_fc(k_z=100, n_out=784, widths=(256, 512, 1024)):
    m = keras.Sequential([keras.Input(shape=(k_z,))])
    for w in widths:
        m.add(layers.Dense(w, use_bias=False))
        m.add(layers.BatchNormalization())
        m.add(layers.LeakyReLU(negative_slope=0.2))
    m.add(layers.Dense(n_out, activation="tanh"))
    return m


def make_discriminator_fc(n_in=784, widths=(1024, 512, 256), p_drop=0.3):
    m = keras.Sequential([keras.Input(shape=(n_in,))])
    for w in widths:
        m.add(layers.Dense(w))
        m.add(layers.LeakyReLU(negative_slope=0.2))
        m.add(layers.Dropout(p_drop))
    m.add(layers.Dense(1))                     # a logit, not a probability
    return m


G, D = make_generator_fc(), make_discriminator_fc()
opt_g = keras.optimizers.Adam(2e-4, beta_1=0.5)
opt_d = keras.optimizers.Adam(2e-4, beta_1=0.5)


@tf.function
def train_step(x, k_z=100, smooth=1.0):
    n = tf.shape(x)[0]

    # --- discriminator ----------------------------------------------------
    x_fake = G(tf.random.normal([n, k_z]), training=True)   # outside the tape
    with tf.GradientTape() as tape:                         # = detach()
        u_real, u_fake = D(x, training=True), D(x_fake, training=True)
        loss_d = (bce(tf.fill(tf.shape(u_real), smooth), u_real)
                  + bce(tf.zeros_like(u_fake), u_fake))
    opt_d.apply_gradients(zip(tape.gradient(loss_d, D.trainable_variables),
                              D.trainable_variables))

    # --- generator, Eq. (17.nonsat) ---------------------------------------
    with tf.GradientTape() as tape:
        u_fake = D(G(tf.random.normal([n, k_z]), training=True), training=True)
        loss_g = bce(tf.ones_like(u_fake), u_fake)
    opt_g.apply_gradients(zip(tape.gradient(loss_g, G.trainable_variables),
                              G.trainable_variables))
    return loss_d, loss_g, tf.sigmoid(u_real), tf.sigmoid(u_fake)

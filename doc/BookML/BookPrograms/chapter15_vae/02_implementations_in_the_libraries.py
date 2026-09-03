"""Chapter 15: listing 2, from the section on implementations in the libraries.

Extracted from doc/BookML/chapter15.tex.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Eq. (15.reparam): h = mu + sigma * eps, with eps an input."""
    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps


d, d_h = 784, 16
enc_in = keras.Input(shape=(d,))
e = layers.Dense(256, activation="relu")(enc_in)
mu = layers.Dense(d_h, name="mu")(e)
logvar = layers.Dense(d_h, name="logvar")(e)         # log sigma^2, unconstrained
h = Sampling()([mu, logvar])
encoder = keras.Model(enc_in, [mu, logvar, h], name="encoder")

dec_in = keras.Input(shape=(d_h,))
dd = layers.Dense(256, activation="relu")(dec_in)
logits = layers.Dense(d)(dd)                          # Bernoulli logits
decoder = keras.Model(dec_in, logits, name="decoder")


class VAE(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def train_step(self, x):
        with tf.GradientTape() as tape:
            mu, logvar, h = self.encoder(x)
            logits = self.decoder(h)
            rec = -tf.reduce_sum(                     # log p(x|h), Bernoulli
                tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits),
                axis=1)
            kl = 0.5 * tf.reduce_sum(                 # Eq. (15.klclosed)
                tf.square(mu) + tf.exp(logvar) - logvar - 1.0, axis=1)
            loss = -tf.reduce_mean(rec - kl)          # minimise the negative ELBO
        g = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(g, self.trainable_weights))
        return {"elbo": -loss}


vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(1e-3))
vae.fit(x_train, epochs=30, batch_size=128)

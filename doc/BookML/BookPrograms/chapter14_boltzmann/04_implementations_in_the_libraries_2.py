"""Chapter 14: listing 4, from the section on implementations in the libraries.

Extracted from doc/BookML/chapter14.tex.
"""

import tensorflow as tf


class RBM(tf.Module):
    def __init__(self, M=784, N=256, k=1):
        self.W = tf.Variable(tf.random.normal([M, N], stddev=0.01))
        self.a = tf.Variable(tf.zeros([M]))
        self.b = tf.Variable(tf.zeros([N]))
        self.k = k

    def sample(self, p):
        return tf.cast(tf.random.uniform(tf.shape(p)) < p, tf.float32)

    def gibbs(self, x):
        h = self.sample(tf.sigmoid(x @ self.W + self.b))
        return self.sample(tf.sigmoid(h @ tf.transpose(self.W) + self.a))

    @tf.function
    def cd_update(self, x, lr=0.01):
        """Eq. (14.rbmgradient) with the negative phase from k Gibbs sweeps."""
        ph_data = tf.sigmoid(x @ self.W + self.b)
        v = x
        for _ in range(self.k):
            v = self.gibbs(v)
        ph_model = tf.sigmoid(v @ self.W + self.b)
        n = tf.cast(tf.shape(x)[0], tf.float32)
        self.W.assign_add(lr * (tf.transpose(x) @ ph_data
                                - tf.transpose(v) @ ph_model) / n)
        self.a.assign_add(lr * tf.reduce_mean(x - v, axis=0))
        self.b.assign_add(lr * tf.reduce_mean(ph_data - ph_model, axis=0))

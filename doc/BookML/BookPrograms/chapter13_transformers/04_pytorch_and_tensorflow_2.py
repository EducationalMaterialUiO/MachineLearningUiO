"""Chapter 13: listing 4, from the section on pytorch and tensorflow.

Extracted from doc/BookML/chapter13.tex.
"""

import tensorflow as tf
from tensorflow.keras import layers


class Block(layers.Layer):
    """Eq. (13.block) in Keras; note key_dim is d_k, not d."""
    def __init__(self, d=64, H=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=H, key_dim=d // H,
                                              dropout=dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dense(d), layers.Dropout(dropout)])

    def call(self, x, training=False):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, use_causal_mask=True, training=training)
        return x + self.mlp(self.ln2(x), training=training)


def build(n_vocab, n_ctx=64, d=64, H=4, d_ff=256, n_blocks=2):
    inp = layers.Input(shape=(n_ctx,), dtype="int32")
    tok = layers.Embedding(n_vocab, d)(inp)
    pos = layers.Embedding(n_ctx, d)(tf.range(n_ctx))
    x = tok + pos
    for _ in range(n_blocks):
        x = Block(d, H, d_ff)(x)
    out = layers.Dense(n_vocab)(layers.LayerNormalization(epsilon=1e-6)(x))
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model

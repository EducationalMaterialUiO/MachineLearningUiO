"""Chapter 12: listing 5, from the section on pytorch and tensorflow.

Extracted from doc/BookML/chapter12.tex.
"""

# Encoder: (28,28,1) -> (14,14,16) -> (7,7,32) -> (4,4,64), Eq. (10.outsize)
conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    keras.layers.Conv2D(16, 3, strides=2, padding="SAME", activation="selu"),
    keras.layers.Conv2D(32, 3, strides=2, padding="SAME", activation="selu"),
    keras.layers.Conv2D(64, 3, strides=2, padding="SAME", activation="selu"),
])
# Decoder: the exact mirror, transposed convolutions doubling the size
conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, 3, strides=2, padding="SAME",
                                 activation="selu", input_shape=[4, 4, 64]),
    keras.layers.Conv2DTranspose(16, 3, strides=2, padding="SAME",
                                 activation="selu"),
    keras.layers.Conv2DTranspose(1, 3, strides=2, padding="SAME",
                                 activation="sigmoid"),
    keras.layers.Lambda(lambda x: x[:, :28, :28, :]),   # crop 32 -> 28
    keras.layers.Reshape([28, 28]),
])
conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])
conv_ae.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.Adam(1e-3))

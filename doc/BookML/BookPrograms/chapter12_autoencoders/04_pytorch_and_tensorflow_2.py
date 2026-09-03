"""Chapter 12: listing 4, from the section on pytorch and tensorflow.

Extracted from doc/BookML/chapter12.tex.
"""

(X_train_full, _), (X_test, _) = keras.datasets.mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]

stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),          # the bottleneck, p = 30
])
stacked_decoder = keras.models.Sequential([             # the exact mirror
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),  # data live in [0,1]
    keras.layers.Reshape([28, 28]),
])
stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy",          # Eq. (12.bce)
                   optimizer=keras.optimizers.Adam(1e-3))
stacked_ae.fit(X_train, X_train, epochs=10,
               validation_data=(X_valid, X_valid))

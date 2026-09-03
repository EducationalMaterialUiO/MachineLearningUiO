"""Chapter 12: listing 3, from the section on pytorch and tensorflow.

Extracted from doc/BookML/chapter12.tex.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# three-dimensional data lying near a plane
np.random.seed(4)
def generate_3d_data(m, w1=0.1, w2=0.3, noise=0.1):
    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    data = np.empty((m, 3))
    data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise*np.random.randn(m)/2
    data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    data[:, 2] = data[:, 0]*w1 + data[:, 1]*w2 + noise*np.random.randn(m)
    return data

X_train = generate_3d_data(60)
X_train = X_train - X_train.mean(axis=0, keepdims=True)   # centre: see the notebox

# no activations anywhere: this is Eq. (12.linae)
encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])
decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])
autoencoder = keras.models.Sequential([encoder, decoder])
autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1.5))
autoencoder.fit(X_train, X_train, epochs=200, verbose=0)   # target == input

codings = encoder.predict(X_train)
# compare with PCA: the subspaces should agree, the bases need not
U, s, Vt = np.linalg.svd(X_train, full_matrices=False)
print("PCA loadings:\n", Vt[:2].T)
print("encoder weights:\n", encoder.layers[0].get_weights()[0])

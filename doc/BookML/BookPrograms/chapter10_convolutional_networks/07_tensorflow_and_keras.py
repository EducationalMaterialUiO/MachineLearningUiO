"""Chapter 10: listing 7, from the section on tensorflow and keras.

Extracted from doc/BookML/chapter10.tex.
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
test_images  = test_images.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                  input_shape=(28, 28, 1)),          # note channels LAST
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10),                                # logits
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.summary()          # prints Eq. (10.paramcount) layer by layer

model.fit(train_images, train_labels, epochs=10, batch_size=64,
          validation_data=(test_images, test_labels))

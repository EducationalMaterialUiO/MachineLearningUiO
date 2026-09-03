"""Chapter 11: listing 6, from the section on an lstm on mnist read row by row.

Extracted from doc/BookML/chapter11.tex.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((-1, 28, 28))        # 28 timesteps of 28 features
x_test = x_test.reshape((-1, 28, 28))
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    LSTM(128, input_shape=(28, 28)),           # 4*128*(28+128+1) = 80384 params
    Dense(10, activation="softmax"),
])
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"test accuracy {test_acc:.4f}")

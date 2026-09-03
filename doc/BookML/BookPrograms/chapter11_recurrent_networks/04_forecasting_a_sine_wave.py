"""Chapter 11: listing 4, from the section on forecasting a sine wave.

Extracted from doc/BookML/chapter11.tex.
"""

import numpy as np
import tensorflow as tf

# 1. Data: a sine wave cut into overlapping windows
time_steps = np.linspace(0, 100, 500)
data = np.sin(time_steps)
seq_length = 20
X, y = [], []
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])
    y.append(data[i+seq_length])
X = np.array(X).reshape(-1, seq_length, 1)     # (samples, timesteps, features)
y = np.array(y).reshape(-1, 1)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 2. Model: a SimpleRNN layer is exactly Eq. (11.rnn)
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(16, input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.summary()      # 16*(1+16+1) = 288 recurrent parameters, then 17

# 3. Training and evaluation
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, verbose=1)
print(f"test loss {model.evaluate(X_test, y_test, verbose=0):.4f}")

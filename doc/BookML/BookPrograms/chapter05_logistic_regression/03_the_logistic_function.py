"""Chapter 5: listing 3, from the section on the logistic function.

Extracted from doc/BookML/chapter5.tex.
"""

import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, 0.1)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
ax[0].plot(z, 1.0 / (1.0 + np.exp(-z)));      ax[0].set_title("sigmoid")
ax[1].plot(z, np.where(z >= 0.0, 1.0, 0.0));  ax[1].set_title("step")
ax[2].plot(z, np.tanh(z));                    ax[2].set_title("tanh")
for a in ax:
    a.grid(True); a.set_xlabel("z")
plt.show()

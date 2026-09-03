"""Load MNIST from a local .npz, so that the programs run without a download.

The chapters' library listings call ``datasets.MNIST(download=True)`` or
``keras.datasets.mnist.load_data()``.  Both fetch the same 60000 + 10000 images
from the network.  When that is not available -- on a cluster node without
outbound access, or behind a proxy -- point ``MNIST_NPZ`` at a local copy and
everything below is unchanged.
"""
import os

import numpy as np

NPZ = os.environ.get("MNIST_NPZ",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "mnist.npz"))


def load_mnist(flat=False, normalise="unit"):
    """Returns (x_train, y_train), (x_test, y_test) with x in float32.

    `normalise`: "unit" maps to [0,1]; "standard" applies the usual MNIST
    mean 0.1307 and standard deviation 0.3081; None leaves the raw bytes.
    """
    d = np.load(NPZ)
    xtr = d["x_train"].astype("float32")
    xte = d["x_test"].astype("float32")
    if normalise is not None:
        xtr, xte = xtr / 255.0, xte / 255.0
    if normalise == "standard":
        xtr = (xtr - 0.1307) / 0.3081
        xte = (xte - 0.1307) / 0.3081
    if flat:
        xtr, xte = xtr.reshape(len(xtr), -1), xte.reshape(len(xte), -1)
    return (xtr, d["y_train"].astype("int64")), (xte, d["y_test"].astype("int64"))

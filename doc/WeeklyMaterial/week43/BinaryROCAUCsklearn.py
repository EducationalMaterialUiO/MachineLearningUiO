"""A small NumPy neural network with scikit-learn binary evaluation.

Install: python -m pip install numpy scikit-learn matplotlib
Run:     python simple_binary_neural_network.py

The example generates two classes, labelled 0 and 1. Class 1 is positive.
The network is written from scratch; scikit-learn supplies the evaluation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, auc,
    confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    # A numerically stable version of 1 / (1 + exp(-z)).
    return np.exp(-np.logaddexp(0, -z))


# 1. Generate data and keep a separate test set.
X, y = make_classification(
    n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.2, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42,
)
scaler = StandardScaler().fit(X_train)  # Fit on training data only.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 2. Train an own network: two inputs -> eight tanh units -> one sigmoid.
rng = np.random.default_rng(42)
W1 = rng.normal(0, 0.5, size=(X_train.shape[1], 8))
b1 = np.zeros(8)
W2 = rng.normal(0, 0.5, size=8)
b2 = 0.0
learning_rate = 0.1

for epoch in range(1000):
    hidden = np.tanh(X_train @ W1 + b1)
    p = sigmoid(hidden @ W2 + b2)

    # Backpropagation for mean binary cross-entropy.
    d_output = (p - y_train) / len(y_train)
    d_hidden = d_output[:, None] * W2 * (1 - hidden ** 2)
    W1 -= learning_rate * (X_train.T @ d_hidden)
    b1 -= learning_rate * d_hidden.sum(axis=0)
    W2 -= learning_rate * (hidden.T @ d_output)
    b2 -= learning_rate * d_output.sum()

# 3. Obtain the continuous class-1 score for every test observation.
# For your own network, replace this line with its class-1 probabilities.
positive_score = sigmoid(np.tanh(X_test @ W1 + b1) @ W2 + b2)
y_pred = (positive_score >= 0.5).astype(int)

# 4. Call scikit-learn's confusion matrix, ROC and AUC functions.
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
fpr, tpr, thresholds = roc_curve(y_test, positive_score, pos_label=1)
auc_value = auc(fpr, tpr)
# This second scikit-learn function gives the same AUC directly from scores:
auc_direct = roc_auc_score(y_test, positive_score)

print("Confusion matrix (rows=true, columns=predicted):\n", cm)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC from ROC coordinates: {auc_value:.4f}")
print(f"AUC directly from scores: {auc_direct:.4f}")

# 5. Plot the matrix and the single binary ROC curve.
fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(
    ax=ax_cm, cmap="Blues", colorbar=False, values_format="d",
)
ax_cm.set_title("Confusion matrix: threshold = 0.5")
ax_roc.plot(fpr, tpr, label=f"ROC (AUC = {auc_value:.3f})")
ax_roc.plot([0, 1], [0, 1], "k--", label="Random ranking")
ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Binary ROC curve", xlim=(0, 1), ylim=(0, 1.02))
ax_roc.legend(loc="lower right")
ax_roc.grid(alpha=0.3)
fig.tight_layout()
figure_path = Path(__file__).with_suffix(".png")
fig.savefig(figure_path, dpi=160)
print(f"Figure saved to {figure_path}")
plt.show()

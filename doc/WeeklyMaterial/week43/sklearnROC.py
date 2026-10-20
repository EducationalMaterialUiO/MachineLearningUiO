#!/usr/bin/env python3
"""Train a NumPy neural network written from scratch; evaluate with scikit-learn.


The bundled 8x8 UCI digits dataset is used, not the 28x28 MNIST dataset.
No dataset download is needed. Binary mode compares digit 3 with digit 8;
digit 8 is the positive class. Multiclass mode uses all ten digits.

To evaluate your own classifier, keep evaluate_classifier() and replace the
training part with:
    probabilities = your_network.predict_proba(X_test)
    evaluate_classifier(y_test, probabilities, your_network.classes_,
                        output_dir="my_results")
For a sigmoid network returning p(class=1), first use
    probabilities = np.column_stack([1 - positive_scores, positive_scores])
and pass classes=np.array([0, 1]). For MNIST, pass its ten probability
columns in digit order 0,...,9 and classes=np.arange(10).

AUC is a scalar area under a ROC curve, not an additional curve.
References:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NumpyNeuralNetwork:
    """Custom dense network: input -> tanh hidden layer -> softmax outputs.

    Training uses mini-batch gradient descent, cross-entropy, optional L2
    regularization, and early stopping on validation loss.
    Training labels are column indices 0,...,n_classes-1.
    """

    def __init__(self, n_features, n_classes, hidden=64, l2=1e-4, seed=42):
        self.rng = np.random.default_rng(seed)
        self.classes_ = np.arange(n_classes)
        self.l2 = l2
        self.parameters = {
            "W1": self.rng.normal(0, np.sqrt(1 / n_features), (n_features, hidden)),
            "b1": np.zeros(hidden),
            "W2": self.rng.normal(0, np.sqrt(1 / hidden), (hidden, n_classes)),
            "b2": np.zeros(n_classes),
        }

    def _forward(self, X):
        p = self.parameters
        hidden = np.tanh(X @ p["W1"] + p["b1"])
        logits = hidden @ p["W2"] + p["b2"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return hidden, log_probabilities

    def predict_proba(self, X):
        """Return (n_images, n_classes), columns matching self.classes_."""
        return np.exp(self._forward(np.asarray(X, dtype=float))[1])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def loss(self, X, y):
        log_probabilities = self._forward(X)[1]
        penalty = sum(np.sum(self.parameters[k] ** 2) for k in ("W1", "W2"))
        return -log_probabilities[np.arange(len(y)), y].mean() + 0.5 * self.l2 * penalty

    def _gradients(self, X, y):
        hidden, log_probabilities = self._forward(X)
        delta = np.exp(log_probabilities)
        delta[np.arange(len(y)), y] -= 1
        delta /= len(y)
        hidden_delta = (delta @ self.parameters["W2"].T) * (1 - hidden ** 2)
        return {
            "W1": X.T @ hidden_delta + self.l2 * self.parameters["W1"],
            "b1": hidden_delta.sum(axis=0),
            "W2": hidden.T @ delta + self.l2 * self.parameters["W2"],
            "b2": delta.sum(axis=0),
        }

    def fit(self, X, y, X_validation, y_validation, epochs=200,
            learning_rate=0.05, batch_size=64, patience=20):
        best_loss, stale_epochs = np.inf, 0
        best_parameters = {k: v.copy() for k, v in self.parameters.items()}
        for epoch in range(1, epochs + 1):
            indices = self.rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                batch = indices[start:start + batch_size]
                gradients = self._gradients(X[batch], y[batch])
                for name in self.parameters:
                    self.parameters[name] -= learning_rate * gradients[name]
            validation_loss = self.loss(X_validation, y_validation)
            if not np.isfinite(validation_loss):
                raise RuntimeError("Training diverged; reduce the learning rate")
            if validation_loss < best_loss - 1e-5:
                best_loss, stale_epochs = validation_loss, 0
                best_parameters = {k: v.copy() for k, v in self.parameters.items()}
                self.best_epoch_ = epoch
            else:
                stale_epochs += 1
            if epoch == 1 or epoch % 25 == 0:
                print(f"Epoch {epoch:3d}: train loss={self.loss(X, y):.4f}, "
                      f"validation loss={validation_loss:.4f}")
            if stale_epochs >= patience:
                break
        self.parameters = best_parameters
        print(f"Restored validation-selected epoch {self.best_epoch_}.")
        return self


def evaluate_classifier(y_true, probabilities, classes, output_dir, show=False):
    """Use sklearn metrics on outputs from ANY classifier.

    y_true: (N,) original class labels.
    probabilities: (N,K), nonnegative, finite, each row sums to one.
    classes: (K,) sorted, unique labels in EXACTLY the same order as columns.
    Every class must occur in this test set to define all OvR ROC/AUCs.
    Returns a metrics dictionary; saves plots, metrics and raw predictions.
    """
    y = np.asarray(y_true)
    raw = np.asarray(probabilities)
    classes = np.asarray(classes)
    if np.iscomplexobj(raw):
        raise ValueError("Probabilities must be real")
    P = raw.astype(float)
    if (classes.ndim != 1 or classes.size < 2
            or not np.array_equal(classes, np.unique(classes))):
        raise ValueError("classes must contain sorted, unique labels")
    if y.ndim != 1 or y.size == 0 or P.shape != (y.size, classes.size):
        raise ValueError("Expected labels (N,) and probabilities (N,K)")
    if (not np.isfinite(P).all() or np.any((P < 0) | (P > 1))
            or not np.allclose(P.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Probability rows must be finite, nonnegative and sum to 1")
    if not np.isin(y, classes).all():
        raise ValueError("A test label is missing from classes")
    Y = y[:, None] == classes[None, :]
    if np.any((Y.sum(axis=0) == 0) | (Y.sum(axis=0) == len(y))):
        raise ValueError("Each class needs both positive and negative test examples")

    # Hard predictions are for the confusion matrix, NOT the ROC inputs.
    y_pred = classes[P.argmax(axis=1)]
    cm = confusion_matrix(y, y_pred, labels=classes)
    result = {"accuracy": float(accuracy_score(y, y_pred)),
              "classes": classes.tolist(), "support": Y.sum(axis=0).tolist(),
              "confusion_matrix": cm.tolist()}

    # Use a noninteractive backend unless a plot window was requested.
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax_cm, cmap="Blues", colorbar=False, values_format="d")
    ax_cm.set_title("Test confusion matrix (argmax predictions)")

    if classes.size == 2:
        # Second column is positive, matching classes[1] (8 in the demo).
        z, score = Y[:, 1], P[:, 1]
        fpr, tpr, _ = roc_curve(z, score, drop_intermediate=False)
        area = auc(fpr, tpr)
        result["positive_class"] = classes[1].item()
        result["roc_auc"] = float(roc_auc_score(z, score))
        np.testing.assert_allclose(area, result["roc_auc"])
        ax_roc.plot(fpr, tpr, lw=2, label=f"Class {classes[1]}: AUC={area:.4f}")
        ax_roc.fill_between(fpr, tpr, alpha=0.12)
    else:
        curves = []
        per_class_auc = []
        for k, label in enumerate(classes):
            fpr, tpr, _ = roc_curve(Y[:, k], P[:, k], drop_intermediate=False)
            area = auc(fpr, tpr)
            curves.append((fpr, tpr))
            per_class_auc.append(area)
            ax_roc.plot(fpr, tpr, lw=1, alpha=0.75,
                        label=f"{label} vs rest: {area:.3f}")
        result["auc_per_class"] = [float(v) for v in per_class_auc]
        result["auc_macro_ovr"] = float(roc_auc_score(
            y, P, labels=classes, multi_class="ovr", average="macro"))
        result["auc_weighted_ovr"] = float(roc_auc_score(
            y, P, labels=classes, multi_class="ovr", average="weighted"))
        np.testing.assert_allclose(result["auc_macro_ovr"], np.mean(per_class_auc))
        # Flattening also works with older sklearn versions.
        fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), P.ravel(), drop_intermediate=False)
        result["auc_micro_ovr"] = float(roc_auc_score(Y.ravel(), P.ravel()))
        np.testing.assert_allclose(auc(fpr_micro, tpr_micro), result["auc_micro_ovr"])
        ax_roc.plot(fpr_micro, tpr_micro, color="black", lw=2.5,
                    label=f"Micro: {result['auc_micro_ovr']:.4f}")
        grid = np.linspace(0, 1, 1001)
        macro_tpr = np.mean([np.interp(grid, fpr, tpr) for fpr, tpr in curves], axis=0)
        ax_roc.plot(grid, macro_tpr, "--", color="magenta", lw=2.5,
                    label=f"Macro display: mean AUC={result['auc_macro_ovr']:.4f}")
        # The plotted macro curve is approximate; its label uses the exact
        # sklearn mean AUC, not numerical integration of the display grid.

    ax_roc.plot([0, 1], [0, 1], "k:", label="Random ranking")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
               xlim=(0, 1), ylim=(0, 1.02), title="Test ROC curves and AUC")
    ax_roc.grid(alpha=0.25)
    ax_roc.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "confusion_matrix_and_roc.png", dpi=180)
    report = classification_report(y, y_pred, labels=classes, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "test_predictions.npz", y_true=y,
                        probabilities=P, classes=classes)
    print(report)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("confusion_matrix", "classes", "support")}, indent=2))
    print(f"Results saved to {output_dir.resolve()}")
    if show:
        plt.show()
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("multiclass", "binary"), default="multiclass")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--show", action="store_true", help="Also display the plot window")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    X, y_original = load_digits(return_X_y=True)
    if args.task == "binary":
        keep = np.isin(y_original, [3, 8])
        X, y_original = X[keep], y_original[keep]
    classes, y_indices = np.unique(y_original, return_inverse=True)
    # Split before fitting preprocessing; never use the test data to tune.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_indices, test_size=0.20, random_state=42, stratify=y_indices)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval)
    scaler = StandardScaler().fit(X_train)
    X_train, X_validation, X_test = [scaler.transform(a)
                                    for a in (X_train, X_validation, X_test)]
    model = NumpyNeuralNetwork(X_train.shape[1], len(classes))
    model.fit(X_train, y_train, X_validation, y_validation, epochs=args.epochs)
    # This is the only interface needed to evaluate your own neural network:
    probabilities = model.predict_proba(X_test)
    output_dir = args.output_dir or Path(__file__).resolve().parent / f"{args.task}_results"
    evaluate_classifier(classes[y_test], probabilities, classes, output_dir, args.show)


if __name__ == "__main__":
    main()

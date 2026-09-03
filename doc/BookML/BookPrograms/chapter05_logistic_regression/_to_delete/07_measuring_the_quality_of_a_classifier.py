"""Chapter 5: listing 7, from the section on measuring the quality of a classifier.

Extracted from doc/BookML/chapter5.tex.
"""

import numpy as np

def confusion_matrix(y_true, y_pred):
    """Return TP, FN, FP, TN for binary labels in {0, 1}."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fn, fp, tn


def classification_report(y_true, y_pred):
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)


def roc_auc(y_true, scores):
    """AUC as the probability that a positive outranks a negative."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(np.sum(y_true == 1))
    n_neg = len(y_true) - n_pos
    return (np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

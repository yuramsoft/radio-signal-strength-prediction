"""
Step 5: evaluation metrics written from scratch (NumPy and pandas only).
"""
import numpy as np
import pandas as pd

from .config import LABELS


# ---- regression ----
def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def skewness(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean((x - x.mean()) ** 3) / x.std() ** 3)


# ---- classification ----
def confusion_matrix(y_true, y_pred, labels=LABELS):
    """Rows are true classes, columns are predicted classes."""
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def accuracy(cm):
    return float(np.trace(cm) / cm.sum())


def classification_metrics(cm, labels=LABELS):
    """Per-class precision, recall, F1 and support as a DataFrame."""
    rows = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"class": label, "precision": precision, "recall": recall,
                     "f1": f1, "support": int(cm[i, :].sum())})
    return pd.DataFrame(rows).set_index("class")


def within_one_class(y_true, y_pred, labels=LABELS):
    """Share of predictions that are correct or in an adjacent class."""
    idx = {l: i for i, l in enumerate(labels)}
    t = np.array([idx[v] for v in y_true])
    p = np.array([idx[v] for v in y_pred])
    return float(np.mean(np.abs(t - p) <= 1))

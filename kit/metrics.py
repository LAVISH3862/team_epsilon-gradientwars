"""Metrics on probability arrays (classification, columns in classes() order) or predictions (regression).
If the competition's metric is missing, add it to score() and HIGHER_IS_BETTER and set comp.METRIC to its name."""
import numpy as np
from sklearn import metrics as M

HIGHER_IS_BETTER = {'macro_f1': True, 'f1': True, 'accuracy': True, 'auc': True,
                    'logloss': False, 'rmse': False, 'mae': False}
FAST = {'macro_f1', 'f1', 'accuracy'}  # label metrics fast() can score from integer labels


def labels(P):
    return P.argmax(1) if P.ndim == 2 else (P > 0.5).astype(int)


def score(name, y, P):
    if name == 'macro_f1': return M.f1_score(y, labels(P), average='macro')
    if name == 'f1': return M.f1_score(y, labels(P))
    if name == 'accuracy': return M.accuracy_score(y, labels(P))
    if name == 'auc': return M.roc_auc_score(y, P[:, 1] if P.ndim == 2 and P.shape[1] == 2 else P, multi_class='ovr')
    if name == 'logloss': return M.log_loss(y, P, labels=np.arange(P.shape[1]))
    if name == 'rmse': return float(np.sqrt(M.mean_squared_error(y, P)))
    if name == 'mae': return M.mean_absolute_error(y, P)
    raise ValueError(f'unknown metric {name!r}: add it to kit/metrics.py')


def fast(name, y, L):
    """A FAST metric from integer labels L via a confusion count: sklearn's values, without its overhead (bootstraps)."""
    if name == 'accuracy':
        return float(np.mean(y == L))
    C = max(int(max(y.max(), L.max())) + 1, 2)
    return from_counts(name, np.bincount(y * C + L, minlength=C * C).reshape(C, C))


def from_counts(name, cm):
    """A FAST metric from a table of counts cm[true label, predicted label]."""
    tp = np.diag(cm)
    if name == 'accuracy':
        return float(tp.sum() / cm.sum())
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2 * tp, den, out=np.zeros(len(cm)), where=den > 0)
    return float(f1[1] if name == 'f1' else f1[den > 0].mean())  # macro: classes in y or L, as sklearn


def better(name, a, b):
    """Is score a better than score b?"""
    return a > b if HIGHER_IS_BETTER.get(name, True) else a < b


def per_class_f1(y, P):
    return M.f1_score(y, labels(P), average=None, labels=np.arange(P.shape[1]), zero_division=0)

"""Blend saved predictions into a new tag. Weights are searched on out-of-fold predictions, and the score printed is
cross-fitted (weights chosen on the other folds, scored on the held-out one), so a blend can't look better than it is.
Held-out rows (make_folds.py --holdout) never choose anything; they get the weights chosen on all the other rows.
  python blend.py --tag blend1 gbm xgb effb0 stack       # hill climbing: start from the best tag, then keep adding the
                                                         # tag (repeats allowed) that helps most; weight = times picked
  python blend.py --tag avg3 gbm effb0 stack --equal     # plain average, e.g. the same model trained with several seeds
  python blend.py --tag geo3 gbm effb0 stack --geo       # geometric mean of probabilities
  python blend.py --tag lr3 gbm effb0 stack --stack --balanced   # logistic regression (Ridge for regression) on top:
                                                                 # use it when hill-climbed weights overfit"""
import argparse
from kit.common import y, folds, load_preds, save_preds, report, scores
from kit import metrics
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import comp

ap = argparse.ArgumentParser()
ap.add_argument('tags', nargs='+')
ap.add_argument('--tag', required=True)
ap.add_argument('--equal', action='store_true')
ap.add_argument('--geo', action='store_true', help='geometric mean of probabilities (classification)')
ap.add_argument('--steps', type=int, default=100, help='hill climbing: most tags added after the first')
ap.add_argument('--stack', action='store_true', help='a regularised linear model on the predictions instead of weights')
ap.add_argument('--C', type=float, default=0.1, help='--stack: inverse regularisation strength (smaller = smoother)')
ap.add_argument('--balanced', action='store_true', help='--stack: class_weight=balanced, usually right for macro-F1')
a = ap.parse_args()
O, T = zip(*[load_preds(t) for t in a.tags])
yy, fo, n = y(), folds(), len(a.tags)
reg = comp.TASK == 'regression'
cv, hold = np.where(fo >= 0)[0], np.where(fo < 0)[0]


def stack_features(Ps):
    """Regression: the predictions side by side. Classification: log-probabilities, which a linear model reads best."""
    return np.column_stack([P if reg else np.log(np.clip(P, 1e-6, 1)) for P in Ps])


def stack_fit(rows):
    if reg:
        model = Ridge(alpha=1 / a.C)
    else:
        model = LogisticRegression(C=a.C, max_iter=2000, class_weight='balanced' if a.balanced else None)
    return make_pipeline(StandardScaler(), model).fit(stack_features([P[rows] for P in O]), yy[rows])


def stack_predict(m, Ps):
    X = stack_features(Ps)
    if reg:
        return m.predict(X)
    out = np.zeros((len(X), O[0].shape[1]))
    out[:, m.classes_] = m.predict_proba(X)  # a class missing from the training rows keeps probability 0
    return out


def combine(Ps, w):
    if a.geo and comp.TASK != 'regression':
        L = sum(wi * np.log(np.clip(P, 1e-6, 1)) for wi, P in zip(w, Ps))
        E = np.exp(L - L.max(1, keepdims=True))
        return E / E.sum(1, keepdims=True)
    return sum(wi * P for wi, P in zip(w, Ps))


def fit(rows):
    """Hill climbing (Caruana's ensemble selection): start from the best single tag, then repeatedly add the tag, repeats
    allowed, that improves the score most, until none does. A tag's weight is how often it was picked."""
    if a.equal:
        return np.full(n, 1 / n)
    Pr, yr = [P[rows] for P in O], yy[rows]
    sc = lambda c: metrics.score(comp.METRIC, yr, combine(Pr, c / c.sum()))
    pick = lambda s: int(np.argmax(s) if metrics.HIGHER_IS_BETTER.get(comp.METRIC, True) else np.argmin(s))
    one = np.eye(n)
    alone = [sc(one[i]) for i in range(n)]
    cnt = one[pick(alone)].copy()
    best = max(alone) if metrics.HIGHER_IS_BETTER.get(comp.METRIC, True) else min(alone)
    for _ in range(a.steps):
        s = [sc(cnt + one[i]) for i in range(n)]
        i = pick(s)
        if not metrics.better(comp.METRIC, s[i], best):
            break
        cnt[i] += 1; best = s[i]
    return cnt / cnt.sum()


oof = np.zeros_like(O[0])
fold_rows = lambda k: np.where((fo >= 0) & (fo != k))[0]
if a.stack:
    for k in range(comp.N_FOLDS):
        oof[fo == k] = stack_predict(stack_fit(fold_rows(k)), [P[fo == k] for P in O])
    m = stack_fit(cv)
    if len(hold):
        oof[hold] = stack_predict(m, [P[hold] for P in O])
    pt = stack_predict(m, T)
else:
    for k in range(comp.N_FOLDS):
        oof[fo == k] = combine([P[fo == k] for P in O], fit(fold_rows(k)))
    w = fit(cv)
    if len(hold):
        oof[hold] = combine([P[hold] for P in O], w)
    pt = combine(T, w)
for i, (t, P) in enumerate(zip(a.tags, O)):
    s, h = scores(P)
    print(f'  {t}: ' + ('' if a.stack else f'weight {w[i]:.2f}, ') + f'alone {comp.METRIC} {s:.4f}'
          + ('' if h is None else f' (held-out {h:.4f})'))
save_preds(a.tag, oof, pt)
report(a.tag + ' (cross-fitted)', oof)

"""Tune a tag's decision rule to the metric and save the result as a new tag. Label metrics (macro_f1, f1, accuracy)
get a bias per class added to the log-probabilities (for binary that is a threshold); logloss gets a temperature.
Everything is cross-fitted: tuned on the other folds, scored on the held-out one, so the gain printed is honest.
Held-out rows (make_folds.py --holdout) are never tuned on: they get the rule tuned on all the other rows.
  python postprocess.py stack --tag stack_thr
  python postprocess.py stack --tag stack_thr --grid 3 --step 0.05    # wider, finer search"""
import argparse
from kit.common import y, folds, classes, load_preds, save_preds, report, scores
from kit import metrics
import numpy as np
import comp

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('--tag', required=True)
ap.add_argument('--grid', type=float, default=2.0, help='biases are searched in [-grid, grid]')
ap.add_argument('--step', type=float, default=0.1)
ap.add_argument('--gate', type=float, default=0.002, help='a cross-fitted gain below this is called noise')
a = ap.parse_args()
if comp.TASK == 'regression':
    raise SystemExit('postprocess.py is for classification: regression predictions have no threshold to tune')
if comp.METRIC == 'auc':
    raise SystemExit('AUC only depends on the ranking, so thresholds and temperatures cannot change it')
O, T = load_preds(a.src)
yy, fo = y(), folds()
L, Lt = np.log(np.clip(O, 1e-6, 1)), np.log(np.clip(T, 1e-6, 1))
label_metric = comp.METRIC != 'logloss'


def softmax(Z):
    E = np.exp(Z - Z.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


def apply(Z, p):
    return softmax(Z + p) if label_metric else softmax(Z / p)


def fit(rows):
    """Label metrics: coordinate search over per-class biases until no single change helps.
    logloss: grid search over one temperature."""
    Lr, yr = L[rows], yy[rows]
    # label metrics only look at the argmax, so the softmax can be skipped while searching
    if label_metric and comp.METRIC in metrics.FAST:  # from label counts: ~50x faster than sklearn on a million rows
        sc = lambda p: metrics.fast(comp.METRIC, yr, metrics.labels(Lr + p))
    else:
        sc = (lambda p: metrics.score(comp.METRIC, yr, Lr + p)) if label_metric else (lambda p: metrics.score(comp.METRIC, yr, apply(Lr, p)))
    if not label_metric:
        ts = np.arange(0.3, 3.0001, 0.05)
        return ts[np.argmax([-sc(t) for t in ts])]
    b, grid = np.zeros(L.shape[1]), np.arange(-a.grid, a.grid + 1e-9, a.step)
    best, moved = sc(b), True
    if comp.METRIC in metrics.FAST and L.shape[1] > 2:
        # one pass over the rows per class instead of one per grid value: a row turns to class c once c's bias
        # exceeds d = (best other class's score) - (c's score), so the label counts at every grid value come from
        # a histogram of d
        C, G, ar = L.shape[1], len(grid), np.arange(len(yr))
        while moved:
            moved = False
            for c in range(C):
                Z = Lr + b
                Z[:, c] = -np.inf
                other = Z.argmax(1)
                g = np.searchsorted(grid, Z[ar, other] - Lr[:, c], 'right')   # first grid value that turns the row
                cnt = np.bincount((g * C + yr) * C + other, minlength=(G + 1) * C * C).reshape(G + 1, C, C)
                base, turned = cnt.sum(0), np.cumsum(cnt[:-1], 0)
                for i, v in enumerate(grid):
                    cm = base - turned[i]
                    cm[:, c] += turned[i].sum(1)
                    s = metrics.from_counts(comp.METRIC, cm)
                    if metrics.better(comp.METRIC, s, best):
                        b[c], best, moved = v, s, True
        return b
    while moved:
        moved = False
        for c in range(len(b)):
            for v in grid:
                b2 = b.copy(); b2[c] = v
                s = sc(b2)
                if metrics.better(comp.METRIC, s, best):
                    b, best, moved = b2, s, True
    return b


oof = np.zeros_like(O)
for k in range(comp.N_FOLDS):
    oof[fo == k] = apply(L[fo == k], fit(np.where((fo >= 0) & (fo != k))[0]))
p = fit(np.where(fo >= 0)[0])
hold = fo < 0
if hold.any():
    oof[hold] = apply(L[hold], p)
if not label_metric:
    print(f'temperature {p:.2f}')
elif len(p) == 2:
    print(f'threshold on P({classes()[1]}): {1 / (1 + np.exp(p[1] - p[0])):.3f} (0.5 before)')
else:
    print('biases: ' + ' '.join(f'{c}={v:+.1f}' for c, v in zip(classes(), p)))
(raw, raw_h), (new, new_h) = scores(O), scores(oof)
gain = (new - raw) * (1 if metrics.HIGHER_IS_BETTER.get(comp.METRIC, True) else -1)
print(f'{a.src} {raw:.4f} -> cross-fitted {new:.4f} ({new - raw:+.4f}): '
      + ('worth keeping.' if gain >= a.gate else f'below the {a.gate} gate, likely noise: keep {a.src}.')
      + ('' if new_h is None else f' Held-out rows: {raw_h:.4f} -> {new_h:.4f} ({new_h - raw_h:+.4f}).'))
save_preds(a.tag, oof, apply(Lt, p))
report(a.tag + ' (cross-fitted)', oof)

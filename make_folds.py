"""Fix the cross-validation folds once (folds.csv). Every model uses them, so out-of-fold predictions stack cleanly.
Stratified for classification; grouped when comp.GROUP is set (rows sharing a group never straddle folds).
--holdout sets a share of the rows aside as fold -1: no model, stack, blend or threshold ever trains on them, and every
model predicts them like test rows (through comp.simulate, the test simulator, when it is filled in). Their score is
the honest one: out-of-fold scores flatter pseudo-labels, noise cleaning and anything tuned on them.
  python make_folds.py --holdout 0.15
  python make_folds.py --holdout 0.15 --testlike    # hold out the most test-like rows (run adversarial.py first)"""
import argparse
from kit.common import train, y
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
import comp

ap = argparse.ArgumentParser()
ap.add_argument('--holdout', type=float, default=0.0, help='share of rows held out from everything, e.g. 0.15')
ap.add_argument('--testlike', action='store_true', help='hold out the rows adversarial.py found most test-like')
a = ap.parse_args()
n, K, S = len(train()), comp.N_FOLDS, comp.SEED
if hasattr(comp, 'fixed_folds'):  # the competition fixes the folds itself (e.g. one fold per month)
    fold = np.asarray(comp.fixed_folds(train()), int)
    pd.DataFrame({'fold': fold}).to_csv('folds.csv', index=False)
    print(f'folds.csv written from comp.fixed_folds, rows per fold: {np.bincount(fold[fold >= 0]).tolist()}')
    raise SystemExit(0)
yy, groups = y(), (train()[comp.GROUP].values if comp.GROUP else None)


def split(k, rows):
    """k (train, validation) splits of the given rows, as row numbers."""
    g = groups[rows] if comp.GROUP else None
    if comp.TASK == 'regression':
        s = GroupKFold(k).split(rows, groups=g) if comp.GROUP else KFold(k, shuffle=True, random_state=S).split(rows)
    elif comp.GROUP:
        s = StratifiedGroupKFold(k, shuffle=True, random_state=S).split(rows, yy[rows], g)
    else:
        s = StratifiedKFold(k, shuffle=True, random_state=S).split(rows, yy[rows])
    return [(rows[t], rows[v]) for t, v in s]


fold, rows = np.zeros(n, int), np.arange(n)
if a.holdout and a.testlike:
    like = pd.Series(np.load('preds/adv_testlikeness_train.npy'))
    if comp.GROUP:  # whole groups, most test-like first
        gl = like.groupby(groups).mean().sort_values(ascending=False)
        take = gl.index[np.cumsum(pd.Series(groups).value_counts()[gl.index].values) <= a.holdout * n]
        hold = np.where(np.isin(groups, take))[0]
    elif comp.TASK == 'regression':
        hold = like.sort_values(ascending=False).index[:int(a.holdout * n)].values
    else:  # the most test-like rows of every class, so the class mix stays
        hold = np.concatenate([like[yy == c].sort_values(ascending=False).index[:int(a.holdout * (yy == c).sum())].values
                               for c in np.unique(yy)])
    fold[hold] = -1
elif a.holdout:
    fold[split(round(1 / a.holdout), rows)[0][1]] = -1
rows = np.where(fold == 0)[0]
for k, (_, va) in enumerate(split(K, rows)):
    fold[va] = k
pd.DataFrame({'fold': fold}).to_csv('folds.csv', index=False)
print('folds.csv written, rows per fold:', np.bincount(fold[fold >= 0]).tolist(),
      f'held out (fold -1): {(fold < 0).sum()}' + (' most test-like' if a.testlike else '') if (fold < 0).any() else '')

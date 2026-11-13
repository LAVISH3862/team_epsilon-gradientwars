"""Time-stability check (IEEE-CIS 1st place): every feature on its own, trained on July, scored on September.
A single-feature LightGBM per feature predicts the 12 classes (labels from LABELS); the score is the mean one-vs-rest
AUC over the classes present. Features at or below --min (default 0.5) did not carry over from one month to a later
one: they are written, comma-joined, to logs/unstable_features.txt for train_gbm.py --drop.
  LABELS=preds/labels_c2.npy python time_check.py"""
import argparse
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import comp
from kit.common import train, y, folds, classes

ap = argparse.ArgumentParser()
ap.add_argument('--feats', default='row+conn+ring+flag')
ap.add_argument('--legit', type=int, default=60000, help='ordinary rows sampled per month')
ap.add_argument('--min', type=float, default=0.5)
a = ap.parse_args()
yy, fo, L = y(), folds(), list(classes()).index('legit')
rng = np.random.default_rng(0)
pick = lambda m: np.sort(np.r_[np.where((fo == m) & (yy != L))[0], rng.choice(np.where((fo == m) & (yy == L))[0], a.legit, replace=False)])
tr_rows, va_rows = pick(0), pick(2)
rows = np.r_[tr_rows, va_rows]
X = comp.features(train().iloc[rows], a.feats.split('+'))
Xa, Xb = X.iloc[:len(tr_rows)], X.iloc[len(tr_rows):]
ya, yb = yy[tr_rows], yy[va_rows]
out = []
for i, col in enumerate(X.columns):
    if X[col].nunique() < 2:
        out.append((col, np.nan)); continue
    m = lgb.LGBMClassifier(n_estimators=40, learning_rate=0.1, num_leaves=15, min_child_samples=50, verbose=-1, n_jobs=10)
    m.fit(Xa[[col]], ya)
    P = m.predict_proba(Xb[[col]])
    aucs = [roc_auc_score(yb == k, P[:, j]) for j, k in enumerate(m.classes_) if 0 < (yb == k).sum() < len(yb)]
    out.append((col, float(np.mean(aucs))))
    if i % 25 == 0:
        print(f'{i}/{X.shape[1]} done', flush=True)
R = pd.DataFrame(out, columns=['feature', 'auc_jul_to_sep']).sort_values('auc_jul_to_sep')
R.to_csv('logs/time_check.csv', index=False)
bad = R[(R.auc_jul_to_sep <= a.min) | R.auc_jul_to_sep.isna()].feature.tolist()
open('logs/unstable_features.txt', 'w').write(','.join(bad))
print(f'{len(bad)} of {len(R)} features at or below {a.min}:', bad)
print(R.head(25).to_string(index=False))

"""Training rows whose out-of-fold predictions confidently contradict their label: probably wrong labels. A model that
memorises them generalises worse. Each row's prediction comes from fold models that never trained on it, averaged over
a few tags. Writes the row numbers for --exclude (train_gbm.py, train_cnn.py), which leaves them out of training while
validation still scores every row. In the Deepfake hackathon (Oct 2026) two rounds gave +0.017 AUC: flag with the
first models, retrain without the flagged rows, re-flag with the better models, retrain again.
  python find_noisy.py gbm effb0 --out preds/noisy_r1.npy                          # round 1
  python train_gbm.py --tag gbm_c1 --balanced --exclude preds/noisy_r1.npy         # retrain without them
  python find_noisy.py gbm_c1 effb0_c1 --out preds/noisy_r2.npy                    # round 2, with the cleaned models
Classification flags a row when the averaged probability of its label is below 1 - thr (binary: |label - P| > thr);
regression flags a residual above --z robust standard deviations. Keep a round only if compare.py calls it a real gain."""
import argparse
from kit.common import train, y, folds, classes, score
import numpy as np, pandas as pd
import comp

ap = argparse.ArgumentParser()
ap.add_argument('tags', nargs='+', help='tags whose out-of-fold predictions are averaged')
ap.add_argument('--thr', type=float, default=0.9, help='classification: flag a row when P(its label) < 1 - thr')
ap.add_argument('--z', type=float, default=4.0, help='regression: flag a row when |residual| > z robust standard deviations')
ap.add_argument('--cap', type=float, default=0.2, help='flag at most this fraction of any class, most contradicted first')
ap.add_argument('--out', default='preds/noisy_idx.npy')
a = ap.parse_args()
yy, fo, reg = y(), folds(), comp.TASK == 'regression'
P = np.mean([np.load(f'preds/{t}_oof.npy') for t in a.tags], axis=0)  # out-of-fold only
if reg:
    r = yy - P
    bad, lim, group = np.abs(r) / (1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12), a.z, np.zeros(len(yy), int)
else:
    bad, lim, group = 1 - P[np.arange(len(yy)), yy], a.thr, yy  # bad = how strongly the models contradict the label
flag = np.zeros(len(yy), bool)
for g in np.unique(group):
    rows = np.where((group == g) & (fo >= 0))[0]   # held-out rows (fold -1) are never trained on, so never flagged
    cand = rows[bad[rows] > lim]
    flag[cand[np.argsort(-bad[cand])][:int(a.cap * len(rows))]] = True
idx = np.where(flag)[0]
np.save(a.out, idx)
cv = fo >= 0
print(f'out-of-fold {comp.METRIC} of the average {score(P[cv], np.where(cv)[0]):.4f}; {len(idx)} rows flagged '
      f'({len(idx) / cv.sum():.1%}), per fold {np.bincount(fo[idx], minlength=comp.N_FOLDS).tolist()}')
info = pd.DataFrame({'row': idx, comp.ID: train()[comp.ID].values[idx] if comp.ID in train() else idx,
                     'label': train()[comp.TARGET].values[idx]})
if reg:
    info['prediction'], info['robust_z'] = P[idx], bad[idx]
    print(f'flagged residuals: {np.round(np.sort(r[idx])[[0, -1]], 3).tolist() if len(idx) else "none"} (min, max)')
else:
    C = classes()
    info['models_say'], info['p_label'] = C[P[idx].argmax(1)], 1 - bad[idx]
    for c in range(len(C)):
        n, m = (yy[cv] == c).sum(), info.label.values == C[c]
        if m.any():
            print(f'  labelled {C[c]}: {m.sum()} of {n} flagged ({m.sum() / n:.1%}); the models say '
                  + ', '.join(f'{k} {v}' for k, v in info.models_say[m].value_counts().head(3).items()))
keep = np.setdiff1d(np.where(cv)[0], idx)
print(f'{comp.METRIC} on the unflagged rows {score(P[keep], keep):.4f} (optimistic: the same predictions chose them)')
info.sort_values('robust_z' if reg else 'p_label', ascending=not reg).to_csv(a.out[:-4] + '.csv', index=False)
print(f'wrote {a.out} and {a.out[:-4]}.csv (look at a few flagged rows before trusting them)')

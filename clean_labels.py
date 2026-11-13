"""Cleaned training labels from a model's out-of-fold probabilities (a tag trained with --balanced, before bias tuning).
About 2.3% of training rows carry a fraud label but look exactly like legit rows (a model cannot tell them from legit
rows at all, AUC 0.50): random label noise that the clean test set does not have. And some rows inside clear rings are
labelled legit (unreported fraud). This writes preds/labels_<name>.npy:
  - fraud-labelled rows the model sees as ordinary (P(legit) >= --noise) become legit;
  - legit-labelled rows the model is sure are fraud (P(legit) < --unreported) take the model's fraud type;
  - everything else keeps its label (a fraud row filed under another fraud type is left alone).
Out-of-fold probabilities come from models that never saw the row, so a label is never used to clean itself.
Use: LABELS=preds/labels_c1.npy python train_gbm.py ...   (kit.common.y() then returns the cleaned labels)
  python clean_labels.py conn --name c1"""
import argparse
from kit.common import classes, load_preds
import kit.common as K
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('teacher')
ap.add_argument('--name', required=True)
ap.add_argument('--noise', type=float, default=0.08, help='fraud label -> legit when P(legit) is at least this')
ap.add_argument('--unreported', type=float, default=0.02, help='legit label -> model fraud type when P(legit) is below this')
a = ap.parse_args()
P, _ = load_preds(a.teacher)
y0, c = K.y_given(), classes()
L = list(c).index('legit')
pl = P[:, L]
top = np.where(np.arange(len(c))[None, :] == L, -1, P).argmax(1)
y1 = y0.copy()
noise = (y0 != L) & (pl >= a.noise)
unrep = (y0 == L) & (pl < a.unreported)
y1[noise], y1[unrep] = L, top[unrep]
np.save(f'preds/labels_{a.name}.npy', y1)
print(f'labels_{a.name}: {noise.sum()} fraud labels -> legit, {unrep.sum()} legit labels -> fraud type, of {len(y0)} rows')
print(pd.DataFrame({'given': np.bincount(y0, minlength=len(c)), 'cleaned': np.bincount(y1, minlength=len(c))}, index=c).to_string())

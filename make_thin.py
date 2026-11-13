"""Test simulator: a half-density copy of train. October has half as many ordinary payments per day as the training
months (8,400 against 17,000; rows per sender match a 50% thinning of September exactly), while rings are intact. This
picks half of the ordinary training rows to drop (preds/thin_rows.npy): rows that are legit after cleaning and that the
teacher is sure about (P(legit) >= 0.99), so no ring row is removed. THIN=1 then makes features.py build, and
comp.features read, the feature blocks of the thinned copy (data/feat_<block>_thin.parquet).
  python make_thin.py ring_c1 --labels preds/labels_c1.npy"""
import argparse
from kit.common import classes, load_preds
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('teacher')
ap.add_argument('--labels', required=True)
ap.add_argument('--share', type=float, default=0.5)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
P, _ = load_preds(a.teacher)
L = list(classes()).index('legit')
ordinary = np.where((np.load(a.labels) == L) & (P[:, L] >= 0.99))[0]
drop = np.sort(np.random.default_rng(a.seed).choice(ordinary, int(a.share * len(ordinary)), replace=False))
np.save('preds/thin_rows.npy', drop)
print(f'thin_rows: dropping {len(drop)} of {len(P)} training rows ({len(ordinary)} ordinary rows, share {a.share})')

"""Does the density shift hurt? October has half the ordinary payments per day, with rings intact. Train on July +
August and predict the September rows that survive the thinning (make_thin.py), three ways:
  full->full   everything at full density                                    (what validation has measured so far)
  full->thin   only September thinned: train at full density, predict at half  (what we do to the test set now)
  thin->thin   every training month thinned: train and predict at half density (train like the test)
Scores are macro-F1 on the labels in use (LABELS), on the same rows every time. Thinned-out ordinary rows are made up
for by counting each surviving ordinary row twice, so the class mix is September's. Per-class biases are tuned on one
random half of the rows and scored on the other.
  python features.py row conn ring                      (full density, already there)
  THIN=1 python features.py row conn ring ; THIN=sep python features.py row conn ring
  LABELS=preds/labels_c1.npy python sim_density.py"""
import numpy as np, pandas as pd, lightgbm as lgb, pyarrow.parquet as pq
import comp
from kit.common import train, y, folds, classes
from kit import metrics

BLOCKS, CAP = ['row', 'conn', 'ring'], 150000
tr, yy, fo, C = train(), y(), folds(), len(classes())
ids, L = tr[comp.ID].values, list(classes()).index('legit')
thin = np.zeros(len(tr), bool)
thin[np.load('preds/thin_rows.npy')] = True
va = np.where((fo == 2) & ~thin)[0]                       # the September rows scored in every setup
ordinary = (yy == L) & (np.load('preds/ring_c1_oof.npy')[:, L] >= 0.99)   # the rows make_thin.py could drop
w_va = np.where(ordinary[va], 2.0, 1.0)


def sample(pool):
    """Training rows: at most CAP per class."""
    rng = np.random.default_rng(comp.SEED)
    return np.sort(np.concatenate([r if len(r) <= CAP else rng.choice(r, CAP, replace=False)
                                   for r in (pool[yy[pool] == c] for c in range(C))]))


def load(suffix, kept, rows):
    """Features of the training rows `rows` (sorted) from the copy `suffix`, whose train rows are those in `kept`.
    Read one row group at a time: the laptop has no room for a whole feature table next to a model."""
    pos, out = np.cumsum(kept)[rows] - 1, []
    for b in BLOCKS:
        f, parts, start = pq.ParquetFile(f'data/feat_{b}{suffix}.parquet'), [], 0
        for g in range(f.num_row_groups):
            t = f.read_row_group(g)
            parts.append(t.take(pos[(pos >= start) & (pos < start + t.num_rows)] - start).to_pandas())
            start += t.num_rows
        out.append(pd.concat(parts))
    X = pd.concat(out, axis=1)
    assert (X.index.values == ids[rows]).all(), 'feature rows do not line up with the training rows'
    return X.reset_index(drop=True)


def fit(X, yv):
    w = (len(yv) / (C * np.maximum(np.bincount(yv, minlength=C), 1)))[yv]
    return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.6, verbose=-1, n_jobs=10, random_state=comp.SEED).fit(X, yv, sample_weight=w)


def f1s(yt, pred, w):
    cm = np.bincount(yt * C + pred, weights=w, minlength=C * C).reshape(C, C)
    tp = np.diag(cm)
    return 2 * tp / np.maximum(2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp), 1e-9)


def tuned(P):
    """Per-class F1 with biases tuned on one half of the rows and applied to the other, both ways."""
    Lg, half, pred = np.log(np.clip(P, 1e-6, 1)), np.random.default_rng(0).random(len(P)) < 0.5, np.zeros(len(P), int)
    for a, b in ((half, ~half), (~half, half)):
        bias, best, moved = np.zeros(C), -1.0, True
        while moved:
            moved = False
            for c in range(C):
                for v in np.arange(-8, 8.01, 0.25):
                    b2 = bias.copy(); b2[c] = v
                    s = f1s(yy[va][a], (Lg[a] + b2).argmax(1), w_va[a]).mean()
                    if s > best + 1e-9:
                        bias, best, moved = b2, s, True
        pred[b] = (Lg[b] + bias).argmax(1)
    return f1s(yy[va], pred, w_va)


everything = np.ones(len(tr), bool)
rf, rt = sample(np.where(fo < 2)[0]), sample(np.where((fo < 2) & ~thin)[0])
setups = {'full->full': ('', everything, rf), 'full->thin': ('_thinsep', ~(thin & (fo == 2)), rf), 'thin->thin': ('_thin', ~thin, rt)}
res = {}
for name, (suffix, kept, rows) in setups.items():
    m = fit(load(suffix, kept, rows), yy[rows])
    P = m.predict_proba(load(suffix, kept, va))
    np.save(f'preds/sim_{name.replace("->", "_")}.npy', P.astype(np.float32))
    f = tuned(P)
    res[name] = [f.mean()] + list(f)
    print(f'{name}: macro-F1 {f.mean():.4f}', flush=True)
print(pd.DataFrame(res, index=['macro_f1'] + list(classes())).round(3).to_string())

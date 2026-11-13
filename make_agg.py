"""Stage-2 features: a teacher's class probabilities spread over the entities a row belongs to.
Every member of a ring shares its type, so a row whose receiver, sender or phone has other rows the teacher calls
merchant_collusion is more likely merchant_collusion itself. For each entity (receiver, sender, phone, the
sender-receiver pair) this writes the mean probability of every class over the entity's OTHER rows (the row itself
left out) within 24 hours and within the same month, the highest fraud probability among them, and how many there are.
Training rows use the teacher's out-of-fold probabilities and only neighbours from the same month, which the same fold
model predicted; test rows use the teacher's test probabilities. Writes data/feat_agg.parquet.
  python make_agg.py flag_c2"""
import sys, time
import numpy as np, pandas as pd
import comp, features as F
from kit.common import classes, load_preds, folds

t0 = time.time()
teacher = sys.argv[1]
O, T = load_preds(teacher)
P = np.vstack([O, T]).astype(np.float64)
C, L = len(classes()), list(classes()).index('legit')
d = F.both()
snd, rcv, dev, ts = F.ids(d)
month = np.r_[folds(), np.full(len(T), 3)]           # 0-2 training months, 3 the test month
pf = 1 - P[:, L]
X = {}
for name, key in (('r', rcv), ('s', snd), ('p', dev), ('pair', snd * 2_000_000 + rcv)):
    k = key * 4 + month                              # neighbours only within the same month
    E = F.Events(k, ts)
    for wname, (lo, hi) in (('24h', (-F.D, F.D)), ('mon', (-F.ALL, F.ALL))):
        n = E.win(k, ts, lo, hi) - 1
        X[f'agg_{name}_{wname}_n'] = n
        for j, cn in enumerate(classes()):
            if j == L:
                continue
            X[f'agg_{name}_{wname}_{cn}'] = np.where(n > 0, (E.win(k, ts, lo, hi, P[:, j]) - P[:, j]) / np.maximum(n, 1), np.nan)
        X[f'agg_{name}_{wname}_fraud'] = np.where(n > 0, (E.win(k, ts, lo, hi, pf) - pf) / np.maximum(n, 1), np.nan)
        X[f'agg_{name}_{wname}_n_fraud'] = E.win(k, ts, lo, hi, (pf >= 0.5).astype(np.float64)) - (pf >= 0.5)
X = pd.DataFrame(X, index=d[comp.ID].values).astype(np.float32)
X.to_parquet('data/feat_agg.parquet')
print(f'feat_agg: {X.shape[1]} features from {teacher}, {time.time() - t0:.0f}s')

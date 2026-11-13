"""Where does October hide fraud we call legit? Compare the rows the model calls legit in September (thinned to
October's density, features from the THIN=sep copy) with those it calls legit in October. A classifier that can tell
them apart points at what changed; the most October-like rows are the suspects."""
import numpy as np, pandas as pd, lightgbm as lgb, pyarrow.parquet as pq
import comp
from kit.common import train, test, folds, classes
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BLOCKS = ['row', 'conn', 'ring']
DROP = ['sender_account_age_days', 'r_account_age_days', 'm_age_days', 'weekday', 'hour', 'device_age_days', 'sim_age_hours',
        'p_since_first', 'sp_since_first', 'pair_since_first', 'qr_vpa_mismatch']
tr, te, fo = train(), test(), folds()
L = list(classes()).index('legit')
thin = np.zeros(len(tr), bool); thin[np.load('preds/thin_rows.npy')] = True
kept_sep = ~(thin & (fo == 2))
O, T = np.load('preds/flag_c1_thr_oof.npy').argmax(1), np.load('preds/flag_c1_thr_test.npy').argmax(1)
a = np.where((fo == 2) & ~thin & (O == L))[0]                 # September rows called legit, at October's density
b = np.where(T == L)[0]
pos_a, pos_b = np.cumsum(kept_sep)[a] - 1, kept_sep.sum() + b
out = []
for blk in BLOCKS:
    f, parts, start = pq.ParquetFile(f'data/feat_{blk}_thinsep.parquet'), [], 0
    pos = np.sort(np.r_[pos_a, pos_b])
    for g in range(f.num_row_groups):
        t = f.read_row_group(g)
        parts.append(t.take(pos[(pos >= start) & (pos < start + t.num_rows)] - start).to_pandas()); start += t.num_rows
    out.append(pd.concat(parts))
X = pd.concat(out, axis=1)
ids = np.r_[tr[comp.ID].values[a], te[comp.ID].values[b]]
assert (X.index.values == ids).all()
X = X.drop(columns=[c for c in DROP if c in X]).reset_index(drop=True)
# only features that mean the same anywhere on the timeline: no all-time totals, spans, first-seen or edge gaps
bad = [c for c in X if c.endswith('_all') or '_since_' in c or c.startswith(('sp_', 'pr_', 'rs_')) or '_gap' in c or c.endswith(('_7d', '_30d', '_dur', '_pos'))
       or c in ('s_amount_vs_median', 'sp_share_of_sender', 'day_of_month')]
X = X.drop(columns=bad)
print(len(X.columns), 'features kept')
yv = np.r_[np.zeros(len(a)), np.ones(len(b))]
p = np.zeros(len(yv))
for trn, val in StratifiedKFold(3, shuffle=True, random_state=0).split(X, yv):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=10).fit(X.iloc[trn], yv[trn])
    p[val] = m.predict_proba(X.iloc[val])[:, 1]
print(f'September (thinned) vs October among rows called legit: AUC {roc_auc_score(yv, p):.4f}  rows {len(a)} vs {len(b)}')
print(pd.Series(m.feature_importances_, X.columns).sort_values(ascending=False).head(25).to_string())
np.save('preds/hunt_p.npy', p.astype(np.float32)); np.save('preds/hunt_rows.npy', np.r_[a, -b - 1])
base = yv.mean()
for thr in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    k = p >= thr
    print(f'score >= {thr}: {int(k.sum())} rows, October share {yv[k].mean():.3f} (base {base:.3f}); excess October rows over a Sep-like mix: {yv[k].sum() - (1 - yv[k]).sum() * base / (1 - base):.0f}')

"""Entity label history: what the training labels say about the same accounts, phones and merchants in OTHER months.
For a training row of month m only the labels of the other two months are used (cross-fitted, no row sees its own
month's labels); a test row uses all three training months. Per entity (sender, receiver, phone, sender-receiver pair):
rows seen, share of rows labelled fraud as given (includes the ordinary-looking fraud labels), share labelled each
fraud type after cleaning. Writes data/feat_hist.parquet.
  python make_hist.py"""
import numpy as np, pandas as pd
import comp
from kit.common import train, test, classes, folds, y_given

tr, te = train(), test()
c = list(classes()); L = c.index('legit')
yg, yc, fo = y_given(), np.load('preds/labels_c1.npy'), folds()
lab = pd.DataFrame({'m': fo, 'given_fraud': (yg != L).astype(float)})
for k in range(len(c)):
    if k != L:
        lab[c[k]] = (yc == k).astype(float)
cols = list(lab.columns[1:])
out_tr = {}; out_te = {}
for key in ('sender_id', 'receiver_id', 'device_id', 'pair'):
    kt = (tr.sender_id + '>' + tr.receiver_id) if key == 'pair' else tr[key]
    ke = (te.sender_id + '>' + te.receiver_id) if key == 'pair' else te[key]
    per = lab.assign(k=kt.values).groupby(['k', 'm'])[cols].agg(['sum']).droplevel(1, axis=1)
    per['n'] = lab.assign(k=kt.values).groupby(['k', 'm']).size()
    tot = per.groupby(level=0).sum()
    # training rows: totals minus their own month
    own = per.reindex(pd.MultiIndex.from_arrays([kt.values, fo])).values
    t_all = tot.reindex(kt.values).values
    other = t_all - own
    n = other[:, -1:]
    rate = np.where(n > 0, other[:, :-1] / np.maximum(n, 1), np.nan)
    nm = 'pair' if key == 'pair' else key.split('_')[0]
    out_tr[f'h_{nm}_n'] = n[:, 0]
    for j, cn in enumerate(cols):
        out_tr[f'h_{nm}_{cn}'] = rate[:, j]
    te_all = tot.reindex(ke.values).values; ne = te_all[:, -1:]
    rate_e = np.where(ne > 0, te_all[:, :-1] / np.maximum(ne, 1), np.nan)
    out_te[f'h_{nm}_n'] = np.nan_to_num(ne[:, 0])
    for j, cn in enumerate(cols):
        out_te[f'h_{nm}_{cn}'] = rate_e[:, j]
    out_tr[f'h_{nm}_n'] = np.nan_to_num(out_tr[f'h_{nm}_n'])
X = pd.concat([pd.DataFrame(out_tr, index=tr[comp.ID].values), pd.DataFrame(out_te, index=te[comp.ID].values)]).astype(np.float32)
X.to_parquet('data/feat_hist.parquet')
print('feat_hist:', X.shape)

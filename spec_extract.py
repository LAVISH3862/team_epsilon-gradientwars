"""Pull the rows the fraud-type specialist needs out of the sub4 feature blocks, column by column (low memory):
train rows that are fraud in labels c1 or that sub4's out-of-fold rule calls fraud, and test rows sub4 calls fraud.
Writes cloud/spec/X.parquet (those rows, sub4's 257 features) and cloud/spec/meta.npz (labels, folds, sub4 calls).
  python spec_extract.py"""
import os
import numpy as np, pandas as pd, pyarrow.parquet as pq

out = 'cloud/spec'
os.makedirs(out, exist_ok=True)
lab = pd.read_parquet('data/train.parquet', columns=['txn_id', 'label'])
ids_tr, n_tr = lab['txn_id'].values, len(lab)
classes = np.array(sorted(lab['label'].unique()))
y_given = np.searchsorted(classes, lab['label'].values).astype(np.int8)
del lab
y_c1, y_t1 = np.load('preds/labels_c1.npy'), np.load('preds/labels_t1.npy')
fo = pd.read_csv('folds.csv')['fold'].values.astype(np.int8)
O, T = np.load('preds/flag_c1_thr_oof.npy'), np.load('preds/flag_c1_thr_test.npy')
s1_oof, s1_test = O.argmax(1).astype(np.int8), T.argmax(1).astype(np.int8)
L = list(classes).index('legit')
tr_rows = np.where((y_c1 != L) | (s1_oof != L))[0]
te_rows = np.where(s1_test != L)[0]
pos = np.concatenate([tr_rows, n_tr + te_rows])
print(len(tr_rows), 'train rows,', len(te_rows), 'test rows')

parts = []
for b in ['row', 'conn', 'ring', 'flag']:
    f = pq.ParquetFile(f'data/feat_{b}.parquet')
    assert f.metadata.num_rows == n_tr + len(s1_test)
    cols = [c for c in f.schema_arrow.names if c not in ('__index_level_0__', 'qr_vpa_mismatch')]
    if b == 'row':
        idx = f.read(columns=['__index_level_0__']).column(0).to_numpy()
        if ids_tr is not None:
            assert (idx[:n_tr] == ids_tr).all()
        ids = idx[pos]
    for k in range(0, len(cols), 16):
        t = f.read(columns=cols[k:k + 16]).take(pos)
        parts.append(t.to_pandas())
        del t
    print(b, len(cols), flush=True)
X = pd.concat(parts, axis=1)
X.index = ids
X.to_parquet(f'{out}/X.parquet')
np.savez(f'{out}/meta.npz', tr_rows=tr_rows, te_rows=te_rows, y_c1=y_c1, y_t1=y_t1,
         y_given=y_given, fo=fo, classes=classes,
         s1_oof=s1_oof, s1_test=s1_test, O=O[tr_rows], T=T[te_rows])
print(X.shape)

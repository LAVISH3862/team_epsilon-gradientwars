"""Binary is_fraud model (any fraud label vs legit, test-like labels t1), month folds, plus amount / account age.
Does it rank fraud above legit among the rows the 12-class model (sub4) calls legit?"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
import comp
from kit.common import train, classes, folds
tr, fo = train(), folds(); L = list(classes()).index('legit')
yt = np.load('preds/labels_t1.npy'); y = (yt != L).astype(int)
X = comp.features(tr, ['row', 'conn', 'ring', 'flag']).drop(columns=['qr_vpa_mismatch'])
X['amount_per_age_day'] = tr.amount.values / (tr.sender_account_age_days.values + 1)
X['amount_per_age_day_bal'] = X.amount_per_age_day / (tr.balance_before.values + 1)
o = np.load('preds/flag_c1_thr_oof.npy').argmax(1)
p = np.zeros(len(y)); rng = np.random.default_rng(0)
for k in range(3):
    tr_rows = np.where(fo != k)[0]
    keep = tr_rows[(y[tr_rows] == 1) | (rng.random(len(tr_rows)) < 0.25)]
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, verbose=-1, n_jobs=10).fit(X.iloc[keep], y[keep])
    p[fo == k] = m.predict_proba(X[fo == k])[:, 1]
np.save('preds/binary_oof.npy', p.astype(np.float32))
print(f'binary is_fraud AUC, all rows: {roc_auc_score(y, p):.4f}')
lg = o == L
print(f'AUC among rows sub4 calls legit (can it find the fraud sub4 misses?): {roc_auc_score(y[lg], p[lg]):.4f} | fraud share there {y[lg].mean():.4f}')
best = max((f1_score(y, p >= t), t) for t in np.linspace(0.05, 0.95, 19))
print(f'binary F1 at best threshold {best[1]:.2f}: {best[0]:.4f} | the 12-class model as a binary flag: {f1_score(y, (o != L).astype(int)):.4f}')
q = p[lg]; yl = y[lg]
for share in (0.001, 0.005, 0.01, 0.02):
    t = np.quantile(q, 1 - share); sel = q >= t
    print(f'top {share * 100:.1f}% of sub4-legit rows by binary score: fraud precision {yl[sel].mean():.4f} ({sel.sum()} rows); base rate {yl.mean():.4f}')
imp = pd.Series(m.feature_importances_, X.columns); print('importance rank of amount_per_age_day:', int((imp > imp['amount_per_age_day']).sum()) + 1, 'of', len(imp))

"""Phishing-family type from the victim's history: a row the base calls phishing_collect, fake_cashback or
loan_app_extortion takes the type of the victim's ring rows in other months when those are all one family type.
Victim = receiver of a Rs1-10 bait payment, otherwise the sender. Out-of-fold rows use only the other months' labels;
test rows use all training months.
  python history_rule.py --base blk_sp1440_dr --tag blk_sp1440_dr_hist"""
import argparse
import numpy as np, pandas as pd
from kit.common import train, test, classes, folds, load_preds, save_preds, report

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True)
ap.add_argument('--tag', required=True)
a = ap.parse_args()
cls = list(classes()); L = cls.index('legit')
FAM = [cls.index(c) for c in ('phishing_collect', 'fake_cashback', 'loan_app_extortion')]
t1, c1 = np.load('preds/labels_t1.npy'), np.load('preds/labels_c1.npy')   # test-like labels; cleaned labels (ring rows)


def victim(df):
    bait = (df.amount.values <= 10) & (df.txn_type.values == 'PAY') & (df.receiver_type.values == 'P2P')
    return np.where(bait, df.receiver_id.values, df.sender_id.values)


tr, te, fo = train(), test(), folds()
hist = np.isin(t1, FAM) & (c1 != L)
H = pd.DataFrame({'v': victim(tr)[hist], 'y': t1[hist], 'f': fo[hist]})


def pure(h):
    g = h.groupby('v').y.agg(['nunique', 'first'])
    return g['first'][g['nunique'] == 1]


def apply(P, v, month=None):
    p0 = P.argmax(1); p = p0.copy(); cand = np.isin(p0, FAM)
    for k in ([None] if month is None else range(3)):
        pt = pure(H if k is None else H[H.f != k])
        m = cand & pd.Series(v).isin(pt.index).values & (True if k is None else month == k)
        p[m] = pt.reindex(v[m]).values
    out = np.zeros_like(P); out[np.arange(len(p)), p] = 1
    return out, int((p != p0).sum())


O, T = load_preds(a.base)
Oo, no = apply(O, victim(tr), fo)
To, nt = apply(T, victim(te))
save_preds(a.tag, Oo, To)
print(f'rows changed: train {no}, test {nt}')
report(a.tag, Oo)

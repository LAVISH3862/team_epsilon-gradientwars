"""Takeover vs SIM swap from the drain arithmetic: inside a multi-row drain, each step's balance is 0.35 of the
previous one in takeover-generated events and 0.25 in SIM-swap ones. Applied to events the base calls takeover or
SIM swap, off the phone blocks (block rows keep their block type).
  LABELS=preds/labels_c1.npy python drain_rule.py --base blk_sp --tag blk_dr"""
import argparse
import numpy as np, pandas as pd
from kit.common import train, test, classes, load_preds, save_preds, report
ap = argparse.ArgumentParser(); ap.add_argument('--base', default='blk_sp'); ap.add_argument('--tag', required=True); a = ap.parse_args()
cls = list(classes()); ATO, SIM = cls.index('account_takeover'), cls.index('sim_swap')

def apply(df, P, blk):
    p = P.argmax(1).copy()
    m = np.isin(p, [ATO, SIM]) & (blk < 0)
    x = pd.DataFrame({'row': np.flatnonzero(m), 's': df.sender_id.values[m], 'ts': pd.to_datetime(df.timestamp.values[m]), 'bal': df.balance_before.values[m]}).sort_values(['s', 'ts'])
    new = (x.s != x.s.shift()) | ((x.ts - x.ts.shift()).dt.total_seconds() > 7200); x['ev'] = new.cumsum()
    x['r'] = x.bal / x.groupby('ev').bal.shift(1)
    med = x.groupby('ev').r.median(); x['rm'] = x.ev.map(med)
    q35 = x.rm.between(0.33, 0.37); q25 = x.rm.between(0.23, 0.27)
    p[x.row[q35].values] = ATO; p[x.row[q25].values] = SIM
    out = np.zeros_like(P); out[np.arange(len(p)), p] = 1
    return out, int((p != P.argmax(1)).sum())

O, T = load_preds(a.base)
Oo, no = apply(train(), O, np.load('preds/blocks_train.npy')); To, nt = apply(test(), T, np.load('preds/blocks_test.npy'))
save_preds(a.tag, Oo, To); print('rows changed: train', no, 'test', nt); report(a.tag, Oo)

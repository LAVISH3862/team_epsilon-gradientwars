"""Block types along the same receiver over a week, for merchant_collusion, velocity_bot and mule_account.
A receiver that takes a ring burst from a block phone of one of these types (and the base model calls that row
fraud) keeps that type for its other bursts within --days: rows the base model calls fraud, off the phone blocks,
of a look-alike type, paid to the same receiver, take the block's type (nearest block row in time wins).
  LABELS=preds/labels_c1.npy python receiver_rule.py --base blk1440_post --blocks _post --tag blk1440_post_rcv"""
import argparse
import numpy as np, pandas as pd
from kit.common import train, test, classes, load_preds, save_preds, report

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='blk1440_post')
ap.add_argument('--blocks', default='_post', help='suffix of the block files from device_blocks.py --out')
ap.add_argument('--first', default='flag_c1_thr', help='model whose fraud / legit calls choose the rows')
ap.add_argument('--days', type=float, default=7)
ap.add_argument('--tag', required=True)
a = ap.parse_args()
cls = list(classes()); L = cls.index('legit'); C = len(cls)
FAM = [{'mule_account', 'layering_chain', 'velocity_bot'}, {'sim_swap', 'account_takeover'},
       {'phishing_collect', 'fake_cashback', 'loan_app_extortion'}, {'qr_swap', 'merchant_collusion', 'refund_abuse'}]
alike = np.zeros((C, C), bool)
for f in FAM:
    for x in f:
        for y in f: alike[cls.index(x), cls.index(y)] = True
TYPES = [cls.index(t) for t in ('merchant_collusion', 'velocity_bot', 'mule_account')]


def apply(df, P, P0, blk):
    p = P.argmax(1).copy(); o = P0.argmax(1); p_in = p.copy()
    ts = pd.to_datetime(df['timestamp']).astype('int64').values // 10**9
    r = df['receiver_id'].values
    sd = np.flatnonzero((blk >= 0) & (o != L) & np.isin(blk, TYPES))
    fr = np.flatnonzero((o != L) & (blk < 0))
    f = pd.DataFrame({'k': r[fr], 'ts': ts[fr], 'row': fr, 'pred': p[fr]})
    S = pd.DataFrame({'k': r[sd], 'ts_s': ts[sd], 'b': blk[sd].astype(int)})
    mg = f.merge(S, on='k')
    mg = mg[(mg.ts - mg.ts_s).abs() <= a.days * 86400]
    mg = mg[alike[mg.pred.values, mg.b.values] & (mg.pred.values != mg.b.values)]
    mg = mg.assign(dt=(mg.ts - mg.ts_s).abs()).sort_values('dt').drop_duplicates('row')
    p[mg.row.values] = mg.b.values
    out = np.zeros_like(P); out[np.arange(len(p)), p] = 1
    return out, int((p != p_in).sum())


O, T = load_preds(a.base)
O0, T0 = load_preds(a.first)
Oo, no = apply(train(), O, O0, np.load(f'preds/blocks_train{a.blocks}.npy'))
To, nt = apply(test(), T, T0, np.load(f'preds/blocks_test{a.blocks}.npy'))
save_preds(a.tag, Oo, To)
print(f'rows changed: train {no}, test {nt}')
report(a.tag, Oo)

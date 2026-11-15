"""Apply the extra-phone blocks (device_blocks.py) to a base model's calls and save the result as a new tag.
Rule 1: a row on a block phone gets the block's fraud type. Rule 2 (--spread MINUTES): rows the base model already
calls fraud, of a look-alike type, that share the sender or receiver account with a block row within that time, take
the block's type too (same event).
  python apply_blocks.py --base flag_c1_thr --tag blk
  python apply_blocks.py --base flag_c1_thr --tag blk_sp --spread 360"""
import argparse
import numpy as np, pandas as pd
from kit.common import train, test, classes, load_preds, save_preds, report

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='flag_c1_thr')
ap.add_argument('--tag', required=True)
ap.add_argument('--spread', type=int, default=0, help='minutes; 0 = rule 1 only')
ap.add_argument('--blocks', default='', help='suffix of the block files from device_blocks.py --out')
a = ap.parse_args()
cls = list(classes()); L = cls.index('legit'); C = len(cls)
FAM = [{'mule_account', 'layering_chain', 'velocity_bot'}, {'sim_swap', 'account_takeover'},
       {'phishing_collect', 'fake_cashback', 'loan_app_extortion'}, {'qr_swap', 'merchant_collusion', 'refund_abuse'}]
alike = np.zeros((C, C), bool)
for f in FAM:
    for x in f:
        for y in f: alike[cls.index(x), cls.index(y)] = True


def apply(df, P, blk):
    p0 = P.argmax(1); p = p0.copy(); inb = blk >= 0
    p[inb] = blk[inb]
    if a.spread:
        ts = pd.to_datetime(df['timestamp']).astype('int64').values // 10**9
        fr = np.flatnonzero((p0 != L) & ~inb)
        f = pd.DataFrame({'s': df['sender_id'].values[fr], 'r': df['receiver_id'].values[fr], 'p2p': df['receiver_type'].values[fr] == 'P2P', 'ts': ts[fr], 'row': fr, 'pred': p0[fr]})
        sd = pd.DataFrame({'s': df['sender_id'].values[inb], 'r': df['receiver_id'].values[inb], 'p2p': df['receiver_type'].values[inb] == 'P2P', 'ts_s': ts[inb], 'b': blk[inb]})
        for ks, kf in (('s', 's'), ('r', 'r'), ('r', 's'), ('s', 'r')):
            s_ = sd if ks == 's' else sd[sd.p2p]; f_ = f if kf == 's' else f[f.p2p]
            mg = f_.merge(s_[[ks, 'ts_s', 'b']].rename(columns={ks: 'k'}), left_on=kf, right_on='k')
            mg = mg[(mg.ts - mg.ts_s).abs() <= a.spread * 60]
            mg = mg[alike[mg.pred.values, mg.b.values]].drop_duplicates('row')
            p[mg.row.values] = mg.b.values
    out = np.zeros_like(P); out[np.arange(len(p)), p] = 1
    return out, int((p != p0).sum())


O, T = load_preds(a.base)
Oo, no = apply(train(), O, np.load(f'preds/blocks_train{a.blocks}.npy'))
To, nt = apply(test(), T, np.load(f'preds/blocks_test{a.blocks}.npy'))
save_preds(a.tag, Oo, To)
print(f'rows changed: train {no}, test {nt}')
report(a.tag, Oo)

"""Merchant-side rows the model calls legit: September thinned to October's density (THIN=sep features) vs October."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
import comp
from kit.common import train, test, folds, classes
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
tr, te, fo = train(), test(), folds(); L = list(classes()).index('legit')
thin = np.zeros(len(tr), bool); thin[np.load('preds/thin_rows.npy')] = True
kept = ~(thin & (fo == 2)); a = np.where((fo == 2) & ~thin)[0]
O, T = np.load('preds/flag_c1_thr_oof.npy').argmax(1), np.load('preds/flag_c1_thr_test.npy').argmax(1)
cols = ['pair_n_1h', 'pair_n_24h', 'pair_n_7d', 'pair_sum_24h', 'pair_round_24h', 'pair_refunds_30d', 's_out_n_1m', 's_out_n_10m', 's_out_n_1h', 's_out_n_24h', 's_receivers_1h', 's_receivers_24h',
        's_refunds_7d', 's_refunds_30d', 'p_n_1m', 'p_n_10m', 'p_n_1h', 'p_n_24h', 'p_senders_24h', 'p_receivers_10m', 'r_in_n_10m', 'r_in_n_1h', 'r_in_n_24h', 'r_senders_1h', 'r_senders_24h', 'r_round_n_24h', 'r_whole_share_24h', 'r_round_share_24h']
F = pq.read_table('data/feat_conn_thinsep.parquet', columns=cols).to_pandas()
pa_, pb_ = np.cumsum(kept)[a] - 1, kept.sum() + np.arange(len(te))
def frame(d, X, pred):
    d = d[['amount', 'txn_type', 'receiver_type', 'merchant_category', 'login_to_txn_sec', 'device_id', 'sender_id', 'timestamp', 'new_payee', 'receiver_id']].reset_index(drop=True)
    d = pd.concat([d, X.reset_index(drop=True)], axis=1); d['legit'] = pred == L
    d['whole'] = d.amount == d.amount.round(); d['round100'] = d.amount % 100 == 0; d['round500'] = d.amount % 500 == 0
    d['extra'] = d.device_id.str[3:].astype(int) >= 420000; hr = d.timestamp.str[11:13].astype(int); d['night'] = (hr >= 23) | (hr < 5)
    d['fast'] = d.login_to_txn_sec <= 8; d['eg'] = d.merchant_category.isin(['ecommerce', 'gaming']); d['refund'] = d.txn_type == 'REFUND'
    return d[d.receiver_type == 'MERCHANT']
A = frame(tr.iloc[a], F.iloc[pa_], O[a]); B = frame(te, F.iloc[pb_], T)
w = np.where(A.legit, 2.0, 1.0)   # thinned-out ordinary rows: each survivor counts twice
NA, NB = (fo == 2).sum(), len(te)
tests = {
 'pair >=2 in 1h': lambda d: d.pair_n_1h >= 2, 'pair >=3 in 24h': lambda d: d.pair_n_24h >= 3, 'pair >=2 in 24h': lambda d: d.pair_n_24h >= 2, 'pair >=4 in 7d': lambda d: d.pair_n_7d >= 4, 'pair >=3 in 7d': lambda d: d.pair_n_7d >= 3,
 'pair >=2 in 24h & round100': lambda d: (d.pair_n_24h >= 2) & d.round100, 'pair >=2 in 24h & whole': lambda d: (d.pair_n_24h >= 2) & d.whole, 'pair >=2 in 7d & round500': lambda d: (d.pair_n_7d >= 2) & d.round500,
 'round500 to merchant': lambda d: d.round500, 'round100 to merchant': lambda d: d.round100, 'whole to merchant': lambda d: d.whole, 'whole & eg': lambda d: d.whole & d.eg, 'whole & eg & fast': lambda d: d.whole & d.eg & d.fast,
 'sender >=3 in 10m': lambda d: d.s_out_n_10m >= 3, 'sender >=2 in 1m': lambda d: d.s_out_n_1m >= 2, 'sender >=4 in 1h': lambda d: d.s_out_n_1h >= 4, 'sender >=6 in 24h': lambda d: d.s_out_n_24h >= 6,
 'phone >=2 in 1m': lambda d: d.p_n_1m >= 2, 'phone >=3 in 10m': lambda d: d.p_n_10m >= 3, 'phone >=4 in 1h': lambda d: d.p_n_1h >= 4, 'phone >=2 senders in 24h': lambda d: d.p_senders_24h >= 2,
 'fast & eg': lambda d: d.fast & d.eg, 'fast & eg & night': lambda d: d.fast & d.eg & d.night, 'login<=4 & merchant': lambda d: d.login_to_txn_sec <= 4, 'login<=4 & eg': lambda d: (d.login_to_txn_sec <= 4) & d.eg,
 'extra phone & eg': lambda d: d.extra & d.eg, 'extra phone & fast': lambda d: d.extra & d.fast, 'extra & whole': lambda d: d.extra & d.whole,
 'refund': lambda d: d.refund, 'refund & sender >=2 refunds in 7d': lambda d: d.refund & (d.s_refunds_7d >= 2), 'refund & >=2 in 30d': lambda d: d.refund & (d.s_refunds_30d >= 2), 'refund & >=3 in 30d': lambda d: d.refund & (d.s_refunds_30d >= 3),
 'refund & pair refunds >=2': lambda d: d.refund & (d.pair_refunds_30d >= 2), 'non-refund row, sender >=2 refunds in 30d': lambda d: ~d.refund & (d.s_refunds_30d >= 2),
 'small merchant day: >=3 round500': lambda d: (d.r_round_n_24h >= 3) & (d.r_in_n_24h < 40), 'small merchant day: >=5 in 1h': lambda d: (d.r_in_n_1h >= 5) & (d.r_in_n_24h < 60), 'small merchant: >=3 in 10m': lambda d: (d.r_in_n_10m >= 3) & (d.r_in_n_24h < 60),
 'small merchant day whole share>=.6 & >=3 rows': lambda d: (d.r_whole_share_24h >= .6) & (d.r_in_n_24h >= 3) & (d.r_in_n_24h < 60),
}
rows = []
for k, f in tests.items():
    ma, mb = f(A).values, f(B).values
    ra, rb = (w * ma).sum() / NA * 1e5, mb.sum() / NB * 1e5
    la, lb = (w * (ma & A.legit.values)).sum() / NA * 1e5, (mb & B.legit.values).sum() / NB * 1e5
    rows.append((k, ra, rb, la, lb, int((mb & B.legit.values).sum()), lb - la))
R = pd.DataFrame(rows, columns=['pattern (merchant rows)', 'sep all', 'oct all', 'sep called legit', 'oct called legit', 'oct legit rows', 'excess per 100k']).round(1)
print(R.sort_values('excess per 100k', ascending=False).to_string(index=False))

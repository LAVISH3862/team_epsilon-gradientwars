"""Fraud-type blocks in the extra-phone ids (a generator artefact, like the QR flag).

Extra phones (device number >= 420000) are numbered in the order the data was generated, one fraud type at a time:
in train, 423000-428533 is a run of blocks (mule, bot, layering, SIM swap, takeover, phishing, QR swap, refund,
cashback, loan app, collusion) and ordinary second phones follow. Test repeats the layout from 473632. A row on a
block phone has that block's type, even when the row itself looks ordinary.

Train block boundaries come from the labels. Test boundaries are found by a left-to-right segmentation (dynamic
programme) of the test phones in id order, using what a block looks like in train: how the base model's predictions
are distributed in it, how many rows a phone has, and the gaps between consecutive ids.
  python device_blocks.py --base flag_c1_thr            # writes preds/blocks_train.npy, preds/blocks_test.npy
  python device_blocks.py --base flag_c1_thr --check    # also: segment September alone and compare with the truth"""
import argparse
import numpy as np, pandas as pd
from kit.common import train, test, classes, folds, load_preds

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='flag_c1_thr')
ap.add_argument('--check', action='store_true')
ap.add_argument('--prior', type=float, default=1.0, help='weight of the block-size prior in test (0 = none)')
ap.add_argument('--posterior', action='store_true', help='test: each phone in its most probable block (forward-backward) '
                'instead of the single best split')
ap.add_argument('--out', default='', help='suffix for the output files: preds/blocks_{train,test}<out>.npy')
a = ap.parse_args()
cls = list(classes()); L = cls.index('legit'); C = len(cls)
ORDER = ['mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'phishing_collect', 'qr_swap',
         'refund_abuse', 'fake_cashback', 'loan_app_extortion', 'merchant_collusion', 'legit']
TRAIN_START, TEST_START, TAIL = 423000, 473632, 400     # TAIL: ordinary second phones kept after the last block
tr, te = train(), test()
O, T = load_preds(a.base)
po, pt = O.argmax(1), T.argmax(1)
num = lambda df: df['device_id'].str[2:].astype(int).values
ntr, nte = num(tr), num(te)
ytr = np.searchsorted(classes(), tr['label'].values)


def phones(n, pred, lo):
    """One row per phone from id `lo` upward, in id order: rows on it and the base model's majority call."""
    m = n >= lo
    g = pd.DataFrame({'n': n[m], 'p': pred[m]}).groupby('n').p.agg(['size', lambda s: np.bincount(s, minlength=C).argmax()])
    g.columns = ['rows', 'pred']
    return g


def emis(g):
    """Observation per phone: (base call, one row or more) and the id gap to the previous phone (capped)."""
    gap = np.minimum(np.diff(g.index.values, prepend=g.index.values[0] - 1), 12)
    return g.pred.values * 2 + (g.rows.values > 1), gap


def true_blocks(g, lab):
    """Train: block of each phone from the labels (majority fraud label of the 15 nearest phones), in ORDER."""
    f = lab.where(lab != L).ffill().bfill().values
    sm = np.array([np.bincount(f[max(0, i - 7):i + 8].astype(int), minlength=C).argmax() for i in range(len(f))])
    st, k = np.zeros(len(sm), int), 0
    for i in range(len(sm)):                                 # walk the blocks in order, never going back
        while k < len(ORDER) - 2 and sm[i] != cls.index(ORDER[k]) and sm[i] in [cls.index(o) for o in ORDER[k + 1:-1]]:
            k += 1
        st[i] = k
    return st


def fit(obs, gap, st):
    K = len(ORDER)
    E = np.ones((K, 2 * C)); G = np.ones((K, 13))
    np.add.at(E, (st, obs), 1); np.add.at(G, (st, gap), 1)
    return np.log(E / E.sum(1, keepdims=True)), np.log(G / G.sum(1, keepdims=True))


def potentials(obs, gap, lE, lG, use_gap, size, w):
    """Per-phone block scores (n x K) and the penalty for a boundary into block k at position i ((n + 1) x K).
    `size`: expected share of phones per fraud block (a soft prior on where each boundary falls)."""
    n, K = len(obs), len(ORDER)
    ll = lE[:, obs].T + (lG[:, gap].T if use_gap else 0)
    pen = np.zeros((n + 1, K))
    if size is not None:                                       # boundary into block k expected at cum[k] * n_fraud
        nf = size['n']; cum = np.cumsum(size['share'])
        for k in range(1, K):
            mu = cum[k - 1] * nf; sd = max(np.sqrt(nf * cum[k - 1] * (1 - cum[k - 1])), 2.0) if k < K - 1 else max(0.08 * nf, 3.0)
            pen[:, k] = -w * 0.5 * ((np.arange(n + 1) - mu) / sd) ** 2
    return ll, pen


def posterior(obs, gap, lE, lG, use_gap=True, size=None, w=1.0):
    """Probability of each phone being in each block of ORDER, over all ordered splits (forward-backward on the same
    scores as segment())."""
    ll, pen = potentials(obs, gap, lE, lG, use_gap, size, w); n, K = ll.shape
    F = np.full((n, K), -np.inf); F[0, 0] = ll[0, 0]
    for i in range(1, n):
        F[i] = np.logaddexp(F[i - 1], np.r_[-np.inf, F[i - 1, :-1] + pen[i, 1:]]) + ll[i]
    B = np.full((n, K), -np.inf); B[-1, K - 1] = 0
    for i in range(n - 2, -1, -1):
        B[i] = np.logaddexp(B[i + 1] + ll[i + 1], np.r_[B[i + 1, 1:] + ll[i + 1, 1:] + pen[i + 1, 1:], -np.inf])
    G = F + B
    G -= G.max(1, keepdims=True)
    P = np.exp(G)
    return P / P.sum(1, keepdims=True)


def segment(obs, gap, lE, lG, use_gap=True, size=None, w=1.0):
    """Best ordered split into the blocks of ORDER."""
    n, K = len(obs), len(ORDER)
    ll, pen = potentials(obs, gap, lE, lG, use_gap, size, w)
    D = np.full((n, K), -np.inf); bk = np.zeros((n, K), int)
    D[0, 0] = ll[0, 0]
    for i in range(1, n):
        for k in range(K):
            stay = D[i - 1, k]; move = D[i - 1, k - 1] + pen[i, k] if k else -np.inf
            bk[i, k], D[i, k] = (k, stay + ll[i, k]) if stay >= move else (k - 1, move + ll[i, k])
    st = np.zeros(n, int); st[-1] = K - 1
    for i in range(n - 1, 0, -1):
        st[i - 1] = bk[i, st[i]]
    return st


def describe(g, st, name):
    print(name)
    for k, o in enumerate(ORDER):
        z = g[st == k]
        if len(z): print(f'  {o:20s} ids {z.index.min()}-{z.index.max()}  phones {len(z):5d}  rows {int(z.rows.sum()):5d}')


# train: truth and the block statistics
g = phones(ntr, po, TRAIN_START)
lab = pd.DataFrame({'n': ntr, 'y': ytr})[ntr >= TRAIN_START].groupby('n').y.agg(lambda s: np.bincount(s, minlength=C).argmax())
st_true = true_blocks(g, lab.reindex(g.index))
isc = (lab.reindex(g.index).values == cls.index('merchant_collusion')) & (st_true == len(ORDER) - 2)
dense = np.convolve(isc, np.ones(11), 'full')[:len(isc)] >= 4          # 4+ collusion phones among the last 11
last = np.flatnonzero(isc & dense)[-1]
st_true[last + 1:] = len(ORDER) - 1                          # ordinary second phones start after the collusion block
keep = np.arange(len(g)) <= last + TAIL
g, st_true = g[keep], st_true[keep]
obs, gap = emis(g)
lE, lG = fit(obs, gap, st_true)
describe(g, st_true, 'train blocks (from labels)')
nfraud = (st_true < len(ORDER) - 1).sum()
share = np.bincount(st_true, minlength=len(ORDER))[:-1] / nfraud
st_fit = segment(obs, gap, lE, lG)
print('train, segmentation vs labels: phones in the right block', round((st_fit == st_true).mean(), 4))

if a.check:   # September alone (a third of the phones, base predictions made without September's labels)
    fo = folds(); mth = pd.DataFrame({'n': ntr, 'f': fo})[ntr >= TRAIN_START].groupby('n').f.first().reindex(g.index).values
    for k in range(3):
        m = mth == k; gk = g[m]; ok, gk_gap = emis(gk)
        lE2, _ = fit(obs[~m], gap[~m], st_true[~m])
        nf = round(nfraud * m[st_true < len(ORDER) - 1].mean())
        for nm, kw in (('predictions only', dict(use_gap=False)), ('predictions + size prior', dict(use_gap=False, size=dict(n=(st_true[m] < len(ORDER) - 1).sum(), share=share)))):
            s = segment(ok, gk_gap, lE2, lG, **kw)
            fr = st_true[m] < len(ORDER) - 1
            print(f'  month {k}: {nm:26s} phones in the right block {(s == st_true[m]).mean():.4f}  (fraud blocks only {(s[fr] == st_true[m][fr]).mean():.4f})')

# test
gt = phones(nte, pt, TEST_START)
ot, gpt = emis(gt)
s0 = segment(ot, gpt, lE, lG)                                 # first pass: no prior on block sizes
nf = int((s0 < len(ORDER) - 1).sum())
st_te = segment(ot, gpt, lE, lG, size=dict(n=nf, share=share), w=a.prior)   # second pass: sizes pulled towards train's shares
describe(gt, s0, f'test blocks, no size prior ({nf} fraud-block phones)')
describe(gt, st_te, f'test blocks, size prior weight {a.prior}')
print('expected phones per block from train shares:', dict(zip(ORDER[:-1], np.round(share * nf).astype(int))))
print('test: phones placed differently by the two passes:', int((s0 != st_te).sum()))
if a.posterior:
    sp = posterior(ot, gpt, lE, lG, size=dict(n=nf, share=share), w=a.prior).argmax(1)
    print('test: phones moved by the posterior assignment:', int((sp != st_te).sum()))
    st_te = sp
code = np.array([cls.index(o) for o in ORDER])
btr = np.full(len(tr), -1, np.int8); m = pd.Series(code[st_true], index=g.index); v = pd.Series(ntr).map(m); btr[v.notna().values] = v.dropna().astype(int).values
bte = np.full(len(te), -1, np.int8); m = pd.Series(code[st_te], index=gt.index); v = pd.Series(nte).map(m); bte[v.notna().values] = v.dropna().astype(int).values
btr[btr == L] = -1; bte[bte == L] = -1
np.save(f'preds/blocks_train{a.out}.npy', btr); np.save(f'preds/blocks_test{a.out}.npy', bte)
print('rows on a block phone: train', int((btr >= 0).sum()), 'test', int((bte >= 0).sum()), '| test rows whose label changes', int(((bte >= 0) & (bte != pt)).sum()))

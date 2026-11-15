# Fraud-type specialist on top of sub4 (runs in Colab). Upload X.parquet and meta.npz first.
# sub4's legit / fraud calls stay fixed; only the type of rows sub4 calls fraud can change.
import os, numpy as np, pandas as pd, lightgbm as lgb
NT = int(os.environ.get('SPEC_TREES', 600))
X = pd.read_parquet('X.parquet'); m = np.load('meta.npz', allow_pickle=True)
tr, te, fo, cls = m['tr_rows'], m['te_rows'], m['fo'], list(m['classes'])
L, C = cls.index('legit'), len(cls)
Y = {n: m['y_' + n].astype(np.int64) for n in ('c1', 't1', 'given')}
s1, s1t = m['s1_oof'].astype(int), m['s1_test'].astype(int)
Xtr, Xte = X.iloc[:len(tr)].reset_index(drop=True), X.iloc[len(tr):].reset_index(drop=True)
ytr, ftr, s1tr = Y['c1'][tr], fo[tr], s1[tr]
FT = [c for c in range(C) if c != L]                          # the 11 fraud types
P = dict(n_estimators=NT, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, n_jobs=-1)


def fit(rows):
    yy = ytr[rows]; w = (len(yy) / (C * np.maximum(np.bincount(yy, minlength=C), 1)))[yy]
    return lgb.LGBMClassifier(random_state=42, **P).fit(Xtr.iloc[rows], yy, sample_weight=w)


def full_probs(model, Xp):                                    # 12 columns, legit column = 0
    out = np.zeros((len(Xp), C)); out[:, model.classes_] = model.predict_proba(Xp); out[:, L] = 0
    return out


# out-of-fold specialist probabilities on train rows sub4 calls fraud, by month
S = np.zeros((len(tr), C))
for k in range(3):
    mdl = fit(np.where((ytr != L) & (ftr != k))[0])
    r = np.where((ftr == k) & (s1tr != L))[0]
    S[r] = full_probs(mdl, Xtr.iloc[r]); print('month', k, 'done', flush=True)
St = full_probs(fit(np.where(ytr != L)[0]), Xte)

# sub4's own type probabilities, renormalised over the 11 types
O, T = m['O'].astype(float), m['T'].astype(float)
O[:, L] = 0; O /= O.sum(1, keepdims=True); T[:, L] = 0; T /= T.sum(1, keepdims=True)


def macro(y, p, w=None):
    cm = np.bincount(y * C + p, weights=w, minlength=C * C).reshape(C, C)
    tp = np.diag(cm); d = cm.sum(0) + cm.sum(1)
    return np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1e-9), 0))


months = {k: np.where(fo == k)[0] for k in range(3)}
pos_in_tr = np.full(len(fo), -1); pos_in_tr[tr] = np.arange(len(tr))


def pred_month(Z, k, b=None):                                 # full prediction vector for month k
    p = s1[months[k]].copy()
    if Z is None: return p
    j = pos_in_tr[months[k]]; sel = (j >= 0) & (p != L)
    Zr = np.log(np.clip(Z[j[sel]], 1e-9, 1)) + (0 if b is None else b)
    p[sel] = Zr.argmax(1); return p


Bl = (S + np.where(S.sum(1, keepdims=True) > 0, O, 0)) / 2
res = []
for name, Z in [('sub4', None), ('spec', S), ('blend', Bl)]:
    for k in range(3):
        p = pred_month(Z, k)
        res.append(dict(model=name, month=k, **{n: macro(y[months[k]], p) for n, y in Y.items()}))
# cross-fitted biases for spec and blend: tuned on the other two months' fraud-called rows (c1 labels)
for name, Z in [('spec+bias', S), ('blend+bias', Bl)]:
    for k in range(3):
        rows = np.where((ftr != k) & (s1tr != L))[0]
        b, best, moved = np.zeros(C), -1, True
        yo = Y['c1'][tr[rows]]; Zl = np.log(np.clip(Z[rows], 1e-9, 1))
        # rows sub4 calls legit in the tuning months contribute fixed counts
        other = np.where((fo != k) & (s1 == L))[0]
        fixed = np.bincount(Y['c1'][other] * C + L, minlength=C * C).astype(float)
        def sc(bb):
            cm = (fixed + np.bincount(yo * C + (Zl + bb).argmax(1), minlength=C * C)).reshape(C, C)
            tp = np.diag(cm); d = cm.sum(0) + cm.sum(1)
            return np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1e-9), 0))
        while moved:
            moved = False
            for c in FT:
                for v in np.arange(-2, 2.01, 0.25):
                    b2 = b.copy(); b2[c] = v; s = sc(b2)
                    if s > best + 1e-9: b, best, moved = b2, s, True
        p = pred_month(Z, k, b)
        res.append(dict(model=name, month=k, **{n: macro(y[months[k]], p) for n, y in Y.items()}))
R = pd.DataFrame(res); print(R.pivot_table(index='model', columns='month', values=['c1', 't1', 'given']).round(4))
print(R.groupby('model')[['c1', 't1', 'given']].mean().round(4))

# paired bootstrap on t1 (all months): spec vs sub4 and blend vs sub4
yt = Y['t1']; allp = {n: np.concatenate([pred_month(Z, k) for k in range(3)]) for n, Z in [('sub4', None), ('spec', S), ('blend', Bl)]}
ya = np.concatenate([yt[months[k]] for k in range(3)])
rng = np.random.default_rng(0)
for n in ['spec', 'blend']:
    d = [macro(ya, allp[n], w) - macro(ya, allp['sub4'], w) for w in (rng.poisson(1, len(ya)) for _ in range(200))]
    print(n, 'vs sub4 on t1: mean gain %.4f, share of draws > 0: %.2f' % (np.mean(d), np.mean(np.array(d) > 0)))

# test: types for the rows sub4 calls fraud (no bias), how many change
for n, Zt in [('spec', St), ('blend', (St + T) / 2)]:
    t = np.argmax(Zt, 1); print(n, 'test rows whose type changes vs sub4:', int((t != s1t[te]).sum()), 'of', len(te))
    np.save(f'test_types_{n}.npy', t)
print('download test_types_spec.npy and test_types_blend.npy')

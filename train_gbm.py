"""Gradient-boosted trees (LightGBM, XGBoost or CatBoost) on comp.features, optionally with other tags' predictions as
extra features (that makes it a stacker) and cross-fitted pseudo-labels. Writes preds/<tag>_{oof,test}.npy.
  python train_gbm.py --tag gbm --balanced
  python train_gbm.py --tag xgb --model xgb --balanced                             # XGBoost; --model cat: CatBoost
  python train_gbm.py --tag stack --with gbm+xgb+effb0 --balanced                  # stacker on features + predictions
  python train_gbm.py --tag stack_pl --with gbm+effb0 --balanced --pseudo stack    # + cross-fitted pseudo-labels
  python train_gbm.py --tag stack_spl --with gbm+effb0 --balanced --pseudo stack --soft --thr 0.5 --parts-seed 0,1,2
  python train_gbm.py --tag gbm_c1 --balanced --exclude preds/noisy_r1.npy         # suspected wrong labels left out
  python train_gbm.py --tag gbm_final --balanced --seeds 5 --refit                 # test from models on all labelled rows
On a GPU (Colab, Kaggle): --params '{"device": "cuda"}' for xgb, '{"task_type": "GPU"}' for cat."""
import argparse, json
from kit.common import train_sim, test, y, folds, classes, save_preds, load_preds, report, seed_all
from kit.pseudo import PseudoLabels, expand
import numpy as np, pandas as pd
import comp

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--model', default='lgb', choices=['lgb', 'xgb', 'cat'], help='LightGBM, XGBoost or CatBoost')
ap.add_argument('--with', dest='others', default='', help='tags whose predictions become features, joined by +')
ap.add_argument('--no-features', action='store_true', help='use only the --with predictions')
ap.add_argument('--pseudo', default='', help='teacher tag: cross-fitted pseudo-labels from preds/<teacher>_test.npy')
ap.add_argument('--thr', type=float, default=0.9, help='teacher confidence needed for a pseudo-label')
ap.add_argument('--soft', action='store_true', help="train on the teacher's probabilities, not its argmax")
ap.add_argument('--parts-seed', default='0', help='how test rows are split into parts; several (0,1,2) are averaged')
ap.add_argument('--seeds', type=int, default=3, help='model seeds averaged per fold')
ap.add_argument('--balanced', action='store_true', help='class-balanced sample weights, usually right for macro-F1')
ap.add_argument('--params', default='{}', help='JSON overrides of the model parameters')
ap.add_argument('--exclude', default='', help='.npy of training rows to leave out of training (find_noisy.py); every row is still validated')
ap.add_argument('--refit', action='store_true', help='test predictions from models refit on every labelled row, held-out '
                'ones included (out-of-fold and held-out scores still come from the fold models)')
ap.add_argument('--feats', default='', help='feature blocks joined by + (comp.BLOCKS), e.g. row+recv')
ap.add_argument('--seed0', type=int, default=0, help='added to every model seed: other seeds than an earlier run of the same recipe')
ap.add_argument('--cat', default='', help='feature columns to treat as categories (LightGBM native), joined by commas')
ap.add_argument('--drop', default='', help='feature columns to leave out, joined by commas')
ap.add_argument('--cap', type=int, default=0, help='at most this many training rows per class, redrawn per seed '
                '(a 96%% majority class trains 10x faster; --balanced weights and postprocess.py undo the prior shift)')
a = ap.parse_args()
seed_all()
if a.feats:
    comp.BLOCKS = a.feats.split('+')

X = pd.DataFrame(index=range(len(train_sim()))) if a.no_features else comp.features(train_sim()).reset_index(drop=True)
Xt = pd.DataFrame(index=range(len(test()))) if a.no_features else comp.features(test()).reset_index(drop=True)
if a.cat:
    for col in a.cat.split(','):
        if col in X:
            cats = sorted(set(X[col].dropna().unique()) | set(Xt[col].dropna().unique()))
            X[col] = pd.Categorical(X[col], categories=cats); Xt[col] = pd.Categorical(Xt[col], categories=cats)
if a.drop:
    X, Xt = (Z.drop(columns=[c for c in a.drop.split(',') if c in Z]) for Z in (X, Xt))
for t in filter(None, a.others.split('+')):
    o, p = load_preds(t)
    o, p = o.reshape(len(o), -1), p.reshape(len(p), -1)
    for j in range(o.shape[1]):
        X[f'{t}_{j}'], Xt[f'{t}_{j}'] = o[:, j], p[:, j]
yy, fo, reg = y(), folds(), comp.TASK == 'regression'
width = 1 if reg else len(classes())
params = {'lgb': dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, n_jobs=8),
          'xgb': dict(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.6,
                      reg_lambda=1.0, tree_method='hist', n_jobs=8),
          'cat': dict(iterations=1000, learning_rate=0.05, depth=6, verbose=0, thread_count=8,
                      allow_writing_files=False)}[a.model]
params.update(json.loads(a.params))


def fit(rows, pseudo, seed):
    """A model on the training rows `rows` plus (test rows, pseudo-labels) from PseudoLabels.for_fold."""
    if a.cap and not reg:
        rng = np.random.default_rng(seed)
        rows = np.sort(np.concatenate([r if len(r) <= a.cap else rng.choice(r, a.cap, replace=False)
                                       for r in (rows[yy[rows] == c] for c in range(width))]))
    Xf, yf, wf = X.iloc[rows], yy[rows], np.ones(len(rows))
    if pseudo is not None:
        r, lab, w = expand(*pseudo)
        Xf, yf, wf = pd.concat([Xf, Xt.iloc[r]], ignore_index=True), np.r_[yf, lab], np.r_[wf, w]
    if a.balanced and not reg:  # n / (classes * class count), as class_weight='balanced'
        wf = wf * (wf.sum() / (width * np.maximum(np.bincount(yf, weights=wf, minlength=width), 1e-9)))[yf]
    if a.model == 'lgb':
        import lightgbm as lgb
        M = lgb.LGBMRegressor if reg else lgb.LGBMClassifier
    elif a.model == 'xgb':
        import xgboost as xgb
        M = xgb.XGBRegressor if reg else xgb.XGBClassifier
    else:
        import catboost as cb
        M = cb.CatBoostRegressor if reg else cb.CatBoostClassifier
    return M(random_state=seed, **params).fit(Xf, yf, sample_weight=wf)


def predict(m, Z):
    return m.predict(Z)[:, None] if reg else m.predict_proba(Z)


keep = np.ones(len(X), bool)
if a.exclude:
    keep[np.load(a.exclude)] = False
hold = np.where(fo < 0)[0]
parts = [int(s) for s in a.parts_seed.split(',')] if a.pseudo else [0]
T = np.load(f'preds/{a.pseudo}_test.npy') if a.pseudo else None
oof, pt = np.zeros((len(X), width)), np.zeros((len(Xt), width))
for ps in parts:  # each test split is a full cross-validation; their predictions are averaged
    pl, w = PseudoLabels(T, a.thr, ps, a.soft) if a.pseudo else None, 1 / (a.seeds * len(parts))
    for k in range(comp.N_FOLDS):
        va, rows = np.where(fo == k)[0], np.arange(len(Xt))
        pseudo = None
        if pl:
            r, lab, rows = pl.for_fold(k)
            pseudo = (r, lab)
        for s in range(a.seeds):
            m = fit(np.where((fo >= 0) & (fo != k) & keep)[0], pseudo, comp.SEED + a.seed0 + s)
            oof[va] += predict(m, X.iloc[va]) * w
            if len(hold):
                oof[hold] += predict(m, X.iloc[hold]) * w / comp.N_FOLDS
            if not a.refit:
                pt[rows] += predict(m, Xt.iloc[rows]) * w / (1 if pl else comp.N_FOLDS)
    if a.refit:  # every labelled row, held-out ones too; with pseudo-labels still cross-fitted over the test parts
        for k in range(comp.N_FOLDS if pl else 1):
            rows, pseudo = np.arange(len(Xt)), None
            if pl:
                r, lab, rows = pl.for_fold(k)
                pseudo = (r, lab)
            for s in range(a.seeds):
                pt[rows] += predict(fit(np.where(keep)[0], pseudo, comp.SEED + a.seed0 + s), Xt.iloc[rows]) * w
if reg:
    oof, pt = oof[:, 0], pt[:, 0]
save_preds(a.tag, oof, pt)
report(a.tag, oof)

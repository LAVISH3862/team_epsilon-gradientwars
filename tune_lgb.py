"""Optuna search over LightGBM settings for the sub4 recipe. Each trial trains on July + August and scores September:
macro-F1 on the test-like labels (LABELS_SCORE) after per-class biases tuned on half of September and applied to the
other half (both ways), so the score is not flattered by the bias search. Prints the best settings.
  LABELS=preds/labels_c1.npy python tune_lgb.py --trials 20"""
import argparse, json, time
import numpy as np, lightgbm as lgb, optuna
import comp
from kit.common import train, y, folds, classes
from kit import metrics

ap = argparse.ArgumentParser()
ap.add_argument('--trials', type=int, default=20)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--cap', type=int, default=150000)
ap.add_argument('--score-labels', default='preds/labels_t1.npy')
a = ap.parse_args()
tr, yy, fo = train(), y(), folds()
C, L = len(classes()), list(classes()).index('legit')
ys = np.load(a.score_labels)
X = comp.features(tr, ['row', 'conn', 'ring', 'flag']).drop(columns=['qr_vpa_mismatch'])
rng = np.random.default_rng(0)
pool = np.where(fo < 2)[0]
fit_rows = np.sort(np.concatenate([r if len(r) <= a.cap else rng.choice(r, a.cap, replace=False)
                                   for r in (pool[yy[pool] == c] for c in range(C))]))
w = (len(fit_rows) / (C * np.maximum(np.bincount(yy[fit_rows], minlength=C), 1)))[yy[fit_rows]]
va = np.where(fo == 2)[0]
half = rng.random(len(va)) < 0.5


def tuned_score(P):
    Lg, pred = np.log(np.clip(P, 1e-9, 1)), np.zeros(len(P), int)
    for f, g in ((half, ~half), (~half, half)):
        b, best, moved = np.zeros(C), -1.0, True
        while moved:
            moved = False
            for c in range(C):
                for v in np.arange(-8, 8.01, 0.5):
                    b2 = b.copy(); b2[c] = v
                    s = metrics.fast('macro_f1', ys[va][f], (Lg[f] + b2).argmax(1))
                    if s > best + 1e-9:
                        b, best, moved = b2, s, True
        pred[g] = (Lg[g] + b).argmax(1)
    return metrics.fast('macro_f1', ys[va], pred)


def objective(t):
    p = dict(learning_rate=t.suggest_float('learning_rate', 0.01, 0.08, log=True),
             num_leaves=t.suggest_int('num_leaves', 15, 255, log=True),
             min_child_samples=t.suggest_int('min_child_samples', 5, 200, log=True),
             colsample_bytree=t.suggest_float('colsample_bytree', 0.3, 1.0),
             subsample=t.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
             reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True),
             reg_alpha=t.suggest_float('reg_alpha', 1e-3, 10, log=True))
    n = t.suggest_int('n_estimators', 300, 3000, log=True)
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=n, verbose=-1, n_jobs=4, random_state=42, **p).fit(X.iloc[fit_rows], yy[fit_rows], sample_weight=w)
    s = tuned_score(m.predict_proba(X.iloc[va]))
    print(f'trial {t.number}: {s:.4f} in {(time.time() - t0) / 60:.1f} min {json.dumps(dict(p, n_estimators=n))}', flush=True)
    return s


base = objective(optuna.trial.FixedTrial(dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, colsample_bytree=0.6,
                                              subsample=0.8, reg_lambda=1.0, reg_alpha=1e-3, n_estimators=600)))
print(f'sub4 settings on this split: {base:.4f}', flush=True)
st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=a.seed))
st.optimize(objective, n_trials=a.trials)
print(f'best {st.best_value:.4f} (sub4 settings {base:.4f}) with {json.dumps(st.best_params)}', flush=True)

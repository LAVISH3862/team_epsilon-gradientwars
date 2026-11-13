"""Shared plumbing: cached data, label encoding, the fixed folds and held-out rows, the preds/<tag>_{oof,test}.npy
convention, scoring and the paired bootstrap."""
import os, random
from functools import lru_cache
import numpy as np, pandas as pd
import comp
from kit import metrics

SIM_SEED = 7  # the held-out simulator's seed: anything that trains on simulated rows must use other seeds

for _d in ('preds', 'models', 'submissions', 'logs'):
    os.makedirs(_d, exist_ok=True)


@lru_cache(None)
def train():
    return comp.load_train().reset_index(drop=True)


@lru_cache(None)
def test():
    return comp.load_test().reset_index(drop=True)


@lru_cache(None)
def classes():
    """Class names in sorted order: column k of every probability array is classes()[k]."""
    return np.array(sorted(train()[comp.TARGET].unique()))


def y_given():
    """The labels as given in the training file."""
    t = train()[comp.TARGET].values
    return t.astype(np.float32) if comp.TASK == 'regression' else np.searchsorted(classes(), t)


def y():
    """The training labels: as given, or the cleaned ones in the file named by the environment variable LABELS
    (clean_labels.py), which training, bias tuning and scoring then all use."""
    return np.load(os.environ['LABELS']) if os.environ.get('LABELS') else y_given()


def folds():
    """Fold of every training row, fixed once by make_folds.py so every model shares the same splits. Fold -1 marks
    held-out rows (make_folds.py --holdout): nothing ever trains on them, not even a blend weight or a threshold, and
    every model predicts them like test rows, so their score stays honest where out-of-fold scores flatter."""
    if not os.path.exists('folds.csv'):
        raise SystemExit('folds.csv is missing: run python make_folds.py first')
    return pd.read_csv('folds.csv')['fold'].values


def cv_rows():
    """The training rows in the cross-validation folds: all but the held-out ones."""
    return np.where(folds() >= 0)[0]


def holdout_rows():
    return np.where(folds() < 0)[0]


@lru_cache(None)
def train_sim():
    """train() as the models see it: held-out rows replaced by comp.simulate(rows, SIM_SEED), a test-like copy, so
    their score predicts the test score. Build model inputs from this, not from train()."""
    df, h = train(), holdout_rows()
    if not len(h) or not hasattr(comp, 'simulate'):
        return df
    sim = comp.simulate(df.iloc[h].copy(), SIM_SEED)
    return pd.concat([df.drop(index=df.index[h]), sim.set_axis(df.index[h])]).sort_index()


def seed_all(seed=comp.SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_preds(tag, oof, test_pred):
    np.save(f'preds/{tag}_oof.npy', np.asarray(oof, np.float32))
    np.save(f'preds/{tag}_test.npy', np.asarray(test_pred, np.float32))


def load_preds(tag):
    return np.load(f'preds/{tag}_oof.npy'), np.load(f'preds/{tag}_test.npy')


def score(P, idx=None):
    """The metric for predictions P of the training rows idx (default: every row)."""
    yy = y()
    return metrics.score(comp.METRIC, yy if idx is None else yy[idx], P)


def scores(oof):
    """(out-of-fold score on the cross-validated rows, score on the held-out rows or None) of a full-length oof."""
    cv, h = cv_rows(), holdout_rows()
    return score(oof[cv], cv), (score(oof[h], h) if len(h) else None)


def report(tag, oof):
    """Print a tag's out-of-fold score, its held-out score when make_folds.py --holdout set rows aside, and per-class F1
    (on the cross-validated rows); append the line to logs/scores.log."""
    cv, ho = scores(oof)
    line = f'{tag}: {comp.METRIC} {cv:.4f}' + ('' if ho is None else f' | held-out {ho:.4f}')
    if comp.TASK != 'regression':
        r = cv_rows()
        line += '  | ' + ' '.join(f'{c}={v:.3f}' for c, v in zip(classes(), metrics.per_class_f1(y()[r], oof[r])))
    if os.environ.get('LABELS'):  # scored on cleaned labels: also show the score on the labels as given
        line += f"  | cleaned labels {os.path.basename(os.environ['LABELS'])}, as given {metrics.score(comp.METRIC, y_given()[r], oof[r]):.4f}"
    if hasattr(comp, 'fixed_folds'):  # folds mean something (e.g. months): show each one
        fo = folds()
        line += '  | folds ' + ' '.join(f'{k}={score(oof[fo == k], np.where(fo == k)[0]):.4f}' for k in range(comp.N_FOLDS))
    print(line, flush=True)
    with open('logs/scores.log', 'a') as f:
        f.write(f'{pd.Timestamp.now():%m-%d %H:%M} {line}\n')


def bootstrap(Pa, Pb, rows, draws=1000):
    """Paired bootstrap over the training rows `rows` (whole groups when comp.GROUP is set): both predictions are scored
    on the same resampled rows. Returns (change b - a on rows, 95% interval of the change, share of draws where b is
    better)."""
    yy, rng = y(), np.random.default_rng(comp.SEED)
    if comp.GROUP:
        g = pd.factorize(train()[comp.GROUP].values[rows])[0]
        members = [np.where(g == i)[0] for i in range(g.max() + 1)]
        draw = lambda: rows[np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])]
    else:
        draw = lambda: rows[rng.integers(0, len(rows), len(rows))]
    sc = lambda i, P: metrics.score(comp.METRIC, yy[i], P[i])
    if comp.METRIC in metrics.FAST:  # label metrics: scored from label counts, ~50x faster than sklearn
        Pa, Pb = metrics.labels(Pa), metrics.labels(Pb)
        sc = lambda i, L: metrics.fast(comp.METRIC, yy[i], L[i])
    d = []
    for _ in range(draws):
        i = draw()
        try:
            d.append(sc(i, Pb) - sc(i, Pa))
        except ValueError:  # a draw without some class can't be scored by AUC/logloss
            pass
    d = np.array(d)
    better = np.mean(d > 0) if metrics.HIGHER_IS_BETTER.get(comp.METRIC, True) else np.mean(d < 0)
    return sc(rows, Pb) - sc(rows, Pa), np.percentile(d, [2.5, 97.5]), float(better)

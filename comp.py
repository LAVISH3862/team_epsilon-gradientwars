"""UPI Fraud Rings (ML Gradient Wars 2026). Everything specific to this competition.

Rules (data/PROBLEM_STATEMENT.md):
- Metric: macro-F1 over 12 labels (legit + 11 fraud types). All-legit scores about 0.08; 0.40 is "excellent".
- Uploads: 10 per team per day, board refreshes every 15 minutes, files up to 15 MB.
- Public board = 30% of test, private = 70%. Final rank = private score of our best public upload (automatic).
- Code: a GitHub or notebook link in the Submission tab before 15 Nov 2026 10:00 AM IST; must reproduce the best upload.
- Data: train only on the provided files; public pretrained weights are fine; no hand-labelling of test rows.
  Using test.csv to build connections between rows is allowed.
- Round: 13 Nov 2026 10:00 AM IST -> 15 Nov 2026 10:00 AM IST. Results 15 Nov 5:00 PM IST.
- Data notes: test is the next month (October: festival spending, new app version, adapted rings); training labels
  are noisy (unreported fraud labelled legit, wrong fraud types), test labels are clean; some columns stop working
  in October (app_version is all new, ip_risk_score is distributed differently)."""
import os
import numpy as np, pandas as pd

ID = 'id'                 # the submission's id; train.csv / test.csv call it txn_id (renamed on load)
TARGET = 'label'
TASK = 'multiclass'
METRIC = 'macro_f1'
SUBMIT = 'label'
N_FOLDS, SEED = 3, 42     # one fold per training month, see fixed_folds()
GROUP = None
HFLIP_OK = False

BLOCKS = os.environ.get('FEATS', 'row').split('+')   # feature blocks built by features.py (data/feat_<block>.parquet)
# THIN=1: features from the half-density copy of train (make_thin.py); THIN=sep: only September thinned (sim_density.py)
THIN = {'': '', '1': '_thin', 'sep': '_thinsep'}[os.environ.get('THIN', '')]


def _load(name):
    """A competition csv, cached as parquet (1.5M rows load in a second instead of ten)."""
    pq = f'data/{name}.parquet'
    if not os.path.exists(pq):
        pd.read_csv(f'data/{name}.csv').to_parquet(pq)
    return pd.read_parquet(pq).rename(columns={'txn_id': ID})


def load_train():
    return _load('train')


def load_test():
    return _load('test')


def load_sample():
    return pd.read_csv('data/sample_submission.csv')


def load_accounts():
    return pd.read_csv('data/accounts.csv')


def load_merchants():
    return pd.read_csv('data/merchants.csv')


def fixed_folds(df):
    """Fold = month: 0 July, 1 August, 2 September (plus the 147 labelled rows of early October). Test is the next
    month and a ring's rows sit close in time, so random folds would leak rings and flatter the score. Fold 2 is the
    headline check: its models train on July and August and predict September."""
    return np.minimum(pd.to_datetime(df['timestamp']).dt.month.values, 9) - 7


def features(df, blocks=None):
    """Feature table for tree models: the blocks in BLOCKS, built over train + test together by features.py and
    cached in data/. Rows are matched by id, one block at a time (the whole table twice over does not fit in 16 GB)."""
    ids, out = df[ID].values, []
    for b in blocks or BLOCKS:
        path = f'data/feat_{b}{THIN}.parquet'
        if not os.path.exists(path):
            raise SystemExit(f'{path} is missing: run python features.py {b}' + (' with THIN=1' if THIN else ''))
        F = pd.read_parquet(path)
        n, i = len(ids), F.index.values
        if n <= len(F) and i[0] == ids[0] and i[n - 1] == ids[-1] and (i[:n] == ids).all():
            F = F.iloc[:n]                                   # the training rows, in file order
        elif n <= len(F) and i[-n] == ids[0] and i[-1] == ids[-1] and (i[-n:] == ids).all():
            F = F.iloc[-n:]                                  # the test rows
        else:
            F = F.reindex(ids)
        out.append(F.reset_index(drop=True))
        del F
    return pd.concat(out, axis=1)


def simulate(df, seed):
    return df


def simulate_images(X, seed):
    return X


def load_images(split):
    raise NotImplementedError('not an image task')

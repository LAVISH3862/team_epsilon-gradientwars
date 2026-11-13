# UPI Fraud Rings (ML Gradient Wars 2026)

Team Epsilon. Labels every October payment as legit or one of 11 fraud types (macro-F1).

## Reproduce
1. Python 3.10+, `pip install -r requirements.txt`.
2. Put `train.csv`, `test.csv`, `accounts.csv`, `merchants.csv`, `sample_submission.csv` in `data/`.
3. Run `bash reproduce.sh` (Git Bash on Windows). It rebuilds every uploaded file in `submissions/`.
   About 2 hours on a 6-core laptop CPU; no GPU needed.

## Approach
- **Folds by month** (July, August, September): test is the next month and a ring's rows sit close in time.
- **Features** (`features.py`), built over train + test rows together, never from labels:
  `row` (single-row signals, whose phone it is), `conn` (bursts at the receiver, sender and phone, money moving on,
  payments in the opposite direction), `ring` (statistics of the event a row belongs to), `flag` (explicit states of
  the QR mismatch flag, which October changes on the raw column).
- **Label cleaning** (`clean_labels.py`): 2.3% of training rows carry a fraud label but cannot be told from legit rows
  by any model (AUC 0.50); they are treated as legit for training. Rows inside clear rings that are labelled legit
  take the model's fraud type. Predictions used for cleaning are always out-of-fold.
- **Models**: LightGBM, XGBoost and CatBoost on the same folds (`train_gbm.py`), class-balanced, legit capped per fold.
- **Decision rule** (`postprocess.py`): per-class biases tuned for macro-F1, cross-fitted, on test-like labels
  (`make_testlike_labels.py`).
- **Checks**: `sim_density.py` (October has half the payments per day: no effect), `compare.py` (paired bootstrap
  before every upload).

`PLAN.md` has the findings in order. `submissions/log.csv` maps each uploaded file to its model and scores.

# UPI Fraud Rings (ML Gradient Wars 2026)

Team Epsilon. Labels every October payment as legit or one of 11 fraud types (macro-F1).

## Reproduce
1. Python 3.10+, `pip install -r requirements.txt`.
2. Put `train.csv`, `test.csv`, `accounts.csv`, `merchants.csv`, `sample_submission.csv` in `data/`.
3. Run `bash reproduce.sh` (Git Bash on Windows). It rebuilds every uploaded file in `submissions/`.
   About 2 hours on a 6-core laptop CPU; no GPU needed.

## Final upload: sub35 (public 0.5481)
sub35 = sub27 (public 0.5478) with the block spread widened to one day and each test phone placed in its most
probable block. Built by `reproduce.sh` (sub27 first, then the sub35 lines):
```
LABELS=preds/labels_c1.npy python device_blocks.py --base flag_c1_thr
LABELS=preds/labels_c1.npy python apply_blocks.py --base flag_c1_thr --tag blk_sp --spread 360
LABELS=preds/labels_c1.npy python submit.py blk_sp --name sub27
LABELS=preds/labels_c1.npy python device_blocks.py --base flag_c1_thr --posterior --out _post
LABELS=preds/labels_c1.npy python apply_blocks.py --base flag_c1_thr --tag blk1440_post --spread 1440 --blocks _post
LABELS=preds/labels_c1.npy python submit.py blk1440_post --name sub35
```
sub36 = sub35 plus `receiver_rule.py` (block types along the same receiver for a week; 1 test row changes).
- Base: `flag_c1_thr` (sub4, public 0.5060): LightGBM on row + conn + ring + flag features, cleaned labels, biases
  tuned on cleaned labels.
- **Extra-phone blocks** (`device_blocks.py`): extra phones (device number >= 420000) are numbered one fraud type at
  a time. Train block edges come from the labels; test edges come from an ordered segmentation of the test phones
  (the base model's calls, rows per phone, id gaps). A row on a block phone takes the block's type.
- **Spread** (`apply_blocks.py --spread 360`): rows the base model already calls fraud, of a look-alike type, sharing
  the sender or receiver with a block row within 6 hours, take the block's type.
- Validation (test-like labels, month folds): sub4 0.5219 -> sub27 0.5500 -> sub35 0.5505.

`WORKLOG.md` lists every experiment and its result.

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

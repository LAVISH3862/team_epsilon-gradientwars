#!/usr/bin/env bash
# Rebuilds every uploaded submission from the competition files in ./data (train.csv, test.csv, accounts.csv,
# merchants.csv, sample_submission.csv). Each step is added, with its exact flags, when its tag is uploaded.
set -euo pipefail
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE

python make_folds.py                 # one fold per training month
python features.py row               # single-row features + phone ownership

# sub1: single-row LightGBM, per-class biases tuned for macro-F1
python train_gbm.py --tag gbm --balanced --cap 150000 --seeds 1
python postprocess.py gbm --tag gbm_thr --grid 8 --step 0.25
python submit.py gbm_thr --name sub1

# sub2: + connection features (receiver / sender / phone / pair windows, chains, reverse payments)
python features.py conn
python train_gbm.py --tag conn --feats row+conn --balanced --cap 150000 --seeds 1
python postprocess.py conn --tag conn_thr --grid 8 --step 0.25
python submit.py conn_thr --name sub2

# sub3: cleaned labels (noisy fraud labels -> legit, unreported ring rows -> fraud type) + ring-level features
python clean_labels.py conn --name c1
python features.py ring
export LABELS=preds/labels_c1.npy
python train_gbm.py --tag ring_c1 --feats row+conn+ring --balanced --cap 150000 --seeds 1
python postprocess.py ring_c1 --tag ring_c1_thr --grid 8 --step 0.25
python submit.py ring_c1_thr --name sub3
unset LABELS

# sub4: + explicit states of the QR mismatch flag (October blanks the raw column on many qr_swap rows)
python features.py flag
export LABELS=preds/labels_c1.npy
python train_gbm.py --tag flag_c1 --feats row+conn+ring+flag --drop qr_vpa_mismatch --balanced --cap 150000 --seeds 1
python postprocess.py flag_c1 --tag flag_c1_thr --grid 8 --step 0.25
python submit.py flag_c1_thr --name sub4
unset LABELS

# sub5: the sub4 recipe as a 4-model blend. The three extra models ran on Kaggle CPU kernels (kaggle_run.py), with
# the same commands as here plus --params '{"n_jobs": 4}' (LightGBM, XGBoost) / '{"thread_count": 4}' (CatBoost).
export LABELS=preds/labels_c1.npy
R="--feats row+conn+ring+flag --drop qr_vpa_mismatch --balanced --cap 150000"
python train_gbm.py --tag k_lgb_s1 $R --seeds 2 --seed0 1
python train_gbm.py --tag k_xgb --model xgb $R --seeds 1
python train_gbm.py --tag k_cat --model cat $R --seeds 1
python make_testlike_labels.py              # preds/labels_t1.npy: given labels + unreported ring rows restored
export LABELS=preds/labels_t1.npy
python blend.py --tag ens1 flag_c1 k_lgb_s1 k_xgb k_cat --equal --geo
python postprocess.py ens1 --tag ens1_t --grid 8 --step 0.25
python submit.py ens1_t --name sub5
unset LABELS
# sub6-sub9 are probes (sub4 with groups of fraud types set to legit), not candidates: see submissions/log.csv

# sub10: sub4 recipe + cross-fitted pseudo-labels on the test month (teacher: the sub5 blend before bias tuning)
LABELS=preds/labels_c1.npy python train_gbm.py --tag flag_pl1 --feats row+conn+ring+flag --drop qr_vpa_mismatch --balanced --cap 150000 --seeds 1 --pseudo ens1 --thr 0.9
LABELS=preds/labels_t1.npy python postprocess.py flag_pl1 --tag flag_pl1_t --grid 8 --step 0.25
LABELS=preds/labels_t1.npy python submit.py flag_pl1_t --name sub10

# sub12: sub4 recipe on second-round cleaned labels (teacher: the sub5 blend before bias tuning)
python clean_labels.py ens1 --name c2 --noise 0.5 --unreported 0.01
LABELS=preds/labels_c2.npy python train_gbm.py --tag flag_c2 --feats row+conn+ring+flag --drop qr_vpa_mismatch --balanced --cap 150000 --seeds 1
LABELS=preds/labels_t1.npy python postprocess.py flag_c2 --tag flag_c2_t --grid 8 --step 0.25
LABELS=preds/labels_t1.npy python submit.py flag_c2_t --name sub12

# sub13: the sub12 model with biases tuned on cleaned labels c2 (as sub4 was), not on test-like labels
LABELS=preds/labels_c2.npy python postprocess.py flag_c2 --tag flag_c2_thr --grid 8 --step 0.25
LABELS=preds/labels_c2.npy python submit.py flag_c2_thr --name sub13
# sub14: sub4 with the legit log-probability raised by 2 (fewer borderline fraud calls): see the snippet in PLAN.md

# sub15: sub4 recipe on type-corrected labels c3 (wrong fraud types moved to the teacher's type); trained on Kaggle
#        with --params '{"n_jobs": 4}'. labels_c3 is built from labels_c2 and the ens1 teacher (see PLAN.md, 14 Nov).
LABELS=preds/labels_c3.npy python train_gbm.py --tag k_flag_c3 --feats row+conn+ring+flag --drop qr_vpa_mismatch --balanced --cap 150000 --seeds 1
LABELS=preds/labels_c3.npy python postprocess.py k_flag_c3 --tag k_flag_c3_thr --grid 8 --step 0.25
LABELS=preds/labels_c3.npy python submit.py k_flag_c3_thr --name sub15

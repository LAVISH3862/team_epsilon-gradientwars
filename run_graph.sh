#!/usr/bin/env bash
# waits for the account-history run, then builds the graph block and trains with it
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
while python -c "import psutil,sys; sys.exit(0 if psutil.pid_exists(9312) else 1)"; do sleep 20; done
python features.py graph
LABELS=preds/labels_c1.npy python train_gbm.py --tag graph_c1 --feats row+conn+ring+flag+graph --drop qr_vpa_mismatch --balanced --cap 150000 --seeds 1
LABELS=preds/labels_c1.npy python postprocess.py graph_c1 --tag graph_c1_thr --grid 8 --step 0.25
LABELS=preds/labels_t1.npy python compare.py flag_c1_thr graph_c1_thr --boot 200

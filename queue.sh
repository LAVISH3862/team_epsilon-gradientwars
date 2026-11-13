#!/usr/bin/env bash
# Overnight queue: GPU jobs strictly one after another (two at once thrash a 4 GB GPU), each with its own log.
# Start it, then run `python keep_awake.py` so Windows doesn't sleep; that exits when the queue finishes.
#   bash queue.sh &        progress: tail logs/queue.log
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p logs
run() { local name=$1; shift
  echo "[$(date +%H:%M)] start $name" >> logs/queue.log
  "$@" > "logs/$name.log" 2>&1
  echo "[$(date +%H:%M)] done $name (exit $?)" >> logs/queue.log; }

run effb0 python train_cnn.py --tag effb0 --arch efficientnet_b0 --size 224 --epochs 8
run r34 python train_cnn.py --tag r34 --arch resnet34 --size 224 --epochs 8
echo "[$(date +%H:%M)] queue finished" >> logs/queue.log

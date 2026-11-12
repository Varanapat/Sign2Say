#!/bin/bash
# Friend's RQ-S findings (hand-local, ST-GCN) under OUR recipe and protocol, seed 0.
# Queue A (fast): S2S-HL on the ablation folds.  Queue B (slow, ~2.5 h/run): ST-GCN.
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u scripts/run_baselines.py --protocol tsl1s_si"
F='==|S2S|STGCN|Error|Traceback'
( $PY --models S2S-HL --folds 0 2 4 6 8 --tag _rqs_s2shl 2>&1 | grep --line-buffered -E "$F" ) > results/run_rqs_a.log &
( $PY --models STGCN-HL STGCN --folds 4 --tag _rqs_stgcn 2>&1 | grep --line-buffered -E "$F"
  $PY --models STGCN-HL --folds 0 2 6 8 --tag _rqs_stgcn 2>&1 | grep --line-buffered -E "$F" ) > results/run_rqs_b.log &
wait
echo "rqs done"

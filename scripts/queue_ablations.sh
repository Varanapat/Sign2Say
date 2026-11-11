#!/bin/zsh
# Run the S2S-LTR ablations (Experiment_Design_Sign2Say.md section 5) once the GPU is free.
# 5 signer-independent folds (0 2 4 6 8), seed 0; the "base" S2S numbers come from the 10-fold run.
# Two chains run in parallel; each writes its own results file (--tag) so nothing is overwritten.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FOLDS=(0 2 4 6 8)

# wait for every running training job ([P] keeps this pattern from matching this script itself)
while pgrep -f "[P]ython -u scripts/run_baselines" >/dev/null || pgrep -f "[k]ill -0 40766" >/dev/null; do
  sleep 60
done

run() {  # $1 = variant, $2 = protocol
  $PY -u scripts/run_baselines.py --protocol $2 --models S2S --variant $1 --folds $FOLDS \
      --tag "_abl_$1" 2>&1 | grep --line-buffered -E "==|S2S|Error|Traceback"
}

( run noface tsl1s_si; run x71 tsl1s_si; run novel tsl1s_si ) > results/run_ablation_A.log &
( run nomask tsl1s_si; run flat tsl1s_si; run base tsl1s_rand ) > results/run_ablation_B.log &
wait
echo "ablations done"

#!/bin/zsh
# C1 (CTC) grid + C2 (sliding window). x16-initialised runs wait for the saved X16 model.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
F="==|C1|C2|Error|Traceback"
for p in rep template; do
  $PY -u scripts/run_cslr.py c1 --protocol $p --init scratch --synth 300 --epochs 100 --seeds 0 1 2 2>&1 | grep --line-buffered -E "$F"
done
while [ ! -f results/models/joint_tsl1s_si_fold0_seed0.pt ]; do sleep 30; done
for p in rep template; do
  $PY -u scripts/run_cslr.py c1 --protocol $p --init x16 --synth 300 --epochs 100 --seeds 0 1 2 2>&1 | grep --line-buffered -E "$F"
  $PY -u scripts/run_cslr.py c1 --protocol $p --init x16 --synth 0 --epochs 100 --seeds 0 1 2 2>&1 | grep --line-buffered -E "$F"
done
$PY -u scripts/run_cslr.py c2 --c2-model x16 2>&1 | grep --line-buffered -E "$F"
$PY -u scripts/run_cslr.py c2 --c2-model x16ft 2>&1 | grep --line-buffered -E "$F"
echo "cslr done"

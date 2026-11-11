#!/usr/bin/env bash
# รันการทดลองต่อกันหลายรอบจากไฟล์คิว แล้วสรุปตารางตอนจบ
#
#   ./run_queue.sh queues/diag.txt            # รันจริง (กันเครื่อง sleep ให้เอง)
#   ./run_queue.sh -n queues/partB_fold4.txt  # dry run: แสดงคำสั่งที่จะรัน ไม่รันจริง
#
# รูปแบบไฟล์คิว (1 บรรทัด = 1 run, # = comment, บรรทัดว่างข้ามได้)
#   B <model> <exp> <fold> <seed> [args เพิ่ม]   -> python -m experiments.train  (Part B, รันจาก root)
#   S <exp> <fold> <seed> [args เพิ่ม]           -> spoter/train_batched.py       (SPOTER, รันจาก spoter/)
#                                                  ค่าตั้งต้น: --device mps --aug --epochs 100 --patience 20 --select f1
#   R <model> <exp> <fold> <seed> [args เพิ่ม]   -> python -m experiments.train_rgb (Part C RGB, รันจาก root)
# ตัวอย่าง
#   B B1 B1_aug 4 0 --aug
#   S A2_spoter_pe 4 1 --pe learned
#
# พฤติกรรม
#   - ข้าม run ที่มี results/<exp>_fold<k>_seed<s>/summary.json แล้ว -> หยุดกลางคันแล้วสั่งใหม่ได้ (รันต่อจากเดิม)
#   - run ไหน error จะข้ามไปตัวถัดไป และสรุปรายชื่อตอนจบ
#   - log ของแต่ละ run: results/logs/<exp>_fold<k>_seed<s>.log
#   - จบแล้วเขียนตารางรวม results/TABLE.md (python -m experiments.aggregate)
#   - Python: ใช้ $PYTHON ถ้าตั้งไว้ ไม่งั้นใช้ ~/venvs/sign2say/bin/python
set -u   # หมายเหตุ: bash 3.2 ของ macOS ถือว่า array ว่างเป็น unbound จึงใช้ ${a[@]+"${a[@]}"}

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-$HOME/venvs/sign2say/bin/python}"
SPOTER_DEFAULTS=(--device mps --aug --epochs 100 --patience 20 --select f1)

DRY=0
if [ "${1:-}" = "-n" ]; then DRY=1; shift; fi
QUEUE="${1:-}"
if [ -z "$QUEUE" ] || [ ! -f "$QUEUE" ]; then
  echo "ใช้: ./run_queue.sh [-n] <ไฟล์คิว>   (ไม่พบไฟล์: '${QUEUE}')" >&2
  exit 1
fi
QUEUE="$(cd "$(dirname "$QUEUE")" && pwd)/$(basename "$QUEUE")"

# กันเครื่อง sleep ตลอดทั้งคิว: เรียกตัวเองใหม่ภายใต้ caffeinate (macOS) ครั้งเดียว
if [ "$DRY" = 0 ] && [ -z "${RUN_QUEUE_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export RUN_QUEUE_CAFFEINATED=1
  exec caffeinate -i "$0" "$QUEUE"
fi

if [ ! -x "$PY" ]; then echo "ไม่พบ Python: $PY (ตั้ง PYTHON=... เอง)" >&2; exit 1; fi
mkdir -p "$ROOT/results/logs"

done_n=0; skip_n=0; fail=(); t_start=$(date +%s)
total=$(grep -cvE '^\s*(#|$)' "$QUEUE")
i=0

while read -r kind rest; do
  case "$kind" in ''|\#*) continue ;; esac
  i=$((i + 1))
  # shellcheck disable=SC2206
  f=($rest)
  if [ "$kind" = "B" ]; then
    model=${f[0]}; exp=${f[1]}; fold=${f[2]}; seed=${f[3]}; extra=("${f[@]:4}")
    workdir="$ROOT"
    cmd=("$PY" -m experiments.train --model "$model" --exp "$exp" --fold "$fold" --seed "$seed" ${extra[@]+"${extra[@]}"})
  elif [ "$kind" = "R" ]; then
    model=${f[0]}; exp=${f[1]}; fold=${f[2]}; seed=${f[3]}; extra=("${f[@]:4}")
    workdir="$ROOT"
    cmd=("$PY" -m experiments.train_rgb --model "$model" --exp "$exp" --fold "$fold" --seed "$seed" ${extra[@]+"${extra[@]}"})
  elif [ "$kind" = "S" ]; then
    exp=${f[0]}; fold=${f[1]}; seed=${f[2]}; extra=("${f[@]:3}")
    workdir="$ROOT/spoter"
    cmd=("$PY" train_batched.py --exp "$exp" --fold "$fold" --seed "$seed" "${SPOTER_DEFAULTS[@]}" ${extra[@]+"${extra[@]}"})
  else
    echo "[$i/$total] ข้ามบรรทัดที่ไม่รู้จัก: $kind $rest"; continue
  fi

  run="${exp}_fold${fold}_seed${seed}"
  if [ -f "$ROOT/results/$run/summary.json" ]; then
    echo "[$i/$total] ข้าม $run (มีผลแล้ว)"; skip_n=$((skip_n + 1)); continue
  fi
  echo "[$i/$total] $(date '+%H:%M') เริ่ม $run"
  echo "          (cd ${workdir#$ROOT/} && ${cmd[*]})"
  [ "$DRY" = 1 ] && continue

  # SPOTER ต้องมี CSV ของ fold นั้นก่อน
  if [ "$kind" = "S" ] && [ ! -d "$ROOT/data/split_fold$fold" ]; then
    (cd "$ROOT" && "$PY" tools/make_csv.py --split-json "splits/tsl_one_s_signer_indep_fold$fold.json") \
      >> "$ROOT/results/logs/$run.log" 2>&1
  fi

  t0=$(date +%s)
  if (cd "$workdir" && "${cmd[@]}") < /dev/null > "$ROOT/results/logs/$run.log" 2>&1; then
    line=$(grep -E "^best epoch|^Test top-1" "$ROOT/results/logs/$run.log" | tail -1)
    echo "          เสร็จใน $(( ($(date +%s) - t0) / 60 )) นาที | $line"
    done_n=$((done_n + 1))
  else
    echo "          ❌ ล้มเหลว — ดู results/logs/$run.log"
    fail+=("$run")
  fi
done < "$QUEUE"

[ "$DRY" = 1 ] && exit 0
echo
echo "สรุป: สำเร็จ $done_n | ข้าม (มีผลแล้ว) $skip_n | ล้มเหลว ${#fail[@]} | รวม $(( ($(date +%s) - t_start) / 60 )) นาที"
for r in ${fail[@]+"${fail[@]}"}; do echo "  ❌ $r"; done
(cd "$ROOT" && "$PY" -m experiments.aggregate --out results/TABLE.md > /dev/null 2>&1) \
  && echo "ตารางรวม: results/TABLE.md"

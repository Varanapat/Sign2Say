"""รวม results/*/summary.json เป็นตาราง markdown

    python -m experiments.aggregate                  # ทุก run + ตาราง mean ± std ต่อ (exp, fold)
    python -m experiments.aggregate --fold 4         # เฉพาะ fold 4
    python -m experiments.aggregate --out results/TABLE.md

ข้าม run ที่ชื่อขึ้นต้นด้วย smoke
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.results import RESULTS

LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def load(root: Path, fold=None):
    runs = []
    for p in sorted(Path(root).glob("*/summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if s["exp"].startswith("smoke") or (fold is not None and s["fold"] != fold):
            continue
        runs.append(s)
    return sorted(runs, key=lambda s: (s["fold"], LEVEL_ORDER.get(s["level"], 9), s["exp"], s["seed"]))


def pct(v):
    return f"{100 * v:.2f}"


def per_run_table(runs):
    lines = ["| exp | level | fold | seed | params | test top-1 | test top-5 | test F1 | val top-1 | val F1 "
             "| best ep | min | latency ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in runs:
        t, v = s["test"], s["val"]
        lines.append(f"| {s['exp']} | {s['level']} | {s['fold']} | {s['seed']} | {s['params'] / 1e6:.2f}M "
                     f"| {pct(t['top1'])} | {pct(t['top5'])} | {pct(t['macro_f1'])} "
                     f"| {pct(v['top1'])} | {pct(v['macro_f1'])} | {s['best_epoch']} "
                     f"| {s['train_minutes']:.1f} | {s['latency_ms_b1']:.2f} |")
    return "\n".join(lines)


def mean_std_table(runs):
    groups = defaultdict(list)
    for s in runs:
        groups[(s["fold"], s["level"], s["exp"])].append(s)
    lines = ["| exp | level | fold | n seeds | test top-1 | test top-5 | test F1 |",
             "|---|---|---|---|---|---|---|"]
    for (fold, level, exp), ss in sorted(groups.items(), key=lambda kv: (kv[0][0], LEVEL_ORDER.get(kv[0][1], 9), kv[0][2])):
        cells = []
        for m in ("top1", "top5", "macro_f1"):
            vals = np.array([s["test"][m] for s in ss]) * 100
            cells.append(f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}" if len(vals) > 1 else f"{vals[0]:.2f}")
        lines.append(f"| {exp} | {level} | {fold} | {len(ss)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=RESULTS)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--out", type=Path, help="เขียนตารางลงไฟล์ (นอกจากพิมพ์ออกจอ)")
    a = ap.parse_args()
    runs = load(a.root, a.fold)
    if not runs:
        raise SystemExit(f"ไม่พบ summary.json ใน {a.root}")
    text = (f"## ทุก run ({len(runs)})\n\n{per_run_table(runs)}\n\n"
            f"## mean ± std ข้าม seed (std ใช้ ddof=1)\n\n{mean_std_table(runs)}\n")
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()

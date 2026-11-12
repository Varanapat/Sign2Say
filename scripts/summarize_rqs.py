"""Summarize the hand-local / ST-GCN runs (friend's RQ-S findings under our recipe and protocol).

Each comparison uses only folds where both models finished: mean test top-1 / macro-F1,
paired difference, folds improved, and McNemar pooled over all test clips of those folds.
Run while the queue is still going: unfinished folds are skipped and the fold count is printed.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
FOLDS = [0, 2, 4, 6, 8]
MODELS = ["S2S", "S2S-HL", "STGCN", "STGCN-HL"]


def load_runs():
    runs = {}
    for p in sorted(R.glob("tsl1s_si*.json"), key=lambda p: p.stat().st_mtime):
        for r in json.loads(p.read_text()):
            if r["model"] in MODELS and r["seed"] == 0 and r.get("variant", "base") == "base":
                runs[(r["split"], r["model"])] = r       # latest wins
    return runs


def correct(split, model):
    p = R / "preds" / f"{split}_{model}_seed0_test.csv"
    if not p.exists():
        return None
    rows = list(csv.reader(open(p)))[1:]
    return {x[0]: int(x[1]) == int(np.argmax(np.array(x[2:], float))) for x in rows}


def compare(runs, a_model, b_model, folds=FOLDS):
    fs = [f for f in folds if (f"tsl1s_si_fold{f}", a_model) in runs and (f"tsl1s_si_fold{f}", b_model) in runs]
    if not fs:
        print(f"{a_model} vs {b_model}: no common folds")
        return
    get = lambda m, k: np.array([runs[(f"tsl1s_si_fold{f}", m)]["test"][k] for f in fs]) * 100
    a, b = get(a_model, "top1"), get(b_model, "top1")
    d = a - b
    only_a = only_b = 0
    for f in fs:
        ca, cb = correct(f"tsl1s_si_fold{f}", a_model), correct(f"tsl1s_si_fold{f}", b_model)
        if ca and cb:
            only_a += sum(ca[k] and not cb[k] for k in ca if k in cb)
            only_b += sum(cb[k] and not ca[k] for k in ca if k in cb)
    p_m = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else float("nan")
    print(f"{a_model:9s} vs {b_model:9s} folds {fs}  {a.mean():5.1f} vs {b.mean():5.1f}  "
          f"Δ {d.mean():+5.1f} pp ({(d > 0).sum()}/{len(fs)} up)  McNemar {only_a}:{only_b} p={p_m:.2g}")


def main():
    runs = load_runs()
    print("| Model | " + " | ".join(f"fold {f}" for f in FOLDS) + " | mean top-1 | mean F1 | folds | s/run |")
    print("|---" * (len(FOLDS) + 5) + "|")
    for m in MODELS:
        cells, t1, f1, sec = [], [], [], []
        for f in FOLDS:
            r = runs.get((f"tsl1s_si_fold{f}", m))
            cells.append(f"{r['test']['top1'] * 100:.1f}" if r else "–")
            if r:
                t1.append(r["test"]["top1"] * 100)
                f1.append(r["test"]["macro_f1"] * 100)
                sec.append(r["seconds"])
        mean = f"{np.mean(t1):.1f}" if t1 else "–"
        print(f"| {m} | " + " | ".join(cells) + f" | {mean} | {np.mean(f1) if f1 else float('nan'):.1f} | "
              f"{len(t1)} | {np.mean(sec) if sec else float('nan'):.0f} |")
    print()
    compare(runs, "S2S-HL", "S2S")
    compare(runs, "STGCN-HL", "STGCN")
    compare(runs, "STGCN", "S2S")
    compare(runs, "STGCN-HL", "S2S-HL")
    compare(runs, "STGCN-HL", "S2S")


if __name__ == "__main__":
    main()

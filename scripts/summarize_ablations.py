"""Summarize the S2S-LTR ablations against base S2S on the same folds.

For each variant: mean test top-1 / macro-F1 over folds, paired difference vs base,
how many folds improved, and a McNemar test pooled over all test clips of those folds.
(With 5 folds the smallest possible Wilcoxon p is 0.0625, so the pooled McNemar is the
more informative test; both are reported.)
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
FOLDS = [0, 2, 4, 6, 8]


def load_runs():
    runs = {}
    for p in sorted(R.glob("tsl1s_*.json"), key=lambda p: p.stat().st_mtime):
        for r in json.loads(p.read_text()):
            if r["model"].startswith("S2S") and r["seed"] == 0:
                runs[(r["split"], r["model"])] = r       # latest wins
    return runs


def correct(split, model):
    p = R / "preds" / f"{split}_{model}_seed0_test.csv"
    if not p.exists():
        return None
    rows = list(csv.reader(open(p)))[1:]
    return {x[0]: int(x[1]) == int(np.argmax(np.array(x[2:], float))) for x in rows}


def compare(runs, label, split_fmt, model, base_split_fmt="tsl1s_si_fold{}", base_model="S2S"):
    fs = [f for f in FOLDS if (split_fmt.format(f), model) in runs and (base_split_fmt.format(f), base_model) in runs]
    if not fs:
        print(f"{label:26s} (not run yet)")
        return
    a = np.array([runs[(split_fmt.format(f), model)]["test"]["top1"] for f in fs]) * 100
    b = np.array([runs[(base_split_fmt.format(f), base_model)]["test"]["top1"] for f in fs]) * 100
    fa = np.array([runs[(split_fmt.format(f), model)]["test"]["macro_f1"] for f in fs]) * 100
    d = a - b
    only_a = only_b = 0
    for f in fs:
        ca, cb = correct(split_fmt.format(f), model), correct(base_split_fmt.format(f), base_model)
        if ca and cb:
            only_a += sum(ca[k] and not cb[k] for k in ca if k in cb)
            only_b += sum(cb[k] and not ca[k] for k in ca if k in cb)
    p_w = wilcoxon(d).pvalue if len(fs) >= 5 and np.any(d != 0) else float("nan")
    p_m = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else float("nan")
    print(f"{label:26s} folds {len(fs)}  top1 {a.mean():5.1f}  macroF1 {fa.mean():5.1f}  "
          f"Δ vs base {d.mean():+5.1f} pp ({(d > 0).sum()}/{len(fs)} folds up)  "
          f"McNemar {only_a}:{only_b} p={p_m:.2g}  Wilcoxon p={p_w:.3f}")


def main():
    runs = load_runs()
    base = [runs[(f"tsl1s_si_fold{f}", "S2S")]["test"]["top1"] * 100 for f in FOLDS
            if (f"tsl1s_si_fold{f}", "S2S") in runs]
    print(f"{'S2S base (C54)':26s} folds {len(base)}  top1 {np.mean(base) if base else float('nan'):5.1f}")
    for v, label in [("noface", "− face (48 pts)"), ("x71", "+ lips/eyes (X71)"),
                     ("novel", "− velocity"), ("nomask", "− valid mask"), ("flat", "− part-aware (flat)")]:
        compare(runs, label, "tsl1s_si_fold{}", f"S2S-{v}")
    compare(runs, "random-matched split", "tsl1s_rand_fold{}", "S2S")


if __name__ == "__main__":
    main()

"""10-fold signer-independent summary for TSL-ONE-S (S2S-LTR vs BiGRU vs LR/RF).

Reads every results/tsl1s_si*.json; when a (model, fold) was run more than once,
the latest run that has a prediction file wins. Reports per-fold metrics, mean ± std,
a Wilcoxon signed-rank test across folds, and a McNemar test pooled over all test clips.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

runs = {}
for p in sorted(R.glob("tsl1s_si*.json"), key=lambda p: p.stat().st_mtime):
    for r in json.loads(p.read_text()):
        fold = int(r["split"].rsplit("fold", 1)[1])
        pred = R / "preds" / f"{r['split']}_{r['model']}_seed{r['seed']}_test.csv"
        key = (r["model"], fold)
        if key not in runs or pred.exists() or not runs[key][1]:
            runs[key] = (r, pred.exists())

models = sorted({m for m, _ in runs}, key=["S2S", "BiGRU", "LR", "RF"].index)
folds = sorted({f for _, f in runs})
print("fold  " + "  ".join(f"{m:>16s}" for m in models) + "   (test top1 / macro-F1, %)")
for f in folds:
    cells = []
    for m in models:
        r = runs.get((m, f))
        cells.append(f"{r[0]['test']['top1']*100:6.1f} / {r[0]['test']['macro_f1']*100:5.1f}" if r else f"{'—':>16s}")
    print(f"{f:4d}  " + "  ".join(f"{c:>16s}" for c in cells))

print()
stats = {}
for m in models:
    fs = [f for f in folds if (m, f) in runs]
    a = {k: np.array([runs[(m, f)][0]["test"][k] for f in fs]) * 100 for k in ("top1", "top5", "macro_f1")}
    stats[m] = {f: runs[(m, f)][0]["test"]["top1"] for f in fs}
    print(f"{m:6s} folds={len(fs):2d}  top1 {a['top1'].mean():5.1f} ± {a['top1'].std():4.1f}   "
          f"top5 {a['top5'].mean():5.1f} ± {a['top5'].std():4.1f}   "
          f"macroF1 {a['macro_f1'].mean():5.1f} ± {a['macro_f1'].std():4.1f}   "
          f"min fold {a['top1'].min():.1f}")


def correct(model, fold):
    p = R / "preds" / f"tsl1s_si_fold{fold}_{model}_seed0_test.csv"
    if not p.exists():
        return None
    rows = list(csv.reader(open(p)))[1:]
    return {x[0]: int(x[1]) == int(np.argmax(np.array(x[2:], float))) for x in rows}


print()
for a, b in [("S2S", "BiGRU"), ("S2S", "LR"), ("BiGRU", "LR")]:
    common = [f for f in folds if f in stats.get(a, {}) and f in stats.get(b, {})]
    if len(common) >= 5:
        d = [stats[a][f] - stats[b][f] for f in common]
        w = wilcoxon(d)
        print(f"{a} − {b}: mean diff {np.mean(d)*100:+.1f} pp over {len(common)} folds "
              f"({sum(x > 0 for x in d)} folds better), Wilcoxon p = {w.pvalue:.4f}")
    only_a = only_b = 0
    used = []
    for f in common:
        ca, cb = correct(a, f), correct(b, f)
        if ca and cb:
            used.append(f)
            only_a += sum(ca[k] and not cb[k] for k in ca)
            only_b += sum(cb[k] and not ca[k] for k in ca)
    if used:
        print(f"   pooled McNemar on folds {used}: only-{a} {only_a}, only-{b} {only_b}, "
              f"p = {binomtest(only_a, only_a + only_b).pvalue:.2g}")

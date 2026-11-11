"""Summarize results/*.json as mean ± std over seeds/folds per (protocol, model)."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
rows = defaultdict(list)
for p in sorted((ROOT / "results").glob("*.json")):
    for r in json.loads(p.read_text()):
        rows[(r["protocol"], r["model"])].append(r)

print(f"{'protocol':12s} {'model':6s} {'runs':>4s}  {'val top1':>13s}  {'test top1':>13s}  "
      f"{'test top5':>13s}  {'test macroF1':>13s}")
for (proto, model), rs in rows.items():
    def ms(key, sub):
        a = np.array([r[sub][key] for r in rs]) * 100
        return f"{a.mean():5.1f} ± {a.std():4.1f}"
    print(f"{proto:12s} {model:6s} {len(rs):4d}  {ms('top1', 'val')}  {ms('top1', 'test')}  "
          f"{ms('top5', 'test')}  {ms('macro_f1', 'test')}")

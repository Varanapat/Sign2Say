"""Summarize C1/C2 CSLR runs: mean ± std over seeds of WER / SER, the S / D / I split,
and the no_space vs with_space breakdown (RQ7 / H7). Reads results/cslr.json and
results/preds/cslr/*.json; prints a markdown table."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results"


def sdi(ref, hyp):
    n, m = len(ref), len(hyp)
    D = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]   # (cost, S, D, I)
    for i in range(1, n + 1):
        D[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        D[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c, s, d, ins = D[i - 1][j - 1]
            best = (c, s, d, ins) if ref[i - 1] == hyp[j - 1] else (c + 1, s + 1, d, ins)
            c, s, d, ins = D[i - 1][j]
            best = min(best, (c + 1, s, d + 1, ins))
            c, s, d, ins = D[i][j - 1]
            best = min(best, (c + 1, s, d, ins + 1))
            D[i][j] = best
    return D[n][m][1:]


rows = json.loads((R / "cslr.json").read_text())
g = defaultdict(list)
for r in rows:
    p = R / "preds" / "cslr" / f"{r['protocol']}_{r['model']}_seed{r['seed']}_test.json"
    pr = json.loads(p.read_text())
    S = D = I = N = 0
    per = defaultdict(lambda: [0, 0])
    for e in pr:
        ref, hyp = e["ref"].split(), e["hyp"].split()
        s, d, i = sdi(ref, hyp)
        S, D, I, N = S + s, D + d, I + i, N + len(ref)
        per[e["variation"]][0] += s + d + i
        per[e["variation"]][1] += len(ref)
    g[(r["protocol"], r["model"])].append(dict(wer=r["test"]["wer"], ser=r["test"]["ser"], S=S / N, D=D / N, I=I / N,
                                               ns=per["no_space"][0] / per["no_space"][1],
                                               ws=per["with_space"][0] / per["with_space"][1] if per["with_space"][1] else np.nan,
                                               nws=sum(1 for e in pr if e["variation"] == "with_space"), n=len(pr),
                                               val=r["val"]["wer"], cfg=r["config"]))
f = lambda xs: f"{100*np.mean(xs):.1f} ± {100*np.std(xs):.1f}" if len(xs) > 1 else f"{100*xs[0]:.1f}"  # noqa: E731
print("| protocol | model | seeds | val WER | test WER | SER | S / D / I | no_space | with_space (n sent) |")
print("|---|---|---|---|---|---|---|---|---|")
for (p, m), v in g.items():
    k = lambda key: [x[key] for x in v]  # noqa: E731
    ws = [x for x in k("ws") if not np.isnan(x)]
    print(f"| {p} | {m} | {len(v)} | {f(k('val'))} | {f(k('wer'))} | {f(k('ser'))} | "
          f"{100*np.mean(k('S')):.1f} / {100*np.mean(k('D')):.1f} / {100*np.mean(k('I')):.1f} | {f(k('ns'))} | "
          f"{f(ws) if ws else '–'} ({v[0]['nws']}/{v[0]['n']}) |")

"""Task 13: combine the two evaluators' forms, mean scores per system, hallucination rate, Cohen's kappa.

Reads human_eval/form_evaluator{1,2}_filled.csv (7 columns, read by position):
    item_id, gloss, sentence, meaning 1-5, naturalness 1-5, hallucination y/n, note
and human_eval/key.csv (item_id -> systems t1|t2|t1+t2, cond gold|ctc).
An item with systems "t1+t2" (both systems gave the same sentence) counts for both systems.
kappa: quadratic weighted for the 1-5 scores, unweighted for the y/n flag.

    python nlp/human_eval_kappa.py            # -> outputs/human_eval_scores.json
    python nlp/human_eval_kappa.py --check    # self-test of the kappa formula
"""

import csv
import json
import sys
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parent
HE = BASE / "human_eval"
FIELDS = {"meaning": 3, "natural": 4, "halluc": 5}  # column positions in the form


def kappa(a, b, cats, quadratic=False):
    """Cohen's kappa = 1 - sum(w*O)/sum(w*E); w = squared distance (quadratic) or 0/1 (unweighted)."""
    n, k = len(a), len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    w = lambda i, j: (i - j) ** 2 if quadratic else int(i != j)
    obs = sum(w(idx[x], idx[y]) for x, y in zip(a, b)) / n
    pa = [sum(x == c for x in a) / n for c in cats]
    pb = [sum(y == c for y in b) / n for c in cats]
    exp = sum(w(i, j) * pa[i] * pb[j] for i in range(k) for j in range(k))
    return 1 - obs / exp if exp else float("nan")  # nan: both raters used a single category


def check():
    assert kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], range(1, 6), True) == 1
    assert abs(kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], range(1, 6), True) + 1) < 1e-9
    assert abs(kappa(list("yynn"), list("ynnn"), "yn") - 0.5) < 1e-9  # po .75, pe .5
    print("check ok")


def read_form(n):
    path = HE / f"form_evaluator{n}_filled.csv"
    if not path.exists():
        sys.exit(f"missing {path.name} (the filled forms come from person 1)")
    out, bad = {}, []
    for row in list(csv.reader(path.open(encoding="utf-8-sig")))[1:]:
        if not row or not row[0].strip():
            continue
        try:
            m, nat, h = (row[FIELDS[f]].strip() for f in ("meaning", "natural", "halluc"))
            m, nat, h = int(m), int(nat), h.lower()
            assert 1 <= m <= 5 and 1 <= nat <= 5 and h in ("y", "n")
            out[row[0].strip()] = {"meaning": m, "natural": nat, "halluc": h}
        except (ValueError, AssertionError, IndexError):
            bad.append(row[0])
    if bad:
        sys.exit(f"{path.name}: {len(bad)} items not filled or out of range, e.g. {bad[:5]}")
    return out


def main():
    if "--check" in sys.argv:
        return check()
    key = {r["item_id"]: r for r in csv.DictReader((HE / "key.csv").open(encoding="utf-8-sig"))}
    e1, e2 = read_form(1), read_form(2)
    assert set(e1) == set(e2) == set(key), "item ids differ between the forms and key.csv"
    ids = sorted(key)

    kap = {"meaning": kappa([e1[i]["meaning"] for i in ids], [e2[i]["meaning"] for i in ids], range(1, 6), True),
           "natural": kappa([e1[i]["natural"] for i in ids], [e2[i]["natural"] for i in ids], range(1, 6), True),
           "halluc": kappa([e1[i]["halluc"] for i in ids], [e2[i]["halluc"] for i in ids], "yn")}

    res = {}
    for sysname in ("t1", "t2"):
        for cond in ("gold", "ctc", "all"):
            sel = [i for i in ids if sysname in key[i]["systems"].split("+") and cond in (key[i]["cond"], "all")]
            both = [e[i] for i in sel for e in (e1, e2)]  # every rating from both evaluators
            res[f"{sysname}/{cond}"] = {"n_items": len(sel),
                                        "meaning": round(mean(x["meaning"] for x in both), 2),
                                        "natural": round(mean(x["natural"] for x in both), 2),
                                        "halluc_rate": round(mean(x["halluc"] == "y" for x in both), 3)}

    print(f"{len(ids)} items, 2 evaluators")
    print(f"{'system/cond':<12}{'n':>4}{'meaning':>9}{'natural':>9}{'halluc':>8}")
    for name, r in res.items():
        print(f"{name:<12}{r['n_items']:>4}{r['meaning']:>9.2f}{r['natural']:>9.2f}{r['halluc_rate']:>8.1%}")
    print("Cohen's kappa: " + " · ".join(f"{k} {v:.3f}" for k, v in kap.items())
          + "  (meaning/natural quadratic weighted, halluc unweighted)")
    out = BASE / "outputs" / "human_eval_scores.json"
    out.write_text(json.dumps({"n_items": len(ids), "kappa": {k: round(v, 3) for k, v in kap.items()},
                               "kappa_type": {"meaning": "quadratic weighted", "natural": "quadratic weighted", "halluc": "unweighted"},
                               "scores": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("-> nlp/outputs/human_eval_scores.json")


if __name__ == "__main__":
    main()

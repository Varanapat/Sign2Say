"""Ablation: choose T2 settings on TRAIN ONLY (leave-one-grammar-group-out). Never reads val/test refs.

For each train sequence, the few-shot examples come from the other grammar groups, so every
prediction is on an unseen structure (the same situation as the test group "เกิด หูหนวก").
Selection rule (fixed before running): highest chrF, then most exact matches; ties keep the baseline.

    python nlp/tune_t2.py        # key in TYPHOON_API_KEY -> outputs/t2_tuning.json
"""

import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from sacrebleu.metrics import CHRF

from run_t2 import API_MODEL, API_URL, OUT, messages, parse, pick_shots, rows
from t1_rules import normalize

BASELINE = (20, 0, False)  # the locked main run
GRID = list(itertools.product((10, 20, 24), (0, 1), (False, True)))  # shots (24 = API max: 50 messages), ref_index, keep_words


def main():
    key = os.environ.get("TYPHOON_API_KEY") or sys.exit("set TYPHOON_API_KEY first")
    client = OpenAI(api_key=key, base_url=API_URL, max_retries=8)
    train = [r for r in rows() if r["split"] == "train"]

    def one(job):
        (n, ri, keep), r = job
        shots = pick_shots(n, ri, exclude=r["grammar_group_id"])
        gloss = " ".join(normalize(r["gloss_words"]))
        resp = client.chat.completions.create(model=API_MODEL, messages=messages(gloss, shots, keep),
                                              temperature=0, max_tokens=1024)
        return parse(resp.choices[0].message.content)

    jobs = [(c, r) for c in GRID for r in train]
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(one, jobs))

    refs = [r["thai_refs"] for r in train]
    assert all(len(x) == 2 for x in refs)
    streams = [[x[i] for x in refs] for i in range(2)]
    ns = lambda s: s.replace(" ", "")
    results = []
    for i, c in enumerate(GRID):
        hyps = outs[i * len(train):(i + 1) * len(train)]
        results.append({"shots": c[0], "ref_index": c[1], "keep_words": c[2],
                        "chrF": round(CHRF().corpus_score(hyps, streams).score, 2),
                        "exact": sum(ns(h) in map(ns, x) for h, x in zip(hyps, refs)),
                        "preds": {r["gloss_seq_id"]: h for r, h in zip(train, hyps)}})
    rank = lambda x: (x["chrF"], x["exact"], (x["shots"], x["ref_index"], x["keep_words"]) == BASELINE)
    best = max(results, key=rank)
    print(f"{'shots':>5} {'ref':>3} {'keep':>5} {'chrF':>6} {'exact':>6}   (train, leave-one-group-out, n={len(train)})")
    for x in sorted(results, key=rank, reverse=True):
        tag = " <- baseline" if (x["shots"], x["ref_index"], x["keep_words"]) == BASELINE else ""
        print(f"{x['shots']:>5} {'a' + str(x['ref_index'] + 1):>3} {str(x['keep_words']):>5} {x['chrF']:>6.2f} {x['exact']:>3}/{len(train)}{tag}")
    chosen = {k: best[k] for k in ("shots", "ref_index", "keep_words", "chrF", "exact")}
    print("chosen:", chosen)
    (OUT / "t2_tuning.json").write_text(json.dumps({"model": API_MODEL, "protocol": "train only, leave-one-grammar-group-out",
                                                    "chosen": chosen, "configs": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("-> nlp/outputs/t2_tuning.json")


if __name__ == "__main__":
    main()

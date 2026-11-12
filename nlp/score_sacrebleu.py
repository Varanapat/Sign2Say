"""Task 11 (official numbers): chrF + BLEU-4 with sacreBLEU, 4 cells {T1, T2} x {gold, CTC} on test.

- refs: sentences.jsonl thai_refs (a1 + a2), multi-reference
- chrF : sacreBLEU CHRF() default (space:no = spaces ignored), raw strings
- BLEU : spaces removed, then PyThaiNLP newmm, sacreBLEU tokenize='none'
- CTC  : mean ± sample SD over seeds 0,1,2 (same as eval_compare.py)
- T2-4B (t2local4b_*) is scored as a supplementary row, not part of the 4 cells
- check: our chrF must equal comparison_scores.json (person 1's own implementation)

    python nlp/score_sacrebleu.py            # -> outputs/official_scores.json
"""

import json
import statistics as st
from pathlib import Path

import pythainlp
import sacrebleu
from pythainlp.tokenize import word_tokenize
from sacrebleu.metrics import BLEU, CHRF

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
SEEDS = (0, 1, 2)
SYSTEMS = {"T1": ("t1", "t1", "t1_from_hyp"), "T2": ("t2", "t2", "t2_from_hyp"),
           "T2-4B": ("t2local4b", "t2", "t2_from_hyp"),  # supplementary: model size
           "T2-tuned": ("t2tuned", "t2", "t2_from_hyp")}  # supplementary: settings chosen by tune_t2.py (train only)
# name -> (file prefix, gold key, ctc key)


def load(p):
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def tok(s):
    return " ".join(w for w in word_tokenize(s.replace(" ", ""), engine="newmm") if w.strip())


CHRF_M, BLEU_M = CHRF(), BLEU(tokenize="none")


def score(hyps, refs_list):
    k = {len(r) for r in refs_list}
    assert k == {2}, f"expected 2 refs per item, got {k}"
    streams = [[r[i] for r in refs_list] for i in range(2)]
    return {"chrF": CHRF_M.corpus_score(hyps, streams).score,
            "BLEU": BLEU_M.corpus_score([tok(h) for h in hyps], [[tok(r) for r in s] for s in streams]).score}


def main():
    sents = {s["gloss_seq_id"]: s for s in load(BASE / "annotations" / "sentences.jsonl")}
    test = [g for g, s in sents.items() if s["split"] == "test"]
    clips = {s: load(OUT / f"t1_ctc_template_C1-x16_seed{s}.jsonl") for s in SEEDS}  # video_id -> gloss_seq_id
    res = {}
    for name, (pre, gk, ck) in SYSTEMS.items():
        gold = {r["gloss_seq_id"]: r[gk] for r in load(OUT / f"{pre}_gold.jsonl")}
        r = {"gold_test": score([gold[g] for g in test], [sents[g]["thai_refs"] for g in test])}
        hyp, fromref = [], []
        for s in SEEDS:
            ctc = {x["video_id"]: x[ck] for x in load(OUT / f"{pre}_ctc_template_C1-x16_seed{s}.jsonl")}
            rows = clips[s]
            assert set(ctc) == {x["video_id"] for x in rows}, f"{pre} seed{s}: video ids differ"
            refs = [sents[x["gloss_seq_id"]]["thai_refs"] for x in rows]
            hyp.append(score([ctc[x["video_id"]] for x in rows], refs))
            fromref.append(score([gold[x["gloss_seq_id"]] for x in rows], refs))
        for m in ("chrF", "BLEU"):
            v = [h[m] for h in hyp]
            r[f"ctc_{m}"] = {"mean": st.mean(v), "sd": st.stdev(v), "seeds": v}
            r[f"ctc_fromref_{m}"] = st.mean(h[m] for h in fromref)
        res[name] = r

    print(f"sacrebleu {sacrebleu.__version__} · pythainlp {pythainlp.__version__} · test: {len(test)} sequences, 45 clips x 3 seeds\n")
    print(f"{'system':<7}{'gold chrF':>11}{'gold BLEU':>11}{'CTC chrF':>17}{'CTC BLEU':>17}{'drop chrF':>11}{'drop BLEU':>11}")
    for name, r in res.items():
        c, b = r["ctc_chrF"], r["ctc_BLEU"]
        print(f"{name:<7}{r['gold_test']['chrF']:>11.2f}{r['gold_test']['BLEU']:>11.2f}"
              f"{c['mean']:>10.2f} ± {c['sd']:<4.2f}{b['mean']:>10.2f} ± {b['sd']:<4.2f}"
              f"{r['ctc_fromref_chrF'] - c['mean']:>11.2f}{r['ctc_fromref_BLEU'] - b['mean']:>11.2f}")
    sig = {"chrF": str(CHRF_M.get_signature()),
           "BLEU": f"{BLEU_M.get_signature()}|pretok:pythainlp-{pythainlp.__version__}-newmm|spaces-removed"}
    print(f"\nchrF: {sig['chrF']}\nBLEU: {sig['BLEU']}")

    # person 1's chrF (own implementation) must match sacreBLEU
    theirs = json.loads((OUT / "comparison_scores.json").read_text(encoding="utf-8"))
    for name in ("T1", "T2"):
        t, r = theirs[name], res[name]
        for ours, key in ((r["gold_test"]["chrF"], "gold_test"), (r["ctc_chrF"]["mean"], "ctc_mean"),
                          (r["ctc_chrF"]["sd"], "ctc_sd"), (r["ctc_fromref_chrF"], "ctc_fromref")):
            assert abs(round(ours, 2) - t[key]) <= 0.01, f"{name} {key}: sacreBLEU {ours:.2f} vs person 1 {t[key]}"
    print("check ok: chrF of T1 and T2 == comparison_scores.json (person 1) to 0.01")

    def rnd(x):
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rnd(v) for v in x]
        return round(x, 2)
    (OUT / "official_scores.json").write_text(json.dumps(
        {"results": rnd(res), "signatures": sig, "note": "T2-4B is supplementary (not in human eval)"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("-> nlp/outputs/official_scores.json")


if __name__ == "__main__":
    main()

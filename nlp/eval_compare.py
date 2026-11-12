#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""งาน 11 (ฝั่ง chrF) + งาน 14: เทียบ T1 กับ T2 แล้วเขียน reports/comparison.md

chrF ใช้ฟังก์ชันเดียวกับ eval_t1_prelim.py ซึ่งตรวจเทียบกับ sacreBLEU 2.6.0 แล้วตรงกัน 0.00 จุด
BLEU-4 (newmm) ยังไม่ได้ทำที่นี่ เพราะเครื่องนี้ติดตั้ง pythainlp ไม่ได้ -> รันบน Windows ด้วย verify_chrf.py

ใช้งาน:  python eval_compare.py
"""
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from eval_t1_prelim import chrf, err_type, strip
from t1_rules import normalize

BASE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def exact(hyps, refs_list):
    return 100 * sum(strip(h) in {strip(r) for r in refs} for h, refs in zip(hyps, refs_list)) / len(hyps)


def ms(v):
    return f"{st.mean(v):.2f} ± {st.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"


def main():
    sents = {s["gloss_seq_id"]: s for s in load(BASE/"annotations"/"sentences.jsonl")}
    t1g = {r["gloss_seq_id"]: r["t1"] for r in load(BASE/"outputs"/"t1_gold.jsonl")}
    t2g = {r["gloss_seq_id"]: r["t2"] for r in load(BASE/"outputs"/"t2_gold.jsonl")}
    meta = json.loads((BASE/"outputs"/"t2_meta.json").read_text(encoding="utf-8"))

    L = ["# T1 (กฎ) vs T2 (Typhoon): ผลเปรียบเทียบ", "",
         f"- T2: `{meta['model']}` · temperature {meta['temperature']} · few-shot {len(meta['shots'])} คู่ (train เท่านั้น) · รันเมื่อ {meta['run_at']}",
         "- T1 และ T2 รับ input เดียวกัน (ผ่าน `t1_rules.normalize()` ยุบคำซ้ำติดกัน)",
         "- chrF: สูตรเดียวกับที่ตรวจเทียบ sacreBLEU 2.6.0 แล้วตรงกัน 0.00 จุด · reference 2 ชุด · ไม่นับช่องว่าง",
         "- exact = ตรงกับ reference ชุดใดชุดหนึ่งทุกตัวอักษร", "",
         "## 1. ตารางหลัก 4 ช่อง (ชุด test)", "",
         "| ระบบ | gloss ที่ถูกต้อง (12 ลำดับ) | gloss ที่ CTC ทาย (45 คลิป × 3 seed) |", "|---|---|---|"]

    res = {}
    test_ids = [g for g in sents if sents[g]["split"] == "test"]
    refs_test = [sents[g]["thai_refs"] for g in test_ids]
    for name, gold in (("T1", t1g), ("T2", t2g)):
        res[f"{name}_gold_test"] = chrf([gold[g] for g in test_ids], refs_test)
        res[f"{name}_gold_test_ex"] = exact([gold[g] for g in test_ids], refs_test)
    hyp = defaultdict(list)
    for s in SEEDS:
        rows1 = load(BASE/f"outputs/t1_ctc_template_C1-x16_seed{s}.jsonl")
        t2c = {r["video_id"]: r["t2_from_hyp"] for r in load(BASE/f"outputs/t2_ctc_template_C1-x16_seed{s}.jsonl")}
        refs = [sents[r["gloss_seq_id"]]["thai_refs"] for r in rows1]
        hyp["T1"].append(chrf([r["t1_from_hyp"] for r in rows1], refs))
        hyp["T2"].append(chrf([t2c[r["video_id"]] for r in rows1], refs))
        hyp["T1_ref"].append(chrf([r["t1_from_ref"] for r in rows1], refs))
        hyp["T2_ref"].append(chrf([t2g[r["gloss_seq_id"]] for r in rows1], refs))
    for name in ("T1", "T2"):
        L.append(f"| {name} | {res[f'{name}_gold_test']:.2f} | {ms(hyp[name])} |")
    L += ["", f"ต่างกันบน gloss ถูกต้อง: T2 − T1 = {res['T2_gold_test'] - res['T1_gold_test']:+.2f} · "
              f"บน gloss ที่ทาย: {st.mean(hyp['T2']) - st.mean(hyp['T1']):+.2f}", "",
          "ตรง reference ทุกตัวอักษร (gloss ถูกต้อง, test): "
          f"T1 {res['T1_gold_test_ex']:.1f}% · T2 {res['T2_gold_test_ex']:.1f}%", "",
          "<b>ผลตกลงเท่าไรเมื่อใช้ gloss ที่ทายแทน gloss ที่ถูกต้อง</b> (วัดบน 45 คลิปเดียวกัน)", "",
          "| ระบบ | จาก gloss ถูกต้อง | จาก gloss ที่ทาย | ลดลง |", "|---|---|---|---|"]
    for name in ("T1", "T2"):
        a, b = st.mean(hyp[f"{name}_ref"]), st.mean(hyp[name])
        L.append(f"| {name} | {a:.2f} | {ms(hyp[name])} | {a - b:.2f} |")

    L += ["", "## 2. ทนต่อ error ของ CTC แค่ไหน (รวม 3 seed)", "",
          '"กู้ได้" = คำแปลจาก gloss ที่ทาย เหมือนกับคำแปลจาก gloss ที่ถูกต้องของระบบเดียวกันทุกตัวอักษร', "",
          "| ชนิด error | คลิป | T1 กู้ได้ | T2 กู้ได้ |", "|---|---|---|---|"]
    cnt = defaultdict(lambda: [0, 0, 0])
    examples = {}
    for s in SEEDS:
        rows1 = load(BASE/f"outputs/t1_ctc_template_C1-x16_seed{s}.jsonl")
        t2c = {r["video_id"]: r["t2_from_hyp"] for r in load(BASE/f"outputs/t2_ctc_template_C1-x16_seed{s}.jsonl")}
        for r in rows1:
            t = err_type(r["ref_words"], r["hyp_words"])
            ok1 = strip(r["t1_from_hyp"]) == strip(r["t1_from_ref"])
            ok2 = strip(t2c[r["video_id"]]) == strip(t2g[r["gloss_seq_id"]])
            c = cnt[t]; c[0] += 1; c[1] += ok1; c[2] += ok2
            if t != "ถูกต้อง":
                examples.setdefault((t, " ".join(r["hyp_words"])),
                                    (r, t2c[r["video_id"]], t2g[r["gloss_seq_id"]], ok1, ok2))
    for t in ("ถูกต้อง", "คำซ้ำติดกัน", "ตกคำ", "คำแทรก", "แทนที่คำ"):
        n, a, b = cnt[t]
        if n:
            L.append(f"| {t} | {n} | {a} ({100*a/n:.0f}%) | {b} ({100*b/n:.0f}%) |")

    L += ["", "## 3. ตัวอย่างที่ T1 กับ T2 ต่างกัน (gloss ที่ CTC ทายผิด)", "",
          "| ชนิด | gloss ที่ทาย | T1 | T2 | ควรได้ (ref) |", "|---|---|---|---|---|"]
    for (t, hw), (r, a, b, ok1, ok2) in sorted(examples.items()):
        if strip(a) != strip(r["t1_from_hyp"]):
            L.append(f"| {t} | {hw} | {r['t1_from_hyp']} | {a} | {' / '.join(sents[r['gloss_seq_id']]['thai_refs'])} |")

    L += ["", "## 4. ชุด test: เทียบรายประโยค (gloss ที่ถูกต้อง)", "",
          "| gloss | T1 | T2 | reference |", "|---|---|---|---|"]
    for g in test_ids:
        L.append(f"| {' '.join(sents[g]['gloss_words'])} | {t1g[g]} | {t2g[g]} | {' / '.join(sents[g]['thai_refs'])} |")

    (BASE/"reports").mkdir(exist_ok=True)
    (BASE/"reports"/"comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    out = {"T1": {"gold_test": round(res["T1_gold_test"], 2), "ctc_mean": round(st.mean(hyp["T1"]), 2),
                  "ctc_sd": round(st.stdev(hyp["T1"]), 2), "ctc_fromref": round(st.mean(hyp["T1_ref"]), 2)},
           "T2": {"gold_test": round(res["T2_gold_test"], 2), "ctc_mean": round(st.mean(hyp["T2"]), 2),
                  "ctc_sd": round(st.stdev(hyp["T2"]), 2), "ctc_fromref": round(st.mean(hyp["T2_ref"]), 2)},
           "_meta": {"chrf": "own impl, verified == sacreBLEU 2.6.0 (diff 0.00)", "t2_model": meta["model"]}}
    (BASE/"outputs"/"comparison_scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"\n-> reports/comparison.md · outputs/comparison_scores.json")


if __name__ == "__main__":
    main()

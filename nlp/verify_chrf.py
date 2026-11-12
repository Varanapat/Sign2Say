#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ตรวจว่า chrF ที่เราคำนวณเองตรงกับ sacreBLEU หรือไม่

ทำไมต้องมีไฟล์นี้: เครื่องที่ใช้พัฒนาตอนแรกติดตั้ง sacrebleu ไม่ได้ (pip ถูกบล็อกที่ proxy)
สูตรใน eval_t1_prelim.py จึงเขียนเอง ตรวจเลขคณิตแล้วแต่ยังไม่เคยเทียบกับไลบรารีจริง

วิธีใช้ (รันในเทอร์มินัลของตัวเอง)
    pip install sacrebleu
    python verify_chrf.py

อ่านผล
    ต่างกัน <= 1.0 จุด  -> ใช้ตัวเลขของ sacreBLEU เป็นทางการได้เลย สูตรเราถือว่าตรง
    ต่างกัน 1-5 จุด     -> น่าจะต่างที่รายละเอียดเล็กน้อย ใช้ของ sacreBLEU และระบุในรายงาน
    ต่างกัน > 5 จุด     -> มีอะไรผิด ต้องหาสาเหตุก่อนใช้ตัวเลขใด ๆ
"""
import json
import statistics as st
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

try:
    from sacrebleu.metrics import CHRF
except ImportError:
    sys.exit("ยังไม่ได้ติดตั้ง sacrebleu — รัน:  pip install sacrebleu")

from eval_t1_prelim import chrf as our_chrf  # noqa: E402


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


METRIC = CHRF()   # ใช้ instance เดียวตลอด เพื่อให้ขอ signature ได้หลังวัดผลเสร็จ


def sacre(hyps, refs_list):
    """refs_list = list ของ list reference ต่อประโยค -> สลับเป็น reference stream ให้ sacreBLEU"""
    k = max(len(r) for r in refs_list)
    streams = [[(r[i] if i < len(r) else r[-1]) for r in refs_list] for i in range(k)]
    return METRIC.corpus_score(hyps, streams).score


def row(name, hyps, refs_list, out, official=None, key=None):
    a, b = sacre(hyps, refs_list), our_chrf(hyps, refs_list)
    flag = "ok" if abs(a - b) <= 1.0 else ("ตรวจ" if abs(a - b) <= 5.0 else "ผิดแน่")
    print(f"| {name:34s} | {len(hyps):3d} | {a:7.2f} | {b:7.2f} | {a - b:+6.2f} | {flag} |")
    out.append(abs(a - b))
    if official is not None:
        official[key] = {"n": len(hyps), "chrf_sacrebleu": round(a, 2)}
    return a


def main():
    sents = {s["gloss_seq_id"]: s for s in load(BASE/"annotations"/"sentences.jsonl")}
    gold = load(BASE/"outputs"/"t1_gold.jsonl")
    print("| ช่อง                               |   n | sacreBLEU | ของเรา |  ต่าง | ผล |")
    print("|------------------------------------|-----|-----------|--------|-------|-----|")
    diffs, official = [], {}
    for split in ("train", "val", "test", "all"):
        rows = [g for g in gold if split == "all" or g["split"] == split]
        row(f"gloss ถูกต้อง · {split}", [g["t1"] for g in rows],
            [sents[g["gloss_seq_id"]]["thai_refs"] for g in rows], diffs, official, f"gold_{split}")
    hyp_scores, ref_scores = [], []
    for f in sorted((BASE/"outputs").glob("t1_ctc_*.jsonl")):
        rows = load(f)
        refs = [sents[r["gloss_seq_id"]]["thai_refs"] for r in rows]
        seed = f.stem.split("_")[-1]
        ref_scores.append(row(f"CTC · {seed} · จาก gloss ถูกต้อง",
                              [r["t1_from_ref"] for r in rows], refs, diffs, official, f"ctc_{seed}_fromref"))
        hyp_scores.append(row(f"CTC · {seed} · จาก gloss ที่ทาย",
                              [r["t1_from_hyp"] for r in rows], refs, diffs, official, f"ctc_{seed}_fromhyp"))
    ms = lambda v: {"mean": round(st.mean(v), 2), "sd": round(st.stdev(v), 2) if len(v) > 1 else 0.0}
    official["ctc_mean_fromref"], official["ctc_mean_fromhyp"] = ms(ref_scores), ms(hyp_scores)
    print(f"\nเฉลี่ย 3 seed: จาก gloss ถูกต้อง {official['ctc_mean_fromref']['mean']:.2f} "
          f"± {official['ctc_mean_fromref']['sd']:.2f} · จาก gloss ที่ทาย "
          f"{official['ctc_mean_fromhyp']['mean']:.2f} ± {official['ctc_mean_fromhyp']['sd']:.2f}")
    worst = max(diffs)
    official["_meta"] = {"signature": str(METRIC.get_signature()),
                         "max_diff_vs_our_impl": round(worst, 4), "system": "T1"}
    out = BASE/"outputs"/"t1_official_scores.json"
    out.write_text(json.dumps(official, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"เขียนตัวเลขทางการลง {out.name}")
    print(f"sacreBLEU signature: {METRIC.get_signature()}")
    print(f"ต่างกันมากที่สุด {worst:.2f} จุด -> " +
          ("สูตรของเราตรงกับ sacreBLEU ใช้ตัวเลข sacreBLEU ได้เลย" if worst <= 1.0 else
           "ต่างเล็กน้อย ใช้ของ sacreBLEU และระบุในรายงาน" if worst <= 5.0 else
           "ต่างมาก ต้องหาสาเหตุก่อนใช้ตัวเลขใด ๆ"))


if __name__ == "__main__":
    main()

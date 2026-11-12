#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""งาน 8: รวมคำแปลอ้างอิง 2 ชุด (refs_annotator1.csv, refs_annotator2.csv) เข้า sentences.jsonl
และเขียนรายงานจุดที่ต่างกัน -> annotations/refs_diff.md

การปรับรูปแบบ (ไม่แตะไฟล์ csv ต้นฉบับ; เก็บข้อความเดิมไว้ใน thai_refs_raw):
  - ชุดที่เขียนเว้นวรรคทุกคำ -> ลบช่องว่างออก (ภาษาไทยไม่เว้นวรรคระหว่างคำ; chrF ไม่นับช่องว่างอยู่แล้ว)
  - แก้คำสะกดผิดตามตาราง TYPOS เท่านั้น

ใช้งาน:  python merge_refs.py
"""
import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
ANN = BASE / "annotations"
TYPOS = {"เทียว": "เที่ยว"}          # สะกดผิด -> ถูก (บันทึกไว้ใน refs_diff.md)


def read(name):
    with open(ANN / name, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    return {r[0]: (r[2].strip(), r[3].strip() if len(r) > 3 else "") for r in rows if r and r[0]}


def clean(text, word_spaced):
    fixes = []
    for bad, good in TYPOS.items():
        if bad in text:
            text = text.replace(bad, good)
            fixes.append(f"{bad}→{good}")
    if word_spaced:
        text = re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text).strip(), fixes


def key(s):
    return re.sub(r"\s+", "", s)


def main():
    sents = [json.loads(l) for l in (ANN / "sentences.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sets = {"a1": read("refs_annotator1.csv"), "a2": read("refs_annotator2.csv")}
    # ชุดไหนเว้นวรรคทุกคำ: ดูจากสัดส่วนประโยคที่มีช่องว่าง >= 2 ช่อง
    spaced = {a: sum(t.count(" ") >= 2 for t, _ in d.values()) > len(d) / 2 for a, d in sets.items()}
    diff, fixed, missing = [], [], []
    for s in sents:
        gid = s["gloss_seq_id"]
        refs, raws, who = [], [], []
        for a, d in sets.items():
            raw, note = d.get(gid, ("", ""))
            if not raw:
                missing.append((gid, a))
                continue
            txt, fx = clean(raw, spaced[a])
            if fx:
                fixed.append((gid, a, ", ".join(fx)))
            refs.append(txt); raws.append(raw); who.append(a)
        s["thai_refs"], s["thai_refs_raw"], s["annotator_ids"] = refs, raws, who
        if len(refs) == 2 and key(refs[0]) != key(refs[1]):
            diff.append(s)
    (ANN / "sentences.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in sents) + "\n", encoding="utf-8")

    n = len(sents)
    L = ["# คำแปลอ้างอิง: จุดที่ 2 ชุดต่างกัน", "",
         f"- ลำดับทั้งหมด {n} · ตรงกันทุกตัวอักษร (ไม่นับช่องว่าง) {n - len(diff)} ({(n - len(diff)) / n:.0%}) · ต่างกัน {len(diff)}",
         f"- ชุดที่เว้นวรรคทุกคำและถูกลบช่องว่าง: {', '.join(a for a, v in spaced.items() if v) or '-'}",
         f"- แก้คำสะกดผิด: {'; '.join(f'{g} ({a}) {f}' for g, a, f in fixed) or '-'}",
         f"- ช่องที่ยังว่าง: {missing or '-'}", "",
         "| id | split | gloss | ชุดที่ 1 (a1) | ชุดที่ 2 (a2) |", "|---|---|---|---|---|"]
    for s in diff:
        L.append(f"| {s['gloss_seq_id']} | {s['split']} | {' '.join(s['gloss_words'])} | {s['thai_refs'][0]} | {s['thai_refs'][1]} |")
    (ANN / "refs_diff.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

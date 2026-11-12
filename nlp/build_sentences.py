#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""งาน 5: สร้าง annotations/sentences.jsonl (1 บรรทัดต่อลำดับคำ) จาก split S-template
และแบบฟอร์มเปล่าสำหรับเขียนคำแปลอ้างอิง (ไม่เขียนทับถ้ามีไฟล์อยู่แล้ว)

ใช้งาน:  python build_sentences.py
"""
import csv
import json
import re
from pathlib import Path

from t1_rules import translate, word

BASE = Path(__file__).resolve().parent
SPLIT = BASE / "data" / "tsl51_sentence_template.json"   # จาก branch core-pipeline: splits/
ANN = BASE / "annotations"


def main():
    sp = json.loads(SPLIT.read_text(encoding="utf-8"))
    seqs = {}
    for clip, split in sp.items():
        sent_id = re.match(r"(.*)_(no_space|with_space)_\d+$", clip).group(1)
        sign_ids = sent_id.split("__")
        key = tuple(word(s) for s in sign_ids)
        e = seqs.setdefault(key, {"variants": [], "splits": set(), "clips": []})
        if sign_ids not in e["variants"]:
            e["variants"].append(sign_ids)
        e["splits"].add(split)
        e["clips"].append(clip)

    order = {"train": 0, "val": 1, "test": 2}
    rows = []
    for key, e in sorted(seqs.items(), key=lambda kv: (order[min(kv[1]["splits"], key=order.get)], kv[0])):
        assert len(e["splits"]) == 1, (key, e["splits"])
        r = translate(list(key))
        assert r["rule"] != "fallback", key
        rows.append({"gloss_words": list(key), "sign_id_variants": e["variants"],
                     "grammar_group_id": r["rule"], "grammar_group": r["group"],
                     "split": e["splits"].pop(), "thai_refs": [], "annotator_ids": [],
                     "verified_by_signer": False, "clips": sorted(e["clips"])})
    for i, r in enumerate(rows, 1):
        r_id = {"gloss_seq_id": f"gs_{i:03d}"}
        r_id.update(r)
        rows[i - 1] = r_id

    ANN.mkdir(exist_ok=True)
    (ANN / "sentences.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"sentences.jsonl: {len(rows)} ลำดับคำ, {sum(len(r['sign_id_variants']) for r in rows)} ลำดับ sign_id, "
          f"{sum(len(r['clips']) for r in rows)} คลิป")

    for who in ("annotator1", "annotator2"):
        f = ANN / f"refs_{who}.csv"
        if f.exists():
            print(f"{f.name}: มีอยู่แล้ว ไม่เขียนทับ")
            continue
        with open(f, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["gloss_seq_id", "gloss (ลำดับคำภาษามือ)", "คำแปลภาษาไทย", "หมายเหตุ"])
            for r in rows:
                w.writerow([r["gloss_seq_id"], " ".join(r["gloss_words"]), "", ""])
        print(f"{f.name}: สร้างแบบฟอร์มเปล่า {len(rows)} แถว")


if __name__ == "__main__":
    main()

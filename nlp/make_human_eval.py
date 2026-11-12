#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""งาน 12: สร้างแบบประเมินโดยคน (blind, สุ่มลำดับ) จากผล T1 และ T2

รายการที่ประเมิน (เฉพาะชุด test ของ S-template)
  A) แปลจาก gloss ที่ถูกต้อง: 12 ลำดับ
  B) แปลจาก gloss ที่ CTC ทายผิด: ทุกคู่ (ลำดับที่ถูกต้อง, ลำดับที่ทาย) ที่ไม่ซ้ำ จากทุก seed
ถ้า T1 และ T2 ให้ประโยคเดียวกันสำหรับรายการเดียวกัน จะถามครั้งเดียวแล้วนับให้ทั้งสองระบบ

ผลลัพธ์ (human_eval/)
  form_evaluator1.csv, form_evaluator2.csv   ส่งให้ผู้ประเมิน (ไม่มีชื่อระบบ)
  key.csv                                    เฉลยว่าแต่ละข้อมาจากระบบไหน (ห้ามส่งให้ผู้ประเมิน)

ไฟล์ T2 ที่ต้องมี (จากคนที่ 2):  outputs/t2_gold.jsonl                     {gloss_seq_id, t2}
                                outputs/t2_ctc_template_C1-x16_seed*.jsonl {video_id, t2_from_hyp}
ใช้งาน:  python make_human_eval.py            (ต้องมีไฟล์ T2)
        python make_human_eval.py --t1-only   (ทดลองดูหน้าตาแบบฟอร์มด้วย T1 อย่างเดียว)
"""
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-only", action="store_true")
    ap.add_argument("--out-dir", default=str(BASE / "human_eval"))
    ap.add_argument("--seed", type=int, default=20261112)
    a = ap.parse_args()

    sents = {s["gloss_seq_id"]: s for s in load(BASE / "annotations" / "sentences.jsonl")}
    systems = ["t1"] if a.t1_only else ["t1", "t2"]
    cand = {}          # (cond, gloss_seq_id, hyp_words) -> {system: sentence}
    for sysname in systems:
        f = OUT / f"{sysname}_gold.jsonl"
        if not f.exists():
            sys.exit(f"ยังไม่มี {f.name} (รอผล T2 จากคนที่ 2 หรือใช้ --t1-only เพื่อทดลอง)")
        for r in load(f):
            if sents[r["gloss_seq_id"]]["split"] == "test":
                key = ("gold", r["gloss_seq_id"], " ".join(sents[r["gloss_seq_id"]]["gloss_words"]))
                cand.setdefault(key, {})[sysname] = r[sysname]
    t1_ctc = {}
    for f in sorted(OUT.glob("t1_ctc_template_C1-x16_seed*.jsonl")):
        t2f = OUT / f.name.replace("t1_", "t2_")
        t2 = {}
        if not a.t1_only:
            if not t2f.exists():
                sys.exit(f"ยังไม่มี {t2f.name}")
            t2 = {r["video_id"]: r["t2_from_hyp"] for r in load(t2f)}
        for r in load(f):
            if r["ref_words"] == r["hyp_words"]:
                continue
            key = ("ctc", r["gloss_seq_id"], " ".join(r["hyp_words"]))
            cand.setdefault(key, {}).setdefault("t1", r["t1_from_hyp"])
            if t2:
                cand[key].setdefault("t2", t2[r["video_id"]])

    items = []
    for (cond, gid, hyp), outs in cand.items():
        by_text = {}
        for sysname, text in outs.items():
            by_text.setdefault(re.sub(r"\s+", " ", text).strip(), []).append(sysname)
        for text, who in by_text.items():
            items.append({"cond": cond, "gloss_seq_id": gid, "gloss_input": hyp,
                          "gloss_true": " ".join(sents[gid]["gloss_words"]), "sentence": text,
                          "systems": "+".join(sorted(who))})
    random.Random(a.seed).shuffle(items)
    for i, it in enumerate(items, 1):
        it["item_id"] = f"q{i:03d}"

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    head = ["item_id", "gloss ที่ถูกต้อง (ความหมายที่ต้องการสื่อ)", "ประโยคที่ให้ประเมิน",
            "ความหมายครบถูก 1-5", "ภาษาไทยเป็นธรรมชาติ 1-5", "เติมข้อมูลที่ไม่มีใน gloss (y/n)", "หมายเหตุ"]
    for k in (1, 2):
        f = out / f"form_evaluator{k}.csv"
        if f.exists():
            print(f"{f.name}: มีอยู่แล้ว ไม่เขียนทับ"); continue
        with open(f, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh); w.writerow(head)
            for it in items:
                w.writerow([it["item_id"], it["gloss_true"], it["sentence"], "", "", "", ""])
    with open(out / "key.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "systems", "cond", "gloss_seq_id", "gloss_true", "gloss_input", "sentence"])
        for it in items:
            w.writerow([it[k] for k in ("item_id", "systems", "cond", "gloss_seq_id", "gloss_true", "gloss_input", "sentence")])
    n_g = sum(it["cond"] == "gold" for it in items)
    print(f"{len(items)} ข้อ (gloss ถูกต้อง {n_g}, gloss ที่ทาย {len(items) - n_g}) -> {out}")


if __name__ == "__main__":
    main()

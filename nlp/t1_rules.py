#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T1: แปลลำดับ gloss (TSL-51) -> ประโยคไทย ด้วยกฎ/แม่แบบ  (RQ8, งาน 9)

ที่มาของกฎ: ตาราง "Sentence Structures" 13 กลุ่ม และตารางหมวดคำ ใน README ของ TSL-51
(ไม่ได้ใช้คำแปลอ้างอิงของชุดใดในการออกแบบ)

ขั้นตอน
  1) normalize : ตัด _var_N / คำอธิบายหลัง "_" ออก แล้วยุบคำซ้ำที่อยู่ติดกัน
  2) template  : จับคู่กับแม่แบบของ 13 กลุ่ม (ต้องตรงทั้งลำดับ)
  3) fallback  : ถ้าไม่ตรงแม่แบบใด เรียงคำใหม่ตามหมวดคำ (ใช้กับ gloss ที่ CTC ทายผิด)

ใช้งาน:  python t1_rules.py            (อ่าน annotations/sentences.jsonl และผลทาย CTC แล้วเขียน outputs/)
"""
import argparse
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ---- หมวดคำ (จากตารางใน README ของ TSL-51 เฉพาะ 30 คำที่ใช้ในประโยค)
SUBJ = {"แม่", "คุณ", "แฟน", "พี่", "น้อง", "ฉัน", "ยาย", "แมว"}
OBJ = {"ขนมปัง", "ข้าว", "ไข่", "ภาษามือ"}
PLACE = {"กรุงเทพ"}
TIME = {"วันนี้", "พรุ่งนี้"}
GREET = {"สวัสดี", "ขอบคุณ", "ขอโทษ"}
HELLO = {"สวัสดี", "ขอบคุณ"}
VERB = {"กิน", "เรียน", "เที่ยว", "เกิด", "ชื่อ", "แต่งงาน"}
STATE = {"สบายดี", "หูหนวก"}
QUES = {"อะไร", "ที่ไหน"}

# ---- แม่แบบ: (รหัสกลุ่ม, ชื่อกลุ่มตาม README, รูปแบบ gloss, ประโยคไทย)
# สมาชิกของรูปแบบ: str = ต้องเป็นคำนั้น, set = ช่องว่าง (คำใดก็ได้ในหมวด) -> {0}, {1}, ... ตามลำดับ
TEMPLATES = [
    ("G01", "<ประธาน> กิน ขนมปัง", ["ขนมปัง", SUBJ, "กิน"], "{0}กินขนมปัง"),
    ("G02", "<ประธาน> ชอบ กิน ไข่", ["ไข่", SUBJ, "กิน", "ชอบ"], "{0}ชอบกินไข่"),
    ("G03", "<ประธาน> เกิด หูหนวก", [SUBJ, "เกิด", "หูหนวก"], "{0}หูหนวกตั้งแต่เกิด"),
    ("G04", "เรียนภาษามือ", ["ภาษามือ", SUBJ, "เรียน"], "{0}เรียนภาษามือ"),
    ("G05", "ชอบเรียนภาษามือ", ["ภาษามือ", SUBJ, "เรียน", "ชอบ"], "{0}ชอบเรียนภาษามือ"),
    ("G06", "<ประธาน> กิน ข้าว", [SUBJ, "ข้าว", "กิน"], "{0}กินข้าว"),
    ("G07", "<ประธาน> ชอบ เที่ยว กรุงเทพ", ["กรุงเทพ", SUBJ, "เที่ยว", "ชอบ"], "{0}ชอบเที่ยวกรุงเทพ"),
    ("G08", "<สวัสดี/ขอบคุณ> <ฉัน> สบายดี", [HELLO, SUBJ, "สบายดี"], "{0} {1}สบายดี"),
    ("G09", "คุณ <แต่งงาน (กับ)/ (เป็น) แฟน (กับ)> ฉัน", [SUBJ, SUBJ, "แต่งงาน"], "{0}แต่งงานกับ{1}"),
    ("G09", "คุณ <แต่งงาน (กับ)/ (เป็น) แฟน (กับ)> ฉัน", [SUBJ, SUBJ, "แฟน"], "{0}เป็นแฟนกับ{1}"),
    ("G10", "ขอโทษ <ฉัน> หูหนวก", ["ขอโทษ", SUBJ, "หูหนวก"], "ขอโทษ {0}หูหนวก"),
    ("G11", "ขอโทษ <ฉัน> ไป กิน ข้าว", ["ขอโทษ", SUBJ, "ข้าว", "กิน", "ไป"], "ขอโทษ {0}ไปกินข้าว"),
    ("G12", "พรุ่งนี้ คุณ ไป กิน ข้าว ที่ไหน", [TIME, SUBJ, "ข้าว", "กิน", "ไป", "ที่ไหน"], "{0}{1}ไปกินข้าวที่ไหน"),
    ("G13", "คุณ <ชื่อ อะไร / เกิด ที่ไหน>", [SUBJ, "ชื่อ", "อะไร"], "{0}ชื่ออะไร"),
    ("G13", "คุณ <ชื่อ อะไร / เกิด ที่ไหน>", [SUBJ, "เกิด", "ที่ไหน"], "{0}เกิดที่ไหน"),
    # 2 รูปที่มีในข้อมูลแต่ไม่อยู่ในตาราง 13 กลุ่มโดยตรง (จัดเข้ากลุ่ม 13 ที่ใกล้ที่สุด)
    ("G13x", "คุณ <ชื่อ อะไร / เกิด ที่ไหน> (รูปใกล้เคียง ไม่อยู่ในตาราง README)", [SUBJ, "ไป", "ที่ไหน"], "{0}ไปที่ไหน"),
    ("G13x", "คุณ <ชื่อ อะไร / เกิด ที่ไหน> (รูปใกล้เคียง ไม่อยู่ในตาราง README)", [SUBJ, "ชื่อ", SUBJ], "{0}ชื่อ{1}"),
]


def word(sign_id):
    """ไข่_var_2 -> ไข่ ; สวัสดี_อายุเท่ากันหรือน้อยกว่า -> สวัสดี"""
    return re.sub(r"_var_\d+$", "", sign_id).split("_")[0]


def normalize(tokens):
    out = []
    for t in tokens:
        w = word(t)
        if not out or out[-1] != w:      # ยุบคำซ้ำติดกัน (CTC มักแทรกคำซ้ำ)
            out.append(w)
    return out


def match(ws, pattern):
    if len(ws) != len(pattern):
        return None
    caps = []
    for w, p in zip(ws, pattern):
        if isinstance(p, str):
            if w != p:
                return None
        elif w in p:
            caps.append(w)
        else:
            return None
    return caps


def fallback(ws):
    """เรียงคำใหม่ตามหมวด: [ทักทาย] [เวลา][ประธาน][ชอบ][ไป][กริยา][กรรม][สถานที่][อื่น ๆ][สภาพ][คำถาม]"""
    rest = list(ws)

    def take(cat, first_only=False):
        got = [w for w in rest if w in cat]
        if first_only:
            got = got[:1]
        for w in got:
            rest.remove(w)
        return got

    greet, time, ques = take(GREET), take(TIME), take(QUES)
    subj = take(SUBJ, first_only=True)
    like, go = take({"ชอบ"}), take({"ไป"})
    verbs, objs, places, states = take(VERB), take(OBJ), take(PLACE), take(STATE)
    if "เกิด" in verbs and "หูหนวก" in states:
        verbs.remove("เกิด")
        states[states.index("หูหนวก")] = "หูหนวกตั้งแต่เกิด"
    body = "".join(time + subj + like[:1] + go[:1] + verbs + objs + places + rest + states + ques)
    return (" ".join(greet) + " " + body).strip()


def translate(tokens):
    """tokens = list ของ sign_id หรือคำ -> dict(text, rule, group, words)"""
    ws = normalize(tokens)
    for gid, gname, pattern, fmt in TEMPLATES:
        caps = match(ws, pattern)
        if caps is not None:
            return {"text": fmt.format(*caps), "rule": gid, "group": gname, "words": ws}
    return {"text": fallback(ws), "rule": "fallback", "group": None, "words": ws}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default=str(BASE / "annotations" / "sentences.jsonl"))
    ap.add_argument("--ctc-dir", default=str(BASE.parent / "cslr" / "cslr"))
    ap.add_argument("--ctc-glob", default="template_C1-x16_seed*_test.json")
    ap.add_argument("--out-dir", default=str(BASE / "outputs"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sents = [json.loads(l) for l in Path(a.sentences).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_words = {" ".join(s["gloss_words"]): s for s in sents}

    rows = []
    for s in sents:
        r = translate(s["gloss_words"])
        rows.append({"gloss_seq_id": s["gloss_seq_id"], "split": s["split"],
                     "gloss_words": s["gloss_words"], "t1": r["text"], "rule": r["rule"]})
    (out / "t1_gold.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    fb = sum(r["rule"] == "fallback" for r in rows)
    print(f"gold: {len(rows)} ลำดับ, ใช้ fallback {fb}")

    for f in sorted(Path(a.ctc_dir).glob(a.ctc_glob)):
        preds = json.loads(f.read_text(encoding="utf-8"))
        rows = []
        for p in preds:
            ref_w = [word(t) for t in p["ref"].split()]
            hyp = translate(p["hyp"].split())
            ref = translate(p["ref"].split())
            rows.append({"video_id": p["video_id"], "variation": p["variation"],
                         "gloss_seq_id": by_words.get(" ".join(ref_w), {}).get("gloss_seq_id"),
                         "ref_words": ref_w, "hyp_words": [word(t) for t in p["hyp"].split()],
                         "t1_from_ref": ref["text"], "t1_from_hyp": hyp["text"], "rule_hyp": hyp["rule"]})
        name = "t1_ctc_" + f.name.replace("_test.json", ".jsonl")
        (out / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        same = sum(r["t1_from_ref"] == r["t1_from_hyp"] for r in rows)
        fb = sum(r["rule_hyp"] == "fallback" for r in rows)
        print(f"{f.name}: {len(rows)} คลิป, คำแปลจาก hyp ตรงกับจาก ref {same}, ใช้ fallback {fb}")


if __name__ == "__main__":
    main()

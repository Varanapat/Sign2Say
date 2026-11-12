#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""งาน 14 (ฝั่ง T1): คะแนนเบื้องต้นและการวิเคราะห์ error ของ T1 -> reports/t1_analysis.md

chrF ที่นี่คำนวณเอง เพราะเครื่องนี้ติดตั้ง sacrebleu ไม่ได้ (pip ถูกบล็อกที่ proxy)

สเปกที่ใช้ (ตั้งใจให้ตรงกับค่าเริ่มต้นของ sacreBLEU CHRF):
  char_order=6, word_order=0, beta=2, whitespace=False (ตัดช่องว่างออกก่อนนับ), eps_smoothing=False
  - นับ character n-gram ขนาด 1..6
  - ข้ามขนาดที่ฝั่ง hyp หรือ ref ไม่มี n-gram เลย (effective order)
  - เฉลี่ย precision และ recall ข้ามขนาดที่นับได้ แล้วรวมเป็น F-beta
  - corpus-level: รวมสถิติทุกประโยคก่อน แล้วค่อยคำนวณคะแนนครั้งเดียว
  - หลาย reference: เลือก reference ที่ให้คะแนนประโยคนั้นดีที่สุด

สถานะการตรวจสอบ (12 พ.ย. 2026)
  ✅ ตรวจเลขคณิตแล้ว: `python eval_t1_prelim.py --selftest` เทียบกับค่าที่คำนวณด้วยมือ 6 กรณี
  ✅ ตรวจเทียบกับ sacreBLEU 2.6.0 แล้ว: `python verify_chrf.py` -> ตรงกันทุกช่อง ต่างกัน 0.00 จุด
     signature: nrefs:2|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0
     ตัวเลขทางการอยู่ที่ outputs/t1_official_scores.json

ใช้งาน:  python eval_t1_prelim.py              วัดผลและเขียน reports/t1_analysis.md
        python eval_t1_prelim.py --selftest   ตรวจสูตรกับค่าที่คำนวณด้วยมือ
"""
import json
import re
import sys
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from t1_rules import normalize

BASE = Path(__file__).resolve().parent
N, BETA = 6, 2.0


def strip(s):
    return re.sub(r"\s+", "", s)


def _stats(h, r):
    out = []
    for n in range(1, N + 1):
        hc = Counter(h[i:i + n] for i in range(len(h) - n + 1))
        rc = Counter(r[i:i + n] for i in range(len(r) - n + 1))
        out.append((sum((hc & rc).values()), sum(hc.values()), sum(rc.values())))
    return out


def _f(stats):
    ps, rs = [], []
    for m, h, r in stats:
        if h > 0 and r > 0:
            ps.append(m / h); rs.append(m / r)
    if not ps:
        return 0.0
    p, r = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + r == 0 else (1 + BETA ** 2) * p * r / (BETA ** 2 * p + r)


def chrf(hyps, refs_list):
    """corpus-level chrF (0-100)"""
    tot = [[0, 0, 0] for _ in range(N)]
    for h, refs in zip(hyps, refs_list):
        best = max((_stats(strip(h), strip(r)) for r in refs), key=_f)
        for t, s in zip(tot, best):
            for i in range(3):
                t[i] += s[i]
    return 100 * _f(tot)


def exact(hyps, refs_list):
    return 100 * sum(strip(h) in {strip(r) for r in refs} for h, refs in zip(hyps, refs_list)) / len(hyps)


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def err_type(ref, hyp):
    if ref == hyp:
        return "ถูกต้อง"
    if normalize(hyp) == ref:
        return "คำซ้ำติดกัน"
    if len(normalize(hyp)) < len(ref):
        return "ตกคำ"
    if len(normalize(hyp)) > len(ref):
        return "คำแทรก"
    return "แทนที่คำ"


def selftest():
    """ค่าคาดหวังคำนวณด้วยมือจากสเปกในหัวไฟล์ (ดูวิธีคิดใน reports/)"""
    cases = [
        # hyp, refs, คะแนนที่คำนวณด้วยมือ, คำอธิบาย
        ("abc", ["abd"], 38.8889, "p=r=(2/3+1/2+0)/3"),
        ("abcd", ["abc"], 89.8438, "p=(3/4+2/3+1/2)/3, r=1 (ref สั้นกว่า order 4 จึงข้าม)"),
        ("แม่กินขนมปัง", ["แม่กินขนมปัง"], 100.0, "เหมือนกันทุกตัวอักษร"),
        ("abc", ["xyz"], 0.0, "ไม่ตรงเลย"),
        ("abc", ["xyz", "abd"], 38.8889, "หลาย ref ต้องเลือกตัวที่ดีกว่า"),
        ("กินข้าว", ["แฟนกินข้าว"], 63.0500, "ตกประธาน: p=1 ทุก order, r=(.7+.667+.625+.571+.5+.4)/6"),
    ]
    ok = True
    for hyp, refs, want, why in cases:
        got = round(chrf([hyp], [refs]), 4)
        mark = "ok  " if abs(got - want) < 1e-3 else "FAIL"
        ok &= mark == "ok  "
        print(f"  {mark} chrF({hyp!r}, {refs!r}) = {got:8.4f}  คาด {want:8.4f}   {why}")
    print("ผ่านทั้งหมด" if ok else "มีข้อที่ไม่ผ่าน")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    sents = {s["gloss_seq_id"]: s for s in load(BASE / "annotations" / "sentences.jsonl")}
    gold = load(BASE / "outputs" / "t1_gold.jsonl")
    L = ["# T1 (กฎ/แม่แบบ): คะแนนเบื้องต้นและการวิเคราะห์ error", "",
         "> chrF คำนวณเองด้วยสูตร chrF2 (ดูหัวไฟล์ `eval_t1_prelim.py`) ตัวเลขทางการให้ใช้จากสคริปต์วัดผลกลาง (งาน 11)",
         "> exact = คำแปลตรงกับ reference ชุดใดชุดหนึ่งทุกตัวอักษร (ไม่นับช่องว่าง)", "",
         "## 1. แปลจาก gloss ที่ถูกต้อง", "", "| ชุด | จำนวนลำดับ | chrF | exact (%) |", "|---|---|---|---|"]
    res = {"gold": {}, "ctc": {}}
    for name in ("train", "val", "test", "all"):
        rows = [g for g in gold if name == "all" or g["split"] == name]
        h = [g["t1"] for g in rows]
        r = [sents[g["gloss_seq_id"]]["thai_refs"] for g in rows]
        res["gold"][name] = {"n": len(rows), "chrf": chrf(h, r), "exact": exact(h, r)}
        L.append(f"| {name} | {len(rows)} | {res['gold'][name]['chrf']:.1f} | {res['gold'][name]['exact']:.1f} |")

    L += ["", "ประโยคใน test ที่ T1 ไม่ตรงกับ reference ใดเลย:", "",
          "| id | gloss | T1 | reference |", "|---|---|---|---|"]
    for g in gold:
        refs = sents[g["gloss_seq_id"]]["thai_refs"]
        if g["split"] == "test" and strip(g["t1"]) not in {strip(x) for x in refs}:
            L.append(f"| {g['gloss_seq_id']} | {' '.join(g['gloss_words'])} | {g['t1']} | {' / '.join(refs)} |")

    L += ["", "## 2. แปลจาก gloss ที่ CTC ทาย (C1-x16, protocol S-template, 45 คลิปต่อ seed)", "",
          "| seed | chrF จาก gloss ถูกต้อง | chrF จาก gloss ที่ทาย | ลดลง | exact จาก gloss ที่ทาย (%) | ใช้กฎสำรอง |",
          "|---|---|---|---|---|---|"]
    by_type = defaultdict(lambda: [0, 0])
    examples = {}
    cols = defaultdict(list)
    for f in sorted((BASE / "outputs").glob("t1_ctc_template_C1-x16_seed*.jsonl")):
        rows = load(f)
        refs = [sents[r["gloss_seq_id"]]["thai_refs"] for r in rows]
        a = chrf([r["t1_from_ref"] for r in rows], refs)
        b = chrf([r["t1_from_hyp"] for r in rows], refs)
        e = exact([r["t1_from_hyp"] for r in rows], refs)
        fb = sum(r["rule_hyp"] == "fallback" for r in rows)
        seed = re.search(r"seed(\d+)", f.name).group(1)
        res["ctc"][seed] = {"chrf_ref": a, "chrf_hyp": b, "exact_hyp": e, "fallback": fb, "n": len(rows)}
        for k, v in (("a", a), ("b", b), ("e", e), ("fb", fb)):
            cols[k].append(v)
        L.append(f"| {seed} | {a:.1f} | {b:.1f} | {a - b:.1f} | {e:.1f} | {fb}/{len(rows)} |")
        for r, rf in zip(rows, refs):
            t = err_type(r["ref_words"], r["hyp_words"])
            ok = strip(r["t1_from_hyp"]) == strip(r["t1_from_ref"])
            by_type[t][0] += 1
            by_type[t][1] += ok
            if t != "ถูกต้อง":
                examples.setdefault((t, " ".join(r["hyp_words"])), (r, ok))
    ms = lambda x: f"{st.mean(x):.1f} ± {st.stdev(x):.1f}" if len(x) > 1 else f"{x[0]:.1f}"
    L.append(f"| **เฉลี่ย ± sd** | {ms(cols['a'])} | {ms(cols['b'])} | {st.mean(cols['a']) - st.mean(cols['b']):.1f} | {ms(cols['e'])} | {ms(cols['fb'])} |")

    L += ["", "## 3. T1 ทนต่อ error แต่ละแบบของ CTC แค่ไหน (รวม 3 seed)", "",
          "\"กู้ได้\" = คำแปลจาก gloss ที่ทาย เหมือนกับคำแปลจาก gloss ที่ถูกต้องทุกตัวอักษร", "",
          "| ชนิด error ของ gloss | จำนวนคลิป | T1 กู้ได้ | % |", "|---|---|---|---|"]
    for t in ("ถูกต้อง", "คำซ้ำติดกัน", "ตกคำ", "คำแทรก", "แทนที่คำ"):
        n, ok = by_type[t]
        if n:
            L.append(f"| {t} | {n} | {ok} | {100 * ok / n:.0f} |")
    L += ["", "ตัวอย่าง (ไม่ซ้ำ):", "", "| ชนิด | gloss ถูกต้อง | gloss ที่ทาย | T1 จาก gloss ที่ทาย | ควรได้ | กู้ได้ |", "|---|---|---|---|---|---|"]
    for (t, _), (r, ok) in sorted(examples.items()):
        L.append(f"| {t} | {' '.join(r['ref_words'])} | {' '.join(r['hyp_words'])} | {r['t1_from_hyp']} | {r['t1_from_ref']} | {'✓' if ok else '✗'} |")

    (BASE / "reports").mkdir(exist_ok=True)
    (BASE / "reports" / "t1_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (BASE / "outputs" / "t1_prelim_scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

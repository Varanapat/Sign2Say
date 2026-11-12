# Sign2Say NLP: gloss → ประโยคไทย (RQ8, T1/T2)

งานส่วน NLP ต่อจาก CSLR ของ branch `core-pipeline`: รับลำดับ gloss ของประโยค TSL-51 แล้วแปลเป็นประโยคไทย
เทียบ **T1 กฎ/แม่แบบ** กับ **T2 LLM few-shot (Typhoon)** ทั้งบน gloss ที่ถูกต้องและ gloss ที่ CTC ทาย

## ผลหลัก (ชุด test)

chrF และ BLEU-4 ด้วย sacreBLEU 2.6.0 · reference 2 ชุด (a1, a2) · gloss ที่ CTC ทาย = 45 คลิป × 3 seed (mean ± SD)

| ระบบ | chrF gloss ถูก | BLEU gloss ถูก | chrF gloss ที่ CTC ทาย | BLEU gloss ที่ CTC ทาย |
|---|---|---|---|---|
| T1 กฎ | 81.83 | 46.96 | 76.05 ± 6.28 | 45.67 ± 5.33 |
| **T2 Typhoon 30B** | **87.52** | **81.73** | **81.05 ± 9.47** | **74.49 ± 6.91** |

ผลประเมินโดยคน (56 ข้อ, ผู้ประเมิน 2 คน, คะแนน 1–5)

| ระบบ / gloss | ความหมาย | ความเป็นธรรมชาติ | เติมข้อมูลที่ไม่มีใน gloss |
|---|---|---|---|
| T1 / ถูก | 5.00 | 5.00 | 0% |
| T2 / ถูก | 5.00 | 4.50 | 0% |
| T1 / CTC ทาย | 2.98 | 2.67 | 50% |
| T2 / CTC ทาย | 2.94 | 2.58 | 50% |

Cohen's κ: ความหมาย 0.892 · ความเป็นธรรมชาติ 0.728 (quadratic weighted) · เติมข้อมูล 1.000

สรุป
- chrF/BLEU ให้ T2 ชนะ แต่ส่วนต่างมาจากกลุ่ม `<ประธาน> เกิด หูหนวก` กลุ่มเดียว ซึ่ง T2 ใช้สำนวน "เกิดมาหูหนวก" ตรงกับ reference
  ส่วน T1 ใช้ "หูหนวกตั้งแต่เกิด" ผู้ประเมินให้ T1 เท่ากันหรือดีกว่าเล็กน้อย
- H8 ไม่ได้รับการสนับสนุน: ทั้งสองระบบกู้ error ของ CTC ได้เท่ากัน และ T2 ตกมากกว่า (chrF ลด 9.15 จุด เทียบกับ T1 7.16)

ผลเสริม (ไม่อยู่ในการประเมินโดยคน)

| ระบบ | chrF gloss ถูก | chrF gloss ที่ CTC ทาย | หมายเหตุ |
|---|---|---|---|
| T2-4B | 65.50 | 65.14 ± 8.16 | Typhoon 4B รันในเครื่อง (Ollama) prompt เดียวกัน |
| T2-tuned | 96.91 | 89.28 ± 11.14 | เลือกค่าบน train อย่างเดียว (leave-one-group-out): 24 ตัวอย่าง, คำแปล a2 |

T2-tuned ทำหลังเปิดดูผล test รอบแรกแล้ว จึงรายงานเป็นผลเสริมเท่านั้น ผลหลักคือ T2 ที่ล็อกไว้ก่อนเห็น reference ของ test

## ไฟล์

| ไฟล์ | เนื้อหา | งาน |
|---|---|---|
| `data/tsl51_sentence_template.json` | split S-template (คัดลอกจาก `core-pipeline:splits/`) | – |
| `build_sentences.py` | สร้าง `annotations/sentences.jsonl` และแบบฟอร์มคำแปลเปล่า | 5 |
| `annotations/refs_annotator{1,2}.csv` | คำแปลอ้างอิง 2 ชุด (ต้นฉบับ ไม่แก้) | 6, 7 |
| `merge_refs.py` | รวมคำแปลเข้า `sentences.jsonl` + `annotations/refs_diff.md` | 8 |
| `t1_rules.py` | T1: กฎ 13 กลุ่มตาม README ของ TSL-51 + กฎสำรอง → `outputs/t1_*.jsonl` | 9 |
| `run_t2.py` | T2: Typhoon few-shot → `outputs/t2_*.jsonl`, `t2_meta.json` (`--local` = 4B ผ่าน Ollama) | 10 |
| `score_sacrebleu.py` | chrF + BLEU-4 (PyThaiNLP newmm) ด้วย sacreBLEU ทุกระบบ → `outputs/official_scores.json` | 11 |
| `verify_chrf.py` | ตรวจ chrF ของเราเทียบกับ sacreBLEU → `outputs/t1_official_scores.json` | 11 |
| `eval_compare.py` | เทียบ T1/T2 และชนิด error ของ CTC → `reports/comparison.md` | 11, 14 |
| `eval_t1_prelim.py` | คะแนน chrF และ error analysis ของ T1 → `reports/t1_analysis.md` | 14 |
| `make_human_eval.py` | แบบประเมินโดยคนแบบ blind → `human_eval/` | 12 |
| `human_eval_kappa.py` | คะแนนเฉลี่ย, อัตราเติมข้อมูล, Cohen's κ → `outputs/human_eval_scores.json` | 13 |
| `tune_t2.py` | ผลเสริม: เลือกค่า T2 บน train (leave-one-group-out) → `outputs/t2_tuning.json` | เสริม |

ลำดับการรัน

```
build_sentences.py → (เขียนคำแปล) → merge_refs.py → t1_rules.py → run_t2.py
→ score_sacrebleu.py → eval_compare.py → make_human_eval.py → (ผู้ประเมินกรอก) → human_eval_kappa.py
```

ติดตั้ง: `pip install openai sacrebleu pythainlp` · T2 ต้องตั้ง environment variable `TYPHOON_API_KEY` (ห้ามใส่ key ในโค้ดหรือ commit)
ผลทาย CTC อ่านจาก `../results/preds/cslr/template_C1-x16_seed{0,1,2}_test.json` (ไม่อยู่ใน branch นี้)
คะแนนทั้งหมดคำนวณซ้ำได้จากไฟล์ใน `outputs/` โดยไม่ต้องเรียก API

## `annotations/sentences.jsonl`

1 บรรทัดต่อลำดับคำ (62 ลำดับ = 76 ลำดับ sign_id = 252 คลิป; train 44 / val 6 / test 12)
ฟิลด์: `gloss_seq_id`, `gloss_words`, `sign_id_variants`, `grammar_group_id`, `grammar_group`, `split`,
`thai_refs` (ปรับรูปแบบแล้ว ใช้วัดผล), `thai_refs_raw`, `annotator_ids`, `verified_by_signer`, `clips`

## T2: วิธีที่ใช้

- โมเดล `typhoon-v2.5-30b-a3b-instruct` · temperature 0 · few-shot 20 คู่จาก **train เท่านั้น** (วนตามกลุ่มไวยากรณ์, คำแปล a1)
- gloss ที่ CTC ทายผ่าน `t1_rules.normalize()` ก่อน (ตัด `_var_N` และยุบคำซ้ำติดกัน) = input เดียวกับ T1
- รันเมื่อ 2026-11-12 01:50 น. ก่อนเปิด reference ของ test; prompt และตัวอย่างทั้งหมดอยู่ใน `outputs/t2_meta.json`
- รูปแบบไฟล์ผล (jsonl, UTF-8):
  - `outputs/t2_gold.jsonl`: `{"gloss_seq_id": "gs_051", "t2": "..."}` ครบ 62 ลำดับ
  - `outputs/t2_ctc_template_C1-x16_seed{0,1,2}.jsonl`: `{"video_id": "...", "t2_from_hyp": "..."}` ครบ 45 คลิปต่อ seed
  - ผลเสริมใช้รูปแบบเดียวกัน ชื่อขึ้นต้นด้วย `t2local4b_` และ `t2tuned_`

## การวัดผล

- chrF: sacreBLEU `CHRF()` ค่าเริ่มต้น (`nrefs:2|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`)
- BLEU-4: ลบช่องว่าง → ตัดคำด้วย PyThaiNLP 5.4.0 `newmm` → sacreBLEU `tokenize='none'`
  (`nrefs:2|case:mixed|eff:no|tok:none|smooth:exp|version:2.6.0`)
- รายงานผลบน **test** เป็นหลัก และเฉลี่ย 3 seed สำหรับ gloss ที่ CTC ทาย
- ข้อที่สองระบบแปลเหมือนกัน (`t1+t2` ใน `human_eval/key.csv`) นับคะแนนให้ทั้งสองระบบ

## ข้อจำกัดที่ต้องเขียนในรายงาน

- กลุ่ม `<ประธาน> เกิด หูหนวก` (7 จาก 12 ลำดับของ test) ไม่มีใน train เลย ถ้อยคำของแม่แบบ T1 กลุ่มนี้กำหนดจาก README ก่อนเห็นคำแปลอ้างอิง และ**ไม่ได้ปรับตาม reference ภายหลัง**
- test มีแค่ 12 ลำดับ คะแนนจึงแกว่งได้มากจากไม่กี่ประโยค โดยเฉพาะ BLEU; ใช้ chrF เป็นตัวชี้วัดหลัก
- คำแปลอ้างอิงเขียนโดยทีมเอง 2 ชุด ไม่ได้ผ่านผู้ใช้ภาษามือตรวจ
- ผู้ประเมินตีความ "ความเป็นธรรมชาติ" ต่างกันในประโยคที่ระบุคนผิด (เช่น "คุณกินข้าว" แทนแฟน ได้ 2 กับ 5) ทำให้ κ ด้านนี้ต่ำที่สุด
- ประโยคทั้งหมดมาจากผู้แสดงคนเดียว; WER ของ CTC ต่างกันมากระหว่าง seed (5.3–16.7%)
- คำแปลอ้างอิงชุด a1 มี 2 ข้อที่ความหมายต่างจาก gloss (gs_038 train, gs_058 test) ผู้เขียนยืนยันให้คงไว้ ดู `annotations/refs_diff.md`

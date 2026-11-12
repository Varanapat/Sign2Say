# Sign2Say NLP: gloss → ประโยคไทย (RQ8, T1/T2)

งานส่วน NLP ต่อจาก CSLR ของ branch `core-pipeline`: รับลำดับ gloss ของประโยค TSL-51 แล้วแปลเป็นประโยคไทย
เทียบ **T1 กฎ/แม่แบบ** กับ **T2 LLM few-shot (Typhoon)** ทั้งบน gloss ที่ถูกต้องและ gloss ที่ CTC ทาย

## ไฟล์

| ไฟล์ | เนื้อหา | งาน |
|---|---|---|
| `data/tsl51_sentence_template.json` | split S-template (คัดลอกจาก `core-pipeline:splits/`) | – |
| `build_sentences.py` | สร้าง `annotations/sentences.jsonl` และแบบฟอร์มคำแปลเปล่า | 5 |
| `annotations/refs_annotator{1,2}.csv` | คำแปลอ้างอิง 2 ชุด (ต้นฉบับ ไม่แก้) | 6, 7 |
| `merge_refs.py` | รวมคำแปลเข้า `sentences.jsonl` + `annotations/refs_diff.md` | 8 |
| `t1_rules.py` | T1: กฎ 13 กลุ่มตาม README ของ TSL-51 + กฎสำรอง → `outputs/t1_*.jsonl` | 9 |
| `eval_t1_prelim.py` | คะแนน chrF และ error analysis ของ T1 → `reports/t1_analysis.md` | 14 |
| `verify_chrf.py` | ตรวจ chrF ของเราเทียบกับ sacreBLEU → `outputs/t1_official_scores.json` | 11 |
| `make_human_eval.py` | แบบประเมินโดยคนแบบ blind → `human_eval/` | 12 |

ลำดับการรัน: `build_sentences.py` → (เขียนคำแปล) → `merge_refs.py` → `t1_rules.py` → `eval_t1_prelim.py` → (ผล T2) → `make_human_eval.py`
ผลทาย CTC อ่านจาก `../cslr/cslr/` (ไม่อยู่ใน git)

## `annotations/sentences.jsonl`

1 บรรทัดต่อลำดับคำ (62 ลำดับ = 76 ลำดับ sign_id = 252 คลิป; train 44 / val 6 / test 12)
ฟิลด์: `gloss_seq_id`, `gloss_words`, `sign_id_variants`, `grammar_group_id`, `grammar_group`, `split`,
`thai_refs` (ปรับรูปแบบแล้ว ใช้วัดผล), `thai_refs_raw`, `annotator_ids`, `verified_by_signer`, `clips`

## ข้อตกลงสำหรับ T2 (คนที่ 2)

- ตัวอย่าง few-shot: ใช้เฉพาะแถวที่ `split == "train"` (gloss_words → thai_refs) **ห้ามใช้ val/test**
- แปลงผลทาย CTC เป็นคำด้วยกติกาเดียวกับ T1: `t1_rules.word()` (ตัด `_var_N` และคำอธิบายหลัง `_`)
  จะยุบคำซ้ำก่อนส่งให้ LLM หรือไม่ เป็นการตัดสินใจของ T2 แต่ต้องระบุในรายงาน
- ไฟล์ผลที่ต้องส่ง (jsonl, UTF-8):
  - `outputs/t2_gold.jsonl`: `{"gloss_seq_id": "gs_051", "t2": "..."}` ครบ 62 ลำดับ (อย่างน้อย val + test)
  - `outputs/t2_ctc_template_C1-x16_seed{0,1,2}.jsonl`: `{"video_id": "...", "t2_from_hyp": "..."}` ครบ 45 คลิปต่อ seed
- การวัดผล: เทียบกับ `thai_refs` ทั้ง 2 แบบ โดยไม่นับช่องว่าง; รายงานผลบน **test** เป็นหลัก และเฉลี่ย 3 seed สำหรับ gloss ที่ทาย

## ข้อจำกัดที่ต้องเขียนในรายงาน

- กลุ่ม `<ประธาน> เกิด หูหนวก` (7 จาก 12 ลำดับของ test) ไม่มีใน train เลย ถ้อยคำของแม่แบบ T1 กลุ่มนี้กำหนดจาก README ก่อนเห็นคำแปลอ้างอิง และ**ไม่ได้ปรับตาม reference ภายหลัง**
- คำแปลอ้างอิงเขียนโดยทีมเอง 2 ชุด ไม่ได้ผ่านผู้ใช้ภาษามือตรวจ
- chrF ของ T1 ตรวจเทียบกับ sacreBLEU 2.6.0 แล้ว ตรงกันทุกช่อง (`nrefs:2|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`)
- ประโยคทั้งหมดมาจากผู้แสดงคนเดียว; WER ของ CTC ต่างกันมากระหว่าง seed (5.3–16.7%)
- คำแปลอ้างอิงชุด a1 มี 2 ข้อที่ความหมายต่างจาก gloss (gs_038 train, gs_058 test) ผู้เขียนยืนยันให้คงไว้ ดู `annotations/refs_diff.md`

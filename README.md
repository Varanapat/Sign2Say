# Sign2Say: Thai Sign Language Recognition from Landmarks

Branch `core-pipeline` เป็นระบบหลักของ Sign2Say: สกัด landmark ด้วย MediaPipe → รู้จำคำเดี่ยว (ISLR) → รู้จำประโยค (CSLR) แล้ววัดผลแบบ **signer-independent** (คนใน test ไม่เคยอยู่ใน train)
งานวิเคราะห์สถาปัตยกรรม (L0 / L1 / L2, RGB) อยู่ที่ branch [`base_line`](https://github.com/Varanapat/Sign2Say/tree/base_line)

วิชา Deep Learning in Medical Image and Video Analysis

## ผลหลัก

ตัวเลขทั้งหมดมาจากการรันจริง ผลเต็มอยู่ใน [`Knowledge Base/Experiment_Design_Sign2Say.md`](Knowledge%20Base/Experiment_Design_Sign2Say.md) §8.9

| งาน | Protocol | ผล |
|---|---|---|
| ISLR: S2S-LTR (โมเดลหลัก) | TSL-ONE-S, 10 fold signer-independent | **84.8 ± 3.0%** top-1, 83.2 ± 3.8 macro-F1 |
| ISLR: BiGRU (baseline) | เหมือนกัน | 77.6 ± 3.1% (S2S ชนะ 10/10 fold, Wilcoxon p = 0.002) |
| X16: encoder ร่วม TSL-ONE-S + TSL-51 | TSL-51 expert → user_sign | 80.9 ± 2.9% (เทรนชุดเดียว 76.2 ± 1.9%) |
| ISLR: S2S-LTR + hand-local (จาก branch `base_line`) | 5 fold (0, 2, 4, 6, 8) | **87.1%** vs 84.2% บน fold เดียวกัน (ดีขึ้น 5/5 fold, McNemar p = 2e-5) |
| ISLR: ST-GCN + hand-local (สูตรเดียวกัน) | 5 fold (0, 2, 4, 6, 8) | 84.3% vs S2S-HL 87.1% (แพ้ 5/5 fold, p = 2e-4) และเทรนช้ากว่า ~15× |
| CSLR: CTC (C1) | ประโยค TSL-51, ผู้แสดงคนเดียว | WER 5.1% (ประโยคเดิม), 11.8% (ลำดับคำใหม่) |
| ความเร็ว | Apple M5 | MediaPipe 42 fps, โมเดล 1.3 ms/คลิป (CPU) |

ข้อจำกัด: ผลหลักเป็น seed เดียว, การเทรนบน MPS ไม่ deterministic (±3 pp ต่อ fold), CSLR มีผู้แสดงคนเดียว

## โครงสร้าง

```
scripts/
  landmarks.py              schema C54 / X71, เติมช่องว่าง, normalize, resample, feature 7 ช่อง
  extract_tsl1s.py          MediaPipe Holistic บนวิดีโอ TSL-ONE-S → dataset/cache/tsl1s_c54_T64.npz
  build_tsl51_cache.py      CSV ของ TSL-51 → dataset/cache/tsl51_c54_T64.npz
  build_sentence_cache.py   ประโยค TSL-51 + split S-rep / S-template
  check_tsl51_schema.py     ตรวจว่าจุดของ TSL-51 ตรงกับ MediaPipe ที่เราใช้
  make_signer_split.py      10 fold แบ่งตามผู้แสดง → splits/
  make_random_split.py      random-matched split (RQ5)
  run_baselines.py          LR / RF / BiGRU / S2S-LTR / S2S-HL / ST-GCN, ablation (--variant)
  stgcn.py                  hand-local feature + ST-GCN (นำข้อค้นพบของ branch base_line มาใช้)
  run_joint.py              X16: encoder ร่วมสอง dataset, 2 head
  run_cslr.py               C1 CTC และ C2 sliding window
  run_robustness.py         รบกวน landmark / วิดีโอ, ผลรายผู้แสดง
  run_efficiency.py         params, ขนาด, latency, FPS ของ MediaPipe
  summarize_*.py            สรุปตารางผลและการทดสอบทางสถิติ (summarize_rqs.py = hand-local / ST-GCN)
  queue_*.sh                คิวการรันที่ใช้จริง
splits/                     split ทุกแบบ (JSON)
dataset/annotations/        label map
results/                    ผลสรุป (*.json, *.log); ไฟล์ทำนายและโมเดลไม่อยู่ใน git
Knowledge Base/             research report, technical design, data pipeline, experiment design, script นำเสนอ
share_for_friend/           brief ที่ส่งให้ branch base_line
```

## ติดตั้ง

ทดสอบบน macOS (Apple Silicon), Python 3.12

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

แพ็กเกจหลัก: `torch 2.15`, `mediapipe 0.10.21`, `numpy 1.26`, `scikit-learn`, `scipy`, `opencv-contrib-python`
ใช้ MPS ถ้ามี ไม่อย่างนั้นใช้ CPU

## ข้อมูล (ไม่อยู่ใน repo)

dataset ทั้งสองชุดใช้สัญญาอนุญาต non-commercial และเห็นหน้าผู้แสดง **ห้ามอัปโหลดวิดีโอหรือ landmark ที่สกัดแล้วขึ้น repo** ให้ดาวน์โหลดจากแหล่งต้นฉบับแล้ววางตามนี้

```
TSL-ONE-S/                     วิดีโอ mp4 4,152 ไฟล์
tsl_one_s_dataset_i3d.json     ไฟล์ split ต้นฉบับของ TSL-ONE-S
Source(Paper)/tsl51/           TSL-51 (metadata/ และ landmark CSV)
```

## วิธีรัน

ทุกคำสั่งรันจาก root ของ repo

**1. สร้างข้อมูล**

```bash
.venv/bin/python scripts/extract_tsl1s.py --workers 8
.venv/bin/python scripts/build_tsl51_cache.py
.venv/bin/python scripts/build_sentence_cache.py
.venv/bin/python scripts/make_signer_split.py
.venv/bin/python scripts/make_random_split.py
```

ถ้าต้องการ X71 (ใช้ใน ablation) ให้รัน `extract_tsl1s.py --schema x71 --cache-only` เพิ่ม

**2. ISLR**

```bash
# 10 fold, S2S-LTR และ BiGRU
.venv/bin/python scripts/run_baselines.py --protocol tsl1s_si --models S2S BiGRU --folds 0 1 2 3 4 5 6 7 8 9
# ablation (noface / x71 / novel / nomask / flat)
.venv/bin/python scripts/run_baselines.py --protocol tsl1s_si --models S2S --folds 0 2 4 6 8 --variant noface
# hand-local และ ST-GCN
.venv/bin/python scripts/run_baselines.py --protocol tsl1s_si --models S2S-HL STGCN-HL --folds 4
# TSL-51 expert → user_sign
.venv/bin/python scripts/run_baselines.py --protocol tsl51_ext --models S2S BiGRU
```

protocol ที่มี: `tsl1s_si` (หลัก), `tsl1s_rand`, `tsl1s_orig`, `tsl51_ext`, `tsl51_loso`
ใส่ `--save-models` เพื่อบันทึกน้ำหนักไว้ที่ `results/models/` (ต้องใช้กับขั้นที่ 3–4)

**3. X16 และ CSLR**

```bash
.venv/bin/python scripts/run_joint.py --folds 0 --save-models
.venv/bin/python scripts/run_cslr.py c1 --protocol template --init x16 --synth 300 --epochs 100 --seeds 0 1 2
.venv/bin/python scripts/run_cslr.py c2 --c2-model x16ft
.venv/bin/python scripts/summarize_cslr.py
```

**4. Robustness และ efficiency** (ต้องมีโมเดล fold 4 ที่บันทึกไว้)

```bash
.venv/bin/python scripts/run_robustness.py
.venv/bin/python scripts/run_efficiency.py   # รันตอนเครื่องว่าง
```

## เอกสาร

| ไฟล์ | เนื้อหา |
|---|---|
| `Knowledge Base/Research_Sign_Language_DL.md` | งานวิจัยที่เกี่ยวข้อง |
| `Knowledge Base/Technical_Design_Sign2Say.md` | สถาปัตยกรรมระบบ |
| `Knowledge Base/Data_Pipeline_Sign2Say.md` | ข้อมูล, schema, split, การตรวจคุณภาพ |
| `Knowledge Base/Experiment_Design_Sign2Say.md` | คำถามวิจัย, สมมติฐาน, ผลทุกรอบ (§8.9) |
| `Knowledge Base/Presentation_Script_Sign2Say.md` | script นำเสนอ |

## ชุดข้อมูลและบทความอ้างอิง

- TSL-ONE-S: *TSL-ONE-S: A Real-World Thai Sign Language Dataset With Deep Learning Benchmarks*, IEEE Access, 2026
- TSL-51 (คำเดี่ยวและประโยค)
- MediaPipe Holistic: Google

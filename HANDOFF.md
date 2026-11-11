# Handoff: RQ-S (Spatial → Temporal vs Spatiotemporal) บน TSL-ONE-S

สรุปสำหรับทีม Sign2Say (11–12 พ.ย. 2026) รายละเอียดทุกการทดลองอยู่ใน [`EXPERIMENTS.md`](EXPERIMENTS.md)
ข้อตกลงร่วม (split, label, metric, รูปแบบไฟล์ผล) ตาม `README_brief` ทุกข้อ

---

## 1. ผลหลัก (fold 4, test = signer 03/08/22, mean ± std ของ 3 seed)

| ID | โมเดล | ระดับ | Params | Test top-1 | Test top-5 | Test macro-F1 | Val top-1 |
|---|---|---|---|---|---|---|---|
| A1 | SPOTER เดิม | L0 | 5.93M | 70.60 ± 3.32 | 88.54 | 68.44 | 77.32 |
| A2 | SPOTER + positional encoding | L1 | 5.94M | 68.99 ± 3.33 | 90.22 | 66.19 | 77.08 |
| B0 | MLP + mean pool | L0 | 0.18M | 67.55 ± 0.48 | 91.11 | 66.36 | 73.08 |
| B1 | MLP → BiGRU | L1 | 2.14M | 68.99 ± 2.29 | 89.90 | 67.53 | 76.36 |
| B2 | MLP → Transformer | L1 | 3.29M | 69.71 ± 1.25 | 90.86 | 66.97 | 75.96 |
| B6 | GCN ต่อเฟรม → BiGRU | L1 + graph | 2.20M | 69.07 ± 2.28 | 90.46 | 66.52 | 76.76 |
| B4 | Divided space-time attention | L2, no graph | 2.52M | 61.54 ± 2.95 | 87.10 | 57.65 | 69.15 |
| **B3** | **ST-GCN** | **L2** | 3.14M | **75.00 ± 1.57** | **93.43** | **72.95** | **82.45** |
| ref | BiGRU ของทีม (1 seed) | L1 | 2.1M | 73.1 | 90.4 | 72.1 | – |

Part C (RGB, seed 0, ยังไม่ครบ ณ 11 พ.ย. 21:00): C2 ResNet-18 → BiGRU (L1) = 37.50 · C3 R(2+1)D-18 (L2) **กำลังรัน** (val 75.24% ที่ epoch 12 จาก 30) · C1 ResNet-18 + mean (L0) **รอรันต่อจาก C3** · C4 R3D-18 **ข้าม** (ดู §4)

## 2. ข้อสรุป

1. **L1 ≈ L0:** การเรียนลำดับเวลาของทั้งคลิปช่วยน้อยและไม่มีนัยสำคัญ (B1/B2 − B0 = +1.4 ถึง +2.2 pp, p ≥ 0.13; A2 − A1 ≈ 0)
2. **L2 แบบ ST-GCN ดีที่สุดอย่างมีนัยสำคัญ:** +5 ถึง +6 pp เหนือ L1 (p < 0.001) ชนะ BiGRU ของทีมและ SPOTER
3. **ที่มาของผลต่าง (แยกตัวแปรกวน):** graph prior อย่างเดียวไม่ช่วย (B6 ≈ B1) แต่ graph + การเรียนพื้นที่และเวลาเฉพาะที่พร้อมกันช่วย (B3 − B6 = +5.9, p = 2e-5) ส่วน B4 (L2 ไม่มี graph) underfit จึง**ไม่ควรใช้ B4 สรุปว่า L2 ไม่ดี**
4. **Late fusion ได้ผลดี:** ensemble 3 seed ของ B3 = 79.81 · B3 + B2 + A1 = **84.38** (เฉลี่ย probability)
5. **ความแปรปรวนสูง:** std ระหว่าง seed ประมาณ 1.5–3.3 pp และ val กับ test ไม่ไปทางเดียวกันบ่อย ต้องรายงาน 3 seed เสมอ

## 3. ข้อค้นพบที่ทีมควรรู้ (กระทบ pipeline ของทีมด้วย)

- **hand-local features สำคัญมาก:** เพิ่ม 2 channel คือพิกัดมือในกรอบของมือเอง (xy − ข้อมือ) / ขนาดมือ ทำให้ val ของ BiGRU ขึ้น +7 ถึง +12 pp (`--hand_local`)
- **z ของ MediaPipe เป็น noise:** ตัด z และ velocity ทิ้ง (`--channels xy_mask`) ทำให้ val ขึ้น +5.5 pp
- **4 คลิปที่ชื่อไม่ตรง:** หลุดจาก **train** ไม่ใช่ val (fold 4 val ครบ 416 คลิป)
- **SPOTER เดิมไม่รู้ลำดับเฟรม** (ยืนยันแล้ว: สลับเฟรมแล้ว output เปลี่ยนแค่ 3.6e-7) และการใส่ PE ก็ไม่ช่วย

## 4. งานที่ยังค้าง และวิธีรันต่อ

| งาน | สถานะ | วิธีรัน |
|---|---|---|
| C3 (RGB, L2) | กำลังรันบน Mac (~30 นาที/epoch) คาดว่าจบ 12 พ.ย. ช่วง 01:00–04:30 | จะ commit ผลเพิ่ม · ถ้าต้องหยุดก่อน: `python -m experiments.eval_rgb_ckpt --model C3` ประเมินจาก best.pth |
| C1 (RGB, L0) | รอ (คิวรันต่อจาก C3 อัตโนมัติ ~70 นาที) | จะ commit ผลเพิ่ม |
| **C4 R3D-18** | **ข้าม** — 3D conv บน Apple MPS ช้ามาก (C3 ~30 นาที/epoch) คาดว่า C4 ใช้ 15–20 ชม. เกินกำหนดส่ง · L2 ฝั่ง RGB ใช้ C3 เป็นตัวแทน | บนเครื่อง CUDA: ลบ `results/C4_fold4_seed0/` แล้ว `python -m experiments.train_rgb --model C4 --fold 4 --seed 0 --device cuda` |
| fold 0, 1, 2 | ยังไม่ได้รัน | แก้เลข fold ในไฟล์คิว แล้ว `./run_queue.sh queues/<ไฟล์>` (SPOTER ใช้บรรทัด `S`) |
| RGB seed 1, 2 | ยังไม่ได้รัน | เพิ่มบรรทัด `R` ในคิว |

## 5. สำหรับกลุ่ม NLP: ใช้ recognizer ต่อ

**Pipeline ที่แนะนำ:** ST-GCN (B3) + hand-local, ensemble 3 seed: test top-1 **79.81%**, top-5 **94.95%** บนคนที่ไม่เคยเห็น (fold 4)

```bash
python -m experiments.predict data/videos/03_01_0001.mp4 --topk 5            # วิดีโอเดียว
python -m experiments.predict clip1.mp4 clip2.mp4 --out preds.json           # หลายไฟล์ -> JSON
```

ผลต่อวิดีโอ:
```json
{"video": "clip1.mp4", "frames": 46,
 "top": [{"gloss": "Mor Maa", "label": 92, "prob": 0.8839},
         {"gloss": "Uncle (Father's younger brother)", "label": 124, "prob": 0.0666}, ...]}
```
- **คำศัพท์:** 184 gloss ใน `data/label_map.json` (ชื่อเป็นภาษาอังกฤษ/ทับศัพท์ เช่น `Mor Maa`, `Thailand`) กลุ่ม NLP ต้องทำ mapping gloss → คำไทยเอง
- **ข้อจำกัด:** รับ **1 คำต่อคลิป** (isolated) ยังไม่รองรับประโยคต่อเนื่อง ถ้าจะใช้กับประโยค ต้องตัดวิดีโอเป็นช่วงก่อน (sliding window / ตรวจช่วงพัก) แล้วส่งทีละช่วง
- **ใช้ top-k + prob ได้:** top-1 ผิด ~20% แต่ top-5 ถูก 94.95% กลุ่ม NLP ใช้ language model เลือกจาก top-5 ตามบริบทของประโยคได้ (beam / rescoring)
- **ความเร็ว:** ~2.7 วินาทีต่อคลิปบน CPU (ส่วนใหญ่เป็นเวลาของ MediaPipe ส่วน ST-GCN ใช้ไม่กี่ ms)
- **ต้องมี:** `results/B3_hl_fold4_seed{0,1,2}/best.pth` (weight), `data/label_map.json`, mediapipe 0.10.14 (มี `mp.solutions`)
- **ตรวจแล้ว:** ทำนายจากวิดีโอจริงได้ top-1 ตรงกับ pipeline ตอนเทรน 20/20 คลิป (`experiments/predict.py` ใช้ MediaPipe + preprocessing ชุดเดียวกับ cache ของทีม คัดลอกไว้ที่ `experiments/c54_landmarks.py`)

## 6. ไฟล์และวิธีใช้

| ที่อยู่ | คืออะไร |
|---|---|
| `results/<exp>_fold<k>_seed<s>/` | `summary.json`, `pred_val.csv`, `pred_test.csv` (prob ครบ 184 คลาส ตาม brief §3.6), `log.csv` |
| `results/TABLE.md` | ตารางรวมทุก run (`python -m experiments.aggregate`) |
| `experiments/` | data (cache C54 + fold), `augment.py` (aug แบบ SPOTER + hand-local), `train.py` (Part B), `train_rgb.py` (Part C), `models/` (B0–B6, C1–C4) |
| `spoter/train_batched.py` | SPOTER + `--fold`, `--pe`, `--select f1`, `--patience` |
| `run_queue.sh`, `queues/` | รันหลาย run ต่อกัน (ข้าม run ที่มีผลแล้ว, log แยก) |
| `tools/make_csv.py --split-json`, `tools/extract_frames.py` | เตรียม CSV ของ SPOTER ตาม fold / ตัดเฟรม RGB |
| `splits/` | fold JSON ของทีม (คัดลอกมา) |

**สูตร Part B ที่ใช้:** `--aug --channels xy_mask --hand_local` (AdamW lr 1e-3, wd 1e-4, batch 32, ≤100 epoch, patience 20, เลือกจาก val macro-F1) เลือกสูตรด้วย val ของ B1 seed 0–1 เท่านั้น
**สภาพแวดล้อม:** Apple M4 (MPS), torch 2.15, torchvision 0.29, Python venv `~/venvs/sign2say`
**ข้อมูล:** `data/` และวิดีโอไม่อยู่ใน git (CC BY-NC) ต้องเตรียม `data/tsl1s_c54_T64.npz` (cache ของทีม) และ `data/videos/` เอง

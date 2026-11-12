# Sign2Say: Architecture Analysis (branch `base_line`)

Branch นี้วิเคราะห์ว่า **โครงสร้างโมเดลแบบไหนเหมาะกับภาษามือไทย** บน TSL-ONE-S แบบ signer-independent โดยคุม input, split และสูตรเทรนให้เท่ากันทุกโมเดล
ระบบหลัก (pipeline สกัด landmark, โมเดล S2S-LTR, CSLR, robustness) อยู่ที่ branch [`core-pipeline`](https://github.com/Varanapat/Sign2Say/tree/core-pipeline)

วิชา Deep Learning in Medical Image and Video Analysis

## คำถามวิจัย (RQ-S)

ควรเรียน "พื้นที่" (ความสัมพันธ์ระหว่างจุดในเฟรมเดียว) กับ "เวลา" (การเคลื่อนไหวข้ามเฟรม) แยกกันหรือพร้อมกัน

| ระดับ | โครงสร้าง | โมเดล |
|---|---|---|
| L0 | spatial อย่างเดียว (ไม่รู้ลำดับเวลา) | B0 MLP + mean pool, A1 SPOTER เดิม, C1 ResNet-18 + mean |
| L1 | spatial → temporal | B1 BiGRU, B2 Transformer, B6 GCN → BiGRU, A2 SPOTER + PE, C2 ResNet-18 → BiGRU |
| L2 | spatial + temporal พร้อมกัน | B3 ST-GCN, B4 divided space-time attention, C3 R(2+1)D-18, C4 R3D-18 |

## ผล (fold 4, test = signer 03, 08, 22, 416 คลิป, 3 seed)

| โมเดล | ระดับ | Params | Test top-1 | Test macro-F1 |
|---|---|---|---|---|
| B0 MLP + mean pool | L0 | 0.18M | 67.6 ± 0.5 | 66.4 |
| A1 SPOTER (เดิม) | L0 | 5.93M | 70.6 ± 3.3 | 68.4 |
| B1 MLP → BiGRU | L1 | 2.14M | 69.0 ± 2.3 | 67.5 |
| B2 MLP → Transformer | L1 | 3.29M | 69.7 ± 1.3 | 67.0 |
| B6 GCN → BiGRU | L1 + graph | 2.20M | 69.1 ± 2.3 | 66.5 |
| A2 SPOTER + PE | L1 | 5.94M | 69.0 ± 3.3 | 66.2 |
| **B3 ST-GCN** | **L2** | 3.14M | **75.0 ± 1.6** | **73.0** |
| B4 divided attention | L2 | 2.52M | 61.5 ± 3.0 | 57.7 |
| C2 ResNet-18 → BiGRU (RGB, 1 seed) | L1 | 12.5M | 37.5 | 36.6 |
| C1, C3, C4 (RGB) | L0 / L2 | 11.3–33.3M | ⏳ กำลังรัน | |

**ข้อค้นพบ**
1. **L2 แบบ ST-GCN ดีกว่า L1 ชัดเจน** (+5–6 pp, McNemar p ≤ 3e-4) ส่วน L1 ดีกว่า L0 แค่เล็กน้อยและไม่มีนัยสำคัญ
2. **ตาราง 2×2 แยกตัวแปรกวน:** graph prior อย่างเดียวไม่ช่วย (B6 ≈ B1) และ attention ที่เรียนพร้อมกันแต่ไม่มีโครงสร้างก็ไม่ช่วย (B4 underfit) สิ่งที่ช่วยคือการเรียนพื้นที่และเวลาเฉพาะที่พร้อมกันบนโครงกระดูก (B3 − B6 = +5.9 pp, p = 2e-5)
3. **Hand-local feature** (พิกัดมือเทียบข้อมือแล้วหารด้วยขนาดมือ) เพิ่ม val ได้ 7–12 pp เพราะในพิกัดเทียบไหล่ มือมีขนาดเล็กจนรายละเอียดรูปมือเรียนยาก
4. **SPOTER ที่ implement จริงไม่รู้ลำดับเวลา:** สลับลำดับเฟรมแล้ว output ต่างกันแค่ระดับ floating-point error และการเพิ่ม positional encoding ก็ไม่ช่วย
5. **RGB ผลแรกต่ำมาก** (37.5%) และ overfit หนัก (train 95%)

ข้อจำกัด: ผลทั้งหมดเป็น fold 4 fold เดียว สูตรเทรนของ branch นี้ (lr 1e-3 คงที่, 100 epoch) ต่างจาก `core-pipeline` ตัวเลขจึงใช้เทียบ **ระหว่างระดับ** เท่านั้น ห้ามเทียบตรง ๆ กับผลของ `core-pipeline` ผลเมื่อนำ hand-local และ ST-GCN ไปเทรนด้วยสูตรของ `core-pipeline` (5 fold): hand-local ช่วย S2S-LTR +2.9 pp (84.2 → 87.1, ดีขึ้น 5/5 fold) และช่วย ST-GCN +3.4 pp บน fold 4 ยืนยันข้อค้นพบที่ 3 แต่ ST-GCN แพ้ S2S-LTR ราว 3 pp เมื่อใช้สูตรเดียวกัน ข้อค้นพบที่ 1 จึงถือได้ภายในสูตรของ branch นี้ รายละเอียดดูใน branch นั้น

รายละเอียดทุกการทดลอง เหตุผล และ McNemar ทุกคู่อยู่ใน [`EXPERIMENTS.md`](EXPERIMENTS.md)

## โครงสร้าง

```
experiments/
  data.py, augment.py       โหลด cache C54, augmentation, hand-local feature
  train.py                  training loop กลางของ landmark model (Part B)
  train_rgb.py, rgb_data.py training loop และ data loader ของ RGB (Part C)
  metrics.py, results.py    top-1/5, macro-F1, latency, บันทึกผล
  aggregate.py              รวมผลทุก run เป็น results/TABLE.md
  models/                   b0_frame_mlp, b1_bigru, b2_transformer, b3_stgcn,
                            b4_divided_attention, b6_gcn_gru, rgb (C1–C4)
spoter/                     SPOTER (fork ของ maty-bohacek/spoter) + train_batched.py, --fold, --pe
tools/                      extract_pose.py, extract_frames.py, make_csv.py, check_npy.py, diff_ids.py
splits/                     10 fold signer-independent (ชุดเดียวกับ core-pipeline)
queues/                     คิวการรันที่ใช้จริง
run_queue.sh                รันคิวต่อกัน ข้าม run ที่เสร็จแล้ว แล้วสรุปตาราง
results/<exp>_fold<k>_seed<s>/  summary.json, log.csv, ไฟล์ทำนาย (ไม่มีน้ำหนักโมเดล)
```

## ติดตั้ง

รายละเอียดแยกตามเครื่อง (macOS / NVIDIA / AMD DirectML) อยู่ใน [`SETUP.md`](SETUP.md)

```bash
python3.12 -m venv ~/venvs/sign2say
~/venvs/sign2say/bin/pip install -r requirements.txt
```

`run_queue.sh` ใช้ Python ที่ `~/venvs/sign2say/bin/python` เป็นค่าเริ่มต้น เปลี่ยนได้ด้วยตัวแปร `PYTHON`

## ข้อมูล (ไม่อยู่ใน repo)

TSL-ONE-S ใช้สัญญาอนุญาต non-commercial และเห็นหน้าผู้แสดง **ห้ามอัปโหลดวิดีโอ, landmark หรือเฟรมที่ตัดแล้ว**

| ไฟล์ | ที่มา |
|---|---|
| `data/tsl1s_c54_T64.npz` | สร้างด้วย `scripts/extract_tsl1s.py` ของ branch `core-pipeline` แล้วคัดลอกมา |
| `data/split_fold4/*.csv` (SPOTER) | `python tools/extract_pose.py` แล้ว `python tools/make_csv.py` |
| `data/rgb_T16_128/` (RGB) | `python tools/extract_frames.py` (16 เฟรม, letterbox 128², ~3.2 GB) |
| `data/label_map.json` | อยู่ใน repo (ตรงกับ label map ของ core-pipeline) |

## วิธีรัน

```bash
# landmark model 1 run
python -m experiments.train --model B3 --exp B3_hl --fold 4 --seed 0 --aug --channels xy_mask --hand_local
# RGB 1 run
python -m experiments.train_rgb --model C3 --exp C3 --fold 4 --seed 0
# รันทั้งคิว (ข้าม run ที่มีผลแล้ว) แล้วสรุปตาราง
./run_queue.sh queues/partB_fold4.txt
./run_queue.sh -n queues/confound_fold4.txt    # dry run
# SPOTER
cd spoter && python train_batched.py --exp A1_spoter --fold 4 --seed 0 --device mps --aug --epochs 100 --patience 20 --select f1
```

ผลแต่ละ run อยู่ที่ `results/<exp>_fold<k>_seed<s>/summary.json` และตารางรวมที่ `results/TABLE.md`

## เอกสาร

| ไฟล์ | เนื้อหา |
|---|---|
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | บันทึกการทดลองทุกครั้ง ผล และการตีความ |
| [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md) | อธิบายโค้ด รวมข้อสังเกตเรื่องโค้ด SPOTER |
| [`SETUP.md`](SETUP.md) | ติดตั้งแยกตามเครื่อง |
| `docs/WORKLOG_SPOTER.md` | บันทึกการ reproduce SPOTER ช่วงแรก (README เดิม) |

## อ้างอิง

- TSL-ONE-S: *TSL-ONE-S: A Real-World Thai Sign Language Dataset With Deep Learning Benchmarks*, IEEE Access, 2026
- SPOTER: Boháček & Hrúz, *Sign Pose-based Transformer for Word-level Sign Language Recognition*, WACV Workshops 2022 ([repo](https://github.com/maty-bohacek/spoter))
- ST-GCN: Yan et al., *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*, AAAI 2018

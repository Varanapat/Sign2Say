# Brief: เทียบโครงสร้าง Spatial → Temporal vs Spatiotemporal (TSL-ONE-S)

> สำหรับ: เพื่อนที่ทำ `Sign2Say-base_line` (SPOTER) · จาก: ทีม Sign2Say · 9 พ.ย. 2026
> แบ่งงาน: **ฝั่งเรา** ทำระบบหลักตามแบบเดิม (landmark S2S-LTR + ablation ด้านข้อมูล/จุดหน้า) · **ฝั่งเพื่อน** ทำการทดลองนี้ = เทียบว่า "เรียนพื้นที่กับเวลาแยกกัน หรือพร้อมกัน" ดีกว่า
> ทั้งสองฝั่งใช้ split / label / metric / รูปแบบไฟล์ผลเดียวกัน เพื่อรวมผลและทำ fusion ได้ภายหลัง

---

## 1. คำถามที่ต้องตอบ

**RQ-S**: บน TSL-ONE-S แบบ signer-independent โครงสร้างแบบไหนดีกว่า เมื่อคุม input และขนาดโมเดลให้เท่ากัน

| ระดับ | โครงสร้าง | ความหมาย |
|---|---|---|
| L0 Spatial only | ฝังแต่ละเฟรม → เฉลี่ยทุกเฟรม | ไม่รู้ลำดับเวลาเลย (ขอบล่าง) |
| L1 Spatial → Temporal | ฝังแต่ละเฟรมก่อน → โมเดลลำดับเวลา | แบบที่เราใช้อยู่ (BiGRU) และแบบ SPOTER |
| L2 Spatiotemporal | ทุกชั้นมองทั้งข้อต่อข้างเคียง **และ** เฟรมข้างเคียงพร้อมกัน | ST-GCN, 3D CNN |
| L3 (เสริม) Temporal → Spatial | เรียนเวลาของแต่ละจุดก่อน → รวมข้อต่อ | ดูผลของการกลับลำดับ |

**สมมติฐาน (ตั้งไว้ก่อนรัน ไม่ใช่ผล)**: L1 และ L2 ดีกว่า L0 ชัดเจน; L2 ดีกว่า L1 เล็กน้อยบน landmark (โครงกระดูกมีโครงสร้างกราฟให้ใช้) ส่วนบน RGB ไม่แน่ เพราะข้อมูลมีแค่ ~3.3K คลิป

---

## 2. การทดลอง (เรียงตามลำดับที่ควรทำ)

### Part A — งานเล็กก่อน (ต่อจากของเดิม)

| ID | งาน | ทำไม |
|---|---|---|
| A1 | SPOTER เดิม บน **fold 4 ของ split เรา** (§3) | ให้ตัวเลข SPOTER ของเพื่อนเทียบกับ BiGRU ของเราได้ตรง (เพื่อนได้ 66.11% บน test ชุดเดียวกัน แต่ val คนละกลุ่ม) |
| A2 | SPOTER + **positional encoding ตามเวลาจริง** (sinusoidal หรือ learnable 64 ตำแหน่ง แทน `self.pos` ขนาด `(1,1,d)`) | ตอบว่าการที่ SPOTER ไม่รู้ลำดับเฟรม (ข้อค้นพบ 6.5 ของเพื่อน — เราตรวจโค้ดแล้วว่าจริง) ทำให้เสียไปเท่าไร; SPOTER เดิม = L0 แบบ attention, SPOTER + PE = L1 |

### Part B — Landmark (ถูก, ทำได้บน CPU/GPU เล็ก) ⭐ หลัก

ทุกตัวใช้ input เดียวกัน: `[B, T=64, N, C]` และ **พารามิเตอร์ใกล้กัน ~1.5–3M**

| ID | ระดับ | โมเดล | สเปกที่แนะนำ |
|---|---|---|---|
| B0 | L0 | per-frame MLP + mean pool | MLP 2 ชั้น (N·C → 256 → 256) → เฉลี่ยตามเวลา → Linear 184 |
| B1 | L1 | per-frame MLP → BiGRU | เหมือนของเรา: Linear 256 → BiGRU 2 ชั้น hidden 256 → mean pool → Linear (2.1M params) |
| B2 | L1 | per-frame MLP → Transformer encoder | 4 ชั้น, d = 256, 4 heads, ff 1024, + positional encoding, [CLS] token |
| B3 | **L2** | **ST-GCN** | กราฟข้อต่อ (มือ 21 จุดตามกระดูกนิ้ว + แขน + หัว), 6–9 บล็อก (spatial graph conv → temporal conv k = 9), channel 64 → 128 → 256; ใช้โค้ดจาก [MMAction2 / pyskl](https://github.com/kennymckormick/pyskl) หรือเขียนเอง |
| B4 | L2 (ทางเลือก) | Space-time Transformer แบบ **divided attention** | token = 1 จุด × 1 เฟรม (64 × 54 ≈ 3.5K token) → แต่ละชั้นทำ attention ตามเวลา (จุดเดียวกันข้ามเฟรม) แล้วตามข้อต่อ (เฟรมเดียวกัน) ไม่ทำ full attention เพราะหนักเกิน |
| B5 | L3 (ทางเลือก) | per-joint temporal conv → รวมจุด | 1D conv ตามเวลาแยกต่อจุด (shared weights) → flatten จุด → MLP |

### Part C — RGB (ใช้ GPU ของเพื่อน)

ใช้ตระกูล ResNet-18 ทั้งหมดเพื่อให้เทียบกันได้ (torchvision มี weight สำเร็จรูปทุกตัว)

| ID | ระดับ | โมเดล | Pretrained | Input |
|---|---|---|---|---|
| C1 | L0 | ResNet-18 ต่อเฟรม → mean pool | ImageNet | 16 เฟรม × 112² |
| C2 | L1 | ResNet-18 ต่อเฟรม → LSTM/GRU (CNN-LSTM ตามบท Ch13) | ImageNet | 16 เฟรม × 112² |
| C3 | L2 แบบแยก conv | `torchvision.models.video.r2plus1d_18` | Kinetics-400 | 16 เฟรม × 112² |
| C4 | L2 เต็ม | `torchvision.models.video.r3d_18` | Kinetics-400 | 16 เฟรม × 112² |
| C5 | (ถ้าไหว) | I3D แบบ paper | Kinetics-400 | 64 × 224² — RTX 4060 8 GB น่าจะต้อง batch เล็ก + accumulation + AMP |

> ⚠️ C1–C4 ใช้ 16 เฟรม 112² ทุกตัวเพื่อให้ยุติธรรม; คลิปเฉลี่ย ~58 เฟรม → sample 16 เฟรมเว้นระยะเท่ากัน ห้ามตัดเอา 16 เฟรมแรก (จะตัดท่าทิ้ง)

**ลำดับที่แนะนำ**: A1 → A2 → B0 → B1 → B3 → B2 → C2 → C3 → C4 → (B4, B5, C1, C5 ถ้ามีเวลา)

---

## 3. ข้อตกลงร่วม (สำคัญที่สุด)

### 3.1 Split
- ใช้ไฟล์ใน `splits/` ของโฟลเดอร์นี้: `tsl_one_s_signer_indep_fold{0-9}.json` — รูปแบบเดียวกับ `tsl_one_s_dataset_i3d.json` เดิม (ช่อง `split` ถูกเขียนใหม่ตามผู้แสดง)
- **เริ่มที่ fold 4**: test = signer 03, 08, 22 (416 คลิป — ชุดเดียวกับ split B ของเพื่อน), val = **02, 25, 29** (ของเพื่อนเดิมคือ 11, 25, 29 → ต้องเปลี่ยนเป็นของ fold 4)
- ถ้ามีเวลา: fold 0, 1, 2 (เรามีผล BiGRU แล้ว) → แล้วจึงครบ 10 fold
- `manifest.json` บอกผู้แสดงของทุก fold

### 3.2 4 คลิปที่ชื่อไม่ตรง JSON
- JSON มี 4 id ที่ไม่มีไฟล์ (`01_08_0266`, `15_08_0267`, `16_01_0020`, `16_01_0028`) และมี 4 ไฟล์ที่ไม่อยู่ใน JSON (`11_04_0134`, `25_01_0028`, `25_08_0277`, `25_08_0278`)
- เราตรวจแล้ว: `CategoryID_GlossID` จับคู่กับ gloss ได้ไม่ซ้ำ (184 คู่) → label ที่เพื่อนใส่ให้ 4 ไฟล์ตามชื่อไฟล์ **สอดคล้องกัน**
- **ข้อตกลงตอนนี้: ตัด 4 ไฟล์นี้ออก** (split ของเราไม่มี) เพื่อให้ชุดคลิปตรงกันทุกบิต — ไม่กระทบ test ของ fold 4 (signer 11 และ 25 ไม่อยู่ใน test) แต่ 3 ไฟล์ของ signer 25 จะหายจาก val ของ fold 4 ไป 3 คลิป

### 3.3 Label
- ใช้ **`label_map.json` ของเพื่อน** (0–183 เรียงตาม CategoryID, GlossID) เป็นมาตรฐานกลาง — เราจะแปลงของเราให้ตรง

### 3.4 การฝึกและเลือกโมเดล
- เลือก checkpoint จาก **val macro-F1** (หรือ val top-1 ถ้าระบุ) — ห้ามใช้ test เลือก
- augmentation เฉพาะ train; ไม่ใช้ horizontal flip (ผู้แสดงถนัดขวาทั้งหมด)
- budget เท่ากันทุกโมเดลใน Part B (เช่น 100 epoch + early stopping patience 20) และใน Part C (เช่น 30 epoch)
- seed: อย่างน้อย 1; ถ้าได้ 3 seed (0, 1, 2)

### 3.5 Metric
Top-1, Top-5, macro-F1 บน test + val; จำนวนพารามิเตอร์; เวลาฝึก (wall-clock, ระบุ GPU); latency ต่อคลิป batch 1 (median ของ 100 รอบหลัง warm-up 10 รอบ)

### 3.6 รูปแบบไฟล์ผล (เพื่อรวมผล + ทำ late fusion)

ต่อ 1 run เก็บไว้ที่ `results/<exp>_fold<k>_seed<s>/`:

`summary.json`
```json
{"exp": "B3_stgcn", "level": "L2", "modality": "landmark", "fold": 4, "seed": 0,
 "input": "54 joints x (x,y), T=64", "params": 2400000,
 "best_epoch": 63, "val": {"top1": 0.0, "top5": 0.0, "macro_f1": 0.0},
 "test": {"top1": 0.0, "top5": 0.0, "macro_f1": 0.0},
 "train_minutes": 0.0, "latency_ms_b1": 0.0, "device": "RTX 4060 Laptop"}
```

`pred_val.csv`, `pred_test.csv` — **หนึ่งแถวต่อคลิป, ความน่าจะเป็นครบ 184 คลาส** (หลัง softmax)
```
video_id,label,p0,p1,...,p183
03_01_0001,0,0.91,0.002,...
```

---

## 4. ข้อสังเกตจากฝั่งเรา (อาจช่วยให้ผลดีขึ้น)

1. **วิดีโอถูก crop แล้วและอัตราส่วนไม่เท่ากัน**: W/H อยู่ระหว่าง 0.51–1.46 (เช่น 352×578) ไม่ใช่ 1280×720 ที่ `make_csv.py` ตั้งไว้ → พิกัด MediaPipe [0, 1] ของแต่ละคลิปยืดแนวนอนไม่เท่ากัน (ต่างกันได้ ~3 เท่า) **แนะนำ**: คูณ x ด้วยความกว้างจริง และ y ด้วยความสูงจริงของแต่ละวิดีโอ (`cv2.CAP_PROP_FRAME_WIDTH/HEIGHT`) ก่อน normalize — เราทำแบบนี้ (`scripts/landmarks.py`) ยังไม่ได้ทดสอบว่าช่วย SPOTER แค่ไหน → **ถ้าทำ จะเป็น ablation ที่ดีมาก (A1 ± aspect fix)**
2. **SPOTER overfit**: train 98.6% แต่ val 73–77% และ val ลดลงช่วงท้ายใน log ของเพื่อน → early stopping / weight decay อาจช่วย
3. **จุดที่หาย**: SPOTER ใส่ 0 → ค่า 0 ปนกับพิกัดจริง ของเราเติม interpolation ช่องว่าง ≤ 5 เฟรม + ส่ง mask (0/1) เป็น feature เพิ่ม
4. ถ้าอยากใช้ landmark ชุดเดียวกับเรา (C54: ไหล่/ศอก/ข้อมือ + คิ้ว/มุมปาก + มือ, x/y/z) มีสคริปต์ใน `scripts/` (`extract_tsl1s.py` + `landmarks.py`, ใช้ `mediapipe==0.10.21`, ~45 นาทีบน 8 cores) — **ไม่บังคับ**: Part B เทียบกันภายในชุดของเพื่อนได้อยู่แล้ว ขอแค่ทุกโมเดลใน Part B ใช้ input เดียวกัน

---

## 5. ผลที่ฝั่งเรามีแล้ว (ไว้เทียบ)

BiGRU (= B1) บน C54, T = 64, 1 seed:

| Split | Top-1 | Top-5 | Macro-F1 |
|---|---|---|---|
| fold 4 | 73.1 | 90.4 | 72.1 |
| fold 0 / 1 / 2 | 86.0 / 74.5 / 76.4 | – | 85.6 / 71.6 / 74.3 |
| split เดิมใน JSON | 81.2 | 94.7 | 79.9 |

> test fold เดียว ~400 คลิป → ช่วงความเชื่อมั่น ~±4 pp; ผลต่างน้อยกว่านั้นจาก fold เดียวสรุปไม่ได้

---

## 6. ไฟล์ในโฟลเดอร์นี้

| ไฟล์ | คืออะไร |
|---|---|
| `README.md` | เอกสารนี้ |
| `splits/tsl_one_s_signer_indep_fold{0-9}.json` | split แบบ signer-independent (รูปแบบ JSON เดิม) |
| `splits/manifest.json` | ผู้แสดงและจำนวนคลิปของทุก fold |
| `scripts/make_signer_split.py` | สคริปต์ที่สร้าง split (seed 42) |
| `scripts/landmarks.py`, `scripts/extract_tsl1s.py` | (ไม่บังคับ) การสกัดและ normalize landmark แบบของเรา |

ข้อมูลเป็น CC-NC — ใช้เพื่อการเรียนเท่านั้น ไม่แจกจ่ายวิดีโอต่อ

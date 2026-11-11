# Code Walkthrough — Sign2Say (SPOTER baseline)

เอกสารนี้อธิบายการทำงานของโค้ดแต่ละไฟล์ และความเชื่อมโยงระหว่างไฟล์ เรียงตามลำดับที่ข้อมูลไหลผ่าน

---

## 1. ภาพรวม Pipeline

```
data/videos/*.mp4
   │  tools/extract_pose.py      (MediaPipe Holistic)
   ▼
data/pose_npy/<video_id>.npy     shape (T, 150) = T เฟรม × 75 จุด × (x, y)
   │  tools/make_csv.py          (+ tsl_one_s_dataset_i3d.json)
   ▼
data/split_signer/tsl_{train,val,test}.csv   + data/label_map.json
   │  spoter/datasets/czech_slr_dataset.py
   │     ├─ spoter/augmentations/__init__.py          (rotate / shear / arm-joint rotate)
   │     └─ spoter/normalization/{body,hand}_normalization.py
   ▼
Tensor (T, 54, 2)   ค่าประมาณ [-0.5, 0.5]
   │  spoter/train.py (ทีละคลิป)  หรือ  spoter/train_batched.py (resample 64 เฟรม → batch)
   ▼
spoter/spoter/spoter_model.py  → logits 184 คลาส
   ▼
spoter/out-batched/<exp>/  (best_val.pth, log.csv, summary.txt, test_predictions.csv)
spoter/eval_ckpt.py        → top-1 / top-5 / macro-F1
```

**ไฟล์เสริม**
- `spoter/dml_adam.py`: Adam ที่ใช้กับ DirectML ได้
- `spoter/diag.py`: ตรวจข้อมูล และทำ overfit test
- `tools/check_npy.py`, `tools/diff_ids.py`: เช็กความครบของ .npy

**ไฟล์จาก SPOTER ต้นฉบับที่ pipeline นี้ไม่ได้ใช้**
- `spoter/normalization/main.py` และฟังก์ชัน `normalize_*_full()`
- `spoter/data_structurization/*` (path ยังชี้ไปเครื่องของผู้เขียนเดิม)

---

## 2. `tools/extract_pose.py`: วิดีโอ → .npy

| บรรทัด | หน้าที่ |
|---|---|
| 19–22 | ตั้ง path default (`data/videos` → `data/pose_npy`) และนามสกุลวิดีโอที่รองรับ |
| 25–28 | `lm_to_array`: ถ้าจับจุดไม่ได้จะคืนค่า 0 ทั้งหมด ถ้าจับได้จะดึง `[x, y]` (0–1, y นับจากบนลงล่าง) และทิ้ง z |
| 32–34 | ข้ามไฟล์ที่ทำไปแล้ว จึงรันต่อจากที่หยุดไว้ได้ |
| 39–51 | Holistic แบบ video mode: แต่ละเฟรม BGR→RGB → process → ต่อ pose 33 + มือซ้าย 21 + มือขวา 21 = 75 จุด → 150 ค่า |
| 56–59 | เขียนลง `.npy.part` ก่อน แล้ว `os.replace` (atomic) เพื่อกันไฟล์เสียถ้าถูกปิดกลางคัน |
| 63 | `__main__` guard จำเป็นสำหรับ multiprocessing แบบ spawn (macOS/Windows) |
| 72–74 | ลบไฟล์ `.part` ที่ค้างจากรอบก่อน |
| 86–93 | `Pool.imap_unordered` แบ่งคลิปให้หลาย process ทำพร้อมกัน |

> ค่า **0 = จุดหาย** เป็น convention ที่ normalization ใช้ต่อ (มีการเช็ก `== 0` หลายจุด)

---

## 3. `tools/make_csv.py`: .npy + JSON → CSV

- **เลือกจุด 75 → 54 จุด**
  - HEAD 5 จุด: nose, eyes, ears
  - BODY 6 จุด: ไหล่ ศอก ข้อมือ
  - `neck` = ค่าเฉลี่ยไหล่ซ้าย/ขวา (MediaPipe ไม่มีจุดนี้ให้)
  - รวมเป็นลำตัว 12 จุด + มือ 21×2 = **54 จุด**
  - 54 × 2 = **108 = `hidden_dim`**
- **Signer split** (signer = เลขหน้าของ `video_id` เช่น `09_07_0238` → `09`)
  - test = 03, 08, 22
  - val = 11, 25, 29
  - ที่เหลือเป็น train
- `put()` เก็บแต่ละจุดเป็นคอลัมน์ `<name>_X`, `<name>_Y` แต่ละช่องเป็น list ของทุกเฟรม (เก็บเป็น string)
- **Label 0..183** เรียงตาม `(หมวด, เลขท่า)` ของ video_id แล้วบันทึกเป็น `data/label_map.json` (ชื่อ → id)
- `--split-mode json` ใช้ split จาก JSON ส่วน `--split-mode signer` ใช้กฎข้างบน (`train_batched.py` ใช้แบบ signer)

---

## 4. `spoter/datasets/czech_slr_dataset.py`: CSV → Tensor

- **บรรทัด 14:** ชื่อจุดมือถูกขยายเป็น `<joint>_0` (ซ้าย) และ `<joint>_1` (ขวา)
- **`load_dataset`**
  - เปลี่ยนชื่อคอลัมน์ `_left_`→`_0_`, `_right_`→`_1_`
  - `ast.literal_eval` แปลง string กลับเป็น list
  - เติมลง array `(T, 54, 2)` ตามลำดับ `BODY_IDENTIFIERS + HAND_IDENTIFIERS`
  - ขั้นนี้ช้า แต่ทำครั้งเดียวตอนสร้าง Dataset
- **`tensor_to_dictionary` / `dictionary_to_tensor`:** แปลงไปมาระหว่าง tensor กับ dict `{ชื่อจุด: (T,2)}` (augmentation และ normalization ทำงานกับ dict)
- **`__getitem__`** (ทำทุกครั้งที่โหลด 1 คลิป)
  1. `np.copy` เพื่อไม่ให้ข้อมูลต้นฉบับถูกแก้
  2. ถ้าเปิด augmentation: มีโอกาส 50% ที่สุ่ม 1 ใน 4 แบบ
     - rotate ±13°
     - perspective 0–0.1
     - squeeze 0–0.15
     - arm-joint rotate (แต่ละข้อมีโอกาส 30% หมุน ±4°)
  3. normalize ลำตัว แล้วตามด้วยมือ
  4. แปลงเป็น tensor แล้ว **ลบ 0.5** (จุดหายจะกลายเป็น -0.5)
  5. `transform` (เช่น GaussianNoise) ถ้ามี

---

## 5. `spoter/normalization/`

### `body_normalization.normalize_single_dict`

ทำทีละเฟรม:
- ถ้าไม่มีไหล่และไม่มี neck/nose จะใช้กรอบของเฟรมก่อนหน้า (ถ้าเป็นเฟรมแรกจะ mark ว่า invalid)
- `head_metric` = ระยะไหล่ซ้าย–ขวา (ถ้าไม่มีไหล่ใช้ระยะ neck–nose แทน)
- สร้างกรอบ:
  - กว้าง 6×head_metric โดยมี neck อยู่ตรงกลาง
  - สูง 6×head_metric โดยอ้างอิงจากตาซ้าย
- clamp ขอบไม่ให้ติดลบ
- แปลงทุกจุดเป็นพิกัดสัมพัทธ์ในกรอบ (ข้ามจุดที่เป็น 0)
- ถ้าเจอกรอบขนาด 0 จะคืน `original_row` (เป็น reference ไปยัง dict เดียวกัน เฟรมที่แก้ไปแล้วจะไม่ถูก rollback)

### `hand_normalization.normalize_single_dict`

ทำทีละมือ ทีละเฟรม:
- หา bounding box ของจุดมือที่ไม่เป็น 0
- ขยายกรอบ 10% และทำให้เป็นจัตุรัส
- แปลงจุดมือเป็นพิกัด 0–1 ภายในกรอบ
- ผลคือรูปทรงมือชัดขึ้น แต่ตำแหน่งมือเทียบกับตัวจะเหลือแค่ใน wrist ของส่วนลำตัว

---

## 6. `spoter/augmentations/__init__.py`

| ฟังก์ชัน | หน้าที่ |
|---|---|
| `__rotate` | หมุนจุดรอบ origin (เมทริกซ์หมุน 2D) |
| `__preprocess_row_sign` / `__wrap_sign_into_row` | แยก dict เป็นส่วน body/hand และรวมกลับ |
| `augment_rotate` | หมุนทุกจุดรอบ (0.5, 0.5) ด้วยมุมเดียวกันทั้งคลิป |
| `augment_shear` | ใช้ `cv2.getPerspectiveTransform` บีบหรือเอียงภาพ ทำกับ**ลำตัวเท่านั้น** และคงจุดที่เป็น 0 ไว้ |
| `augment_arm_joint_rotate` | ไล่ neck → shoulder → elbow → wrist แต่ละข้อมีโอกาส 30% ที่จะหมุนข้อถัด ๆ ไปทั้งหมดรอบข้อนั้น |

---

## 7. `spoter/spoter/spoter_model.py`: โมเดล

```python
self.row_embed   = nn.Parameter(torch.rand(50, hidden_dim))  # ไม่ถูกใช้ใน forward
self.pos         = nn.Parameter(...)                          # shape (1, 1, 108)
self.class_query = nn.Parameter(torch.rand(1, hidden_dim))    # query สำหรับจำแนกคลาส
self.transformer = nn.Transformer(108, 9, 6, 6)               # d_model, nhead, enc 6, dec 6
self.linear_class = nn.Linear(108, num_classes)
```

**`forward`**
1. `(T, 54, 2)` → flatten เป็น `(T, 108)` โดย 1 เฟรม = 1 token
2. `unsqueeze(1)` → `(T, 1, 108)` ตาม format `(seq, batch, dim)`
3. `pos + h` → encoder (self-attention ระหว่างเฟรม)
4. decoder ใช้ `class_query` token เดียว cross-attend กับ output ของ encoder (คล้าย CLS token)
5. `linear_class` → logits 184 คลาส

**`SPOTERTransformerDecoderLayer`**
- ลบ self-attention ออก เพราะมี query ตัวเดียว
- ใส่ `SimpleNamespace(batch_first=False)` แทน ให้ PyTorch 2.x อ่านค่า `batch_first` ได้
- บรรทัด `tgt = tgt + dropout1(tgt)` อยู่ในตำแหน่งที่เคยเป็น self-attention (quirk ของต้นฉบับ)

---

## 8. การเทรน

### `spoter/train.py` + `spoter/spoter/utils.py` (แบบต้นฉบับ: batch = 1 คลิป)

- ตั้ง seed → เลือก device (`FORCE_CPU` → DirectML → CUDA → CPU; ไม่มี MPS)
- ใช้ CrossEntropy + Adam/DmlAdam + ReduceLROnPlateau
- Dataset ชุด train มี GaussianNoise และ augmentation ส่วน val/test ไม่มี
- `train_epoch`
  - `squeeze(0)` เอามิติ batch ออก
  - หาร loss ด้วย `ACCUM_STEPS` (env var) แล้ว `step` ทุก N คลิป (gradient accumulation)
- loop หลัก: train → val → เก็บ checkpoint ที่ดีที่สุดในแต่ละช่วง 10 epoch เป็น `checkpoint_{t|v}_{i}.pth` (save ทั้งโมเดล)
- ตอนจบ: วัดทุก checkpoint บน test แล้ววาดกราฟลง `out-img/`

### `spoter/train_batched.py` (batch จริง)

| ส่วน | หน้าที่ |
|---|---|
| `resample_frames` | เลือก 64 เฟรมที่เว้นระยะเท่ากันด้วย `linspace` (คลิปสั้นมีเฟรมซ้ำ คลิปยาวข้ามบางเฟรม) |
| `make_collate` | resample + ใส่ noise → stack เป็น `(B, 64, 54, 2)` |
| `batched_forward` | ทำแบบเดียวกับ `SPOTER.forward` แต่รองรับ B คลิป: `h=(T,B,108)` และ `class_query=(1,B,108)` ไม่ต้องแก้ไฟล์โมเดล |
| `check` | ยืนยันว่าผลแบบทีละคลิปกับแบบ batch ต่างกันไม่เกิน 1e-4 |
| `train` | Adam lr 1e-4, batch 16 → วัด val ทุก epoch → บันทึก `best_val.pth` (state_dict บน CPU) + `log.csv` → ตอนจบโหลด best มาวัด test → `summary.txt`, `test_predictions.csv` |

ไม่มี LR scheduler (LR คงที่)

---

## 9. ไฟล์เสริม

- **`eval_ckpt.py`**: วัดผล checkpoint ของ `train.py`
  - `find_ckpt` เลือก `checkpoint_v_*` ที่ mtime ล่าสุด
  - `load_model` โหลดได้ทั้งแบบทั้งโมเดลและแบบ state_dict
  - `load_names` กลับ label_map ให้เป็น id → ชื่อ
  - `macro_f1` คิด F1 ของแต่ละคลาสแล้วเฉลี่ย
  - `train_batched.py` import `load_names` และ `macro_f1` จากไฟล์นี้
- **`dml_adam.py`**: Adam ที่เขียนด้วย `mul_/add_/addcdiv_` แทน `lerp` ที่ DirectML ไม่รองรับ (`python dml_adam.py` ทดสอบว่าผลเท่ากับ `torch.optim.Adam`)
- **`diag.py`**
  - `data` ตรวจ NaN, คลิปที่ไม่ขยับ, สัดส่วนจุดหาย และช่วงค่า input
  - `overfit` ให้จำ 32 คลิป ถ้า pipeline ถูกต้อง acc ต้องเข้าใกล้ 100%
- **`spoter/utils.py`**: แบ่ง val แบบ stratified จาก train (ใช้เฉพาะตอน `--validation_set split-from-train` หรือ `--experimental_train_split`)
- **`spoter/spoter/gaussian_noise.py`**: บวก noise `N(mean, std)` ทุกค่า

---

## 10. ข้อสังเกต (สำหรับวิเคราะห์ผลหรือปรับปรุงในอนาคต)

1. **Positional encoding ไม่ได้ encode ลำดับเวลา:** `self.pos` เป็นเวกเตอร์เดียว `(1,1,108)` ที่บวกให้ทุกเฟรมเท่ากัน โมเดลจึงไม่รู้ลำดับเฟรม (พฤติกรรมของ SPOTER ต้นฉบับ)
2. **แกน y ของ body normalization:** สูตรเขียนไว้สำหรับ Apple Vision (y ชี้ขึ้น) แต่ MediaPipe ให้ y ชี้ลง
   - กรอบด้านหนึ่งจึงติดลบแล้วถูก clamp เป็น 0
   - ไหล่/ศอก/ข้อมือน่าจะมีค่า y > 1
   - ไม่ทำให้โค้ดพัง แต่ scale ไม่ตรงกับ paper ตรวจได้ด้วย `python diag.py data` (ดูช่วงค่า input)
3. **จุดหาย + rotate:** `augment_rotate` หมุนจุด (0,0) ด้วย จุดที่หายจึงกลายเป็นค่าที่ไม่ใช่ 0 และถูกนับเป็นจุดจริงตอน hand normalization
4. **`train.py` ไม่ได้ส่ง scheduler เข้า `train_epoch`:** ReduceLROnPlateau ไม่ทำงาน LR คงที่
5. **`train.py` มีค่า default `--num_classes 64`:** ต้องส่ง `184` เองทุกครั้ง

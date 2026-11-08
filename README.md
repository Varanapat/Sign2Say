# โปรเจกต์ Sign Language Recognition (SPOTER บน TSL-ONE-S)

วิชา Deep Learning in Medical Image and Video Analysis
หัวข้อ: Video analysis ภาษามือ (Thai Sign Language) งานของเรา: ทำ **baseline** และวิเคราะห์ว่าแบบไหนดี
Paper อ้างอิง: *TSL-ONE-S: A Real-World Thai Sign Language Dataset With Deep Learning Benchmarks* (IEEE Access, 2026)

> ไฟล์นี้บันทึกทุกอย่างที่ทำไปแล้ว ผลที่ตรวจสอบแล้ว ปัญหาที่เจอ และขั้นตอนที่ต้องรันต่อ
> อัปเดตล่าสุด: 8 พ.ย. 2026 (เช้า) — มีสคริปต์เทรนแบบ batch จริง `train_batched.py` (batch 16 × 64 เฟรม) เร็วกว่าเดิม ~3 เท่า ใช้ GPU ได้คุ้ม; เหลือเทรนจริง 80 epoch

---

## 0. สรุปสถานะ (อ่านตรงนี้ก่อน)

| งาน | สถานะ |
|---|---|
| สภาพแวดล้อม (NVIDIA/CUDA และ AMD/DirectML/CPU) | ✅ |
| สกัด pose ครบ 4,152 คลิป (`data/pose_npy/`) | ✅ |
| CSV split B (signer) 3,317 / 419 / 416, 184 classes, ไฟล์ขาด 0 | ✅ |
| แก้โค้ด SPOTER ให้รันบน PyTorch 2.x + 184 classes | ✅ (ดูข้อ 3) |
| แก้ปัญหาโมเดลไม่เรียนรู้ (batch 1 + Adam) ด้วย gradient accumulation 16 | ✅ (ดูข้อ 6) |
| สคริปต์วัดผลที่เลือก checkpoint จาก validation (`eval_ckpt.py`) | ✅ |
| ทดลอง 5 epoch split B: test **42.79%**, top-5 76.68%, macro-F1 0.363 | ✅ |
| สคริปต์เทรนแบบ batch จริง `train_batched.py` (ตรวจแล้วว่าให้ผลเท่าโมเดลเดิม) | ✅ |
| วัดความเร็ว: GPU 75 วิ/epoch, CPU 96 วิ/epoch (เดิม ~240 วิ) | ✅ |
| พบว่า SPOTER ที่ implement จริงไม่มี positional encoding ตามเวลา (ข้อ 6.5) | ✅ |
| เทรนจริง 80 epoch split B (มี augmentation) ด้วย `train_batched.py` | ✅ |
| ตรวจ 4 คลิปที่ชื่อไฟล์ไม่ตรง JSON (ข้อ 4.6) | ✅ |
| split A (json) สร้าง CSV ใหม่ + เทรน | ⏳ |

---

## 1. เป้าหมายและการตัดสินใจ

- ทำ baseline และเปรียบเทียบ ตามที่ paper ทำกับ 2 โมเดล:
  - **I3D** (RGB, Kinetics-400 pretrained): 79.69% top-1, F1 0.79, 12.35M params, เทรน ~21 ชม. 41 นาที (RTX A6000)
  - **SPOTER** (pose + Transformer): 80.83% top-1, F1 0.78, 5.92M params, เทรน ~1 ชม. 48 นาที (RTX 2070)
- **ตกลงกับเพื่อนว่าเลือกทำ SPOTER ก่อน** เพราะเทรนเร็ว ใช้ข้อมูล pose ได้ และมี repo ทางการ
- ส่วน Pose-LSTM, RGB model, fusion และหลาย seed ทำวันอื่น
- ค่าตาม paper ที่ใช้เป็นเป้าหมาย reproduce: **80 epochs, batch size 16, learning rate 1e-4, Adam**, เลือก checkpoint จาก validation
  - batch 16 ทำผ่าน **gradient accumulation 16 step** (ข้อ 3.5) เพราะโค้ด SPOTER รับทีละ 1 คลิป

---

## 2. สภาพแวดล้อม (ทำแล้ว)

มี 2 เครื่องในทีม ตั้งค่าต่างกันตาม GPU

### 2.1 เครื่อง NVIDIA (RTX 4060 Laptop)

- โฟลเดอร์โปรเจกต์: `C:\Users\khims\OneDrive\Desktop\3_1\DLM\project2`
- venv: `python -m venv venv` แล้ว `.\venv\Scripts\Activate.ps1`
  (ถ้าติด execution policy: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)
- PyTorch แบบ CUDA: `torch 2.5.1+cu121`, ตรวจแล้ว `torch.cuda.is_available() = True`
  - ติดตั้งจาก index ของ PyTorch (ไม่อยู่บน PyPI): `--extra-index-url https://download.pytorch.org/whl/cu121`

### 2.2 เครื่อง AMD (Radeon RX 6700 XT)

- โฟลเดอร์โปรเจกต์: `C:\Users\MyHome\OneDrive\Desktop\DPL_PJ2\Sign2Say`, Python 3.12.8
- **venv อยู่นอก OneDrive** ที่ `C:\venvs\sign2say` (เปิดด้วย `C:\venvs\sign2say\Scripts\Activate.ps1`)
  - เหตุผล: venv ใน OneDrive ทำให้ไฟล์ `.pyd` ของ pandas เสีย (`No module named 'pandas._libs.pandas_parser'`)
- RX 6700 XT (RDNA2) **ไม่รองรับ ROCm PyTorch บน Windows** (รองรับเฉพาะ RX 7000/9000) จึงใช้ `torch-directml`
  - ติดตั้ง: ลบ `torch==...+cu121` และ `--extra-index-url` ออกจาก requirements แล้ว `pip install torch-directml`
- **ความเร็วขึ้นกับวิธีป้อนข้อมูล**:
  - ป้อน**ทีละคลิป** (`train.py` เดิม): CPU เร็วกว่า DirectML เพราะโมเดลเล็ก เวลาส่วนใหญ่หมดกับการรับส่งข้อมูล CPU↔GPU

    | overfit test 32 คลิป × 40 epoch | เวลา |
    |---|---|
    | CPU + `torch.optim.Adam` | 81 วิ |
    | DirectML + `DmlAdam` | 138 วิ |
    | DirectML + `torch.optim.Adam` (lerp fallback ไป CPU) | 213 วิ |

  - ป้อน**เป็น batch จริง** (`train_batched.py`, 16 คลิป × 64 เฟรม): GPU เร็วกว่า และเร็วกว่าทีละคลิปมาก

    | วิธี (split B, 1 epoch รวม val) | เวลา/epoch | 80 epoch |
    |---|---|---|
    | ทีละคลิป + accumulation 16, CPU (`train.py`) | ~240 วิ | ~5–6 ชม. |
    | batch 16 × 64 เฟรม, CPU | 96 วิ | ~2 ชม. |
    | **batch 16 × 64 เฟรม, DirectML** | **75 วิ** | **~1 ชม. 40 นาที** |

  → **ใช้ `train_batched.py --device dml`** เป็นหลัก

### 2.3 ทั้งสองเครื่อง

- แพ็กเกจหลัก: `pandas numpy matplotlib scikit-learn tqdm opencv-python mediapipe==0.10.14`
  - ห้ามติดตั้ง `opencv-python` คู่กับ `opencv-contrib-python` (ไฟล์ `cv2` ทับกัน)
- **ข้าม `requirements.txt` ของ SPOTER** เพราะล็อก `pandas==1.1.5` (สร้างไม่ได้บน Python รุ่นใหม่) และ `torch==1.8.1`

> ทุกครั้งที่เปิดเทอร์มินัลใหม่ ให้เปิด venv ก่อน
> หมายเหตุ: บน Windows ถ้า `python xxx.py` หาไฟล์ไม่เจอ ข้อความ error จะแสดง path Python หลักของเครื่อง (`...\Python312\python.exe`) แม้เปิด venv อยู่ — ไม่ใช่ปัญหา venv แต่คือไฟล์ไม่อยู่ในโฟลเดอร์ปัจจุบัน

---

## 3. SPOTER repo และการแก้ไข (ทำแล้ว)

- โคลน: `git clone https://github.com/maty-bohacek/spoter` (อยู่ใน `spoter/`)
- **ระวังโฟลเดอร์ซ้อน**: ใน repo มีโฟลเดอร์ย่อย `spoter/spoter/` (โค้ดโมเดล) — รัน `train.py` จาก `spoter/` ชั้นแรกเท่านั้น
- Argument ที่มี: `--experiment_name`, `--num_classes`, `--hidden_dim`, `--seed`, `--training_set_path`, `--testing_set_path`, `--validation_set {from-file,split-from-train,none}`, `--validation_set_path`, `--epochs`, `--lr`, `--log_freq`, `--save_checkpoints`, `--scheduler_factor`, `--scheduler_patience`, `--gaussian_mean`, `--gaussian_std`, `--plot_stats`, `--plot_lr`
  (`--help` ขึ้น `unrecognized arguments` แต่แสดงรายการได้ — ไม่ใช่ปัญหา)

### รายการแก้ไขโค้ด SPOTER ทั้งหมด (ต้องระบุในรายงาน)

| # | ไฟล์ | แก้อะไร | เหตุผล |
|---|---|---|---|
| 3.1 | `train.py` (~บรรทัด 114) | optimizer จาก SGD เป็น Adam: CPU/CUDA ใช้ `optim.Adam`, DirectML ใช้ `DmlAdam` | ตาม paper |
| 3.2 | `spoter/spoter_model.py` | หลัง `del self.self_attn` เพิ่ม `self.self_attn = SimpleNamespace(batch_first=False)` และเติม `**kwargs` ใน `forward` ของ `SPOTERTransformerDecoderLayer` | PyTorch 2.x อ่าน `self_attn.batch_first` และส่ง `tgt_is_causal`/`memory_is_causal` (โค้ดเดิมเขียนสำหรับ torch 1.8) ไม่เปลี่ยนสถาปัตยกรรม |
| 3.3 | `spoter/utils.py` (`evaluate`) | `stats = {i: [0, 0] for i in range(101)}` → `defaultdict(lambda: [0, 0])` | เดิมรองรับแค่ ~100 classes (WLASL100) → `KeyError: 172` |
| 3.4 | `datasets/czech_slr_dataset.py` | ลบ `- 1` ออกจาก label ใน `__getitem__` | loader เดิมคาดว่า label เริ่มที่ 1; CSV ของเราเริ่มที่ 0 ทำให้มี label `-1` (DirectML ไม่ฟ้อง error) |
| 3.5 | `spoter/utils.py` (`train_epoch`) | เขียนใหม่ให้รองรับ **gradient accumulation** ผ่าน `$env:ACCUM_STEPS` (default 1) | เทียบเท่า batch 16 ตาม paper; ถ้าไม่ทำโมเดลไม่เรียนรู้ (ข้อ 6.1) |
| 3.6 | `train.py` (เลือก device) | `$env:FORCE_CPU="1"` → CPU, ไม่งั้นลอง DirectML แล้วค่อย CUDA/CPU | CPU เร็วกว่า DirectML สำหรับ SPOTER |
| 3.7 | `train.py` (สร้าง train set) | `augmentations=True/False` สลับตามการทดลอง | ใช้ทดสอบหาสาเหตุ (ปัจจุบันตั้งเป็น False ต้องเปิดกลับก่อนเทรนจริง) |

### ไฟล์ที่เพิ่มในโฟลเดอร์ `spoter/`

- `dml_adam.py` — Adam สูตรเดียวกับ `torch.optim.Adam` แต่ไม่ใช้ `lerp` (DirectML ไม่รองรับ) รัน `python dml_adam.py` เพื่อทดสอบว่าผลเท่ากัน
- `diag.py` — วินิจฉัย: `python diag.py data` (ตรวจข้อมูลที่เข้าโมเดล) และ `python diag.py overfit --device cpu|dml --opt torch|dml`
- `eval_ckpt.py` — **วัดผลบน test ด้วย checkpoint ที่เลือกจาก validation** + top-5, macro-F1, คู่ที่สับสนบ่อย, บันทึกผลรายคลิปเป็น CSV (ใช้กับผลจาก `train.py` เท่านั้น)
- `train_batched.py` — **สคริปต์เทรนหลักตอนนี้** เทรนแบบ batch จริง ไม่ต้องแก้ไฟล์ SPOTER เพิ่ม:
  - ใช้ loader / normalization / augmentation ของ `CzechSLRDataset` และโมเดล `SPOTER` เดิม
  - resample ทุกคลิปเป็น 64 เฟรม (เลือกเฟรมเว้นระยะเท่ากัน: คลิปสั้น = เฟรมซ้ำ, คลิปยาว = ข้ามเฟรม) แล้วรวม batch 16
  - `batched_forward()` ทำแบบเดียวกับ `SPOTER.forward` แต่รับ (B, T, 54, 2): h = (T, B, 108), class_query = (1, B, d)
  - `--check` ตรวจว่าผลเท่ากับโมเดลเดิมทีละคลิป (ผ่าน)
  - Adam 1e-4 (DirectML ใช้ `DmlAdam`), gaussian noise std 0.001 ตาม SPOTER, `--aug` เปิด augmentation
  - เลือก checkpoint จาก val แล้ววัด test ด้วย 64 เฟรมเหมือนตอนเทรน
  - ผลอยู่ที่ `out-batched/<exp>/`: `summary.txt`, `log.csv` (ใช้พล็อตกราฟ), `test_predictions.csv`, `best_val.pth`
  - checkpoint บันทึกเป็น tensor บน CPU (tensor บน DirectML โหลดกลับด้วย `weights_only=True` ไม่ได้)

### ⚠️ อย่าใช้ตัวเลขที่ SPOTER พิมพ์ตอนจบ

- `The top result was recorded at ...` = เลือก checkpoint ที่ **test สูงสุด** (ใช้ test เลือกโมเดล = ผิดหลัก)
- ตอนจบ SPOTER วัดแค่ `checkpoint_*_0` ซึ่งเป็นโมเดลช่วงต้น (เช่น try5_accum16: SPOTER รายงาน 3.1% แต่ checkpoint ที่ val ดีที่สุดได้ 42.79%)
- **ให้ใช้ `python eval_ckpt.py --exp <ชื่อ experiment>` เท่านั้น** (เลือก `checkpoint_v_*` ล่าสุด = val ดีที่สุด)
- checkpoint อยู่ที่ `spoter/out-checkpoints/<experiment_name>/`

---

## 4. ข้อมูล (ตรวจแล้ว)

### 4.1 ไฟล์ที่มี

- `TSL-ONE-S/` วิดีโอคลิปสั้นแยกต่อคำ (`.mp4`) เช่น `01_01_0001.mp4`
- `tsl_one_s_dataset_i3d.json` ไฟล์ split ระดับ gloss (`[{gloss, instances:[{video_id, split}]}]`)

### 4.2 สิ่งที่ตรวจพบจาก JSON

- 184 glosses, 4,152 คลิป, split = train/val/test = **3,327 / 468 / 357** ตรงกับ paper (ชื่อ split ใน JSON คือ `train`/`val`/`test`)
- แต่ละ gloss มี 16–27 คลิป; signer ส่วนใหญ่เซ็นครบ 184 คำ (บางคนไม่ครบ เช่น signer 27 มี 12 คลิป)

### 4.3 รูปแบบชื่อไฟล์

ชื่อ `AA_BB_CCCC` = `signer_ฟิลด์กลาง_เลขท้าย`

- **ฟิลด์กลางและเลขท้ายไม่ใช่หมวดหมู่ 1–8 หรือ gloss ID ใน Table 9 ของ paper**
- แต่ละ gloss มีคู่ (ฟิลด์กลาง, เลขท้าย) ของตัวเองคู่เดียว เช่น `01_01_0001` = Country, `01_04_0134` = Wat Pho
- **ต้องใช้ JSON แมป video_id → gloss เสมอ ห้ามใช้เลขท้ายเป็น label**
- label 0–183 เรียงตาม (ฟิลด์กลาง, เลขท้าย) บันทึกที่ `data/label_map.json` (ใช้ไฟล์เดียวกันทุกการทดลอง)

### 4.4 ข้อสังเกตสำคัญ: split ใน JSON ไม่ใช่ signer-independent

paper ระบุว่า signer ใน test ไม่อยู่ใน train แต่ใน JSON นี้ **signer 28 จาก 29 คนอยู่ทั้งใน train และ test** ต้องระบุเป็นข้อสังเกตในรายงาน จึงเตรียม 2 split:

| Split | วิธีแบ่ง | จำนวน (train/val/test) | ใช้ทำอะไร |
|---|---|---|---|
| **A: `json`** | ตาม JSON เดิม | 3,327 / 468 / 357 | reproduce ตัวเลข ~80% ของ paper |
| **B: `signer`** | test = signer 03, 08, 22; val = 11, 25, 29; train = ที่เหลือ 23 คน | **3,317 / 419 / 416** (ได้จริง) | วัด generalization กับ signer ใหม่จริง |

ผลจริงของ split B ต่างจากแผน (3,321 / 415 / 416) เพราะเรื่อง 4 คลิปในข้อ 4.6 ซึ่งไม่กระทบ test

### 4.5 รูปแบบไฟล์ pose `.npy`

- อยู่ที่ **`data/pose_npy/`** (4,152 ไฟล์) — สคริปต์ใน `tools/` ชี้ไปที่ path นี้แล้ว
  - มี `data/landmarks/` อีกชุด (4,152 ไฟล์ รูปแบบเดียวกัน) — ใช้ `data/pose_npy/` ชุดเดียวตลอด ระบุในรายงานว่าใช้ชุดไหน
- shape `(จำนวนเฟรม, 150)` = เฟรม × **75 จุด** × (x, y) แบบ interleaved
- 75 จุด = **pose 33 + มือซ้าย 21 + มือขวา 21** (MediaPipe Holistic) ตรวจแล้วว่าจุดข้อมือของแต่ละมือตรงกับ pose จุดที่ 15 (ซ้าย) และ 16 (ขวา)
- พิกัด normalize ตามขนาดเฟรม; ค่าเกิน 1 ได้ (จุดอยู่นอกภาพ); ค่าศูนย์ = จับไม่ได้ในเฟรมนั้น
- paper ใช้ **54 จุด** ต่อเฟรม (108 มิติ): หัว 5 + ลำตัวบน 7 (neck = กึ่งกลางไหล่) + มือละ 21 → `make_csv.py` เลือกจากดัชนี pose: nose 0, ตา 2/5, หู 7/8, ไหล่ 11/12, ศอก 13/14, ข้อมือ 15/16

### 4.6 ชื่อไฟล์ไม่ตรงกับ JSON 4 คลิป (ต้องตรวจยืนยัน)

- video_id ใน JSON ที่ไม่มีไฟล์ชื่อตรงกัน: `16_01_0028`, `15_08_0267`, `01_08_0266`, `16_01_0020` (signer 16/15/01 = train)
- ใน `data/pose_npy/` มี `.npy` อีก 4 ไฟล์ที่ชื่อไม่อยู่ใน JSON (หาด้วย `tools/diff_ids.py`)
- แก้ให้ตรงกันแล้ว → ไฟล์ขาด 0 แต่ **4 คลิปไปอยู่ใน val** (ชื่อไฟล์ขึ้นต้นด้วย signer ของ val)
- **TODO**: เปิดวิดีโอ 4 คลิปนี้ ยืนยันว่า (1) ผู้เซ็นเป็นคนตาม signer ID ในชื่อไฟล์จริง และ (2) ท่าตรงกับ gloss ที่ใส่ใน JSON ถ้าชื่อไฟล์พิมพ์ผิดให้แก้ชื่อ แล้วคลิปจะกลับไปอยู่ใน train
- ตรวจคลิปซ้ำ (signer เดียวกัน gloss เดียวกันเกิน 1 คลิป) ด้วยสคริปต์สั้นใน `tools/` (ต้องได้ `{}`)
- แก้ใน JSON ฉบับสำเนา ไม่แก้ไฟล์ต้นฉบับ

---

## 5. โครงสร้างโปรเจกต์

```
Sign2Say/   (หรือ project2/)
├─ spoter/                         ← repo ที่โคลนมา (แก้ตามข้อ 3)
│   ├─ train.py
│   ├─ train_batched.py                     ← สคริปต์เทรนหลัก (batch จริง)
│   ├─ dml_adam.py, diag.py, eval_ckpt.py   ← ไฟล์ที่เพิ่ม
│   ├─ spoter/  (spoter_model.py, utils.py)
│   ├─ datasets/czech_slr_dataset.py
│   ├─ out-checkpoints/<experiment>/       ← ผลจาก train.py
│   └─ out-batched/<experiment>/           ← ผลจาก train_batched.py
├─ TSL-ONE-S/                      ← วิดีโอ .mp4
├─ tsl_one_s_dataset_i3d.json      ← split JSON
├─ tools/
│   ├─ extract_pose.py             ← สกัด pose จาก mp4 -> data/pose_npy/*.npy
│   ├─ make_csv.py                 ← .npy + JSON -> CSV สำหรับ SPOTER
│   ├─ check_npy.py                ← นับ .npy เทียบ JSON แยกตาม split/signer
│   └─ diff_ids.py                 ← หา video_id ที่ JSON กับไฟล์ไม่ตรงกัน
└─ data/
    ├─ pose_npy/                   ← ผลจาก extract_pose.py (4,152 ไฟล์)
    ├─ landmarks/                  ← pose อีกชุด (ไม่ได้ใช้)
    ├─ label_map.json
    ├─ split_json/   (tsl_train.csv, tsl_val.csv, tsl_test.csv)
    └─ split_signer/ (tsl_train.csv, tsl_val.csv, tsl_test.csv)
```
venv ของเครื่อง AMD อยู่นอกโปรเจกต์ที่ `C:\venvs\sign2say`

### สคริปต์ใน `tools/`

**`extract_pose.py`** — MediaPipe Holistic (model_complexity=1, confidence 0.5) สกัด 75 จุด/เฟรม บันทึก `data/pose_npy/<video_id>.npy` ขนาด `(เฟรม, 150)`; multiprocessing (≤6 workers), `--limit N` สำหรับทดสอบ, ข้ามไฟล์ที่ทำแล้ว, เขียนไฟล์ชั่วคราวก่อน rename

**`make_csv.py`** — อ่าน JSON + `.npy` แล้วสร้าง CSV 3 ไฟล์ ตามคอลัมน์ใน Table 10 ของ paper; `--split-mode json` หรือ `signer`; รายงานจำนวนคลิป/class ต่อ split และไฟล์ที่ขาด

---

## 6. ข้อค้นพบระหว่างทาง (ใช้ในรายงานได้)

### 6.1 batch 1 + Adam 1e-4 → โมเดลไม่เรียนรู้เลย

- 5 epoch (ทั้งมีและไม่มี augmentation, ทั้ง DirectML และ CPU): loss ค้างที่ ~5.26 (= ln 184 ≈ 5.21), acc ~0.5% (= เดาสุ่ม 1/184), ทายคลาสเดียวเกือบทุกคลิป
- ตัดสาเหตุอื่นออกด้วย `diag.py`:
  - ข้อมูลปกติ: ไม่มี NaN, ไม่มีคลิปที่ค่าคงที่, จุดหาย 4.7%, ช่วงค่า [-0.53, 1.21]
  - overfit 32 คลิปได้ 100% ทุก device/optimizer → โมเดล/optimizer ทำงานถูก
- สาเหตุ: Adam ปรับ weight ขนาดใกล้เคียงกันทุก step แม้ gradient จาก 1 คลิปของ 184 คลาสจะเป็น noise → ติดที่ "ทายคลาสเดียว"
  (paper ใช้ Adam คู่ batch 16; repo ต้นฉบับใช้ SGD คู่ batch 1 — ผสม Adam กับ batch 1 จึงไม่เวิร์ก)
- **แก้ด้วย gradient accumulation 16 step** → epoch 5 val 42.0% (ดูข้อ 8)

### 6.2 SPOTER รายงานผล test ผิด checkpoint (ข้อ 3)

### 6.3 ความเร็ว: ทีละคลิป CPU ชนะ, batch จริง GPU ชนะ (ข้อ 2.2)

batch จริงเร็วกว่าทีละคลิป ~3 เท่า แม้บน CPU

### 6.4 SPOTER ต้นฉบับเขียนสำหรับ WLASL100 / torch 1.8

ต้องแก้หลายจุดก่อนใช้กับ 184 classes และ PyTorch 2.x (ข้อ 3.2–3.4)

### 6.5 SPOTER ที่ implement จริงไม่รู้ลำดับเวลาของเฟรม (สำคัญสำหรับการวิเคราะห์)

ใน `spoter/spoter_model.py`:
```python
self.row_embed = nn.Parameter(torch.rand(50, hidden_dim))
self.pos = nn.Parameter(torch.cat([self.row_embed[0].unsqueeze(0).repeat(1, 1, 1)], dim=-1).flatten(0, 1).unsqueeze(0))
```
- `self.pos` มีขนาด `(1, 1, d)` = เวกเตอร์เดียวบวกให้**ทุกเฟรมเท่ากัน** → ไม่ได้บอกตำแหน่งเวลา
- `row_embed` ไม่ถูกใช้ใน `forward` เลย (ไม่เคยถูกอัปเดต)
- ผล: encoder self-attention มองเฟรมเป็น "เซต" ไม่มีลำดับ — สลับลำดับเฟรมแล้วผลทายเท่าเดิม → แยกท่าที่รูปมือเหมือนกันแต่ทิศการเคลื่อนไหวต่างกันไม่ได้ (เช่น ย่า/ยาย)
- **ต่างจากที่ paper TSL-ONE-S บรรยาย** (paper บอกว่าใช้ sinusoidal positional encoding)
- ต่อยอดได้: เมื่อทุกคลิปยาว 64 เฟรมแล้ว เพิ่ม positional encoding จริง (1 เวกเตอร์ต่อตำแหน่ง) ได้ง่ายใน `train_batched.py` → เทียบกับ baseline

---

## 7. ขั้นตอนต่อไป

> รันจาก `spoter/` โดยเปิด venv แล้ว เสียบสายชาร์จและตั้งไม่ให้เครื่อง sleep

### ขั้นที่ 1: ตรวจ 4 คลิปในข้อ 4.6 (ไม่บังคับก่อนเทรน แต่ควรทำก่อนรายงาน)

### ขั้นที่ 2: เทรนจริง split B 80 epoch ด้วย `train_batched.py` (~1 ชม. 40 นาที บน GPU)

```powershell
python train_batched.py --exp b16_signer_aug_80 --device dml --epochs 80 --aug
```
- เครื่อง NVIDIA ใช้ `--device cuda`; ถ้าไม่มี GPU ใช้ `--device cpu` (~2 ชม.)
- augmentation คำนวณบน CPU ทีละคลิป อาจทำให้ช้ากว่า 75 วิ/epoch เล็กน้อย
- ผลอยู่ใน `out-batched\b16_signer_aug_80\summary.txt` (เลือก checkpoint จาก val แล้ว — ใช้ตัวเลขนี้ในรายงาน)
- อย่ารันพร้อมกับการเทรนอื่น (แย่ง CPU)

### ขั้นที่ 3: เปรียบเทียบไม่มี augmentation

```powershell
python train_batched.py --exp b16_signer_noaug_80 --device dml --epochs 80
```
→ วิเคราะห์ว่า augmentation ช่วยไหม

### ขั้นที่ 4: split A (json)

```powershell
cd ..
python tools/make_csv.py --split-mode json      # ต้องได้ ~3,327 / 468 / 357, 184 classes, ขาด 0
cd spoter
python train_batched.py --exp b16_json_aug_80 --device dml --epochs 80 --aug `
  --train_csv ..\data\split_json\tsl_train.csv `
  --val_csv ..\data\split_json\tsl_val.csv `
  --test_csv ..\data\split_json\tsl_test.csv
```

### ขั้นที่ 4.5: หลาย seed (ถ้ามีเวลา)

รันขั้นที่ 2 ซ้ำด้วย `--seed 1`, `--seed 2` และ `--exp` คนละชื่อ → รายงาน mean ± std

### (ทางเลือก) `train.py` แบบทีละคลิป + accumulation

ยังใช้ได้ (ข้อ 3.5) แต่ช้ากว่า ~3 เท่า: `$env:FORCE_CPU="1"; $env:ACCUM_STEPS="16"` แล้ว `python train.py ...` และวัดผลด้วย `eval_ckpt.py --exp <ชื่อ>` (ต้องเปิด `augmentations=True` ใน `train.py` เอง)

### ขั้นที่ 5: ประเมินผลและวิเคราะห์

- top-1, top-5, **macro-F1**, คู่ที่สับสนบ่อยสุด (`eval_ckpt.py` ให้ครบ + CSV ผลรายคลิป)
- เทียบกับ paper (SPOTER: 80.83% / F1 0.78) และอธิบายความต่าง (split ต่างกัน, seed, CPU vs GPU, test เล็ก ~1 คลิป ≈ 0.25%)
- คู่ที่ paper ระบุว่าสับสนง่าย: ปู่/ย่า–ตา/ยาย, หมู/หมูป่า, judo/wrestling, Phor Phaan / Sor Soo
  (ใน 5 epoch แรกเริ่มเห็น Beef → Wild boar meat)
- เทียบ split A กับ split B → ความต่างสะท้อน generalization ต่อ signer ใหม่

### ขั้นที่ 6: เขียนรายงาน/สไลด์

โครง: ที่มา → ข้อมูล → โมเดล/การตั้งค่า → ผล → error analysis → ข้อจำกัด → สรุป

### ขั้นที่ 7: งานวันอื่น (ลำดับตามความคุ้ม)

1. **SPOTER + positional encoding จริง** (แก้ข้อ 6.5) เทียบกับ baseline — เป็นข้อค้นพบที่ชัดถ้าผลดีขึ้น
1. Pose-LSTM หรือ 1D-CNN (เบา เป็น lower bound)
2. RGB model ย่อ (R3D-18 / R(2+1)D) หรือ I3D ถ้ามีเวลา/GPU
3. Late fusion (pose + RGB) เฉลี่ย logits
4. รันหลาย seed (เช่น 3) รายงาน mean ± std และทดสอบนัยสำคัญ (bootstrap CI / McNemar)
5. แยกผลตาม signer / หมวดคำ / ท่ามือเดียว vs สองมือ (paper ไม่ได้ทำ)

---

## 8. ตารางบันทึกผลการทดลอง

ตัวเลข "ของเรา" วัดด้วย checkpoint ที่ val ดีที่สุด (`eval_ckpt.py` สำหรับ `train.py`, `summary.txt` สำหรับ `train_batched.py`) เว้นแต่ระบุ

**ทีละคลิป** = `train.py` ใช้ความยาวคลิปเดิม; **b16×64** = `train_batched.py` batch 16 × 64 เฟรม — คนละการทดลอง เทียบกันตรง ๆ ต้องระบุ

| การทดลอง | Split | Epochs | Optimizer / batch | Aug | Device | Val | Top-1 (test) | Top-5 | Macro-F1 | เวลา | หมายเหตุ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Paper: SPOTER | signer-indep. (ตาม paper) | 80 | Adam 1e-4 / 16 | ✓ | RTX 2070 | – | 80.83% | – | 0.78 | ~1:48 ชม. | อ้างอิง |
| Paper: I3D | signer-indep. (ตาม paper) | 80 | Adam 1e-4 / 16 | – | RTX A6000 | – | 79.69% | – | 0.79 | ~21:41 ชม. | อ้างอิง |
| try5_signer | B | 5 | Adam 1e-4 / 1 | ✓ | DirectML | 0.7% | ~0.7%* | – | – | ~7.6 นาที/epoch | ไม่เรียนรู้ |
| try5_noaug | B | 5 | Adam 1e-4 / 1 | ✗ | CPU | 0.7% | 0.48%* | – | – | – | ไม่เรียนรู้ |
| **try5_accum16** | B | 5 | Adam 1e-4 / **16 (accum)** | ✗ | CPU | 42.0% | **42.79%** | 76.68% | 0.363 | ~20 นาที | เรียนรู้ได้, ยังไม่อิ่มตัว |
| speed_cpu | B | 1 | Adam 1e-4 / b16×64 | ✗ | CPU | 3.82% | 4.33% | 9.86% | 0.011 | 96 วิ | วัดความเร็ว |
| speed_dml | B | 1 | Adam(Dml) 1e-4 / b16×64 | ✗ | DirectML | 2.63% | – | – | – | 75 วิ | วัดความเร็ว (โหลด checkpoint error → แก้แล้ว) |
| **b16_signer_aug_80** | B | 80 | Adam(Dml) 1e-4 / b16×64 | ✓ | DirectML | | | | | | ⏳ baseline หลัก |
| b16_signer_noaug_80 | B | 80 | Adam(Dml) 1e-4 / b16×64 | ✗ | DirectML | | | | | | ⏳ |
| b16_json_aug_80 | A | 80 | Adam(Dml) 1e-4 / b16×64 | ✓ | DirectML | | | | | | ⏳ |

\* ตัวเลขจากที่ SPOTER พิมพ์เอง (checkpoint ช่วงต้น) — ใช้แค่ยืนยันว่าไม่เรียนรู้

---

## 9. ข้อควรระวังและข้อจำกัดที่ต้องบันทึกในรายงาน

- baseline หลักใช้ `train_batched.py`: **resample ทุกคลิปเป็น 64 เฟรม** (เลือกเฟรมเว้นระยะเท่ากัน) แล้ว batch 16 ตาม paper (paper บอกแค่ "interpolation or padding" ไม่ระบุวิธี)
- รอบที่ใช้ `train.py` ใช้ **gradient accumulation 16 step แทน batch 16** (โค้ด SPOTER รับทีละคลิป) เทียบเท่าการเฉลี่ย gradient ของ 16 คลิป โดยไม่ resample
- SPOTER ที่ใช้ไม่มี positional encoding ตามเวลา (ข้อ 6.5) ต่างจากที่ paper บรรยาย — ใช้โค้ดต้นฉบับตามเดิมเพื่อเป็น baseline
- แก้โค้ด SPOTER หลายจุด (ข้อ 3) เพื่อให้รันบน PyTorch 2.x / 184 classes โดยไม่เปลี่ยนสถาปัตยกรรม
- เครื่อง AMD เทรนบน DirectML ด้วย `DmlAdam` (สูตรเท่ากับ `torch.optim.Adam` ทดสอบแล้ว) หรือ CPU ด้วย `torch.optim.Adam`
- split ใน JSON ที่ได้มาไม่แบ่งตาม signer (ข้อ 4.4) จึงทำ split B เพิ่มเอง ผล split B **เทียบกับ 80.83% ของ paper ตรง ๆ ไม่ได้**
- 4 คลิปชื่อไฟล์ไม่ตรงกับ JSON (ข้อ 4.6) ต้องระบุวิธีแก้ในรายงาน
- pose สกัดเองด้วย MediaPipe 0.10.14 อาจต่างจากไฟล์ของ dataset เล็กน้อย (paper มีขั้น crop ตัวคนก่อน) SPOTER normalize ตามขนาดศีรษะอยู่แล้ว ลดผลกระทบส่วนนี้ได้
- test set เล็ก (357–416 คลิป, ~2 คลิป/class) ผลต่างเล็ก ๆ อาจไม่มีนัยสำคัญ; ถ้ารันได้แค่ seed เดียวให้ระบุเป็นข้อจำกัด
- ทุก signer ถนัดขวา และ 25 จาก 29 คนเป็นครู ข้อจำกัดด้านการ generalize ตามที่ paper ระบุ
- ข้อมูลเป็นใบอนุญาต CC-NC ใช้เพื่อการเรียนได้ ใบหน้าผู้เซ็นเป็นข้อมูลส่วนบุคคล ไม่ควรเผยแพร่คลิปต่อ
- ใช้ `label_map.json` เดียวกันตลอดทุกการทดลอง และจด config/seed/device ทุกครั้ง

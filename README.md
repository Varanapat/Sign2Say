# โปรเจกต์ Sign Language Recognition (SPOTER บน TSL-ONE-S)

วิชา Deep Learning in Medical Image and Video Analysis
หัวข้อ: Video analysis ภาษามือ (Thai Sign Language) งานของเรา: ทำ **baseline** และวิเคราะห์ว่าแบบไหนดี
Paper อ้างอิง: *TSL-ONE-S: A Real-World Thai Sign Language Dataset With Deep Learning Benchmarks* (IEEE Access, 2026)

> ไฟล์นี้บันทึกทุกอย่างที่ทำไปแล้ว ผลที่ตรวจสอบแล้ว ปัญหาที่เจอ และขั้นตอนที่ต้องรันต่อ

---

## 1. เป้าหมายและการตัดสินใจ

- ทำ baseline และเปรียบเทียบ ตามที่ paper ทำกับ 2 โมเดล:
  - **I3D** (RGB, Kinetics-400 pretrained): 79.69% top-1, F1 0.79, 12.35M params, เทรน ~21 ชม. 41 นาที (RTX A6000)
  - **SPOTER** (pose + Transformer): 80.83% top-1, F1 0.78, 5.92M params, เทรน ~1 ชม. 48 นาที (RTX 2070)
- **ตกลงกับเพื่อนว่าเลือกทำ SPOTER ก่อน** เพราะเทรนเร็ว ใช้ข้อมูล pose ได้ และมี repo ทางการ
- แผนวันนี้: ทำ SPOTER ให้เสร็จ 1 ตัวให้ได้แน่ ๆ ส่วน Pose-LSTM, RGB model, fusion และหลาย seed ทำวันอื่น
- ค่าตาม paper ที่ใช้เป็นเป้าหมาย reproduce: **80 epochs, batch size 16, learning rate 1e-4, Adam**, เลือก checkpoint จาก validation

---

## 2. สภาพแวดล้อม (ทำแล้ว)

- ระบบ: Windows (PowerShell), GPU **NVIDIA GeForce RTX 4060 Laptop**
- โฟลเดอร์โปรเจกต์: `C:\Users\khims\OneDrive\Desktop\3_1\DLM\project2`
- สร้าง virtual environment และเปิดใช้:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  (ถ้าติด execution policy: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)
- ติดตั้ง PyTorch แบบ CUDA: ได้ `torch 2.5.1+cu121`, ตรวจแล้ว `torch.cuda.is_available() = True`
- ติดตั้งแพ็กเกจอื่น (ไม่ล็อกเวอร์ชัน):
  ```powershell
  pip install pandas numpy matplotlib scikit-learn tqdm opencv-python
  pip install mediapipe==0.10.14
  ```
- **ข้าม `requirements.txt` ของ SPOTER** เพราะล็อก `pandas==1.1.5` (สร้างไม่ได้บน Python รุ่นใหม่ ขึ้น `No module named 'pkg_resources'`) และ `torch==1.8.1` (เรามี 2.5.1 แล้ว)

> ทุกครั้งที่เปิดเทอร์มินัลใหม่ ให้เปิด venv ก่อน (`.\venv\Scripts\Activate.ps1` จากโฟลเดอร์ `project2`)

---

## 3. SPOTER repo (ทำแล้ว)

- โคลน: `git clone https://github.com/maty-bohacek/spoter` (อยู่ใน `project2/spoter/`)
- `python train.py --help` ขึ้นข้อความ `unrecognized arguments: --help` แต่แสดงรายการ argument ที่ใช้ได้ (เป็นความแปลกของ parser ใน repo ไม่ใช่ปัญหา)
- Argument ที่มี: `--experiment_name`, `--num_classes`, `--hidden_dim`, `--seed`, `--training_set_path`, `--testing_set_path`, `--validation_set {from-file,split-from-train,none}`, `--validation_set_path`, `--epochs`, `--lr`, `--log_freq`, `--save_checkpoints`, `--scheduler_factor`, `--scheduler_patience`, `--gaussian_mean`, `--gaussian_std`, `--plot_stats`, `--plot_lr`
- ตรวจ `train.py` ด้วย `Select-String` พบ:
  - บรรทัด 114 ใช้ **SGD** (paper ใช้ Adam)
  - ไม่ได้ตั้ง `batch_size` ใน `DataLoader` จึงเป็น **batch size 1** (paper ระบุ 16)

### การแก้ไข `spoter/train.py` (ต้องทำ)

บรรทัด 114 เปลี่ยนจาก SGD เป็น Adam (ปล่อยชื่อตัวแปร `sgd_optimizer` ไว้เหมือนเดิม จะได้ไม่ต้องแก้บรรทัดอื่น):

```python
sgd_optimizer = optim.Adam(slrt_model.parameters(), lr=args.lr)
```

**ข้อแตกต่างจาก paper ที่ต้องรายงานตามจริง:** batch size คง 1 ตามโค้ดต้นฉบับ (โค้ดของ SPOTER น่าจะเขียนโดยสมมติ batch = 1 จึงไม่เปลี่ยนเป็น 16 เฉย ๆ) ถ้ามีเวลาค่อยแก้ให้รองรับ batch 16

---

## 4. ข้อมูล (ตรวจแล้ว)

### 4.1 ไฟล์ที่มี

- `TSL-ONE-S/` เป็นวิดีโอคลิปสั้นแยกต่อคำ (`.mp4`) เช่น `01_01_0001.mp4`, `01_04_0133.mp4`
- `tsl_one_s_dataset_i3d.json` เป็นไฟล์ split ระดับ gloss (รูปแบบ: `[{gloss, instances:[{video_id, split}]}]`)

### 4.2 สิ่งที่ตรวจพบจาก JSON

- 184 glosses, 4,152 คลิป (video_id ไม่ซ้ำ), split = train/val/test = **3,327 / 468 / 357** ตรงกับ paper
- แต่ละ gloss มี 16–27 คลิป; signer ส่วนใหญ่เซ็นครบ 184 คำ (บางคนไม่ครบ เช่น signer 27 มี 12 คลิป)

### 4.3 รูปแบบชื่อไฟล์ (ข้อสรุปที่ถูกต้อง)

ชื่อ `AA_BB_CCCC` = `signer_ฟิลด์กลาง_เลขท้าย`

- **ฟิลด์กลางและเลขท้ายไม่ใช่หมวดหมู่ 1–8 หรือ gloss ID ใน Table 9 ของ paper** (ฟิลด์กลางมีค่า 01, 04, 06, 07, 08, 13, 25, 28; เลขท้าย 1–989)
- แต่ละ gloss มีคู่ (ฟิลด์กลาง, เลขท้าย) ของตัวเองคู่เดียว ตัวอย่างที่ตรวจแล้ว: `01_01_0001` = Country, `0002` = Thailand, `0003` = Laos, `0028` = Israel, `01_04_0133` = Wat Phra Kaew, `0134` = Wat Pho
- **จึงต้องใช้ JSON แมป video_id → gloss เสมอ ห้ามใช้เลขท้ายเป็น label**
- label ที่สร้างเอง: เรียง gloss ตาม (ฟิลด์กลาง, เลขท้าย) ให้เป็น 0–183 และบันทึกไว้ที่ `data/label_map.json`

> บันทึกความผิดพลาดระหว่างทาง: ตอนแรกเข้าใจผิดว่าเลขท้ายคือ gloss ID จึงตรวจแก้แล้ว สคริปต์ปัจจุบันใช้ JSON ถูกต้อง

### 4.4 ข้อสังเกตสำคัญ: split ใน JSON ไม่ใช่ signer-independent

paper ระบุว่า signer ใน test ไม่อยู่ใน train แต่ใน JSON นี้ **signer 28 จาก 29 คนอยู่ทั้งใน train และ test** (ทุกคนอยู่ใน val ด้วย) เหมือนแบ่งแบบสุ่มตามคลิป ยังไม่ทราบสาเหตุ (ชื่อไฟล์ `_i3d` อาจต่างจาก split ที่ใช้เทรน SPOTER จริง หรือเป็นความคลาดเคลื่อนใน paper) ต้องระบุเป็นข้อสังเกตในรายงาน

จึงเตรียม 2 split:

| Split | วิธีแบ่ง | ใช้ทำอะไร |
|---|---|---|
| **A: `json`** | ตาม JSON เดิม (3,327/468/357) | reproduce ตัวเลข ~80% ของ paper |
| **B: `signer`** | แบ่งตาม signer: test = 03, 08, 22 (416 คลิป); val = 11, 25, 29 (415 คลิป); train = ที่เหลือ 23 คน (3,321 คลิป) | วัด generalization กับ signer ใหม่จริง (ทุก gloss มีใน train ครบ) |

### 4.5 รูปแบบไฟล์ pose `.npy` (วิเคราะห์จากไฟล์ตัวอย่าง `01_01_0001`, `01_01_0002`)

- shape `(จำนวนเฟรม, 150)` = เฟรม × **75 จุด** × (x, y)
- 75 จุด = **pose 33 + มือซ้าย 21 + มือขวา 21** (MediaPipe Holistic) ตรวจแล้วว่าจุดข้อมือของแต่ละมือตรงกับ pose จุดที่ 15 (ซ้าย) และ 16 (ขวา)
- พิกัด normalize ตามขนาดเฟรม; ค่า y เกิน 1 ได้ (จุดช่วงล่างลำตัวอยู่นอกภาพ); ค่าศูนย์ = จับมือไม่ได้ในเฟรมนั้น
- paper ใช้ **54 จุด** ต่อเฟรม (108 มิติ): หัว 5 (nose, leftEye, rightEye, leftEar, rightEar) + ลำตัวบน 7 (neck, shoulders, elbows, wrists) + มือละ 21 → ใน `make_csv.py` เลือกจากดัชนี pose: nose 0, ตา 2/5, หู 7/8, ไหล่ 11/12, ศอก 13/14, ข้อมือ 15/16 และ **neck = กึ่งกลางไหล่** (MediaPipe ไม่มีจุดคอ)

---

## 5. โครงสร้างโปรเจกต์

```
project2/
├─ spoter/                         ← repo ที่โคลนมา (แก้ train.py บรรทัด 114)
├─ TSL-ONE-S/                      ← วิดีโอ .mp4
├─ venv/
├─ tsl_one_s_dataset_i3d.json      ← split JSON
├─ tools/
│   ├─ extract_pose.py             ← สกัด pose จาก mp4 -> .npy
│   └─ make_csv.py                 ← .npy + JSON -> CSV สำหรับ SPOTER
├─ pose_npy/                       ← ผลจาก extract_pose.py (สร้างอัตโนมัติ)
└─ data/                           ← ผลจาก make_csv.py (สร้างอัตโนมัติ)
    ├─ label_map.json
    ├─ split_json/   (tsl_train.csv, tsl_val.csv, tsl_test.csv)
    └─ split_signer/ (tsl_train.csv, tsl_val.csv, tsl_test.csv)
```

### สคริปต์ที่สร้าง (ทำแล้ว)

**`tools/extract_pose.py`** — อ่านทุกคลิปใน `TSL-ONE-S/`, ใช้ MediaPipe Holistic (model_complexity=1, confidence 0.5) สกัด 75 จุด/เฟรม, บันทึก `pose_npy/<video_id>.npy` ขนาด `(เฟรม, 150)`
- รัน multiprocessing (ค่า default ไม่เกิน 6 workers), `--limit N` สำหรับทดสอบ
- **ข้ามไฟล์ที่ทำแล้ว** จึงรันซ้ำต่อได้ถ้าถูกปิดกลางคัน; เขียนไฟล์ชั่วคราวก่อนแล้วค่อย rename กันไฟล์เสีย
- ต้องมี `if __name__ == "__main__":` (จำเป็นบน Windows)

**`tools/make_csv.py`** — อ่าน JSON + `.npy` แล้วสร้าง CSV 3 ไฟล์ (train/val/test) ตามชื่อคอลัมน์ใน Table 10 ของ paper (`nose_X`, `leftEye_Y`, `thumbCMC_left_X`, ... และ `video_id`, `labels`, `label_name`, `split`, `video_width`, `video_height`, `fps`, `length`) โดยแต่ละเซลล์เป็นลิสต์ค่าของทุกเฟรม
- `--split-mode json` (split A) หรือ `--split-mode signer` (split B)
- รายงานจำนวนคลิป/จำนวน class ต่อ split และรายการ `.npy` ที่ยังไม่มี

---

## 6. ผลที่ตรวจสอบแล้วจนถึงตอนนี้

- [x] GPU ใช้ได้กับ PyTorch (RTX 4060 Laptop, torch 2.5.1+cu121)
- [x] SPOTER โคลนแล้ว และ `train.py` โหลดโมดูลผ่าน (ติดตั้ง `matplotlib`, `opencv-python` แล้ว)
- [x] ตรวจ JSON แล้ว (นับจำนวน, ชื่อไฟล์, การทับซ้อนของ signer)
- [x] ทดสอบ `extract_pose.py --limit 5` สำเร็จ ได้ `.npy` 5 ไฟล์ใน `pose_npy/` (ข้อความ log ยาว ๆ จาก MediaPipe/protobuf เป็น warning ปกติ ไม่ใช่ error)
- [x] ตรวจ `01_01_0001`: `.npy` shape **(69, 150)**, วิดีโอ **69 เฟรม** ตรงกัน, ศูนย์ 8.5% (ไฟล์ตัวอย่างจาก dataset เดิมมี 6.5%; ต่างเล็กน้อยเพราะน่าจะต่างเวอร์ชัน/ขั้น crop)
- [ ] สกัด pose ทั้งชุด 4,152 คลิป
- [ ] สร้าง CSV และเช็กว่าเข้ากับ data loader ของ SPOTER
- [ ] เทรนจริงและประเมินผล

---

## 7. ขั้นตอนต่อไป: เริ่มรันทั้งหมด

> ทุกคำสั่งรันโดยเปิด venv แล้ว ใช้ PowerShell. เสียบสายชาร์จและตั้งไม่ให้เครื่อง sleep ก่อนเริ่มงานยาว

### ขั้นที่ 1: สกัด pose ทั้งชุด (ใช้เวลานานที่สุด เริ่มก่อนอย่างอื่น)

รันจากโฟลเดอร์ `project2`:

```powershell
$env:GLOG_minloglevel="2"      # ลดข้อความ log ที่รก
python tools/extract_pose.py
```

- จะพิมพ์ความคืบหน้าทุก 20 คลิป (`20/4152`, `40/4152`, ...) ให้จับเวลา 20 คลิปแล้วคูณ ~208 เพื่อประเมินเวลารวม (คาดว่าหลายชั่วโมง)
- ถ้าถูกปิดกลางคัน **รันคำสั่งเดิมซ้ำ** ได้ จะข้ามไฟล์ที่เสร็จแล้ว
- ถ้าเครื่องอืด/ซิงก์ช้า ลองพักการซิงก์ OneDrive ชั่วคราว (มีไฟล์เล็ก ๆ ถูกเขียน 4,152 ไฟล์)

### ขั้นที่ 2: ทดสอบ CSV + loader ของ SPOTER ระหว่างรอ (เปิดเทอร์มินัลใหม่)

เพื่อหา error ของ data loader ล่วงหน้า โดยไม่ต้องรอสกัดครบ:

```powershell
# จากโฟลเดอร์ project2 (เปิด venv)
python tools/make_csv.py --split-mode json
cd spoter
python train.py --experiment_name smoke --num_classes 184 `
  --training_set_path ..\data\split_json\tsl_train.csv `
  --validation_set from-file `
  --validation_set_path ..\data\split_json\tsl_train.csv `
  --testing_set_path ..\data\split_json\tsl_train.csv `
  --epochs 1 --lr 0.0001
```

- ตอนนี้มี `.npy` ไม่กี่ไฟล์ บาง split อาจเป็น 0 แถว (ปกติ) จึงใช้ `tsl_train.csv` เป็นทั้ง train/val/test เพื่อดูแค่ว่าโค้ดรันจบ ตัวเลข accuracy ไม่มีความหมาย
- ถ้าขึ้น error (เช่น `KeyError` ชื่อคอลัมน์ หรือ `Target out of bounds` ที่อาจต้องใช้ `--num_classes 185`) ให้เก็บข้อความ error ไว้แก้สคริปต์แปลงให้ตรงกับ loader
- เพื่อดูว่า loader ต้องการคอลัมน์อะไร:
  ```powershell
  Select-String -Path datasets\czech_slr_dataset.py -Pattern "video_size|labels|columns|literal_eval|_X|_Y"
  ```

### ขั้นที่ 3: สร้าง CSV จริงเมื่อสกัดครบ (จาก `project2`)

```powershell
python tools/make_csv.py --split-mode json
python tools/make_csv.py --split-mode signer
```

เช็กผล: split A ต้องได้ใกล้ **3,327 / 468 / 357** และ **184 classes**; split B ได้ประมาณ **3,321 / 415 / 416**; รายการ "ไม่มีไฟล์ .npy" ต้องเป็น 0

### ขั้นที่ 4: เทรนจริง (จากโฟลเดอร์ `spoter`)

ก่อนเทรนต้องแก้ `train.py` บรรทัด 114 เป็น Adam แล้ว (ข้อ 3)

**Split A (reproduce paper):**
```powershell
cd spoter
python train.py --experiment_name tsl_json --num_classes 184 `
  --training_set_path ..\data\split_json\tsl_train.csv `
  --validation_set from-file `
  --validation_set_path ..\data\split_json\tsl_val.csv `
  --testing_set_path ..\data\split_json\tsl_test.csv `
  --epochs 80 --lr 0.0001 --seed 42
```

**Split B (signer-independent) ถ้ามีเวลา:** ใช้คำสั่งเดียวกัน เปลี่ยน `split_json` เป็น `split_signer` และ `--experiment_name tsl_signer`

- ใช้ตัวเลือก `--epochs 80 --lr 0.0001` ตาม paper; ถ้าเวลาไม่พอให้ลด epoch และจดไว้เป็นข้อจำกัด
- เลือก checkpoint จาก validation แล้วรายงานผลบน test **ครั้งเดียว** ห้ามใช้ test เลือกไฮเปอร์พารามิเตอร์
- ตรวจดูให้แน่ใจว่าที่เก็บ checkpoint/ผลลัพธ์อยู่ที่ไหน และไม่อยู่ในโฟลเดอร์ที่ OneDrive ซิงก์หนักจนไฟล์ล็อก (ถ้ามีปัญหาแปลก ๆ ให้ย้ายโปรเจกต์ไปไว้นอก OneDrive เช่น `C:\dev\project2`)

### ขั้นที่ 5: ประเมินผลและวิเคราะห์

- คำนวณ top-1, **macro-F1**, confusion matrix และคู่ class ที่สับสนบ่อยสุด (ใช้ `data/label_map.json` แปลง id เป็นชื่อ gloss)
- เทียบกับ paper (SPOTER: 80.83% / F1 0.78 / 5.92M params) และอธิบายความต่างที่เป็นไปได้ (batch size 1 แทน 16, seed, จำนวน epoch, test เล็ก 357 คลิป ต่างกัน 1% ≈ 4 คลิป)
- คู่ที่ paper ระบุว่าสับสนง่าย ให้ตรวจเป็นกรณีศึกษา: ปู่/ย่า–ตา/ยาย, หมู/หมูป่า, judo/wrestling, อักษร พอ พาน (Phor Phaan) กับ ศอ โซ่ (Sor Soo)
- เทียบผล split A กับ split B และเขียนวิเคราะห์ว่าความต่างสะท้อน generalization ต่อ signer ใหม่อย่างไร

### ขั้นที่ 6: เขียนรายงาน/สไลด์

โครงสั้น: ที่มา → ข้อมูล → โมเดล/การตั้งค่า → ผล → error analysis → ข้อจำกัด → สรุป (ใส่ตารางผลด้านล่าง)

### ขั้นที่ 7: งานวันอื่น (ลำดับตามความคุ้ม)

1. Pose-LSTM หรือ 1D-CNN (เบา เป็น lower bound)
2. RGB model ย่อ (R3D-18 / R(2+1)D) หรือ I3D ถ้ามีเวลา/GPU
3. Late fusion (pose + RGB) เฉลี่ย logits
4. รันหลาย seed (เช่น 3) รายงาน mean ± std และทดสอบนัยสำคัญ (bootstrap CI / McNemar)
5. แก้ให้รองรับ batch size 16 ตาม paper

---

## 8. ตารางบันทึกผลการทดลอง (กรอกเมื่อรันเสร็จ)

| การทดลอง | Split | Epochs | Optimizer / batch | Top-1 (test) | Macro-F1 | เวลาเทรน | หมายเหตุ |
|---|---|---|---|---|---|---|---|
| Paper: SPOTER | signer-independent (ตาม paper) | 80 | Adam 1e-4 / 16 | 80.83% | 0.78 | ~1:48 ชม. (RTX 2070) | อ้างอิง |
| Paper: I3D | signer-independent (ตาม paper) | 80 | Adam 1e-4 / 16 | 79.69% | 0.79 | ~21:41 ชม. (RTX A6000) | อ้างอิง |
| ของเรา: SPOTER | A (json) | | Adam 1e-4 / 1 | | | | |
| ของเรา: SPOTER | B (signer) | | Adam 1e-4 / 1 | | | | |

---

## 9. ข้อควรระวังและข้อจำกัดที่ต้องบันทึกในรายงาน

- ใช้ **batch size 1** (ตามโค้ดต้นฉบับ) ต่างจาก paper ที่ระบุ 16
- split ใน JSON ที่ได้มาไม่แบ่งตาม signer (ดูข้อ 4.4) จึงทำ split B เพิ่มเอง
- pose สกัดเองด้วย MediaPipe 0.10.14 อาจต่างจากไฟล์ที่ dataset สร้างไว้เล็กน้อย (paper มีขั้นตอน crop ตัวคนก่อน) SPOTER normalize ตามขนาดศีรษะของแต่ละคนอยู่แล้ว ลดผลกระทบส่วนนี้ได้
- test set เล็ก (357 คลิป, ~2 คลิป/class) ผลต่างเล็ก ๆ อาจไม่มีนัยสำคัญ; ถ้ารันได้แค่ seed เดียวให้ระบุเป็นข้อจำกัด
- ทุก signer ถนัดขวา และ 25 จาก 29 คนเป็นครู ข้อจำกัดด้านการ generalize ตามที่ paper ระบุ
- ข้อมูลเป็นใบอนุญาต CC-NC ใช้เพื่อการเรียนได้ ใบหน้าผู้เซ็นเป็นข้อมูลส่วนบุคคล ไม่ควรเผยแพร่คลิปต่อ
- ใช้ `label_map.json` เดียวกันตลอดทุกการทดลอง เพื่อให้เทียบข้ามวันได้ และจด config/seed ทุกครั้ง
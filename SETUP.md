# คู่มือเริ่มใช้งาน baseline SPOTER บน TSL-ONE-S

คู่มือนี้สำหรับคนที่ clone repo นี้ไปแล้วอยากรัน baseline ต่อ ตั้งแต่ติดตั้งจนได้ผลบน test set
รายละเอียดเบื้องหลัง ข้อค้นพบ และตารางผลทั้งหมดอยู่ใน [`README.md`](README.md)

> คำสั่งทั้งหมดเขียนสำหรับ **Windows PowerShell** (บน macOS/Linux เปลี่ยน `\` เป็น `/` และใช้ `source .../bin/activate`)

---

## สรุปขั้นตอน

| # | ขั้นตอน | เวลาโดยประมาณ |
|---|---|---|
| 1 | clone repo และ checkout branch `base_line` | 1 นาที |
| 2 | สร้าง venv และติดตั้งแพ็กเกจ (เลือก PyTorch ตาม GPU) | 10 นาที |
| 3 | วางข้อมูล dataset (ไม่อยู่ใน repo) | – |
| 4 | ได้ไฟล์ pose `data/pose_npy/`: **4A โหลด zip** (แนะนำ) หรือ 4B สกัดเองด้วย MediaPipe | 4A: ไม่กี่นาที / 4B: หลายชั่วโมง |
| 5 | สร้าง CSV ตาม split | 2–5 นาที |
| 6 | ตรวจ pipeline (`--check`, เทรน 1 epoch) | 5 นาที |
| 7 | เทรน 80 epoch และอ่านผล | 1.5–2 ชั่วโมง |

---

## 1. Clone repo

```powershell
git clone https://github.com/Varanapat/Sign2Say.git
cd Sign2Say
git checkout base_line
```

**แนะนำ:** อย่าวาง venv ไว้ในโฟลเดอร์ที่ OneDrive / Google Drive ซิงก์ (ไฟล์ `.pyd` ของแพ็กเกจอาจเสีย) — ตัวโปรเจกต์อยู่ใน OneDrive ได้ แต่ venv ให้อยู่ข้างนอก

---

## 2. ติดตั้งสภาพแวดล้อม

ต้องมี **Python 3.12** (MediaPipe ยังไม่รองรับ 3.13 ขึ้นไป) ตรวจด้วย `python --version`

### 2.1 สร้าง venv นอกโฟลเดอร์ที่ซิงก์

```powershell
python -m venv C:\venvs\sign2say
C:\venvs\sign2say\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
ถ้าติด execution policy: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

> ทุกครั้งที่เปิด terminal ใหม่ ต้องรัน `C:\venvs\sign2say\Scripts\Activate.ps1` ก่อน
> ใน VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → เลือก `C:\venvs\sign2say\Scripts\python.exe`

### 2.2 ติดตั้ง PyTorch ตาม GPU ที่มี

ดูก่อนว่าเครื่องมีการ์ดจออะไร: `Get-CimInstance Win32_VideoController | Select-Object Name`

| GPU | คำสั่งติดตั้ง | ใช้ `--device` |
|---|---|---|
| **NVIDIA** (`nvidia-smi` ใช้ได้) | `pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121` | `cuda` |
| **AMD** (Windows, เช่น RX 6000) | `pip install torch-directml` | `dml` |
| ไม่มี GPU / ไม่แน่ใจ | `pip install torch torchvision` | `cpu` |

AMD RX 7000/9000 บน Windows 11 ใช้ ROCm PyTorch ของ AMD ได้ด้วย (`--device cuda`) แต่ baseline นี้ทดสอบกับ DirectML เท่านั้น

### 2.3 แพ็กเกจอื่น

```powershell
pip install pandas numpy matplotlib scikit-learn tqdm opencv-python mediapipe==0.10.14
```
- **อย่า**ติดตั้ง `opencv-python` คู่กับ `opencv-contrib-python` (ไฟล์ `cv2` ทับกัน)
- **ไม่ต้อง**ใช้ `spoter/requirements.txt` (ล็อกเวอร์ชันเก่า `torch==1.8.1`, `pandas==1.1.5` ที่ติดตั้งไม่ได้บน Python 3.12)
- ถ้า `import mediapipe` ขึ้น error เรื่อง numpy ให้ `pip install "numpy<2"`

### 2.4 ตรวจ

```powershell
python -c "import torch, cv2, mediapipe, pandas; print(torch.__version__, cv2.__version__, mediapipe.__version__)"
```
- NVIDIA: `python -c "import torch; print(torch.cuda.is_available())"` ต้องได้ `True`
- AMD: `python -c "import torch_directml; print(torch_directml.device_name(0))"` ต้องเห็นชื่อการ์ด

---

## 3. วางข้อมูล dataset

ข้อมูลในโฟลเดอร์ `data/` **ไม่อยู่ใน repo** (ยกเว้น `data/label_map.json`) เพราะวิดีโอเป็นใบอนุญาต CC BY-NC และมีใบหน้าผู้เซ็น ห้ามเผยแพร่ต่อ — ขอจากผู้จัดทำ dataset TSL-ONE-S หรือจากทีม

โครงสร้างที่ต้องได้หลังวางข้อมูลครบ (ขั้นที่ 3–5):

```
Sign2Say/
├─ data/
│   ├─ videos/                     ← วิดีโอ .mp4 (เช่น 01_01_0001.mp4) 4,152 คลิป   [ขั้นที่ 3]
│   ├─ pose_npy/                   ← pose .npy 4,152 ไฟล์                         [ขั้นที่ 4]
│   ├─ split_signer/               ← CSV สำหรับ SPOTER (สร้างเอง)                  [ขั้นที่ 5]
│   └─ label_map.json              ← มากับ repo
├─ spoter/                         ← โค้ด SPOTER ที่แก้แล้ว (มากับ repo)
├─ tools/                          ← สคริปต์เตรียมข้อมูล (มากับ repo)
└─ tsl_one_s_dataset_i3d.json      ← ไฟล์ split ของ dataset: [{gloss, instances:[{video_id, split}]}]
```

สิ่งที่ต้องวางเอง:
1. **วิดีโอ** → `data/videos/` (ถ้าใช้ทางเลือก 4A แค่เทรนโมเดล ไม่จำเป็นต้องมีวิดีโอ แต่ควรมีไว้ดูคลิปที่ทายผิด)
2. **`tsl_one_s_dataset_i3d.json`** → โฟลเดอร์หลัก `Sign2Say/` (ต้องมีเสมอ ใช้แมป video_id → gloss ในขั้นที่ 5)

ตรวจ:
```powershell
(Get-ChildItem data\videos -Filter *.mp4).Count      # ควรได้ 4,152
Test-Path tsl_one_s_dataset_i3d.json                 # ต้องได้ True
```

---

## 4. ได้ไฟล์ pose (`data/pose_npy/`) — เลือก 4A หรือ 4B

### 4A (แนะนำ): โหลดไฟล์ pose ที่สกัดไว้แล้ว

ข้ามการสกัดที่ใช้เวลาหลายชั่วโมง ได้ผลเหมือนกับที่ใช้ทำ baseline ทุกไฟล์

1. โหลด `pose_npy.zip` จาก: **Google drive** (ลิงก์จำกัดสิทธิ์เฉพาะทีม — ข้อมูลมาจาก dataset ใบอนุญาต CC BY-NC)
2. แตกไฟล์ไว้ที่ `data/`:
   ```powershell
   Expand-Archive pose_npy.zip -DestinationPath data\ -Force
   ```
3. ตรวจว่าไฟล์ `.npy` อยู่ที่ `data\pose_npy\` โดยตรง (ไม่ซ้อนเป็น `data\pose_npy\pose_npy\`) และครบ:
   ```powershell
   (Get-ChildItem data\pose_npy -Filter *.npy).Count     # ต้องได้ 4,152
   Get-ChildItem data\pose_npy -Filter *.npy | Select-Object -First 3 Name
   ```
   ถ้าซ้อนกัน 2 ชั้น: `Move-Item data\pose_npy\pose_npy\* data\pose_npy\` แล้วลบโฟลเดอร์ที่ว่าง

แล้วข้ามไปขั้นที่ 5 ได้เลย

> ไฟล์ใน zip สกัดด้วย MediaPipe Holistic 0.10.14 (model_complexity=1, confidence 0.5) — ถ้าสกัดเองด้วยเวอร์ชันอื่น ค่าอาจต่างเล็กน้อย ผลเทรนจะไม่ตรงกับ baseline เป๊ะ

### 4B: สกัดเองด้วย MediaPipe (ต้องมีวิดีโอใน `data/videos/`)

```powershell
$env:GLOG_minloglevel="2"           # ลด log ที่รก
python tools/extract_pose.py --limit 5    # ทดสอบ 5 คลิปก่อน
python tools/extract_pose.py              # ทั้งชุด (นานหลายชั่วโมง)
```
- ผลลัพธ์: `data/pose_npy/<video_id>.npy` ขนาด `(จำนวนเฟรม, 150)` = 75 จุด (pose 33 + มือซ้าย 21 + มือขวา 21) × (x, y)
- ถ้าถูกปิดกลางคัน **รันคำสั่งเดิมซ้ำ** ได้ จะข้ามไฟล์ที่เสร็จแล้ว
- ข้อความ log ยาว ๆ จาก MediaPipe/protobuf เป็น warning ปกติ
- เสียบสายชาร์จและตั้งไม่ให้เครื่อง sleep

ตรวจว่าสคริปต์อ่านวิดีโอจาก `data/videos` และเขียน/อ่าน pose ที่ `data/pose_npy` ตรงกัน:
```powershell
Select-String -Path tools\extract_pose.py, tools\make_csv.py -Pattern "pose_npy|videos|TSL-ONE-S"
```

ตรวจว่าครบ:
```powershell
(Get-ChildItem data\pose_npy -Filter *.npy).Count     # ควรได้ 4,152
```

---

## 5. สร้าง CSV สำหรับ SPOTER

```powershell
python tools/make_csv.py --split-mode signer
```

ต้องได้ประมาณ:
```
train: 3317 clips, 184 classes
val: 419 clips, 184 classes
test: 416 clips, 184 classes
ไม่มีไฟล์ .npy: 0 []
```
- ผลอยู่ที่ `data/split_signer/` (`tsl_train.csv`, `tsl_val.csv`, `tsl_test.csv`) และ `data/label_map.json`
- **split B (signer)**: test = signer 03, 08, 22; val = 11, 25, 29; train = คนที่เหลือ → test เป็นคนที่โมเดลไม่เคยเห็นจริง
- ทำไมไม่ใช้ split ตาม JSON: JSON ที่ได้มาแบ่งแบบสุ่มรายคลิป (signer 28/29 คนอยู่ทั้งใน train และ test) ไม่ตรงกับที่ paper บอก — ดู README ข้อ 4.4
- ถ้ามีไฟล์ขาด ดู README ข้อ 4.6 (ชื่อไฟล์ไม่ตรงกับ JSON 4 คลิป)

---

## 6. ตรวจ pipeline ก่อนเทรนจริง

รันทุกอย่างจากโฟลเดอร์ `spoter/`:
```powershell
cd spoter
```

### 6.1 ตรวจว่าการเทรนแบบ batch ให้ผลเท่าโมเดลต้นฉบับ
```powershell
python train_batched.py --check
```
ต้องขึ้น **ผ่าน**

### 6.2 เทรน 1 epoch เพื่อดูว่ารันจบและจับเวลา
```powershell
python train_batched.py --exp smoke --device dml --epochs 1     # เปลี่ยน dml เป็น cuda / cpu ตามเครื่อง
```
อ้างอิง (RX 6700 XT): GPU ~75 วิ/epoch, CPU ~96 วิ/epoch, val หลัง epoch แรก ~3%

### 6.3 (ถ้ามีปัญหา) ตรวจข้อมูลและ overfit test
```powershell
python diag.py data                                  # ไม่มี NaN, จุดหาย ~5%, ช่วงค่า ~[-0.5, 1.2]
python diag.py overfit --device cpu --opt torch      # 32 คลิป ต้องขึ้นถึง ~100% ใน 40 epoch
```

---

## 7. เทรน baseline และอ่านผล

```powershell
python train_batched.py --exp b16_signer_aug_80 --device dml --epochs 80 --aug
```

| ตัวเลือก | ความหมาย | ค่าที่ใช้ (ตาม paper) |
|---|---|---|
| `--device` | `dml` (AMD), `cuda` (NVIDIA), `cpu` | ตามเครื่อง |
| `--epochs` | จำนวน epoch | 80 |
| `--batch_size` | ขนาด batch | 16 |
| `--lr` | learning rate (Adam) | 1e-4 |
| `--frames` | resample ทุกคลิปเป็นกี่เฟรม | 64 |
| `--aug` | เปิด augmentation ของ SPOTER | เปิด |
| `--seed` | random seed | 42 |
| `--train_csv` / `--val_csv` / `--test_csv` | path ของ CSV | `..\data\split_signer\...` |

ผลอยู่ที่ `spoter/out-batched/<exp>/`:
- `summary.txt` — **ตัวเลขที่ใช้รายงาน**: test top-1, top-5, macro-F1, คู่ที่ทายผิดบ่อย (ใช้ checkpoint ที่ val ดีที่สุด)
- `log.csv` — loss / accuracy ทุก epoch (ใช้พล็อตกราฟ)
- `test_predictions.csv` — ผลทายรายคลิป
- `best_val.pth` — weights (ไม่ขึ้น git)

**อย่าใช้ตัวเลขที่ `train.py` ต้นฉบับพิมพ์ตอนจบ** ("The top result ...") — มันเลือก checkpoint ด้วย test และวัดแค่ checkpoint ช่วงต้น ดู README ข้อ 3

### การทดลองต่อยอดที่แนะนำ
```powershell
python train_batched.py --exp b16_signer_noaug_80 --device dml --epochs 80          # ไม่มี augmentation
python train_batched.py --exp b16_signer_aug_80_s1 --device dml --epochs 80 --aug --seed 1   # seed อื่น
```
บันทึกผลลงตาราง README ข้อ 8 ทุกครั้ง พร้อม device, seed และเวลา

---

## 8. สิ่งที่ถูกแก้ใน `spoter/` (ต่างจาก repo ต้นฉบับ)

โฟลเดอร์ `spoter/` คือ [SPOTER](https://github.com/matyasbohacek/spoter) (Apache 2.0) ที่แก้แล้ว — **ไม่ต้อง clone SPOTER ใหม่**

| ไฟล์ | การแก้ไข | เหตุผล |
|---|---|---|
| `spoter/spoter_model.py` | `self.self_attn = SimpleNamespace(batch_first=False)` + `**kwargs` ใน decoder layer | ให้รันบน PyTorch 2.x ได้ |
| `spoter/utils.py` | `stats` ใช้ `defaultdict`; `train_epoch` รองรับ gradient accumulation (`$env:ACCUM_STEPS`) | รองรับ 184 classes; ทางเลือกแทน batch 16 |
| `datasets/czech_slr_dataset.py` | ลบ `- 1` ออกจาก label | CSV ของเรา label เริ่มที่ 0 |
| `train.py` | optimizer เป็น Adam; เลือก device (`$env:FORCE_CPU`, DirectML) | ตาม paper; รองรับ AMD |
| **เพิ่ม** `train_batched.py` | เทรนแบบ batch จริง (16 × 64 เฟรม) + วัดผล test | **สคริปต์หลัก** เร็วกว่า `train.py` ~3 เท่า |
| **เพิ่ม** `eval_ckpt.py` | วัดผล checkpoint ของ `train.py` ด้วย checkpoint ที่ val ดีที่สุด | `train.py` รายงานผลผิด checkpoint |
| **เพิ่ม** `dml_adam.py` | Adam ที่ไม่ใช้ `lerp` | DirectML ไม่รองรับ `lerp` |
| **เพิ่ม** `diag.py` | ตรวจข้อมูล / overfit test | ใช้ไล่ปัญหา |

**ข้อควรรู้:** SPOTER ต้นฉบับไม่มี positional encoding ตามเวลา (`self.pos` เป็นเวกเตอร์เดียว บวกทุกเฟรมเท่ากัน) — โมเดลไม่รู้ลำดับเฟรม ดู README ข้อ 6.5 ถ้าจะปรับปรุงต่อ

### ทางเลือก: `train.py` แบบต้นฉบับ (ทีละคลิป)
ยังใช้ได้แต่ช้ากว่า ต้องตั้ง accumulation ไม่งั้นโมเดลไม่เรียนรู้:
```powershell
$env:ACCUM_STEPS="16"      # สำคัญ: ถ้าเป็น 1 (batch 1 + Adam) โมเดลค้างที่เดาสุ่ม
$env:FORCE_CPU="1"         # AMD: CPU เร็วกว่า DirectML สำหรับแบบทีละคลิป
python train.py --experiment_name signer_accum16 --num_classes 184 `
  --training_set_path ..\data\split_signer\tsl_train.csv `
  --validation_set from-file --validation_set_path ..\data\split_signer\tsl_val.csv `
  --testing_set_path ..\data\split_signer\tsl_test.csv `
  --epochs 80 --lr 0.0001 --seed 42
python eval_ckpt.py --exp signer_accum16
```

---

## 9. แก้ปัญหาที่เจอบ่อย

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `No matching distribution found for torch==2.5.1+cu121` | ต้องใส่ `--extra-index-url https://download.pytorch.org/whl/cu121` (ไม่อยู่บน PyPI) |
| `No module named 'pandas._libs.pandas_parser'` | venv อยู่ในโฟลเดอร์ OneDrive → สร้าง venv ใหม่นอก OneDrive |
| `can't open file '...\spoter\spoter\train.py'` | อยู่ในโฟลเดอร์ `spoter/spoter/` → `cd ..` (รันจาก `spoter/` ชั้นแรก) |
| error แสดง path `...\Python312\python.exe` ทั้งที่เปิด venv | ปกติบน Windows — จริง ๆ คือหาไฟล์ .py ไม่เจอในโฟลเดอร์ปัจจุบัน |
| `'SPOTERTransformerDecoderLayer' object has no attribute 'self_attn'` | `spoter_model.py` เป็นเวอร์ชันต้นฉบับ → ใช้ไฟล์ใน repo นี้ |
| `KeyError: 172` ใน `evaluate` | `utils.py` เป็นเวอร์ชันต้นฉบับ (`range(101)`) |
| มี label `-1` ใน stats | `czech_slr_dataset.py` ยังมี `- 1` |
| loss ค้าง ~5.26, acc ~0.5% | ใช้ `train.py` แบบ batch 1 + Adam → ใช้ `train_batched.py` หรือตั้ง `ACCUM_STEPS=16` |
| warning `aten::lerp ... fall back to run on the CPU` | DirectML + Adam ปกติ → `train_batched.py` ใช้ `DmlAdam` ให้อัตโนมัติ |
| `Weights only load failed ... _rebuild_device_tensor_from_numpy` | `train_batched.py` เวอร์ชันเก่า → ใช้เวอร์ชันใน repo (บันทึก checkpoint บน CPU) |
| test accuracy ต่ำผิดปกติเทียบกับ val | อ่านผลจาก `train.py` ตรง ๆ → ใช้ `summary.txt` หรือ `eval_ckpt.py` |
| warning `enable_nested_tensor is True ...` / `torch.load ... weights_only=False` | ไม่ต้องสนใจ |

---

## 10. git: อะไรขึ้น / ไม่ขึ้น repo

ตาม `.gitignore`: **ไม่ขึ้น** `data/*` (วิดีโอ, pose, CSV — ยกเว้น `data/label_map.json`), `tsl_one_s_dataset_i3d*.json`, `.npy`, `.zip` ของข้อมูล, `.pth`, venv
ไฟล์ `pose_npy.zip` แชร์ผ่านลิงก์ที่จำกัดสิทธิ์ (เช่น Google Drive ของทีม) เท่านั้น อย่า commit เข้า repo
ก่อน commit ตรวจทุกครั้ง:
```powershell
git ls-files | Where-Object { $_ -match '\.(mp4|npy|npz|pth|pt)$|data/split|TSL-ONE-S' }
```
ต้องไม่มีผลลัพธ์ แล้วค่อย `git pull --rebase origin base_line` → `git push origin base_line`

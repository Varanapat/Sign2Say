# สรุป Chapter 11: Object Detection with Deep Learning

> สรุปแบบเน้น "ทำไม" มากกว่า "อะไร" — อ่านจบแล้วควรอธิบายได้ว่าแต่ละโมเดลแก้ปัญหาอะไรของรุ่นก่อน และเอาไปเทรนกับข้อมูลตัวเองได้

---

## 0. ภาพใหญ่ก่อน: Detection ต่างจาก Classification ยังไง

| | Image Classification | Object Detection |
|---|---|---|
| ถามว่า | "ภาพนี้คืออะไร?" | "มีอะไร **อยู่ตรงไหน** บ้าง?" |
| Output | 1 label ต่อภาพ | หลายกล่อง ต่อภาพ แต่ละกล่อง = `(x, y, w, h)` + class + confidence |
| ปัญหาหลัก | แยกคลาสให้ถูก | แยกคลาส **+** หาตำแหน่ง (regression) **+** ไม่รู้ล่วงหน้าว่ามีกี่วัตถุ |

**แก่นของ Object Detection = Classification + Localization (Regression) ทำพร้อมกันหลายจุดในภาพเดียว**

ทุกโมเดลในบทนี้ตอบคำถาม 2 ข้อเหมือนกัน ต่างกันแค่ "วิธี":
1. **จะดูตรงไหนบ้าง?** (หา candidate region)
2. **ตรงนั้นคืออะไร และกล่องควรขยับไปเท่าไหร่?** (classify + refine box)

แบ่งเป็น 2 ตระกูล:
- **Multi-Stage (Two-Stage)**: หา region ก่อน แล้วค่อยจำแนก → แม่นแต่ช้า (R-CNN family)
- **Single-Stage**: ทำนายทุกอย่างในรอบเดียว → เร็วมาก (YOLO)

---

## 1. R-CNN (2014, Ross Girshick) — "ไอเดียตั้งต้น"

### Concept
"ถ้าเรามี CNN ที่ classify ภาพเก่งอยู่แล้ว ก็แค่ตัดภาพเป็นชิ้น ๆ แล้วให้ CNN classify ทีละชิ้น"

### Pipeline 4 ขั้น
```
ภาพ → [1] Selective Search (~2,000 regions)
    → [2] Warp แต่ละ region ให้เป็น 224×224
    → [3] CNN (AlexNet ตัด FC สุดท้ายออก) → Feature Vector
    → [4] SVM จำแนกคลาส (+ bbox regression ปรับกล่อง)
```

**[1] Selective Search** — ไม่ใช่ deep learning แต่เป็นอัลกอริทึมคลาสสิก
- เริ่มจาก graph-based segmentation (Felzenszwalb) แบ่งภาพเป็นชิ้นเล็ก ๆ
- รวมชิ้นที่อยู่ติดกันและคล้ายกันเข้าด้วยกันเรื่อย ๆ โดยดู **color, texture, size, shape compatibility**
- ได้ Region Proposal ~1,000–2,000 กล่อง

**[2] Warping** — ทำไมต้องบีบให้เป็น 224×224?
→ เพราะ **Fully Connected layer รับ input ขนาดตายตัว** (จำนวน neuron fix) ส่วน convolution จริง ๆ รับขนาดไหนก็ได้ *(ช่องว่างในสไลด์ตอบข้อนี้)*

**[4] SVM** — หาเส้นแบ่ง (hyperplane `WX + b = 0`) ที่มี **margin กว้างที่สุด** ระหว่าง 2 กลุ่ม
- Support vectors = จุดข้อมูลที่อยู่บนขอบ H1 (`WX+b ≥ +1`) / H2 (`WX+b ≤ −1`)
- Margin `M = d₊ + d₋` → เลือก hyperplane ที่ M มากสุด
- ใช้ 1 SVM ต่อ 1 คลาส (one-vs-rest)

### จุดอ่อน (= สิ่งที่รุ่นถัดไปต้องแก้)
- ❌ **รัน CNN 2,000 ครั้งต่อภาพ** → ช้ามาก (~47 วินาที/ภาพ)
- ❌ Region ที่ซ้อนกันถูกคำนวณ feature ซ้ำ ๆ เปลืองเปล่า
- ❌ เทรนแยก 3 ส่วน (CNN, SVM, bbox regressor) ไม่ end-to-end
- ❌ Warp ทำให้ภาพบิดสัดส่วน

---

## 2. Fast R-CNN (2015, Ross Girshick) — "คำนวณ CNN ครั้งเดียวพอ"

### Concept หลัก (จำประโยคนี้)
> **"แทนที่จะตัดภาพแล้วค่อยเข้า CNN → ให้ภาพทั้งภาพเข้า CNN ครั้งเดียว แล้วไปตัดบน Feature Map แทน"**

Convolution 1 ครั้ง/ภาพ แทน 2,000 ครั้ง → เร็วขึ้นมหาศาล

### Pipeline 8 ขั้น
```
ภาพทั้งภาพ → CNN → Feature Map ──────────┐
ภาพทั้งภาพ → Selective Search → Proposals ┤
                                          ↓
              RoI Projection (ทาบกล่องลง feature map แล้ว crop)
                                          ↓
              RoI Pooling (ทำให้ทุกชิ้นขนาดเท่ากัน)
                                          ↓
              Flatten → Fully Connected
                         ├→ Softmax           → class
                         └→ Linear Regression → x, y, w, h (ปรับกล่อง)
```

### 2 กลไกใหม่ที่ต้องเข้าใจ

**RoI Projection** — แปลงพิกัดกล่องจากภาพจริงไปเป็นพิกัดบน feature map ด้วยการเทียบสัดส่วน
- เช่น CNN ย่อภาพลง 16 เท่า (stride 16) → กล่องที่ `x=320` บนภาพ จะอยู่ที่ `x=20` บน feature map
- นี่คือเหตุผลที่ต้องรู้ขนาด input แน่นอน

**RoI Pooling** — แก้ปัญหา "region แต่ละอันขนาดไม่เท่ากัน แต่ FC ต้องการขนาดเท่ากัน"
1. Crop feature map ตามกล่อง (ขนาดไม่เท่ากัน เช่น 7×5, 12×9)
2. แบ่งเป็น grid `n × n` ช่องเท่า ๆ กัน (ขนาดช่อง ≈ `m/n`)
3. Max pool แต่ละช่อง → ได้ output `n × n` เสมอ ไม่ว่า input จะใหญ่แค่ไหน

> 💡 นี่คือการแก้ปัญหา warp ของ R-CNN อย่างสวยงาม: แทนที่จะบีบ "ภาพ" ไปบีบ "feature" แทน
> (รุ่นหลังพัฒนาเป็น **RoI Align** ใน Mask R-CNN ที่ใช้ bilinear interpolation แทนการปัดเศษ → แม่นขึ้น)

### สิ่งที่ดีขึ้น
- ✅ CNN รันครั้งเดียว
- ✅ เปลี่ยน SVM เป็น Softmax → classification + bbox regression เทรนร่วมกันได้ (multi-task loss)

### จุดอ่อนที่เหลือ
- ❌ **Selective Search ยังรันบน CPU และกลายเป็นคอขวด** (~2 วินาที จากเวลาทั้งหมด) — CNN เร็วแล้วแต่ตัวหา region ช้า

---

## 3. Faster R-CNN (2015, Shaoqing Ren et al.) — "ให้ Neural Network หา region เองด้วย"

### Concept หลัก
> **เปลี่ยน Selective Search → Region Proposal Network (RPN)** ซึ่งเป็น conv layer เล็ก ๆ ที่ใช้ feature map ร่วมกับตัว detector

ทุกอย่างเป็น neural network ทั้งหมดแล้ว → รันบน GPU ได้ เทรน end-to-end ได้ (~5–7 FPS)

Pipeline ที่เหลือเหมือน Fast R-CNN ทุกอย่าง ต่างแค่ขั้นที่ 2

### Region Proposal Network (RPN) — 6 ขั้น

**[2.1] Intermediate Layer** — conv 3×3 (512 filters, padding) บน feature map สุดท้าย

**[2.2] Anchor Boxes** ⭐ (concept สำคัญที่สุดของบทนี้ ใช้ต่อในโมเดลอื่นอีกเยอะ)
- ทุกตำแหน่งบน feature map วาง "กล่องตั้งต้น" ไว้หลายแบบ
- กำหนดด้วย **Scale** (เล็ก/กลาง/ใหญ่) × **Aspect Ratio** (1:1, 1:2, 2:1)
- ปกติ 3 scales × 3 ratios = **9 anchors ต่อจุด**
- โมเดลไม่ต้องทำนายกล่องจากศูนย์ → **ทำนายแค่ว่าต้องขยับ anchor ไปเท่าไหร่ (offset)** ซึ่งเรียนง่ายกว่ามาก

**ในเชิงโค้ด** — ไม่ได้ sliding window จริง แต่ใช้ 1×1 conv:
```
Feature map (w×h×c)
   → 3×3 conv (c filters)                 → w×h×c
       ├→ 1×1 conv (a filters)            → w×h×a      = objectness score ของแต่ละ anchor
       └→ 1×1 conv (4·a filters)          → w×h×(4a)   = offset (x,y,w,h) ของแต่ละ anchor
(a = จำนวนแบบ anchor เช่น 9)
```

**[2.3] Classification** — anchor นี้เป็น foreground (มีวัตถุ) หรือ background → **Objectness Score**

**[2.4] Regression** — ปรับ `[x, y, w, h]` ของ anchor ให้พอดีวัตถุ

**[2.5] Top-N** — เอาเฉพาะ foreground เรียงตาม score มาก→น้อย เลือก n ตัวแรก

**[2.6] NMS** — ลบกล่องซ้ำ (ดูหัวข้อถัดไป)

> 💡 สังเกต: RPN คือ **mini-detector แบบ class-agnostic** (รู้แค่ "มีของ/ไม่มีของ") แล้วส่งต่อให้ head ตัวที่สองระบุคลาส → นี่คือความหมายของ "two-stage"

---

## 4. เครื่องมือกลางที่ใช้ทุกโมเดล: IoU และ NMS ⭐

### IoU (Intersection over Union)
```
IoU = พื้นที่ซ้อนทับ / พื้นที่รวม = (R1 ∩ R2) / (R1 ∪ R2)
```
- 0 = ไม่ทับกันเลย, 1 = ทับกันสนิท
- ใช้ 3 ที่: (1) วัดว่ากล่องทำนายถูกไหม (เช่น IoU ≥ 0.5 ถือว่าเจอ) (2) ใน NMS (3) จับคู่ anchor กับ ground truth ตอนเทรน

```python
def iou(a, b):  # box = (x1, y1, x2, y2)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)
```

### NMS (Non-Maximum Suppression)
ปัญหา: โมเดลมักทำนายหลายกล่องรอบวัตถุเดียวกัน → ต้องเหลือแค่กล่องที่ดีที่สุด

```
1. เรียงกล่องตาม score มาก → น้อย
2. หยิบกล่องที่ score สูงสุด เก็บไว้เป็นคำตอบ
3. ลบกล่องอื่นที่ IoU กับกล่องนั้น > threshold (เช่น 0.5) ทิ้ง  ← ซ้อนกัน = วัตถุเดียวกัน
4. วนซ้ำกับกล่องที่เหลือจนหมด
```

```python
def nms(boxes, scores, thr=0.5):
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [i for i in order if iou(boxes[best], boxes[i]) <= thr]
    return keep
```

**ผลของการปรับ threshold:**
- IoU threshold **สูง** (0.7) → ลบน้อย → อาจเหลือกล่องซ้ำ
- IoU threshold **ต่ำ** (0.3) → ลบเยอะ → วัตถุที่อยู่ชิดกันจริง ๆ (เช่น คนในฝูงชน) อาจหายไป

---

## 5. ตารางเปรียบเทียบวิวัฒนาการ R-CNN family

| | R-CNN | Fast R-CNN | Faster R-CNN |
|---|---|---|---|
| หา Region | Selective Search | Selective Search | **RPN (neural net)** |
| CNN รันกี่ครั้ง/ภาพ | ~2,000 | **1** | 1 |
| ทำให้ขนาดเท่ากันด้วย | Warp ภาพ | **RoI Pooling** | RoI Pooling |
| Classifier | SVM | **Softmax** | Softmax |
| End-to-end | ❌ | เกือบ (ยกเว้น SS) | ✅ |
| คอขวดที่แก้ | — | CNN ซ้ำซ้อน | Selective Search ช้า |

**Pattern ที่ควรเห็น:** แต่ละรุ่น = หาคอขวดที่ช้าที่สุด แล้วแทนด้วยสิ่งที่ share computation ได้ / เรียนรู้ได้

---

## 6. YOLO — You Only Look Once (Single-Stage)

### Concept หลัก
> **"ไม่ต้องหา region ก่อน — มองภาพครั้งเดียว ทำนายทุกกล่องทุกคลาสพร้อมกันเป็น regression ปัญหาเดียว"**

Unified network, end-to-end → **เร็วระดับ real-time** (YOLOv1 ~45 FPS)

### YOLOv1 ทำงานยังไง
1. แบ่งภาพเป็น grid **S × S** (v1: S = 7)
2. แต่ละ cell ทำนาย:
   - **B กล่อง** (v1: B = 2) แต่ละกล่อง = `x, y, w, h, confidence`
   - **C class probabilities** = `P(class | object)` (ต่อ cell ไม่ใช่ต่อกล่อง!)
3. Output tensor = `S × S × (B·5 + C)` → v1 กับ VOC 20 คลาส = **7 × 7 × 30**

### ความสัมพันธ์ วัตถุ – เซลล์ – กล่อง – คลาส (ข้อสอบชอบออก)
- **วัตถุ ↔ เซลล์**: วัตถุถูกรับผิดชอบโดยเซลล์ที่ **จุดกึ่งกลาง** ของวัตถุตกอยู่
- **วัตถุ ↔ กล่อง**: ในเซลล์นั้น กล่องที่มี **IoU กับ ground truth สูงสุด** เป็นคนรับผิดชอบ อีกกล่องถือเป็น background
- **เซลล์ ↔ คลาส**: 1 เซลล์มีได้ **1 คลาส** เท่านั้น

### ค่าในกล่อง
- `x, y` = offset ของจุดศูนย์กลาง **เทียบกับมุมของ cell** (0–1)
- `w, h` = normalize ด้วยความกว้าง/สูงของ **ทั้งภาพ** (0–1)
- `confidence` = `P(object) × IoU(pred, truth)` → สะท้อนทั้ง "มีของไหม" และ "กล่องแม่นไหม"
- ตอนใช้งาน: **class score ของแต่ละกล่อง = confidence × P(class|object)** → แล้วเข้า NMS

### สถาปัตยกรรม
24 conv layers + 2 FC layers (input 448×448)

### Loss Function (5 ส่วน, ใช้ SSE ทั้งหมด)
| ส่วน | วัดอะไร | จุดสังเกต |
|---|---|---|
| 1 | `(x, y)` ตำแหน่ง | คูณ **λ_coord = 5** ให้น้ำหนักตำแหน่งเยอะ |
| 2 | `(√w, √h)` ขนาด | ใช้ **รากที่สอง** เพราะ error 10px ในกล่องเล็กสำคัญกว่าในกล่องใหญ่ |
| 3 | confidence (มีวัตถุ) | target = IoU |
| 4 | confidence (ไม่มีวัตถุ) | คูณ **λ_noobj = 0.5** เพราะ cell ว่างมีเยอะมาก ไม่ให้มันครอบงำ gradient |
| 5 | class probabilities | คิดเฉพาะ cell ที่มีวัตถุ |

> 💡 Insight: λ ทั้งสองตัวคือการแก้ **class imbalance** ระหว่าง "มีวัตถุ" กับ "พื้นหลัง" — ปัญหานี้เจอในทุก detector (RetinaNet แก้ด้วย Focal Loss)

### จุดอ่อน YOLOv1 (และรุ่นหลังแก้อย่างไร)
- ❌ วัตถุเล็กที่อยู่กระจุกกัน (ฝูงนก) → 1 cell รับได้น้อย → **v2+ เพิ่ม anchor, v3+ ทำนายหลาย scale (FPN)**
- ❌ 2 วัตถุต่างคลาสใน cell เดียว → **v2+ ให้คลาสต่อกล่องแทนต่อ cell**
- ❌ Localization แย่กว่า two-stage

### YOLOv8 (Ultralytics, 2023) — ที่ใช้จริงในบทนี้
- **Anchor-free**: ทำนายจุดศูนย์กลางตรง ๆ ไม่ต้องตั้ง anchor เอง (ลด hyperparameter)
- **Decoupled head**: แยก head classification กับ regression
- Backbone ใช้ C2f module, ทำนาย multi-scale
- ทำได้ 4 งาน: Detection, Instance Segmentation, Classification, Pose Estimation
- มี 5 ขนาด: **n / s / m / l / x** (เล็ก-เร็ว → ใหญ่-แม่น)

---

## 7. Two-Stage vs Single-Stage — เลือกใช้อะไรเมื่อไหร่

| | Two-Stage (Faster R-CNN) | Single-Stage (YOLO) |
|---|---|---|
| ความเร็ว | ช้ากว่า | **Real-time** |
| ความแม่น (โดยเฉพาะวัตถุเล็ก) | ดีกว่า (ดั้งเดิม) | YOLO รุ่นใหม่ตามทันแล้ว |
| เหมาะกับ | งานต้องการความแม่นสูง, ไม่รีบ (ภาพการแพทย์, วิเคราะห์ offline) | กล้อง CCTV, รถขับเอง, มือถือ, edge device |
| ใช้งานจริงตอนนี้ | Detectron2, torchvision | **Ultralytics** (ง่ายที่สุด) |

---

## 8. Hands-on: เทรน YOLOv8 กับ Dataset ของตัวเอง

### ติดตั้ง & ลองเล่น
```bash
pip install ultralytics      # Python ≥ 3.8, PyTorch ≥ 1.8
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

### 6 ขั้นตอน

**① dataset.yaml**
```yaml
train: datasets/train/images   # แนะนำใช้ absolute path
val:   datasets/valid/images
test:  datasets/test/images    # optional
nc: 1                          # จำนวนคลาส
names: ['car']                 # ชื่อคลาส (index 0, 1, 2, ...)
```

**② Label format YOLO** — 1 ภาพ = 1 ไฟล์ `.txt`, 1 บรรทัด = 1 วัตถุ
```
<class> <x_center> <y_center> <width> <height>
0 0.481 0.634 0.690 0.713
```
- ทุกค่า **normalize 0–1** ด้วยขนาดภาพ (มุมบนซ้าย = (0,0), ล่างขวา = (1,1))
- ใช้ **จุดกึ่งกลาง** ไม่ใช่มุมซ้ายบน! (ต่างจาก COCO JSON ที่เป็น `x_min, y_min, w, h` และ Pascal VOC ที่เป็น `x1,y1,x2,y2`)

แปลงจาก pixel `(x1,y1,x2,y2)` → YOLO:
```python
def to_yolo(x1, y1, x2, y2, img_w, img_h):
    return ((x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h,
            (x2 - x1) / img_w, (y2 - y1) / img_h)
```

**③ โครงสร้างโฟลเดอร์** — Ultralytics หา label เองโดยเปลี่ยน `/images/` → `/labels/`
```
datasets/
├── train/
│   ├── images/im0.jpg
│   └── labels/im0.txt     ← ชื่อไฟล์ต้องตรงกัน
├── valid/{images,labels}/
└── test/{images,labels}/
```

**④ เลือกโมเดล** — `yolov8n.pt` (เร็วสุด) ถึง `yolov8x.pt` (แม่นสุด); เริ่มจาก n/s เพื่อทดลองก่อน

**⑤ เทรน**
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")          # โหลด pretrained บน COCO = transfer learning
results = model.train(data="dataset.yaml", epochs=50, batch=16, imgsz=640)
```
ผลลัพธ์อยู่ที่ `runs/detect/train/weights/best.pt`

**⑥ วัดผล & ใช้งาน**
```python
model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="dataset.yaml", imgsz=640, batch=16, conf=0.25, iou=0.5)

results = model(img)                # inference
results[0].save("out.jpg")          # results เป็น list (1 ตัวต่อ 1 ภาพ)
for box in results[0].boxes:
    print(box.cls, box.conf, box.xyxy)
```

### พารามิเตอร์ที่ต้องเข้าใจ
| พารามิเตอร์ | ความหมาย | ค่า default |
|---|---|---|
| `conf` | ตัดกล่องที่ confidence ต่ำกว่านี้ทิ้ง | 0.001 ตอน val |
| `iou` | threshold ของ NMS | 0.6 ตอน val (0.7 ตอน predict) |
| `imgsz` | ขนาดภาพ input | 640 |
| `batch` | จำนวนภาพต่อรอบ | 16 |

> ⚠️ **เรื่อง `conf` ที่สไลด์บอกว่า 0.001 "ไม่แนะนำ"**: ถ้าจะ **วัด mAP** ค่าต่ำ ๆ แบบนี้คือสิ่งที่ถูกต้อง (mAP ต้องการกล่องทุกระดับ confidence เพื่อสร้าง precision-recall curve) แต่ถ้าจะ **ดูผลลัพธ์/ใช้งานจริง** ต้องตั้งสูงขึ้น (0.25–0.5) ไม่งั้นกล่องขยะเต็มภาพ — ทำตามที่อาจารย์กำหนดในงาน แต่เข้าใจเหตุผลไว้

---

## 9. ต่อยอด / ใช้งานจริง

**ตัววัดผลที่ต้องรู้ (ไม่มีในสไลด์แต่ต้องใช้แน่นอน)**
- **Precision** = ทำนายว่าเจอ ถูกจริงกี่ % / **Recall** = ของจริงทั้งหมด หาเจอกี่ %
- **AP** = พื้นที่ใต้ Precision-Recall curve ของ 1 คลาส
- **mAP@0.5** = AP เฉลี่ยทุกคลาส โดยถือว่าเจอเมื่อ IoU ≥ 0.5
- **mAP@0.5:0.95** = เฉลี่ย IoU threshold 0.5, 0.55, …, 0.95 (เข้มกว่า, มาตรฐาน COCO)

**Debug เวลาผลไม่ดี**
| อาการ | สาเหตุที่เป็นไปได้ | ลองแก้ |
|---|---|---|
| mAP ≈ 0 | label format ผิด (ลืม normalize, ใช้มุมแทนกึ่งกลาง, ชื่อไฟล์ไม่ตรง) | เปิดดู `runs/.../train_batch0.jpg` ว่ากล่องวาดถูกที่ไหม |
| พลาดวัตถุเล็ก | ภาพถูกย่อเกิน | เพิ่ม `imgsz`, ใช้โมเดลใหญ่ขึ้น |
| กล่องซ้ำหลายกล่อง | NMS iou สูงไป | ลด `iou` |
| วัตถุชิดกันหายไป | NMS iou ต่ำไป | เพิ่ม `iou` |
| train ดี val แย่ | overfit / data น้อย | เพิ่ม data, augmentation, epochs น้อยลง |

**ไอเดียต่อยอด**
- Concept ที่ใช้ซ้ำได้ทุกที่: **anchor, IoU, NMS, RoI pooling, multi-task loss** — เจออีกใน Mask R-CNN (segmentation), RetinaNet, SSD, DETR
- เปลี่ยน task ใน Ultralytics แค่เปลี่ยนโมเดล: `yolov8n-seg.pt` (segmentation), `yolov8n-pose.pt` (pose)
- รุ่นใหม่กว่า (YOLO11 ฯลฯ) ใช้ API เดียวกัน เปลี่ยนแค่ชื่อไฟล์โมเดล
- Export ไป deploy: `model.export(format="onnx")` / `"tflite"` / `"coreml"`
- Label ข้อมูลเอง: Roboflow, CVAT, Label Studio (export เป็น YOLO format ได้เลย)
- ต่อกับ tracking (นับรถ/คนผ่านเส้น): `model.track(source="video.mp4")`

---

## 10. สรุปเป็นประโยคเดียวต่อหัวข้อ (ไว้ทวนก่อนสอบ)

- **R-CNN**: Selective Search หา ~2,000 region → warp → CNN ทีละ region → SVM — ถูกแต่ช้ามาก
- **Fast R-CNN**: CNN ทั้งภาพครั้งเดียว → ตัดบน feature map → RoI Pooling → Softmax + bbox regression
- **Faster R-CNN**: แทน Selective Search ด้วย RPN + anchor boxes → ทั้งระบบเป็น neural network
- **IoU**: วัดการซ้อนทับ = intersection / union
- **NMS**: เก็บกล่อง score สูงสุด ลบกล่องที่ทับมันเกิน threshold วนจนหมด
- **YOLO**: grid S×S, ทุก cell ทำนายกล่อง + คลาสพร้อมกันในรอบเดียว → real-time
- **YOLO loss**: λ_coord=5 เน้นตำแหน่ง, √w √h ให้กล่องเล็กสำคัญ, λ_noobj=0.5 ลดอิทธิพลพื้นหลัง
- **Ultralytics**: yaml + label txt (class, xc, yc, w, h normalize) + `images/`↔`labels/` → `model.train()` → `model.val()`

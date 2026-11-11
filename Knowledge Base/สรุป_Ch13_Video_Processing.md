# สรุป Chapter 13: Deep Learning for Video Processing

> สรุปแบบเน้น "ทำไม" — อ่านจบแล้วควรตอบได้ว่าแต่ละสถาปัตยกรรมแก้ปัญหาอะไร เลือกใช้ตัวไหนเมื่อไหร่ และเขียนโค้ดเทรน video classifier เองได้

---

## 0. ภาพใหญ่: วิดีโอ = ภาพ + เวลา

**วิดีโอต่างจากภาพนิ่งตรงที่มีมิติเวลาเพิ่มเข้ามา** ดังนั้นต้องดึง feature 2 แบบพร้อมกัน

| | Spatial Feature (เชิงพื้นที่) | Temporal Feature (เชิงเวลา) |
|---|---|---|
| ตอบคำถาม | "ในเฟรมนี้มีอะไร?" | "เฟรมนี้ต่างจากเฟรมก่อนยังไง / เคลื่อนไหวยังไง?" |
| ตัวอย่าง | ขอบ, สี, รูปร่าง, พื้นผิว | การเคลื่อนที่, ลำดับท่าทาง |
| เครื่องมือ | CNN (2D) | RNN / LSTM |
| ทั้งสองอย่างพร้อมกัน | **ConvLSTM**, **3D CNN** | |

> 💡 ทำไมต้องมี temporal? — ภาพนิ่งภาพเดียวของคน "ยืนบนกระดานกระโดด" บอกไม่ได้ว่ากำลังจะกระโดดหรือเพิ่งขึ้นมา ต้องดูหลายเฟรมถึงรู้ว่าเป็น action อะไร

### รูปร่างข้อมูล (Tensor) — ต้องแม่นเรื่องนี้ ไม่งั้นโค้ดพังทุกครั้ง
| ข้อมูล | Rank | Shape |
|---|---|---|
| ค่าเดียว | 0 (scalar) | `()` |
| vector | 1 | `(N,)` |
| ภาพขาวดำ | 2 (matrix) | `(H, W)` |
| ภาพสี | 3 | `(H, W, C)` |
| **วิดีโอสี** | **4** | `(T, H, W, C)` — T = จำนวนเฟรม |
| **batch ของวิดีโอ** | **5** | `(B, T, C, H, W)` หรือ `(B, C, T, H, W)` |

⚠️ ลำดับแกนต่างกันตามเครื่องมือ:
- OpenCV อ่านมาเป็น `(H, W, C)` และเป็น **BGR** ไม่ใช่ RGB
- ConvLSTM ในสไลด์ใช้ `(B, T, C, H, W)` (`batch_first=True`)
- `nn.Conv3d` ของ PyTorch ต้องการ `(B, C, T, H, W)` → ต้อง `x.permute(0, 2, 1, 3, 4)`

### งานที่ใช้จริง
- **Action Recognition** — เตะบอล, กระโดด, หมุนตัว
- **Event Detection** — รถชนจาก CCTV, ผู้สูงอายุล้ม (CNN + LSTM)
- **Behavior Analysis** — พฤติกรรมการเดินของคนในสถานี (3D CNN)

### 4 แนวทางหลักในการสร้างโมเดลวิดีโอ (แผนที่ของทั้งบท)
```
① Video → CNN ทีละเฟรม → Feature Vector ต่อเฟรม → LSTM → ผลลัพธ์     (แยก spatial / temporal)
② Signal / Feature ที่สกัดมาแล้ว → LSTM → ผลลัพธ์                     (ข้าม CNN)
③ Video → ConvLSTM → (NN) → ผลลัพธ์                                   (รวมในชั้นเดียว แบบ recurrent)
④ Video → 3D CNN → (NN) → ผลลัพธ์                                     (รวมในชั้นเดียว แบบ convolution)
```

---

## 1. RNN — "เครือข่ายที่มีความจำ"

### ทำไมต้องมี RNN
NN / CNN ปกติออกแบบมาสำหรับข้อมูลที่:
- ใส่เข้ามา **พร้อมกันทั้งก้อน**
- **ลำดับไม่สำคัญ**
- **ขนาดเท่ากันตายตัว**

แต่ข้อมูลอนุกรมเวลา (วิดีโอ, ข้อความ, ราคาหุ้น) ลำดับสำคัญมาก และความยาวไม่เท่ากัน → ต้องมีโครงสร้างที่ **"จำสิ่งที่เห็นมาแล้ว"**

เปรียบเหมือนคนอ่านหนังสือ — เราเข้าใจคำปัจจุบันได้เพราะจำคำก่อนหน้าได้

### กลไก
RNN มี cell **ตัวเดียว** ที่ถูกเรียกซ้ำทุก time step (unfold ออกมาดูได้เป็นสายยาว) โดยรับ 2 อย่าง:

```
Hₜ = tanh( Xₜ·Wₓ  +  Hₜ₋₁·Wₕ  + b )      ← สกัด feature จาก (ข้อมูลตอนนี้ + ความจำเดิม)
Yₜ = softmax( Hₜ·Wᵧ + bᵧ )                ← ตีความ feature ออกมาเป็นคำตอบ
```

- `Hₜ` = **hidden state** = "ความจำ" ที่ส่งต่อไปเรื่อย ๆ
- ส่วน `Yₜ` (RNN Cell Forward) **เป็นแค่ส่วนต่อขยาย** — ปรับให้ตรงงานได้ หรือไม่ใส่ก็ได้ถ้าไม่ต้องการ output ทุก step (เช่น classify ทั้งวิดีโอ ใช้แค่ `H` ตัวสุดท้าย)
- **Weight `Wₓ, Wₕ` ใช้ร่วมกันทุก time step** → model ไม่โตตามความยาว input

### ข้อดี / ข้อเสีย
| ข้อดี | ข้อเสีย |
|---|---|
| รับ input ยาวไม่เท่ากันได้ | คำนวณช้า (ต้องทำทีละ step, ขนานไม่ได้) |
| ขนาดโมเดลไม่โตตาม input | **ลืมข้อมูลช่วงต้น ๆ** เมื่อ sequence ยาว |
| แชร์ weight ข้ามเวลา | มองอนาคตไม่ได้ (ใช้แค่อดีต) |

### ปัญหาใหญ่: ลืมของเก่า (Long-term dependency)
ข้อมูลจากต้น sequence ต้องผ่าน `tanh` และคูณ `Wₕ` ซ้ำหลายสิบรอบ → อิทธิพลหดลงเรื่อย ๆ (ตอนเทรนคือ **vanishing gradient**) → **LSTM เกิดมาเพื่อแก้ข้อนี้**

---

## 2. LSTM (1997, Hochreiter) — "RNN ที่เลือกได้ว่าจะจำหรือลืมอะไร"

### Concept หลัก
> เพิ่ม **"สายพานความจำระยะยาว" (Cell State)** ที่ข้อมูลไหลผ่านได้แทบไม่ถูกบีบ แล้วใช้ **ประตู (gate)** 3 บานควบคุมว่าจะ **ลบ / เขียน / อ่าน** อะไรบนสายพานนั้น

### ส่วนประกอบ (ตามที่อาจารย์แบ่ง)
- **2 State**
  - **Cell State `Cₜ`** (Long memory) — สรุปข้อมูลทั้งหมดตั้งแต่ต้นจนถึงตอนนี้
  - **Hidden State `hₜ`** (Short memory) — "ลักษณะเฉพาะ" ที่เกี่ยวกับตอนล่าสุด ใช้ทำนาย/ส่งต่อ (ยังไม่ใช่ output โดยตรง)
- **3 Gate** (ใช้ sigmoid → ค่า 0 ถึง 1 = "เปิดกี่ %")
- **2 Modulate** (ใช้ tanh → ค่า −1 ถึง 1) *(ชื่อนี้อาจารย์ตั้งเอง ตำราทั่วไปเรียก candidate / cell input)*

### สมการ (ทุก gate ใช้ input เหมือนกัน คือ `xₜ` + `hₜ₋₁` แต่มี weight ของตัวเอง)

| ส่วน | สมการ | หน้าที่ | เปรียบเทียบ |
|---|---|---|---|
| Forget gate | `fₜ = σ(W_xf·xₜ + W_hf·hₜ₋₁ + b_f)` | เก็บ Cell State เก่าไว้กี่ % | **Delete** |
| Input gate | `iₜ = σ(W_xi·xₜ + W_hi·hₜ₋₁ + b_i)` | ให้ข้อมูลใหม่เข้ากี่ % | **Write** |
| Input modulate | `zₜ = tanh(W_xz·xₜ + W_hz·hₜ₋₁ + b_z)` | ข้อมูลใหม่ที่จะเขียน (candidate) | เนื้อหาที่จะเขียน |
| Output gate | `oₜ = σ(W_xo·xₜ + W_ho·hₜ₋₁ + b_o)` | เปิดเผย Cell State ออกไปกี่ % | **Read** |

**อัปเดตสถานะ — 2 บรรทัดที่สำคัญที่สุดในบท:**
```
Cₜ = fₜ ⊙ Cₜ₋₁  +  iₜ ⊙ zₜ        ← (ของเก่าที่เหลือ) + (ของใหม่ที่ยอมให้เข้า)
hₜ = oₜ ⊙ tanh(Cₜ)                 ← Output modulate: อ่านความจำออกมาบางส่วน
```
(`⊙` = คูณทีละตัว element-wise)

> 💡 **ทำไม LSTM จำได้นาน?** — ดูบรรทัด `Cₜ`: เป็นการ **บวก** ไม่ใช่การคูณ weight ซ้ำ ๆ ถ้า `fₜ ≈ 1` ข้อมูลเก่าก็ไหลผ่านได้เกือบสมบูรณ์ gradient ก็ไหลย้อนกลับได้ไกล (หลักการเดียวกับ skip connection ใน ResNet)

> 💡 **ทำไม gate ใช้ sigmoid แต่เนื้อหาใช้ tanh?** — gate คือ "สัดส่วน" ต้องอยู่ 0–1; เนื้อหาต้องเป็นได้ทั้งบวกและลบ (เพิ่ม/ลดค่าในความจำ) จึงใช้ −1 ถึง 1

### ตัวอย่างความคิด
ดูวิดีโอคนกระโดดน้ำ:
- เฟรมแรก ๆ เห็นสระ → **input gate** เขียน "อยู่ที่สระ" ลง cell
- กลางคลิปเห็นคนวิ่งบนกระดาน → **เขียน** "กำลังวิ่ง", **forget gate** ยังเก็บ "สระ" ไว้
- เฟรมท้ายลอยกลางอากาศ → cell มีครบ "สระ + วิ่ง + ลอย" → **output gate** อ่านออกไป → classify ว่า Diving

---

## 3. ConvLSTM (2015, Xingjian Shi) — "LSTM ที่เห็นเป็นภาพ"

### ปัญหาของ LSTM ธรรมดากับวิดีโอ
LSTM รับ input เป็น **vector** → ถ้าเอาเฟรมไปใส่ ต้อง flatten ภาพเป็นเส้นยาว → **ข้อมูลว่า pixel ไหนอยู่ติดกับ pixel ไหน (โครงสร้างเชิงพื้นที่) หายหมด** และพารามิเตอร์มหาศาล (112×112×3 = 37,632 ค่าต่อเฟรม)

### Concept หลัก
> **เอาสมการ LSTM มาทั้งชุด แต่เปลี่ยนการคูณ weight (`W·x`) เป็น convolution (`W * x`)**

ผลคือ input, hidden state, cell state เป็น **feature map 3 มิติ `(C, H, W)`** ไม่ใช่ vector → แต่ละตำแหน่งบนภาพมีความจำของตัวเอง และอัปเดตโดยดูเพื่อนบ้านรอบ ๆ (ตาม kernel)

```
iₜ = σ(W_xi * Xₜ + W_hi * Hₜ₋₁ + W_ci ⊙ Cₜ₋₁ + b_i)     (* = convolution)
fₜ = σ(W_xf * Xₜ + W_hf * Hₜ₋₁ + W_cf ⊙ Cₜ₋₁ + b_f)
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ tanh(W_xc * Xₜ + W_hc * Hₜ₋₁ + b_c)
oₜ = σ(W_xo * Xₜ + W_ho * Hₜ₋₁ + W_co ⊙ Cₜ + b_o)
Hₜ = oₜ ⊙ tanh(Cₜ)
```
(paper ต้นฉบับมี "peephole" `W_c ⊙ C` ด้วย แต่โค้ดที่ใช้ในวิชาตัดออก)

> ⚠️ สไลด์เขียนว่า "เปลี่ยน element-wise multiplication เป็น convolution" — ที่ถูกคือเปลี่ยน **matrix multiplication (W·x)** เป็น convolution ส่วนการคูณ element-wise ของ gate (`f ⊙ C`) ยังอยู่เหมือนเดิม

### ใช้กับอะไร
งานที่ทั้ง "ตำแหน่ง" และ "เวลา" สำคัญ: พยากรณ์ฝน/เรดาร์อากาศ (งานต้นฉบับของ paper), ภาพดาวเทียม, ทำนายเฟรมถัดไป, กล้องวงจรปิด, video classification

---

## 4. Hands-on: สร้าง ConvLSTM Video Classifier ด้วย PyTorch

5 ขั้นตอนเหมือนการเทรนโมเดลทั่วไป: **Dataset → Model → Loss/Optimizer → Train → Evaluate**

### ① Dataset — ต้องเขียนเอง เพราะไม่มี dataset สำเร็จรูปสำหรับวิดีโอ 5 มิติ
โครงสร้างโฟลเดอร์: `train/<ชื่อคลาส>/<วิดีโอ>.mp4` (ชื่อโฟลเดอร์ = label)

```python
class VideoDataset(Dataset):
    def __init__(self, root_dir, labels, transform=None, max_frames=0, resize=(224, 224)):
        # เดินทุกโฟลเดอร์คลาส เก็บ path วิดีโอ + label ไว้ (ยังไม่โหลดจริง ประหยัด RAM)
        ...
    def __len__(self):
        return len(self.video_paths)
    def __getitem__(self, idx):
        frames = self.load_video(self.video_paths[idx])         # numpy (T, H, W, C)
        frames = [self.transform(f) for f in frames]             # transform ทีละเฟรม
        return torch.stack(frames), self.labels.index(label)     # (T, C, H, W), int
    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        # วน cap.read() → ret=False แปลว่าหมดวิดีโอ → resize → append
        # หยุดเมื่อครบ max_frames ; ปิดด้วย cap.release() ใน finally
```

```python
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32),
    transforms.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
])
train_dataset = VideoDataset('./action_classification_dataset/train',
                             ['Biking', 'Diving', 'HorseRiding'],
                             transform, max_frames=30, resize=(112, 112))
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
```
- `num_workers` = จำนวน process ที่ช่วยกันโหลดข้อมูลบน CPU (0/1 = process เดียว) ไม่ควรเกินจำนวน core
- วิดีโอกิน RAM มาก → `resize` เล็ก (112) และ `batch_size` เล็ก (4) เป็นเรื่องปกติ

### ② Model — PyTorch ไม่มี ConvLSTM สำเร็จรูป ต้องเขียน cell เอง
```python
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2        # รักษาขนาด H, W
        # conv เดียว ออก 4 เท่า → ได้ทั้ง 4 gate ในครั้งเดียว (เร็วกว่าทำ 4 conv)
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=padding, bias=bias)

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)                  # ต่อ x กับ h ตามแกน channel
        cc_i, cc_f, cc_o, cc_g = torch.split(self.conv(combined), self.hidden_dim, dim=1)
        i, f, o = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)                                     # = zₜ (input modulate)
        c_next = f * c_cur + i * g                               # ← สมการ LSTM เป๊ะ ๆ
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
```
> 💡 สังเกต: `torch.cat([x, h])` แล้ว conv ครั้งเดียว = `W_x * x + W_h * h` (เพราะ conv เป็น linear) — trick นี้ใช้ใน LSTM ทั่วไปด้วย

ต่อเป็นโมเดล classification:
```python
class ConvLSTMModel(nn.Module):
    def __init__(self, num_classes, image_size, hidden_dim=[64, 64]):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim=3, hidden_dim=hidden_dim, kernel_size=(3, 3),
                                 num_layers=2, batch_first=True)
        self.fc = nn.Linear(image_size[0] * image_size[1] * hidden_dim[-1], num_classes)

    def forward(self, x):                                   # x: (B, T, C, H, W)
        _, last_state_list = self.convlstm(x)
        h_last = last_state_list[-1][0]                     # h ของ layer สุดท้าย ณ เฟรมสุดท้าย
        return self.fc(h_last.view(x.size(0), -1))          # flatten → FC
```
(ต้นฉบับ `ConvLSTM`: https://github.com/ndrplz/ConvLSTM_pytorch)

### ③ Loss & Optimizer — เลือกตามชนิดงาน
| งาน | Loss |
|---|---|
| Multiclass (1 คลาสต่อวิดีโอ) | `CrossEntropyLoss` |
| Multilabel (หลาย label พร้อมกัน) | `BCEWithLogitsLoss` |
| Regression (ทำนายค่าตัวเลข) | `MSELoss` / `L1Loss` |

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

### ④ Train
```python
model.train()
for epoch in range(num_epochs):
    for videos, labels in train_dataloader:
        optimizer.zero_grad()
        loss = criterion(model(videos), labels)
        loss.backward()
        optimizer.step()
torch.save(model.state_dict(), 'video_classification_model.pt')
```

### ⑤ Evaluate & ใช้งาน
```python
model.eval()
with torch.no_grad():                         # ไม่ต้องคำนวณ gradient ตอนทดสอบ
    for videos, labels in test_dataloader:
        _, predicted = torch.max(model(videos), 1)
        ...                                   # ใช้ sklearn.metrics วัดผล

# ทำนายวิดีโอเดียว
video = torch.stack([transform(f) for f in load_video(path, (112, 112), 30)])
video = video.unsqueeze(0)                    # เพิ่มแกน batch → (1, T, C, H, W)
score = torch.softmax(model(video).squeeze(0), 0)
print(classes[score.argmax()], score.max().item() * 100)
```

### ⚠️ จุดที่โค้ดในสไลด์อาจพังตอนใช้จริง
| ปัญหา | เกิดอะไรขึ้น | แก้ |
|---|---|---|
| `view(self.batch_size, -1)` | batch สุดท้ายมีไม่ครบ 4 → error shape | ใช้ `x.size(0)` (แบบที่เขียนด้านบน) หรือ `drop_last=True` |
| ลูปเทรนใช้ `dataloader` แต่สร้างไว้ชื่อ `train_dataloader` | `NameError` | ใช้ชื่อให้ตรงกัน |
| วิดีโอสั้นกว่า `max_frames` | แต่ละคลิปได้ T ไม่เท่ากัน → `DataLoader` stack ไม่ได้ | pad เฟรมสุดท้ายซ้ำ หรือ sample ให้ได้จำนวนเท่ากันเสมอ |
| อ่านแค่ 30 เฟรมแรก | ได้แค่ ~1 วินาทีแรก อาจยังไม่เห็น action | **sample เฟรมกระจายทั้งคลิป** (ดูโค้ดด้านล่าง) |
| OpenCV ให้ BGR | ถ้าจะใช้ pretrained model ที่เทรนด้วย RGB สีจะเพี้ยน | `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` |
| ไม่ได้ `.to(device)` | รันบน CPU ช้ามาก | `model.to(device)`, `videos.to(device)` |
| accuracy เฉลี่ยราย batch | batch สุดท้ายเล็กกว่า → ค่าเพี้ยนเล็กน้อย | นับ `correct`/`total` ราย sample |

```python
def sample_frames(path, num_frames=16, resize=(112, 112)):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, num_frames).astype(int)   # กระจายเท่า ๆ กันทั้งคลิป
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            f = frames[-1]                                       # สำรองกรณีอ่านไม่ได้
        frames.append(cv2.cvtColor(cv2.resize(f, resize), cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)
```

---

## 5. 3D CNN — "ฟิลเตอร์ที่เลื่อนผ่านเวลาด้วย"

### Concept หลัก
| | 2D Convolution | 3D Convolution |
|---|---|---|
| ฟิลเตอร์ | `k × k` (× ทุก channel) | `t × k × k` (× ทุก channel) |
| เลื่อนตามแกน | x, y | **x, y และ เวลา (t)** |
| เห็นอะไร | รูปร่างในเฟรมเดียว | รูปร่าง **+ การเปลี่ยนแปลงระหว่างเฟรมติดกัน** |

> ฟิลเตอร์ 3D ตัวเดียวดึงได้ทั้ง spatial และ temporal ในครั้งเดียว → เรียกว่า **Spatio-Temporal operation**

ถ้าใช้ 2D conv กับวิดีโอโดยเอาเฟรมไปซ้อนเป็น channel → เวลาจะถูกรวมหายตั้งแต่ layer แรก; 3D conv **รักษามิติเวลาไว้** ให้ layer ถัด ๆ ไปค่อย ๆ เรียนรู้ motion ที่ซับซ้อนขึ้น

### วิธีคำนวณ (ตัวอย่างในสไลด์)
ข้อมูล `D` 4×4×4, ฟิลเตอร์ `F` 3×3×3, stride 1 → ผลลัพธ์ `O` 2×2×2 (`4 − 3 + 1 = 2` ทุกแกน)

ค่า 1 ช่องของผลลัพธ์ =
```
O[0,0,0] = (D[0] ส่วน 3×3 * F[0]) + (D[1] ส่วน 3×3 * F[1]) + (D[2] ส่วน 3×3 * F[2])
         = ผลรวมของ 2D conv 3 ชั้นที่ซ้อนกันตามแกนเวลา
```
แล้วเลื่อนต่อทั้งแกน x, y และแกนเวลา (`O[1]` = ใช้ `D[1], D[2], D[3]`)

**3D Max Pooling** — หลักการเดียวกัน แต่หาค่าสูงสุดในก้อน 3D (เช่น 2×2×2) → ย่อทั้งพื้นที่และเวลา

### ใน PyTorch
```python
m = nn.Conv3d(16, 33, 3, stride=2)
#            ↑   ↑   ↑  └ stride ทุกแกน
#            │   │   └ kernel 3×3×3
#            │   └ output channels (จำนวนฟิลเตอร์)
#            └ input channels
m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))   # กำหนดแยกแต่ละแกน (T, H, W)
x = torch.randn(20, 16, 10, 50, 100)     # (B, C, T, H, W)  ← channel มาก่อนเวลา!
```

ขนาด output แต่ละแกน = `⌊(in + 2·pad − kernel) / stride⌋ + 1`

### ข้อดี / ข้อเสีย
| ข้อดี | ข้อเสีย |
|---|---|
| แม่นกว่า 2D CNN สำหรับข้อมูล 3 มิติ | **พารามิเตอร์เยอะมาก** (kernel ใหญ่ขึ้น t เท่า) |
| ได้ spatial + temporal ในครั้งเดียว | กิน memory และเวลามาก |
| ขนานได้ดีกว่า LSTM (ไม่ต้องทำทีละ step) | **เสี่ยง overfit** + มี pretrained model น้อย |

---

## 6. Action Recognition: ปัญหาและทางแก้ระดับงานวิจัย

### ปัญหาหลัก
1. Dataset วิดีโอที่ใช้กัน (UCF-101, HMDB-51) **เล็ก** → โมเดลซับซ้อน overfit
2. สถาปัตยกรรมภาพนิ่ง (2D ConvNet) พัฒนาไปไกลแล้ว แต่ใช้กับวิดีโอตรง ๆ ไม่ได้
3. 3D ConvNet พารามิเตอร์เยอะ เทรนยาก

### Two-Stream
แยกเป็น 2 สาย แล้วค่อยรวม (fusion) ตอนท้าย:
- **RGB stream** — ภาพปกติ → บอก "มีอะไร" (appearance)
- **Optical Flow stream** — ภาพที่บอกทิศทาง/ความเร็วการเคลื่อนที่ของแต่ละ pixel ระหว่างเฟรม → บอก "เคลื่อนที่ยังไง" (motion)

### I3D — Inflated 3D ConvNet ⭐
> **"ไม่ต้องออกแบบ 3D CNN ใหม่จากศูนย์ — เอา 2D CNN ที่ดีอยู่แล้ว (Inception-v1) มา 'เป่าลม' ให้กลายเป็น 3D"**

- **Inflation**: ฟิลเตอร์/pooling `N × N` → `T × N × N`
- จุดสำคัญคือเลือก temporal stride / pooling ให้ receptive field ทางเวลาเหมาะสม
- ใช้คู่กับ Two-Stream: RGB → 3D ConvNet, Optical Flow → 3D ConvNet → Average Pooling 3D → 1×1×1 conv → เฉลี่ยคะแนนสองสาย → Softmax
- Input ตัวอย่าง: `(25, 3, 16, 224, 224)` = batch 25, RGB, 16 เฟรม, 224×224

> 💡 จุดแข็งที่แท้จริงของ inflation คือ **ใช้ weight pretrained จาก ImageNet ได้** (ก๊อป weight 2D ซ้ำ T ชั้นแล้วหารด้วย T) → แก้ปัญหา "3D CNN มี pretrained น้อย" และ "dataset เล็ก" ไปพร้อมกัน

### 1×1×1 Convolution ทำอะไร
kernel ไม่ครอบเวลา (T=1) ไม่ครอบพื้นที่ (1×1) → **ผสมเฉพาะ channel** (linear combination)
- **Channel reduction** — ลด channel ลด computation/memory
- **Channel mixing** — สร้าง feature ใหม่จากการรวมหลาย channel
- **Non-linearity** — เมื่อต่อกับ BatchNorm + ReLU
- ทำให้ Inception branch "สั้นและเบา"

### สถาปัตยกรรมรุ่นใหม่ (ไว้ค้นต่อ)
| ปี | โมเดล | ไอเดียเด่น |
|---|---|---|
| 2019 | **SlowFast** | 2 สาย: Slow (เฟรมน้อย, ดูรายละเอียด) + Fast (เฟรมถี่, ดู motion, channel น้อย) |
| 2020 | **X3D** | ขยาย 2D เป็น 3D ทีละแกนอย่างประหยัด ได้โมเดลเล็กแต่แม่น |
| 2020 | Temporal Pyramid Network | จับ action ที่ความเร็วต่างกันด้วย feature หลาย temporal scale |
| 2021 | **Video Swin Transformer**, MViTv2 | ใช้ Transformer/attention แทน conv |
| 2022 | **VideoMAE** | pretrain แบบ self-supervised โดยปิดบัง patch 90% แล้วให้เดา → ใช้ข้อมูลไม่มี label ได้ |
| 2023 | UniFormer | ผสม conv (local) + attention (global) |

---

## 7. เลือกใช้อะไรดี

| สถานการณ์ | แนะนำ | เหตุผล |
|---|---|---|
| Data น้อย, GPU น้อย, อยากได้ผลเร็ว | **CNN pretrained (ResNet) ทีละเฟรม → LSTM** | ใช้ ImageNet pretrained เต็มที่ เบาที่สุด |
| สนใจตำแหน่ง + เวลา (พยากรณ์อากาศ, ทำนายเฟรมถัดไป) | **ConvLSTM** | ความจำเป็นแผนที่ ไม่ทิ้งโครงสร้างพื้นที่ |
| Action ที่ motion สั้น ๆ สำคัญ, มี GPU พอ | **3D CNN** | จับ motion ระหว่างเฟรมติดกันได้ดี ขนานได้ |
| อยากได้ความแม่นสูง, ใช้ของสำเร็จ | **pretrained บน Kinetics** (R(2+1)D, X3D, VideoMAE) แล้ว fine-tune | วิดีโอ pretrain สำคัญมาก dataset เราเล็กเสมอ |

**เทียบตัวหลักในบท**
| | CNN + LSTM | ConvLSTM | 3D CNN |
|---|---|---|---|
| Spatial | CNN | Conv ภายใน cell | Conv 3D |
| Temporal | LSTM บน vector | LSTM บน feature map | Conv ตามแกนเวลา |
| ความยาว sequence | ยาวได้ | ยาวได้ | ตายตัว (clip) |
| ความเร็วเทรน | ปานกลาง | ช้า (ทีละ step + conv) | เร็วกว่า (ขนาน) แต่กิน memory |
| Pretrained | ✅ เยอะ (ฝั่ง CNN) | ❌ ต้องเทรนเอง | มีบ้าง (Kinetics) |

---

## 8. ต่อยอด / ใช้งานจริง

**ของสำเร็จรูปที่ช่วยได้มาก**
- `torchvision.models.video` — `r3d_18`, `r2plus1d_18`, `mvit_v2_s`, `swin3d_t` พร้อม weight pretrained บน Kinetics-400
  ```python
  from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
  model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
  model.fc = nn.Linear(model.fc.in_features, num_classes)   # เปลี่ยนหัวให้เท่าจำนวนคลาสเรา
  ```
- HuggingFace `transformers` — `VideoMAEForVideoClassification`
- PyTorchVideo — SlowFast, X3D
- อ่านวิดีโอเร็วกว่า OpenCV: `decord`, `torchvision.io.read_video`

**เทคนิคที่ช่วยให้ผลดีขึ้น**
- **Frame sampling**: sample กระจายทั้งคลิป; ตอน test ใช้หลาย clip แล้วเฉลี่ยคะแนน
- **Augmentation**: ต้องใช้ transform **เดียวกันกับทุกเฟรมในคลิป** (crop/flip ตำแหน่งเดียวกัน) ไม่งั้น motion เพี้ยน
- **Pre-extract feature**: รัน CNN ดึง feature ทุกเฟรมเก็บไว้ครั้งเดียว แล้วค่อยเทรน LSTM บน feature → เร็วขึ้นหลายสิบเท่า
- **ลด memory**: `resize` 112, `num_frames` 8–16, mixed precision (`torch.autocast`)

**ไอเดียโปรเจกต์**
- ตรวจจับการล้มของผู้สูงอายุ (binary classification บนคลิปสั้น)
- นับท่าออกกำลังกาย — ใช้ YOLO pose (Ch11) ดึง keypoint ทุกเฟรม → LSTM (= แนวทาง ② ที่ใช้ signal แทนภาพ เบามาก)
- ตรวจจับเหตุการณ์จาก CCTV — YOLO หาวัตถุ + tracking + LSTM วิเคราะห์ลำดับ
- พยากรณ์ภาพเรดาร์ฝนล่วงหน้า (ConvLSTM แบบ encoder-decoder)

---

## 9. สรุปเป็นประโยคเดียวต่อหัวข้อ (ไว้ทวนก่อนสอบ)

- **วิดีโอ** = ภาพต่อกันตามเวลา → Tensor rank 4 `(T, H, W, C)`, มี batch เป็น 5 มิติ; ต้องดึงทั้ง spatial และ temporal feature
- **RNN**: `Hₜ = tanh(XₜWₓ + Hₜ₋₁Wₕ + b)` ใช้ cell เดียวซ้ำทุก step แชร์ weight แต่ลืมข้อมูลช่วงต้นเมื่อ sequence ยาว
- **LSTM**: เพิ่ม Cell State + 3 gate (forget=ลบ, input=เขียน, output=อ่าน); `Cₜ = f⊙Cₜ₋₁ + i⊙z`, `hₜ = o⊙tanh(Cₜ)` — การบวกทำให้จำได้นาน
- **Gate ใช้ sigmoid (0–1 = สัดส่วน), เนื้อหาใช้ tanh (−1–1)**
- **ConvLSTM**: LSTM ที่เปลี่ยนการคูณ weight เป็น convolution → state เป็น feature map รักษาโครงสร้างพื้นที่
- **PyTorch ConvLSTM**: ต่อ `[x, h]` → conv เดียวออก `4×hidden` → split เป็น i, f, o, g
- **3D CNN**: ฟิลเตอร์ `t×k×k` เลื่อนผ่าน x, y, เวลา → spatio-temporal ในครั้งเดียว แม่นแต่พารามิเตอร์เยอะ เสี่ยง overfit
- **Conv3d** input `(B, C, T, H, W)` — channel มาก่อนเวลา
- **Two-Stream**: RGB (appearance) + Optical Flow (motion) แล้ว fusion
- **I3D**: inflate 2D Inception → 3D เพื่อใช้ pretrained weight ได้; 1×1×1 conv ผสม/ลด channel

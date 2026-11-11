# Experimental Design & Model Evaluation: Sign2Say

> จัดทำตาม `ExperimenceDesign.md` · 8 พ.ย. 2026
> ต่อจาก [Technical_Design_Sign2Say.md](Technical_Design_Sign2Say.md) (architecture) และ [Data_Pipeline_Sign2Say.md](Data_Pipeline_Sign2Say.md) (ข้อมูล, schema, split)
>
> **ป้ายกำกับ**
> - 📄 **จากงานวิจัย** (✅ ตรวจกับตารางต้นฉบับแล้ว)
> - 📚 **จากบทเรียน** ใน Knowledge Base
> - 🛠 **ข้อเสนอการออกแบบ**
> - 🔮 **Expected** = คาดการณ์จากหลักฐานในงานวิจัย **ไม่ใช่ผลทดลอง**
>
> ⚠️ **เอกสารนี้ยังไม่มีผลทดลองใดเลย** ทุกช่อง "Actual" ว่างไว้ (—) จนกว่าจะรันจริง ตัวเลขที่ปรากฏเป็นผลจากบทความอื่น (คนละ dataset/protocol) หรือค่าที่คำนวณได้ทางสถิติเท่านั้น

---

## 0. ภาพรวม

| | |
|---|---|
| งานหลัก | ISLR signer-independent บน TSL-ONE-S (184 คำ) |
| งานรอง | CSLR บน TSL-51 ประโยค (signer-dependent), gloss → ไทย |
| Protocol หลัก | 10 signer-folds (`splits/tsl_one_s_signer_indep_fold{0-9}.json`) × 3 seeds |
| Metric หลัก | ISLR: **macro-F1** และ Top-1; CSLR: **WER**; แปล: **chrF** |
| จำนวนการทดลองหลัก | 18 การทดลอง (§3) จัดเป็น 4 ระดับความสำคัญ — ทำ P0–P1 ให้ครบก่อน |

---

## 1. Research Questions & Hypotheses

แต่ละ RQ เขียนให้ **พิสูจน์ได้**: มีตัวแปรต้น, ตัวแปรควบคุม, metric และเกณฑ์ตัดสินที่ตั้งไว้ก่อนรัน

| RQ | คำถาม | Hypothesis (H) | เกณฑ์ตัดสิน (ตั้งก่อนรัน) | หลักฐานที่ทำให้ตั้ง H แบบนี้ |
|---|---|---|---|---|
| **RQ1** | RGB-based กับ Landmark-based ให้ผลต่างกันอย่างไรบน TSL ระดับคำ แบบ signer-independent | H1: landmark ได้ macro-F1 **ไม่ต่ำกว่า** RGB เกิน 2 pp แต่ใช้ compute น้อยกว่า ≥ 10× | ผลต่างเฉลี่ย 10 fold + 95% CI ของผลต่าง (paired) | 📄 ✅ TSL-ONE-S (split เดิม): SPOTER 80.83% vs I3D 79.69%; AUTSL: SL-GCN 95.45% vs RGB 3D CNN 94.77% |
| **RQ2** | Hybrid fusion (landmark + hand-crop RGB) ช่วยไหม | H2: late fusion เพิ่ม macro-F1 > 0 อย่างมีนัยสำคัญ และช่วยมากที่สุดในคู่คำที่ต่างกันแค่รูปมือ | paired test p < 0.05 + วิเคราะห์ confusion รายคู่ | 📄 ✅ SAM-SLR: ensemble หลาย modality ดีกว่าเดี่ยว (RGB-D ensemble 97.10 vs SL-GCN 95.45); TwoStream CSLR: two-stream 18.79 vs keypoint 28.00 WER |
| **RQ3** | Temporal modeling แบบ LSTM/GRU เทียบกับ Transformer | H3: บนข้อมูลขนาด TSL-ONE-S (~3.4K คลิป train) BiGRU กับ Transformer ต่างกัน < 2 pp; ทั้งสองดีกว่า "ไม่มี temporal model" (mean pooling) ชัดเจน | ผลต่างและ CI | 📄 TSL10: LSTM/GRU 99% บน 10 คำ (ข้อมูลเล็ก, protocol ต่าง); SPOTER (Transformer) ใช้ได้ดีบน TSL-ONE-S; 📚 Ch13 / Transformers for CV: Transformer ต้องการข้อมูลมากกว่า |
| **RQ4** | Pretraining ช่วยลดความต้องการข้อมูลได้แค่ไหน | H4: เมื่อฝึกด้วยผู้แสดงน้อย (3–7 คน) pretrained encoder ดีกว่า scratch ชัดเจน และช่องว่างแคบลงเมื่อใช้ผู้แสดงทั้งหมด | learning curve 4 จุด; วัด "จำนวนผู้แสดงที่ต้องใช้เพื่อให้ scratch ได้เท่า pretrained" | 📚 Pre-trained CNN & Transfer Learning; 📄 ✅ SLRT: feature ที่ pretrain บนข้อมูลภาษามือดีกว่า ImageNet |
| **RQ5** | ผล signer-independent ต่างจาก random split แค่ไหน | H5: random split ได้ Top-1 สูงกว่า signer-independent (ช่องว่าง > 0) และช่องว่างของ RGB กว้างกว่า landmark | ผลต่าง (random − SI) ต่อโมเดล | 📄 ✅ TSL-ONE-S รายงานว่า I3D มีช่องว่าง train–val กว้างกว่า SPOTER (สัญญาณ overfit ลักษณะภาพ); split เดิมของ TSL-ONE-S เป็น signer-dependent (ตรวจแล้ว ✅) |
| **RQ6** | จุดหน้า/ปากช่วยภาษามือไทยไหม | H6: X71 (เพิ่มริมฝีปาก/ตา) ดีกว่า C54 และ C54 ดีกว่า "ไม่มีหน้า" | ablation 3 ระดับ | 📄 ✅ TwoStream (PHOENIX-2014T keypoint): + ปาก WER 36.88 → 28.20, + หน้า → 27.19 (ภาษา DGS, งาน CSLR) |
| **RQ7** | CSLR: ต้องใช้ CTC หรือ segment-then-classify ก็พอ | H7: segment-then-classify ใกล้ CTC ในประโยค `with_space` แต่แย่กว่าชัดเจนใน `no_space` | WER แยกตามชนิดประโยค | 📄 TSL-ONE-S ตัดคลิปด้วยกฎ "มือวางบนตัก" (ใช้ได้เมื่อมีจังหวะพัก) |
| **RQ8** | gloss → ไทย: rule กับ LLM few-shot ต่างกันอย่างไร และ error จาก recognition ส่งต่อแค่ไหน | H8: บน gloss จริง LLM ≈ rule ด้าน chrF แต่ LLM คล่องกว่า (human eval); บน gloss ที่ทำนาย rule ทนน้อยกว่า | chrF + human eval, gloss จริง vs ทำนาย | 📄 ✅ SLRT Gloss2Text บน PHOENIX14T BLEU-4 24.54 (gloss จริง) |
| **RQ9** | ฝึก encoder ร่วม TSL-ONE-S + TSL-51 ช่วยไหม (label space ต่างกัน) | H9: ช่วย TSL-51 (ข้อมูลน้อยกว่า) มากกว่าช่วย TSL-ONE-S | ผลบน test ของแต่ละ dataset แยกกัน | 🛠 มาจากข้อค้นพบว่าคำศัพท์สอง dataset ไม่ซ้ำกัน (Data Pipeline §0 ✅) |

---

## 2. Baselines

| ระดับ | โมเดล | Input | Configuration (🛠 เว้นแต่ระบุ) | เหตุผลที่เลือก |
|---|---|---|---|---|
| Sanity | **Chance / majority** | – | 1/184 = 0.54% (TSL-ONE-S), 1/52 = 1.9% (TSL-51) | ขอบล่าง — ถ้าโมเดลใกล้ค่านี้แปลว่า pipeline ผิด |
| **Simple** | **Logistic Regression / Random Forest บนสถิติของ landmark** | C54 normalize แล้ว → ต่อจุด: mean, std, min, max ของ x, y และ velocity → ~1,300 ค่าต่อคลิป | sklearn; RF 500 ต้น; LR L2, C จาก val | ไม่มี temporal model → วัดว่าข้อมูล "ตำแหน่ง/รูปมือเฉลี่ย" อย่างเดียวได้เท่าไร (ตอบ RQ3 ด้านล่างสุด) |
| Simple (ทางเลือก) | **1-NN + DTW** | C54 มือ + ข้อมือ | ระยะ DTW บน sequence ที่ normalize | ไม่ต้องฝึก, เป็น baseline คลาสสิกของ gesture recognition |
| **Standard DL** | **BiGRU 2 ชั้น** | C54, `[B,64,54,7]` → flatten ต่อเฟรม → Linear 256 | hidden 256, dropout 0.3, Adam lr 1e-3, batch 32, 100 epochs, early stop บน val macro-F1 (patience 15) | 📚 Ch13 (RNN/LSTM/GRU); 📄 TSL10 ใช้ LSTM/GRU บน MediaPipe |
| Standard DL | **SPOTER** (โค้ดทางการ) | S54/C54 | ตามบทความ TSL-ONE-S: Adam lr 1e-4, batch 16, 80 epochs 📄 ✅ | **ทำให้เทียบกับบทความได้ตรง** (TSL-ONE-S ระบุว่าข้อมูลเข้ากับ SPOTER ทางการ 📄) — ต้องรันซ้ำบน SI split เอง |
| **Advanced** | **S2S-LTR** (โมเดลที่เสนอ) | C54 `[B,64,54,7]` | part-aware MLP → Transformer 4 ชั้น d = 256, 4 heads, ff 1024 → CLS head (~3.5M params ตาม TDD §6.2); AdamW lr 3e-4, wd 0.05, cosine, warmup 5 epochs, label smoothing 0.1, batch 32, 150 epochs | โมเดลหลักของระบบ |
| Advanced (RGB) | **I3D** (Kinetics-400 pretrained) | RGB `[B,3,64,224,224]` | ตามบทความ: Adam lr 1e-4, batch 16, 80 epochs 📄 ✅ (ถ้า GPU ไม่พอ: batch 8 + grad accumulation ×2) | baseline ของ Architecture A และ RQ1 |
| Advanced (Hybrid) | **S2S-LTR + hand-crop ResNet18 / MobileNetV3** | C54 + crops `[B,T,2,3,112,112]` | late fusion ก่อน (logit เฉลี่ยถ่วงน้ำหนัก w เลือกบน val) → early concat → cross-attention | Alternative ของ TDD; ตอบ RQ2 |

**กติกาเดียวกันทุกโมเดล** (ทำให้เปรียบเทียบได้): split เดียวกัน, ใช้ val เลือก epoch/hyperparameter, budget tuning เท่ากัน (สูงสุด 12 config ต่อโมเดลบน fold 0 แล้ว fix config ไปทุก fold), seed {0, 1, 2}

---

## 3. Experimental Matrix

ไม่ทำทุก combination — เลือกเฉพาะแถวที่เปลี่ยน **ตัวแปรเดียว** จาก reference เพื่อให้ตอบ RQ ได้

**Reference config (R)**: C54 · part-aware MLP · Transformer 4 ชั้น · ไม่มี fusion · CLS head · scratch · velocity + mask · online augment · SI-10fold

| ID | Input | Backbone / spatial | Temporal | Fusion | Head/Decoder | Pretrained | ตอบ | Priority |
|---|---|---|---|---|---|---|---|---|
| X0 | C54 สถิติ | – | – | – | LR / RF | – | baseline | P0 |
| X1 | C54 | flatten | BiGRU | – | CLS | scratch | RQ3, baseline | P0 |
| X2 | S54/C54 | SPOTER | Transformer | – | SPOTER | scratch | baseline เทียบบทความ | P0 |
| **R** | C54 | part-aware | Transformer | – | CLS | scratch | โมเดลหลัก | P0 |
| X3 | C54 | part-aware | **mean pool** (ไม่มี temporal) | – | CLS | scratch | RQ3 | P1 |
| X4 | C54 | part-aware | **BiGRU** | – | CLS | scratch | RQ3 | P1 |
| X5 | C54 | part-aware | **TCN** (1D conv 4 ชั้น) | – | CLS | scratch | RQ3 | P2 |
| X6 | **C54 ไม่มีหน้า** (48 จุด) | part-aware | Transformer | – | CLS | scratch | RQ6 | P1 |
| X7 | **X71** | part-aware | Transformer | – | CLS | scratch | RQ6 | P1 |
| X8 | C54 | part-aware | Transformer | – | CLS | **pretrain บน Google ISLR (ASL landmarks)** | RQ4 | P2 |
| X9 | C54 | part-aware | Transformer | – | CLS | **pretrain self-supervised (masked frame) บน TSL-ONE-S train + TSL-51** | RQ4 | P3 |
| X10 | **RGB** | I3D | (3D conv) | – | CLS | Kinetics | RQ1 | P1 |
| X11 | RGB | I3D | – | – | CLS | **scratch** | RQ4 (ฝั่ง RGB) | P3 |
| X12 | C54 + **hand crop** | part-aware + ResNet18 | Transformer | **late** | CLS | ImageNet | RQ2 | P1 |
| X13 | C54 + hand crop | เหมือน X12 | Transformer | **early** | CLS | ImageNet | RQ2 | P2 |
| X14 | C54 + hand crop | เหมือน X12 | Transformer | **cross-attn** | CLS | ImageNet | RQ2 | P3 |
| X15 | C54 **hand crop อย่างเดียว** | ResNet18 | Transformer | – | CLS | ImageNet | RQ2 (ต้องมีเพื่อแยกผลของ landmark) | P2 |
| X16 | C54 (TSL-ONE-S + TSL-51) | part-aware | Transformer | – | **2 heads** (184 + 52) | scratch | RQ9 | P2 |
| C1 | C54 ประโยค | part-aware | Transformer | – | **CTC** | encoder จาก R/X16 | RQ7 | P1 |
| C2 | C54 ประโยค | Architecture D | – | – | **segment → CLS** | encoder จาก X16 | RQ7 | P1 |
| T1 | gloss จริง / ทำนาย | – | – | – | **rule** | – | RQ8 | P1 |
| T2 | gloss จริง / ทำนาย | – | – | – | **LLM few-shot** | LLM สำเร็จรูป | RQ8 | P1 |

**RQ5 (random vs SI)**: รัน X1, R, X10 ซ้ำบน 3 protocol — SI-10fold, random-matched (ขนาดเท่ากัน), original split ของ TSL-ONE-S → 3 × 3 = 9 runs-ชุด (P0 สำหรับ R, P1 สำหรับที่เหลือ)

**RQ4 learning curve**: R vs X8 (และ X10 vs X11 ถ้ามี GPU) ที่จำนวนผู้แสดงใน train = 3, 7, 14, ทั้งหมด (~23) — ลดด้วยการ **ตัดผู้แสดง** ไม่ใช่ตัดคลิป เพราะความหลากหลายของคนคือข้อมูลที่หายากจริง; test fold คงเดิม

**ค่าใช้จ่ายโดยประมาณ** (🛠 คำนวณ ไม่ใช่วัด): landmark model 1 run ≈ ไม่กี่นาที–< 1 ชม. บน GPU 8 GB (SPOTER บน TSL-ONE-S ใช้ 1:48 ชม. บน RTX 2070 ที่ 80 epochs 📄 ✅) → P0–P1 landmark ทั้งหมด ≈ 10 รูปแบบ × 10 fold × 3 seed = 300 runs → ต้องลดเป็น **5 fold × 3 seed** ถ้าเวลาไม่พอ; I3D 1 run ≈ 21:41 ชม. บน A6000 📄 ✅ → **I3D รันแค่ 3 fold × 1 seed** และระบุชัด

---

## 4. Evaluation Metrics

### 4.1 Recognition (ISLR)

| Metric | นิยาม | ข้อดี | ข้อจำกัด | ใช้ใน Sign2Say |
|---|---|---|---|---|
| Accuracy / Top-1 | ทายถูก / ทั้งหมด | เข้าใจง่าย, เทียบบทความได้ | 📚 Evaluation Metrics: หลอกได้เมื่อคลาสไม่สมดุล (TSL-ONE-S 16–27 คลิปต่อคำ — ไม่สมดุลเล็กน้อย; หมวด 07 มี 1,225 คลิป ✅) | รายงานเสมอ |
| Top-5 | คำตอบอยู่ใน 5 อันดับแรก | สะท้อนการใช้งานจริงที่ UI แสดง top-3/5 ให้ผู้ใช้เลือก (FR2 ใน TDD) | ไม่บอกว่าแยกคำใกล้กันได้ | รายงานเสมอ (+ Top-3 ตาม UI) |
| Precision / Recall / F1 (ต่อคลาส) | 📚 F1 = harmonic mean ของ P และ R | หาคำที่โมเดลอ่อน | 184 ค่า อ่านยาก | ตารางภาคผนวก + 10 คำที่แย่สุด |
| **Macro-F1** | เฉลี่ย F1 ทุกคลาสเท่ากัน | ไม่ถูกคลาสใหญ่ครอบงำ | แปรปรวนสูงเมื่อคลาสมีตัวอย่าง test น้อย (~2 คลิป/คำ/fold) | **metric หลัก** (เฉลี่ย fold แล้วค่อยเฉลี่ย seed) |
| Confusion matrix | ตาราง จริง × ทาย | เห็นคู่ที่สับสน | 184 × 184 อ่านไม่ได้ทั้งตาราง | แสดง (ก) ต่อหมวด 8 หมวด (ข) 20 คู่ที่สับสนมากสุด — ใช้ตัดสินใจเปิด Hybrid (TDD §10.2) |
| Per-signer accuracy | accuracy แยกตามผู้แสดงใน test | วัดความเป็นธรรม/ความทนต่อคน | – | รายงาน min, median, max (worst-signer) |
| Rejection (null) | precision/recall ของคลาส `null_act` + accuracy เมื่อตั้ง threshold | สำคัญกับ FR4 | ขึ้นกับ threshold | TSL-51 เท่านั้น (TSL-ONE-S ไม่มีคลาส null) |

### 4.2 Continuous recognition (CSLR)

| Metric | นิยาม | ข้อดี | ข้อจำกัด |
|---|---|---|---|
| **WER** | (S + D + I) / N จาก edit distance ระดับ gloss | มาตรฐานของ CSLR (เทียบงานวิจัยได้) | เกิน 100% ได้; ไม่สนความหมาย (ทาย `ฉัน_var_2` แทน `ฉัน_var_1` นับผิดเต็ม) → 🛠 รายงาน **WER ระดับคำ** (รวม variant) คู่กับระดับ `sign_id` |
| SER (Sentence Error Rate) | สัดส่วนประโยคที่ผิดอย่างน้อย 1 gloss | ตรงกับประสบการณ์ผู้ใช้ (ประโยคผิดคำเดียวก็สื่อผิด) | หยาบ, ประโยคยาวเสียเปรียบ |
| แยก S / D / I | นับแยกชนิด | ชี้สาเหตุ: D สูง = ตัดคำหาย; I สูง = ช่วงเปลี่ยนท่าถูกนับเป็นคำ | – |
| Signer-independent | – | – | **วัดไม่ได้** กับประโยค TSL-51 (คนเดียว ✅) → ระบุเป็นข้อจำกัด; ถ้าเก็บประโยคจากคนอื่นได้ใช้เป็น external |

### 4.3 Translation (gloss → ไทย)

| Metric | ข้อดี | ข้อจำกัด | ใช้อย่างไร |
|---|---|---|---|
| BLEU-4 | มาตรฐาน เทียบงาน SLT ได้ | ภาษาไทยต้องตัดคำก่อน → ผลขึ้นกับ tokenizer; ประโยคสั้น 3–6 คำทำให้ 4-gram หายาก | sacreBLEU บนข้อความที่ตัดด้วย PyThaiNLP `newmm`, **ระบุ signature/tokenizer** |
| ROUGE-L | วัด longest common subsequence — ไวต่อการเรียงคำ | เหมือน BLEU เรื่องการตัดคำ | รายงานเสริม |
| **chrF** | ระดับตัวอักษร **ไม่ต้องตัดคำไทย**, ดีกับประโยคสั้น | ไม่จับความหมายที่ใช้คำต่างกัน (แม่ vs มารดา) | **metric หลัก** |
| Human evaluation | วัดความหมายและความเป็นธรรมชาติได้จริง | แพง, ต้องมีผู้ประเมิน | 2 ด้าน 1–5 คะแนน: **Adequacy** (ความหมายครบถูก) และ **Fluency** (ภาษาไทยเป็นธรรมชาติ) + ธง **hallucination** (เติมข้อมูลที่ไม่มีใน gloss); ผู้ประเมิน ≥ 2 คน (อย่างน้อย 1 คนใช้ภาษามือได้), ไม่รู้ว่าระบบไหนสร้าง (blind), สุ่มลำดับ, รายงาน agreement (Cohen's κ หรือ Krippendorff's α) |

### 4.4 Protocol ทางสถิติ

| ประเด็น | กฎ (🛠) |
|---|---|
| การรายงาน | mean ± std ข้าม fold (และ seed); ไม่รายงานแค่ fold ที่ดีที่สุด |
| ความแม่นยำของตัวเลข | ช่วงความเชื่อมั่น 95% ของ accuracy ที่ ~80% บน test fold เดียว (~400 คลิป) ≈ **±3.9 pp**; รวมทุก fold (4,152 คลิป) ≈ **±1.2 pp**; user_sign (547 คลิป) ≈ ±3.4 pp (คำนวณจาก normal approximation) → **ผลต่าง < ~4 pp จาก fold เดียวสรุปไม่ได้** |
| ทดสอบนัยสำคัญ | paired: Wilcoxon signed-rank บน 10 fold และ McNemar บนผลต่อคลิปที่รวมทุก fold; หลายคู่เทียบ → Holm–Bonferroni |
| ใช้ test | เลือกทุกอย่างบน val; test ประเมินครั้งเดียวต่อ config ที่ fix แล้ว |
| การเทียบกับบทความ | ตัวเลข 79.69/80.83% ของ TSL-ONE-S มาจาก split เดิม → เทียบได้กับ X2/X10 บน **original split** เท่านั้น |

---

## 5. Ablation Studies

ทุก ablation เปลี่ยนตัวแปรเดียวจาก R (หรือจาก X12 สำหรับ fusion) และใช้ SI-10fold

| คำถามใน prompt | Ablation | แถวที่ใช้ | การอ่านผล |
|---|---|---|---|
| การเพิ่ม **Landmark** ช่วยจริงไหม | (ก) RGB hand-crop อย่างเดียว vs hand-crop + landmark; (ข) I3D vs I3D + landmark late fusion | X15 vs X12; X10 vs (X10 ⊕ R) | ถ้า (crop + landmark) > crop อย่างเดียว → landmark ให้ข้อมูลตำแหน่ง/การเคลื่อนที่ที่ภาพ crop ไม่มี |
| การเพิ่ม **Facial features** ช่วยจริงไหม | ไม่มีหน้า (48 จุด) → C54 (คิ้ว 4 + มุมปาก 2) → X71 (+ ริมฝีปาก/ตา) | X6 → R → X7 | ดูผลรวม + **ผลเฉพาะคำที่ต่างกันที่สีหน้า** (ต้องให้ผู้รู้ TSL ระบุรายการคำ) |
| การเพิ่ม **Temporal encoder** ช่วยจริงไหม | ไม่มี (สถิติ/mean pool) → TCN → BiGRU → Transformer | X0, X3 → X5 → X4 → R | ช่องว่าง X3 → R คือประโยชน์ของการ model ลำดับเวลา |
| เปลี่ยน **Fusion strategy** | late vs early vs cross-attention | X12, X13, X14 | ทำ X13–X14 ต่อเมื่อ X12 > R อย่างมีนัยสำคัญเท่านั้น (TDD §5.4) |
| **Pretrained** มีประโยชน์แค่ไหน | landmark: scratch vs ASL-pretrain vs self-supervised; RGB: Kinetics vs scratch; hand-crop: ImageNet vs scratch | R vs X8 vs X9; X10 vs X11; X12 vs X12-scratch | รายงานที่ทุกจุดของ learning curve (RQ4) — ประโยชน์ที่ข้อมูลน้อยสำคัญกว่าที่ข้อมูลเต็ม |
| (เพิ่ม) Feature design | ± velocity, ± mask, ± part-aware (เทียบ flatten), ± hand-local coords | R vs R−vel, R−mask, R−part, R−local | ตรวจว่าการออกแบบใน TDD §6.2 มีเหตุผลจริง |
| (เพิ่ม) Augmentation | ไม่มี / online / online + copy สำเร็จรูปของ TSL-51 / + hflip | R และ X16 | copy สำเร็จรูปผ่าน MediaPipe จริง จึงอาจช่วยต่างจาก online augment |
| (เพิ่ม) จำนวนเฟรม | T = 32 / 64 / 96 | R | trade-off ความเร็ว–ความแม่น |

---

## 6. Robustness Testing

ข้อจำกัดที่ต้องรู้ก่อน: ไฟล์ TSL-ONE-S ที่มี **ไม่มี label ต่อคลิป** ของแสง/ฉากหลัง/อุปกรณ์ (บทความรายงานแค่สัดส่วน ❓) → ใช้ 3 แหล่งรวมกัน

| แหล่ง | วิธี | ข้อดี / ข้อเสีย |
|---|---|---|
| **A. Perturbation สังเคราะห์** | แก้วิดีโอ test แล้ว **สกัด landmark ใหม่** (วัดทั้ง MediaPipe + โมเดล); สำหรับ RGB ใช้วิดีโอที่แก้ตรง ๆ | ควบคุมระดับได้ / ไม่เหมือนสภาพจริงทั้งหมด |
| **B. Slice ตามธรรมชาติ** | แยกผล test ตามกลุ่มที่มีข้อมูล: ผู้แสดง, hearing status (deaf 13 / hearing 16 📄), ภูมิภาค/โรงเรียน (ถอดจาก Table 8), handedness ของคำ, อัตรามือหาย | ไม่ต้องสร้างข้อมูล / กลุ่มเล็ก CI กว้าง |
| **C. ชุดที่เก็บเอง** | ตาม Data Pipeline §4.3 (ผู้แสดงใหม่ × อุปกรณ์ × แสง × ฉากหลัง × มุม) | ใกล้การใช้งานจริงที่สุด / ต้องใช้เวลาเก็บ |

| ปัจจัย | วิธีทดสอบ | ระดับ | Metric |
|---|---|---|---|
| Lighting | A: ปรับ gamma/brightness, ลด contrast, เพิ่ม noise แสงน้อย; C: แสงห้อง / แสงน้อย / ย้อนแสง | 5 ระดับ (เช่น brightness 0.4×–1.6×) | Top-1 + **อัตรามือหาย** ต่อระดับ |
| Background | A: แทนฉากหลังด้วย MediaPipe Selfie Segmentation + ภาพรก; B: ผลต่อผู้แสดงที่บันทึกหลายสถานที่ (บทความระบุว่ามี 📄); C: ฉากเรียบ/รก | 3 ระดับ (เลียนแบบ low/medium/high ของบทความ) | Top-1 — คาดว่า RGB ไวกว่า landmark (H5) |
| Camera angle | A: หมุนในระนาบ ±15°, ±30° (2D เท่านั้น); C: บันทึกมุมเอียง ~30° (yaw) จริง | 4 ระดับ + มุมจริง | Top-1 |
| Different signers | B: per-signer accuracy ใน SI test; external: TSL-51 expert → user_sign | ทุกผู้แสดง | min / median / max, std ข้ามคน |
| Occlusion | A: วางกล่องทึบบังมือ/หน้า (ใน RGB ก่อนสกัด) เป็นช่วงเวลา 10–40% ของคลิป; จำลองใน landmark ด้วยการ mask | 4 ระดับ | Top-1 vs สัดส่วนที่บัง |
| Signing speed | A: resample landmark/วิดีโอ test ที่ 0.5×, 0.75×, 1.25×, 1.5×, 2× | 5 ระดับ | Top-1 — ตรวจว่า augment ช่วง 0.8–1.25 ครอบคลุมพอไหม |
| Recording devices | C: webcam laptop vs มือถือ; cross: TSL-51 expert (1920×1080) → user (640×480, ✅); A: ลดความละเอียด 720p → 480p → 360p + JPEG/H.264 คุณภาพต่ำ | 3–4 ระดับ | Top-1 + อัตรามือหาย |
| (เพิ่ม) Handedness | flip วิดีโอ test (จำลองผู้ใช้ถนัดซ้าย) | 2 | Top-1 — คาดว่าตกมากถ้าไม่ฝึกด้วย flip (ผู้แสดงถนัดขวาทั้งหมด 📄) |

รายงานในรูป **กราฟเส้น metric vs ระดับ** ต่อโมเดล (R, X10, X12) + อัตรามือหายของ MediaPipe บนแกนรอง เพื่อแยกว่าความผิดพลาดมาจาก perception หรือจากโมเดล

---

## 7. Efficiency Evaluation

| Metric | วิธีวัด (🛠) | หน่วย |
|---|---|---|
| Training time | wall-clock จนถึง epoch ที่ดีที่สุดบน val + เวลารวม; ระบุ GPU, batch, precision (fp32/AMP) | ชั่วโมง |
| Inference latency | batch = 1, warm-up 20 รอบ, วัด 200 รอบ, รายงาน **median และ p95**; แยก (ก) MediaPipe ต่อเฟรม (ข) โมเดลต่อคลิป (ค) end-to-end จากจบท่าถึงได้ผล | ms |
| FPS | จำนวนเฟรมที่ MediaPipe + buffer ประมวลผลได้ต่อวินาทีบน **CPU laptop เป้าหมาย** (NFR2 ใน TDD ≥ 20 fps) | fps |
| Model parameters | `sum(p.numel())` แยกส่วน (encoder, head, RGB stream) | M |
| GPU memory | `torch.cuda.max_memory_allocated()` ตอนฝึก (batch ที่ใช้จริง) และตอน inference | GB |
| Model size | ขนาดไฟล์ state dict fp32 / fp16 / int8 (ถ้า quantize) | MB |
| FLOPs (เสริม) | fvcore หรือ ptflops ต่อคลิป | GFLOPs |

**ตารางที่ต้องเติมหลังวัด** (ค่าที่มีอยู่แล้วมาจากบทความหรือการคำนวณ ไม่ใช่การวัดของเรา):

| โมเดล | Params | Train time | Latency (CPU) | FPS | GPU mem | Size |
|---|---|---|---|---|---|---|
| X1 BiGRU | 2.16M ✅ | 24 นาที/fold (median, 88 epoch) ✅ | 2.33 ms (p95 2.36) ✅ | – | 68 MB (MPS, batch 32) ✅ | 8.3 / 4.1 MB ✅ |
| X2 SPOTER | 5.92M 📄 ✅ | 1:48 ชม. RTX 2070 📄 ✅ (split เดิม) / ของเรา — | — | — | — | — |
| R S2S-LTR | 3.28M ✅ | 13 นาที/fold (median, 139 epoch) ✅ | 1.31 ms (p95 1.33) ✅ | – | 249 MB (MPS, batch 32) ✅ | 12.5 / 6.3 MB ✅ |
| X16 joint | 3.29M ✅ | 27 นาที (fold 0, 150 epoch) ✅ | 1.32 ms ✅ | – | – | 12.6 / 6.3 MB ✅ |
| C1 CTC | 3.88M ✅ | 4–15 นาที/run ✅ | 1.94 ms ต่อประโยค 10 วินาที ✅ | – | – | 14.8 / 7.4 MB ✅ |
| X10 I3D | 12.35M 📄 ✅ | 21:41 ชม. A6000 📄 ✅ / ของเรา — | — | — | — | — |
| X12 Hybrid | — | — | — | — | — | — |
| MediaPipe Holistic | – | – | 23.7 ms/เฟรม (p95 25.9) ✅ | **42 fps** ✅ | – | – |

✅ = วัดเอง (รอบที่ 7 ใน §8.9)

---

## 8. Final Experimental Plan

### 8.1 Research Questions → §1 (RQ1–RQ9)

### 8.2 Hypotheses → §1 (H1–H9 พร้อมเกณฑ์ตัดสินที่ตั้งก่อนรัน)

### 8.3 Baseline Models → §2 (Chance, LR/RF, DTW, BiGRU, SPOTER, S2S-LTR, I3D, Hybrid)

### 8.4 Experimental Matrix → §3 (X0–X16, C1–C2, T1–T2 + RQ4/RQ5 protocol)

### 8.5 Evaluation Metrics → §4

### 8.6 Ablation Studies → §5

### 8.7 Robustness Tests → §6

### 8.8 Expected Results 🔮 (คาดการณ์ ไม่ใช่ผล)

> ส่วนนี้คือ **สิ่งที่คาดไว้ก่อนรัน** พร้อมเหตุผล เพื่อให้หลังรันเทียบได้ว่าผลตรงหรือแย้งกับความคาดหมาย — ห้ามคัดลอกช่องนี้ไปเป็นผล

| RQ | Expected 🔮 | เหตุผล | ถ้าผลแย้ง แปลว่า |
|---|---|---|---|
| RQ1 | landmark (R) ≈ I3D (X10) ภายใน ±3 pp; compute ต่างกันมาก | ผลสองโมเดลบน TSL-ONE-S split เดิมห่างกัน 1.14 pp 📄 ✅ | ถ้า I3D ดีกว่าชัด → ข้อมูลที่ landmark ไม่มี (รูปมือละเอียด/สีหน้า) สำคัญใน TSL → เลื่อน Hybrid เป็นหลัก |
| RQ2 | late fusion ดีขึ้นเล็กน้อย (≤ 3 pp) และกระจุกในคู่คำที่รูปมือต่างกัน | ISLR ทั้งสอง modality ใกล้กัน; ensemble มักช่วย 📄 | ถ้าไม่ช่วย → RGB ไม่มีข้อมูลเสริมที่โมเดลใช้ได้ด้วยข้อมูลขนาดนี้ |
| RQ3 | X3 (mean pool) แย่กว่า R ชัดเจน; BiGRU ≈ Transformer | ภาษามือมีลำดับเวลาเป็นสาระ; ข้อมูล ~3.4K คลิปไม่มากพอให้ Transformer ได้เปรียบ | ถ้า mean pool ใกล้ R → คำใน TSL-ONE-S แยกกันได้ด้วยตำแหน่ง/รูปมือเฉลี่ย (คำง่ายเกิน) |
| RQ4 | pretrained ช่วยชัดที่ 3–7 คน, ช่วยน้อยที่ใช้ทุกคน; ASL-pretrain อาจช่วยน้อยกว่าที่คิดเพราะภาษาต่างกัน | 📚 transfer learning ช่วยตอนข้อมูลน้อย | ถ้า ASL-pretrain ทำให้แย่ลง → negative transfer; ใช้ self-supervised บน TSL แทน |
| RQ5 | random/original split ได้ Top-1 สูงกว่า SI; ช่องว่างของ I3D > landmark | ผู้แสดงคนเดียวกันอยู่ทั้ง train และ test ใน split เดิม ✅; I3D overfit ภาพมากกว่า 📄 ✅ | ถ้าไม่ต่าง → ความแตกต่างระหว่างคนส่งผลน้อยกว่าที่คิด (ข้อค้นพบที่ดี) |
| RQ6 | C54 ≈ ไม่มีหน้า (6 จุดน้อยเกิน); X71 ดีขึ้นเล็กน้อยในคำบางกลุ่ม | ในงาน ISLR คำศัพท์ส่วนใหญ่ของ TSL-ONE-S (คน สถานที่ ตัวอักษร) เป็นท่ามือเป็นหลัก 💭; ผลใหญ่ของ TwoStream อยู่ใน CSLR | ถ้า X71 ช่วยมาก → ต้องใช้ schema ขยายในระบบจริง (แต่ TSL-51 ใช้ไม่ได้) |
| RQ7 | segment-then-classify ใกล้ CTC ใน `with_space`; CTC ดีกว่าใน `no_space` | กฎพักมือใช้ไม่ได้เมื่อไม่มีจังหวะพัก | ถ้า segment ดีกว่าทั้งคู่ → ข้อมูลประโยคน้อยเกินให้ CTC เรียน |
| RQ8 | rule ≈ LLM บน chrF กับ gloss จริง; LLM fluency สูงกว่าแต่มี hallucination บ้าง; ทั้งคู่ตกเมื่อใช้ gloss ที่ทำนาย | โดเมนปิด 13 กลุ่มไวยากรณ์ 📄 README | ถ้า LLM hallucinate บ่อย → ใช้ rule เป็นค่าเริ่มต้น |
| RQ9 | ช่วย TSL-51 มากกว่า TSL-ONE-S | TSL-51 ต้นฉบับน้อยกว่า (1,155 vs 4,152 คลิป ✅) | ถ้าทำให้ TSL-ONE-S แย่ลง → label space ต่างกันจนแย่ง capacity |

### 8.9 Actual Results

**รอบที่ 1 — baseline (9 พ.ย. 2026)** · Apple M5 (BiGRU บน MPS) · MediaPipe 0.10.21 · schema C54, T = 64 · สคริปต์ `scripts/run_baselines.py`, ผลดิบ `results/*.json`
ค่า = mean ± std ข้าม fold หรือ seed (%) · **รอบนี้ยังไม่ครบ protocol**: TSL-ONE-S รันแค่ 3 จาก 10 fold และ 1 seed (ดูรอบที่ 2 สำหรับ 10 fold)

| ID | Protocol | Runs | Top-1 | Top-5 | Macro-F1 | หมายเหตุ |
|---|---|---|---|---|---|---|
| X0-LR | TSL-ONE-S original split | 1 | 52.7 | 81.5 | 50.8 | |
| X0-RF | TSL-ONE-S original split | 1 | 37.3 | 66.7 | 34.6 | |
| X1 BiGRU | TSL-ONE-S original split | 1 | **81.2** | 94.7 | 79.9 | ใกล้ SPOTER 80.83 / I3D 79.69 ในบทความ → ผ่านจุดตรวจ §8.11 |
| X0-LR | TSL-ONE-S SI fold 0–2 | 3 | 53.1 ± 6.6 | 79.9 ± 7.4 | 50.4 ± 7.2 | |
| X0-RF | TSL-ONE-S SI fold 0–2 | 3 | 48.4 ± 5.3 | 74.4 ± 3.1 | 44.1 ± 6.8 | |
| X1 BiGRU | TSL-ONE-S SI fold 0–2 | 3 | **78.9 ± 5.0** | 93.9 ± 2.8 | 77.2 ± 6.0 | ต่อ fold: 86.0 / 74.5 / 76.4; best epoch 70–86 จาก 100 (อาจยังฝึกไม่สุด) |
| X0-LR / X0-RF / X1 BiGRU | TSL-ONE-S SI fold 4 (test = signer 03, 08, 22) | 1 each | 46.6 / 40.1 / **73.1** | 73.8 / 69.2 / 90.4 | 46.7 / 39.0 / 72.1 | test 416 คลิปชุดเดียวกับ split B ของ Sign2Say-base_line (เพื่อน): SPOTER ได้ 66.11 / 89.42 / 63.8 — val ต่างกัน (เพื่อน 11, 25, 29; เรา 02, 25, 29), seed เดียวทั้งคู่ |
| **R S2S-LTR** | TSL-ONE-S SI fold 4 | 1 | **80.5** | 92.1 | 78.5 | 3.28M params, best epoch 119/150 (15 นาทีบน M5); BiGRU รันซ้ำได้ 73.3 / 91.1 / 72.2 |
| | ↳ paired McNemar บน 416 คลิปเดียวกัน | | | | | S2S vs BiGRU: ถูกเฉพาะ S2S 50 คลิป / เฉพาะ BiGRU 20, p = 0.0004 · S2S vs SPOTER (เพื่อน): 78 / 18, p < 0.0001 · BiGRU vs SPOTER: 71 / 41, p = 0.006 — SPOTER ใช้ val คนละกลุ่ม |
| **R S2S-LTR** | TSL-51 expert → user_sign | 1 (seed 0) | **72.9** | 94.7 | 70.7 | BiGRU seed 0 ได้ 54.3 (3 seeds: 59.4 ± 3.9) |
| X0-LR | TSL-51 expert → user_sign | 1 | 33.8 | 68.7 | 28.3 | val (ผู้แสดงเดิม) 95.1 |
| X0-RF | TSL-51 expert → user_sign | 1 | 38.8 | 75.7 | 33.5 | val 92.9 |
| X1 BiGRU | TSL-51 expert → user_sign | 3 seeds | **59.4 ± 3.9** | 88.1 ± 1.9 | 56.5 ± 3.6 | val 95.7 |
| X0-LR | TSL-51 LOSO (test expert_01) | 1 | 27.0 | 54.1 | 17.2 | |
| X0-RF | TSL-51 LOSO | 1 | 42.0 | 75.9 | 32.3 | |
| X1 BiGRU | TSL-51 LOSO | 3 seeds | 39.5 ± 5.7 | 77.7 ± 1.2 | 28.8 ± 5.5 | val 91.4 |
| X2, X3–X15 | – | – | — | — | — | ยังไม่ได้รัน (X16 ดูรอบที่ 4) |

**รอบที่ 2 — 10 fold signer-independent (10 พ.ย. 2026)** · seed 0 · ทุก fold ใช้ split ใน `splits/`, ผลดิบ `results/tsl1s_si_s2s.json`, `results/tsl1s_si_bigru.json`, สรุปด้วย `scripts/summarize_10fold.py`

| ID | Protocol | Runs | Top-1 | Top-5 | Macro-F1 | หมายเหตุ |
|---|---|---|---|---|---|---|
| **R S2S-LTR** | TSL-ONE-S SI 10 fold | 10 folds × 1 seed | **84.8 ± 3.0** | 94.4 ± 1.4 | **83.2 ± 3.8** | ต่อ fold (top-1): 91.9 / 84.7 / 83.4 / 86.1 / 80.5 / 83.4 / 83.6 / 87.3 / 81.5 / 85.1; fold 5 และ 7 ใช้ครบ 150 epoch (best 139, 128) → อาจยังฝึกไม่สุด |
| X1 BiGRU | TSL-ONE-S SI 10 fold | 10 folds × 1 seed | 77.6 ± 3.1 | 92.9 ± 1.8 | 76.2 ± 3.6 | ต่อ fold: 83.3 / 76.3 / 78.0 / 81.6 / 73.3 / 75.7 / 73.8 / 79.6 / 76.1 / 78.4 — fold 0–2 เป็นรอบรันซ้ำ (เพื่อเก็บไฟล์ทำนาย); รอบแรกได้ 86.0 / 74.5 / 76.4 → seed เดียวกันบน MPS ไม่ deterministic, ต่างได้ ±3 pp ต่อ fold |
| | ↳ S2S − BiGRU | | +7.2 pp | | | S2S ดีกว่าทั้ง 10/10 fold, Wilcoxon p = 0.002 · McNemar รวมทั้ง 10 fold (4,148 คลิป): ถูกเฉพาะ S2S 454 / เฉพาะ BiGRU 157, p ≈ 2e-34 → H ของ RQ (โมเดลหลักดีกว่า baseline) ได้รับการสนับสนุน, ยังเป็น seed เดียว |
| X0-LR / X0-RF | TSL-ONE-S SI | 4 folds (0, 1, 2, 4) | 51.5 ± 6.4 / 46.3 ± 5.8 | 78.4 / 73.1 | 49.5 / 42.8 | ไม่ได้รันครบ 10 fold (ห่างจาก deep models มากจนไม่จำเป็น) |

**รอบที่ 3 — Ablation ของ S2S-LTR (10–11 พ.ย. 2026)** · TSL-ONE-S SI fold 0, 2, 4, 6, 8 · seed 0 · เปลี่ยนทีละอย่างจาก base · ผลดิบ `results/tsl1s_*_abl_*.json`, สรุปด้วย `scripts/summarize_ablations.py` · McNemar รวมทุกคลิปของ 5 fold (5 fold ทำให้ Wilcoxon p ต่ำสุดได้แค่ 0.0625)

| Variant | Top-1 | Macro-F1 | Δ top-1 vs base | fold ที่ดีขึ้น | McNemar (ถูกเฉพาะ variant : เฉพาะ base) | สรุป |
|---|---|---|---|---|---|---|
| base S2S-LTR (C54, 54 จุด) | 84.2 | — | — | — | — | อ้างอิง (จากรอบที่ 2) |
| − face (48 จุด) | 85.1 | 84.0 | +0.9 | 3/5 | 80 : 61, p = 0.13 | จุดหน้า 6 จุดไม่ได้ช่วย |
| + lips/eyes/nose/hips (X71) | 85.1 | 83.6 | +0.9 | 3/5 | 103 : 87, p = 0.28 | จุดหน้าเพิ่มก็ไม่ได้ช่วยชัดเจน → RQ6 ไม่ได้รับการสนับสนุนบน TSL-ONE-S |
| − velocity | 83.0 | 81.2 | −1.2 | 0/5 | 69 : 94, p = 0.06 | velocity น่าจะช่วยเล็กน้อย (ไม่ถึงนัยสำคัญ) |
| − valid mask | 84.9 | 83.5 | +0.7 | 2/5 | 91 : 78, p = 0.36 | mask ไม่ได้ช่วย (จุดที่หายเป็น 0 อยู่แล้ว) |
| − part-aware (MLP เดียว) | 83.1 | 82.1 | −1.1 | 0/5 | 100 : 123, p = 0.14 | part-aware น่าจะช่วยเล็กน้อย (ไม่ถึงนัยสำคัญ) |
| random-matched split (RQ5) | 82.6 | 81.3 | −1.6 | 2/5 | test คนละชุด เทียบรายคลิปได้แค่ 25 คลิปที่ซ้ำกัน | ไม่พบ inflation จากการเห็นผู้แสดงใน train บน TSL-ONE-S → H5 ไม่ได้รับการสนับสนุน |

หมายเหตุ: (1) รอบแรกของ "− valid mask" ได้ ~1% เพราะบั๊ก (augmentation คูณพิกัดด้วยช่อง valid ทำให้ input เป็นศูนย์ทั้งหมด) แก้ใน `augment()` ให้หา mask จากพิกัดที่เป็น 0 — ตรงกับช่อง valid ทุกค่าบนทั้งสอง cache, ผล base และ variant อื่นจึงไม่เปลี่ยน; ตัวเลขในตารางเป็นรอบที่แก้แล้ว (2) seed เดียวบน MPS ไม่ deterministic (BiGRU รันซ้ำต่างได้ ±3 pp ต่อ fold) → ผลต่าง ≤ 1.6 pp ทั้งหมดยังสรุปไม่ได้ ต้องใช้หลาย seed ถ้าจะยืนยัน

**รอบที่ 4 — X16 เทรนร่วม TSL-ONE-S + TSL-51 (11 พ.ย. 2026, RQ9)** · S2S-LTR encoder เดียว + 2 head (184 + 52) แต่ละคลิปคิด loss เฉพาะ head ของตัวเอง · TSL-ONE-S SI fold 0, 2, 4, 6, 8 + TSL-51 expert → user_sign (split เดียวกับการเทรนแยก) · เลือก epoch ด้วยค่าเฉลี่ย val macro-F1 สองชุด · สคริปต์ `scripts/run_joint.py`, ผลดิบ `results/joint.json`

| Test set | X16 joint (5 runs) | เทรนชุดเดียว | Δ | การทดสอบ |
|---|---|---|---|---|
| TSL-ONE-S SI (fold 0, 2, 4, 6, 8) | 83.7 (ต่อ fold 90.1 / 86.4 / 82.0 / 84.3 / 75.6) | 84.2 (R, fold เดียวกัน) | −0.5 pp | McNemar ถูกเฉพาะ joint 86 : เฉพาะ single 99, p = 0.38 → ไม่ต่าง |
| TSL-51 user_sign (547 คลิป) | **80.9 ± 2.9** (83.2 / 82.4 / 78.1 / 84.1 / 76.8) | 76.2 ± 1.9 (5 seeds: 72.9 / 76.1 / 78.2 / 77.9 / 75.7) | **+4.7 pp** | Mann-Whitney (5 vs 5) p ≈ 0.06 · McNemar vs single seed 0: 370 : 152, p ≈ 5e-22 |

สรุป: **H9 ได้รับการสนับสนุนเบื้องต้น** — การเทรนร่วมช่วย TSL-51 (ข้อมูลน้อยกว่า) และไม่ทำให้ TSL-ONE-S แย่ลง. ข้อจำกัด: (1) 5 run ของ joint ต่างกันที่ fold ของ TSL-ONE-S ไม่ใช่ seed, ส่วน single TSL-51 ต่างกันที่ seed (ซึ่งเปลี่ยน val split ด้วย) → เทียบแบบไม่จับคู่, p ≈ 0.06 ยังไม่ถึง 0.05 (2) val ของ TSL-51 อิ่มที่ ~100% เร็ว (ผู้แสดงเดียวกับ train) จึงแทบไม่ช่วยเลือก epoch — การเลือกจริงขึ้นกับ val ของ TSL-ONE-S; fold 8 หยุดเร็วที่ epoch 40 และได้ 75.6 (single 81.5) (3) ✅ ตรวจแล้ว (11 พ.ย.): ลำดับจุดหน้า/pose/มือของสอง dataset ตรงกัน และ CSV ของ TSL-51 มาจาก MediaPipe ที่ให้ผลเท่ากับ 0.10.21 (ดู Data Pipeline D2) → ผลนี้ไม่ได้มาจาก schema ไม่ตรงกัน (4) MPS ไม่ deterministic

**รอบที่ 5 — Robustness (11 พ.ย. 2026, §6)** · R S2S-LTR vs X1 BiGRU, TSL-ONE-S SI **fold 4** (test 416 คลิป, signer 03, 08, 22), seed 0, โมเดลที่บันทึกไว้ใน `results/models/` · สคริปต์ `scripts/run_robustness.py`, ผลดิบ `results/robustness.json` · ค่า = top-1 (%) · ทุกเงื่อนไขใช้ test ชุดเดียวกัน แต่เป็น fold เดียว seed เดียว → ผลต่าง < ~4 pp ระหว่างเงื่อนไขอาจเป็น noise (L2)

ก) รบกวนที่ landmark (ไม่รัน MediaPipe ใหม่)

| ปัจจัย | ระดับ | R S2S-LTR | X1 BiGRU |
|---|---|---|---|
| clean | – | **80.5** | **73.3** |
| หมุนในระนาบ | −30° / −15° / +15° / +30° | 71.6 / 78.8 / 81.2 / 79.8 | 58.4 / 71.2 / 72.1 / 61.1 |
| ลด frame rate | 1/2 / 1/3 / 1/4 | 80.5 / 80.5 / 81.5 | 73.1 / 71.4 / 72.4 |
| ความเร็ว | 0.5× / 0.75× / 1.25× / 1.5× / 2× | 80.5 / 81.0 / 80.0 / 81.0 / 81.2 | 72.4 / 72.4 / 73.6 / 74.3 / 72.4 |
| time warp (ไม่สม่ำเสมอ) | 0.2 / 0.4 / 0.6 | 80.0 / 79.3 / 78.8 | 72.6 / 71.4 / 71.2 |
| jitter (σ × ความกว้างไหล่) | 0.01 / 0.02 / 0.05 / 0.1 | 80.0 / 78.8 / 75.2 / 63.2 | 73.3 / 73.6 / 70.7 / 58.2 |
| มือหายรายเฟรม | p = 0.1 / 0.2 / 0.3 / 0.5 | 80.5 / 80.3 / 80.0 / 77.6 | 72.4 / 72.4 / 71.4 / 64.7 |
| บังมือขวาต่อเนื่อง | 10 / 20 / 30 / 40% ของคลิป | 80.3 / 76.9 / 74.0 / 67.3 | 70.2 / 61.5 / 55.3 / 47.8 |
| บังทั้งสองมือต่อเนื่อง | 10 / 20 / 30 / 40% | 79.8 / 76.4 / 71.6 / 65.1 | 69.7 / 60.8 / 53.6 / 44.2 |
| กลับซ้าย-ขวา (จำลองคนถนัดซ้าย) | mirror + สลับมือ | **17.1** | **13.9** |

ข) รบกวนที่วิดีโอ แล้วสกัด MediaPipe ใหม่ (`dataset/features/landmarks/tsl1s_robust/`) · อัตราตรวจเจอมือ = สัดส่วนเฟรมที่มีมือ (clean: ซ้าย 0.75 / ขวา 0.85)

| ปัจจัย | ระดับ | R S2S-LTR | X1 BiGRU | มือซ้าย / ขวา |
|---|---|---|---|---|
| ความสว่าง | 0.4× / 0.6× / 1.6× | 78.1 / 79.1 / 81.5 | 66.6 / 70.9 / 73.6 | 0.72/0.78 · 0.74/0.83 · 0.75/0.85 |
| ความละเอียด | 0.5× / 0.33× | 80.0 / 79.8 | 73.8 / 73.1 | 0.75/0.86 · 0.75/0.85 |
| เบลอ | Gaussian k = 9 | 78.1 | 72.6 | 0.74/0.84 |

ค) แยกตามผู้แสดง (รวมผล test ของทั้ง 10 fold, 29 คน, รอบที่ 2)

| โมเดล | mean | min (คน) | median | max (คน) | std ข้ามคน |
|---|---|---|---|---|---|
| R S2S-LTR | 84.3 | 56.9 (28) | 85.1 | 95.7 (12) | 8.1 |
| X1 BiGRU | 76.2 | 58.8 (28) | 77.2 | 87.4 (15) | 7.9 |

S2S ดีกว่า BiGRU ใน 28/29 คน; ความแม่นยำรายคนของสองโมเดลสัมพันธ์กัน r = 0.78 → ความยากมาจากตัวผู้แสดงมากกว่าตัวโมเดล; คนที่ยากสุด 4 คนของ S2S: 28, 10, 21, 19

สรุป
- **ความเร็ว / frame rate ไม่มีผลเลย** (±1 pp) เพราะ pipeline resample ทุกคลิปเป็น T = 64 อยู่แล้ว ส่วน time warp แบบไม่สม่ำเสมอลดลงแค่ ~2 pp → ไม่ใช่จุดอ่อน; ตอบคำถามใน §6 ว่า augment 0.8–1.25 พอ
- **การบังมือต่อเนื่องคือจุดอ่อนหลัก**: บังทั้งสองมือ 40% → S2S −15 pp, BiGRU −29 pp; ในขณะที่มือหายแบบสุ่มรายเฟรม (p = 0.5) ลดน้อยกว่ามาก (−3 / −9 pp) → โมเดลเติมช่องว่างสั้น ๆ ได้ แต่ไม่ได้ถ้าหายเป็นช่วงยาว
- **มุมกล้อง**: S2S ทน ±15° (−2 pp) แต่ BiGRU ตกที่ ±30° ถึง 12–15 pp; ผลของ S2S ไม่สมมาตร (−30° ตก 9 pp, +30° ตก 1 pp)
- **แสง / ความละเอียด / เบลอ**: กระทบน้อย (S2S ≤ 2.4 pp) แม้แสงน้อย 0.4× ทำให้ตรวจเจอมือขวาลดจาก 0.85 → 0.78; BiGRU ไวต่อแสงน้อยกว่า S2S (−6.7 vs −2.4 pp)
- **ผู้ใช้ถนัดซ้ายใช้ไม่ได้** (17%) ตามที่คาดใน §6 — ผู้แสดงทุกคนถนัดขวา → ระบบจริงต้องมี flip augmentation หรือให้ผู้ใช้เลือกมือถนัดแล้ว mirror input ก่อนเข้าโมเดล
- **S2S ทนกว่า BiGRU ในทุกปัจจัย** โดยช่องว่างกว้างขึ้นเมื่อรบกวนหนัก (clean +7 pp → บังมือ 40% +21 pp)
- ข้อจำกัด: fold เดียว, seed เดียว, ยังไม่มี X10 (RGB) / X12 ให้เทียบ H5 ด้าน robustness; รบกวนทั้งหมดเป็นแบบสังเคราะห์ (L5); ยังไม่ได้ทดสอบฉากหลัง (ต้องใช้ segmentation) และ hearing status / ภูมิภาค (ต้องมีข้อมูล Table 8)

**รอบที่ 6 — CSLR บนประโยค TSL-51 (11 พ.ย. 2026, RQ7 / C1, C2)** · 252 ประโยค (user_sentence, ผู้แสดงคนเดียว, 31 sign_id) · cache `dataset/cache/tsl51_sentences.npz` จาก `scripts/build_sentence_cache.py` · สคริปต์ `scripts/run_cslr.py`, ผลดิบ `results/cslr.json` + `results/preds/cslr/`, สรุปด้วย `scripts/summarize_cslr.py` · ค่า = WER / SER บน test (%), mean ± std ข้าม 3 seed (C2 ไม่มีการเทรนแบบสุ่ม จึงรัน 1 ครั้ง)

Protocol (Data Pipeline §5)
- **S-rep** (`splits/tsl51_sentence_rep.json`, 84 / 84 / 84): ประโยคแม่แบบเดิม แต่เป็นการแสดงครั้งใหม่ → วัดว่าจำประโยคที่เคยเห็นได้ไหม
- **S-template** (`splits/tsl51_sentence_template.json`, 186 / 21 / 45): ลำดับคำของ test (12 ลำดับ) และ val (6) ไม่อยู่ใน train แต่ทุกคำมีใน train → วัดการประกอบคำเป็นประโยคใหม่

Models
- **C1 CTC**: part-aware encoder (เหมือน R) → Conv1d stride 2 × 2 (ลดเวลา 4×) → Transformer 4 ชั้น → CTC, 3.88M params, 100 epoch, เลือก checkpoint ด้วย val WER · ตัวแปร: encoder เริ่มจาก X16 (fold 0) หรือ scratch, และมี / ไม่มี **ประโยคสังเคราะห์** (300 ประโยค/epoch จากการต่อคลิป user_sign ของผู้แสดงคนเดียวกัน, ไม่ใช้ลำดับคำของ val/test)
- **C2 segment → classify**: sliding window บนโมเดลคำเดี่ยว (head TSL-51 ของ X16, จำกัดเหลือ 31 คำ + null) → threshold → รวมช่วงที่ติดกัน; window / τ / ความยาวขั้นต่ำ จูนบน val ของแต่ละ protocol · `x16` = ไม่เคยเห็นผู้แสดงประโยคนี้, `x16ft` = fine-tune 30 epoch บนคลิปคำเดี่ยวทั้งหมดของ TSL-51 (รวม user_sign ของผู้แสดงคนนี้)

| Protocol | Model | Runs | val WER | **test WER** | SER | S / D / I | no_space | with_space |
|---|---|---|---|---|---|---|---|---|
| S-rep | C1 X16 + synth (หลัก) | 3 | 3.1 ± 0.7 | **5.1 ± 0.6** | 15.1 ± 1.5 | 0.7 / 0.9 / 3.5 | 5.5 ± 0.8 | 1.2 ± 1.7 |
| S-rep | C1 scratch + synth | 3 | 4.3 ± 1.3 | 5.1 ± 0.3 | 14.7 ± 2.0 | 0.5 / 1.8 / 2.8 | 5.0 ± 0.3 | 6.2 ± 1.7 |
| S-rep | C1 X16 ไม่มี synth | 3 | 1.5 ± 0.6 | **3.5 ± 1.3** | 11.1 ± 3.4 | 0.5 / 1.7 / 1.3 | 3.2 ± 1.0 | 6.2 ± 4.6 |
| S-rep | C2 X16 | 1 | 38.8 | 42.9 | 86.9 | 15.2 / 13.5 / 14.2 | 44.7 | 25.9 |
| S-rep | C2 X16-ft | 1 | 25.3 | 30.8 | 69.0 | 6.2 / 12.1 / 12.5 | 30.9 | 29.6 |
| S-template | C1 X16 + synth (หลัก) | 3 | 1.9 ± 0.7 | **11.8 ± 4.8** | 28.9 ± 7.9 | 2.2 / 4.7 / 4.9 | 11.8 ± 4.8 | – |
| S-template | C1 scratch + synth | 3 | 7.4 ± 1.3 | 13.6 ± 3.0 | 34.8 ± 5.8 | 3.8 / 6.2 / 3.6 | 13.6 ± 3.0 | – |
| S-template | C1 X16 ไม่มี synth | 3 | 2.3 ± 1.3 | 36.4 ± 7.6 | 67.4 ± 5.8 | 23.8 / 8.9 / 3.8 | 36.4 ± 7.6 | – |
| S-template | C2 X16 | 1 | 45.8 | 39.3 | 82.2 | 18.0 / 15.3 / 6.0 | 39.3 | – |
| S-template | C2 X16-ft | 1 | 31.9 | 27.3 | 80.0 | 5.3 / 18.0 / 4.0 | 27.3 | – |

ต่อ seed (test WER): S-rep C1 X16 + synth 4.5 / 5.9 / 4.8, ไม่มี synth 3.1 / 2.1 / 5.2 · S-template C1 X16 + synth 5.3 / 13.3 / 16.7, scratch 16.0 / 15.3 / 9.3, ไม่มี synth 37.3 / 26.7 / 45.3 · C2 hyperparameter ที่เลือก: X16 rep W60 τ0.2 run5, template W75 τ0.35 run2; X16-ft rep W60 τ0.5 run3, template W45 τ0.2 run8 · WER ระดับคำ (รวม variant) เท่ากับ WER ระดับ sign_id ทุก run → ไม่มีกรณีทายผิด variant ของคำเดียวกัน

สรุป
- **H7 ไม่ได้รับการสนับสนุน**: segment → classify (C2) แย่กว่า CTC มากทั้งสองชนิดประโยค — แม้ใน `with_space` (C2 26–30% vs C1 1–6%) ซึ่ง H7 คาดว่าจะใกล้กัน. C2 แบบ X16 เดิมดีขึ้นมากใน `with_space` (25.9 vs 44.7) ตามทิศที่คาด แต่ห่างจาก CTC มาก; แบบ fine-tune แล้วไม่ต่างกัน (29.6 vs 30.9). ข้อจำกัดสำคัญ: test ของ `with_space` มีแค่ **8 ประโยค (27 คำ)** ใน S-rep และ **ไม่มีเลย** ใน S-template → เปรียบเทียบสองชนิดประโยคได้หยาบมาก
- **ลำดับคำใหม่ยากกว่าราว 2–3×** (S-template 11.8 vs S-rep 5.1) และแปรปรวนข้าม seed สูง (5.3–16.7) เพราะ test มีแค่ 12 ลำดับคำ; val WER (1.9) ต่ำกว่า test มาก → val 6 ลำดับเล็กเกินจะเลือก checkpoint ได้น่าเชื่อถือ
- **ประโยคสังเคราะห์คือปัจจัยที่ใหญ่ที่สุดสำหรับลำดับคำใหม่**: S-template 36.4 → 11.8 (substitution 23.8 → 2.2: ไม่มี synth โมเดลจำลำดับคำของ train แล้วทายคำตามลำดับที่เคยเห็น). แต่ใน S-rep ไม่มี synth กลับดีกว่าเล็กน้อย (3.5 vs 5.1, ไม่ชัดเจนด้วย 3 seed) → synth ทำให้ประโยคที่เคยเห็นเสียความแม่นไปเล็กน้อย แลกกับการทั่วไปได้
- **encoder จาก X16 ช่วยน้อย**: S-template 11.8 vs scratch 13.6 (ความแปรปรวนซ้อนกัน), S-rep เท่ากัน; ข้อดีที่เห็นชัดคือ **เทรนเร็วขึ้น ~3×** (S-rep 213–303 s vs 790–902 s) และ val WER ต่ำกว่า
- **C2 อ่อนเพราะช่วงเปลี่ยนท่า**: ลบ + แทรกรวม ~21–28% (ตัดคำหาย / นับช่วงเปลี่ยนท่าเป็นคำ) ในขณะที่ CTC เรียนช่วงเปลี่ยนท่าจากประโยคจริงได้; fine-tune บนผู้แสดงคนนี้ลด substitution 15.2 → 6.2 แต่ไม่ลด deletion/insertion
- ข้อจำกัด (L3): ทุกประโยคมาจาก **ผู้แสดงคนเดียว** และ user_sign ที่ใช้สังเคราะห์ก็เป็นคนเดียวกัน → ผลทั้งหมดเป็น signer-dependent, ไม่บอกการใช้กับคนอื่น; คำศัพท์ 31 คำ, ประโยคยาว 3–6 คำ

**รอบที่ 7 — Efficiency (11 พ.ย. 2026, §7)** · Apple M5 (macOS 26), PyTorch MPS / CPU, รันตอนเครื่องว่าง · batch 1, warm-up 20, วัด 200 รอบ, median (p95) · สคริปต์ `scripts/run_efficiency.py`, ผลดิบ `results/efficiency.json` · input ต่อคลิป = 64 เฟรม × 54 จุด × 7 ช่อง; C1 = 300 เฟรม (ประโยค 10 วินาทีที่ 30 fps)

| โมเดล | Params | ขนาด fp32 / fp16 | Latency CPU | Latency MPS | หน่วยความจำตอนเทรน (batch 32) | เวลาเทรน |
|---|---|---|---|---|---|---|
| R S2S-LTR | 3.28M | 12.5 / 6.3 MB | 1.31 (1.33) ms | 1.38 (2.27) ms | 249 MB | 13 นาที/fold (median 9 fold, 139 epoch) |
| X1 BiGRU | 2.16M | 8.3 / 4.1 MB | 2.33 (2.36) ms | 7.55 (7.76) ms | 68 MB | 24 นาที/fold (median, 88 epoch) |
| X16 joint | 3.29M | 12.6 / 6.3 MB | 1.32 (1.35) ms | 1.34 (1.45) ms | – | 27 นาที (fold 0) |
| C1 CTC | 3.88M | 14.8 / 7.4 MB | 1.94 (2.02) ms | 1.82 (2.08) ms | – | 4–15 นาที/run |

| ขั้น pipeline | เวลา |
|---|---|
| MediaPipe Holistic 0.10.21 (Python, CPU / XNNPACK), วิดีโอ TSL-ONE-S 448×522 | 23.7 ms/เฟรม (p95 25.9) = **42 fps** |
| MediaPipe ที่ 640×480 (ขนาด webcam) | 23.1 ms/เฟรม (p95 23.7) = 43 fps |
| preprocess (เติมช่องว่าง + normalize + resample + feature) ต่อคลิป 69 เฟรม | 0.79 ms |
| R S2S-LTR ต่อคลิป (CPU) | 1.31 ms |

สรุป
- **ผ่าน NFR2 (≥ 20 fps) ด้วย ~2×**: คอขวดคือ MediaPipe (~24 ms/เฟรม) ซึ่งรันไปพร้อมกับการทำท่า; หลังจบท่า เหลือแค่ preprocess + โมเดล ≈ **2 ms** ต่อคำ → ผู้ใช้ไม่รู้สึกถึง delay ของโมเดล
- S2S-LTR เร็วกว่า BiGRU บน CPU (1.3 vs 2.3 ms) และบน MPS มาก (1.4 vs 7.6 ms) แม้มี params มากกว่า เพราะ Transformer ขนานข้ามเวลาได้ ส่วน GRU ต้องคำนวณทีละเฟรม; ทุกโมเดล < 15 MB fp32 → ใส่ในเว็บ/มือถือได้
- BiGRU ใช้หน่วยความจำตอนเทรนน้อยกว่า (68 vs 249 MB) แต่เทรนนานกว่า (ต่อ fold) เพราะช้ากว่าต่อ epoch
- ข้อจำกัด: วัดบนเครื่องเดียว (M5 ซึ่งเร็วกว่า laptop ทั่วไป) → ต้องวัดซ้ำบน laptop เป้าหมาย; เวลาเทรนจาก log บางรอบรัน 2 งานพร้อมกัน จึงสูงกว่าจริง; หน่วยความจำวัดจาก MPS driver (รวม cache) ไม่ใช่ `max_memory_allocated` ของ CUDA; ยังไม่ได้วัด FLOPs และ int8

### 8.10 Risks and Limitations

| # | Risk / Limitation | ผลกระทบต่อข้อสรุป | การจัดการ |
|---|---|---|---|
| L1 | Compute ไม่พอสำหรับ I3D 10 fold | RQ1/RQ5 ฝั่ง RGB มีความเชื่อมั่นต่ำกว่า | 3 fold × 1 seed, ระบุชัด; ใช้ Colab/เครื่องแล็บ |
| L2 | Test fold เล็ก (~400 คลิป, ~2 คลิป/คำ) | ผลต่าง < 4 pp ใน fold เดียวสรุปไม่ได้; macro-F1 ต่อคำแปรปรวนสูง | รวมผลทุก fold + paired test |
| L3 | CSLR มีผู้แสดงคนเดียว | ผล CSLR ไม่บอกการใช้กับคนอื่น | ระบุเป็น signer-dependent; เก็บประโยคจากคนอื่นถ้าได้ |
| L4 | ไม่มีคำแปลไทยต้นฉบับ (ต้องเขียนเอง) | metric แปลวัดเทียบกับคำแปลของเราเอง | 2 reference + ผู้ใช้ภาษามือตรวจ; รายงาน agreement |
| L5 | Robustness ส่วนใหญ่เป็นสังเคราะห์ | อาจไม่ตรงกับสภาพจริง | ใช้ชุดที่เก็บเองเป็นตัวยืนยัน |
| L6 | Hyperparameter tuning ไม่เท่ากันโดยไม่ตั้งใจ | โมเดลที่จูนมากกว่าได้เปรียบ | budget เท่ากัน (12 config บน fold 0) และบันทึกทุก run |
| L7 | Multiple comparisons (หลาย RQ × หลายคู่) | เจอ "นัยสำคัญ" โดยบังเอิญ | Holm–Bonferroni; ตั้ง H ก่อนรัน (เอกสารนี้) |
| L8 | ผลของ ASL-pretrain ถูกเข้าใจผิดว่าเป็นผล ASL | รายงานผิด | รายงานเฉพาะผลบน test ของ TSL; ระบุแหล่ง pretrain |
| L9 | MediaPipe เวอร์ชันต่างจาก TSL-51 | ผลของการรวม dataset ปนกับผลของ extractor | ✅ ตรวจแล้ว: สกัดใหม่ตรงกับ CSV (median ต่าง 1e-4) — ความเสี่ยงนี้ปิดได้ |
| L10 | คำศัพท์ไม่ครอบคลุมการสื่อสารจริง (184 + 51 คำ) | ระบบเป็น proof-of-concept | ระบุขอบเขตในรายงาน |

### 8.11 ลำดับการรัน (🛠)

| ขั้น | งาน | ต้องเสร็จก่อน |
|---|---|---|
| 1 | Pipeline ข้อมูล + Quality checklist Q1–Q16 | – |
| 2 | P0: X0, X1, X2, R บน fold 0 → ตรวจว่าตัวเลขสมเหตุสมผล (X2 บน original split ควรใกล้ 80.83% 📄) | 1 |
| 3 | P0 ครบ 10 fold + RQ5 สำหรับ R | 2 |
| 4 | P1: X3, X4, X6, X7, X10 (3 fold), X12, C1, C2, T1, T2 | 3 |
| 5 | วิเคราะห์ confusion → ตัดสินใจทำ P2 (X13, X15, X16, X8, X5) | 4 |
| 6 | Robustness (§6) บน R, X10, X12 | 4 |
| 7 | Efficiency (§7) บนเครื่องเป้าหมาย | 4 |
| 8 | P3 ถ้ามีเวลา | 5 |

> ✅ ขั้นที่ 2 มีจุดตรวจ: รัน SPOTER บน **original split** แล้วควรได้ใกล้ตัวเลขในบทความ — ถ้าห่างมาก แปลว่า pipeline ข้อมูลของเราต่างจากของผู้เขียน และต้องแก้ก่อนเชื่อผลอื่นทั้งหมด

---

## อ้างอิง

- Chalotonpised et al., TSL-ONE-S, IEEE Access 2026, DOI 10.1109/ACCESS.2026.3670970 — SPOTER/I3D baselines, hyperparameters, เวลาฝึก, ผู้แสดง
- Boháček & Hrúz, SPOTER, WACVW 2022
- Carreira & Zisserman, *Quo Vadis, Action Recognition? (I3D)*, CVPR 2017, arXiv 1705.07750
- Jiang et al., SAM-SLR, CVPRW 2021, arXiv 2103.08833 — ผลแยก modality บน AUTSL
- Chen et al., TwoStream-SLR, NeurIPS 2022, arXiv 2211.01367 — ablation keypoint/video/หน้า-ปาก
- Camgöz et al., Sign Language Transformers, CVPR 2020, arXiv 2003.13830 — Gloss2Text
- Popović, *chrF: character n-gram F-score for automatic MT evaluation*, WMT 2015
- Post, *A Call for Clarity in Reporting BLEU Scores* (sacreBLEU), WMT 2018
- Thammasit & Sompong, วารสาร IT มจพ. 2026 — LSTM/GRU บน MediaPipe (TSL10)
- บทเรียนใน Knowledge Base: Evaluation Metrics (precision/recall/F1, confusion matrix, imbalance), Ch13 Video Processing (RNN/LSTM/GRU, 3D CNN), Transformers for CV, Pre-trained CNN & Transfer Learning

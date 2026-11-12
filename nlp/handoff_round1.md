# ส่งต่อให้คนที่ 2 (รอบที่ 1: สำหรับทำ T2)

ในชุดนี้ `annotations/sentences_for_t2.jsonl` **ซ่อนคำแปลอ้างอิงของ test (12 ลำดับ)** ไว้
เพื่อให้ออกแบบ prompt ได้โดยไม่เห็นเฉลย  ใช้ train (44) เป็นตัวอย่าง few-shot และใช้ val (6) ลองปรับ prompt

ขั้นตอน
1. ทำ T2 บน Typhoon ตามข้อตกลงใน README.md (หัวข้อ "ข้อตกลงสำหรับ T2")
2. รันให้ครบแล้ว **ล็อกผล**: outputs/t2_gold.jsonl และ outputs/t2_ctc_template_C1-x16_seed{0,1,2}.jsonl
   พร้อมบันทึก prompt / ชื่อรุ่น Typhoon / วันที่รัน
3. ส่งไฟล์ t2_* กลับให้คนที่ 1 แล้วจะได้รอบที่ 2: sentences.jsonl ฉบับเต็ม + ผล T1 สำหรับเขียนสคริปต์วัดผล (งาน 11)

แปลงชื่อ sign เป็นคำด้วย t1_rules.word() เพื่อให้ T1/T2 รับ input แบบเดียวกัน
ผลทาย CTC อยู่ที่ cslr/  (ใช้ template_C1-x16_seed0/1/2_test.json เป็นหลัก)

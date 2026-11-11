"""ตัดเฟรม RGB สำหรับ Part C: วิดีโอ -> data/rgb_T16_128/<video_id>.npy  uint8 [16, 128, 128, 3]

- เลือก 16 เฟรมเว้นระยะเท่ากันตลอดคลิป (README_brief: ห้ามตัดเอา 16 เฟรมแรก)
- วิดีโอถูก crop มาแล้วและอัตราส่วนไม่เท่ากัน (W/H 0.5–1.5) -> เติมขอบดำให้เป็นสี่เหลี่ยมจัตุรัส
  (letterbox) ก่อนย่อ จึงไม่ยืดภาพและไม่ตัดมือทิ้ง
- เก็บที่ 128² เพื่อให้ตอนเทรนสุ่ม crop 112² ได้ (ตอนประเมินใช้ center crop)
- รันซ้ำได้: ข้ามไฟล์ที่ทำแล้ว

    python tools/extract_frames.py --limit 5        # ทดสอบ
    python tools/extract_frames.py                  # ทั้งชุด (~3.3 GB)
"""
import argparse
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE / "data" / "videos"
OUT_DIR = BASE / "data" / "rgb_T16_128"
T, SIZE = 16, 128


def letterbox(img, size):
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def extract(path):
    out = OUT_DIR / (path.stem + ".npy")
    if out.exists():
        return path.stem, "skip"
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:                                   # อ่านทั้งคลิป (36–92 เฟรม) เพราะจำนวนเฟรมจาก metadata เชื่อไม่ได้เสมอ
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    if not frames:
        return path.stem, "ไม่มีเฟรม"
    idx = np.linspace(0, len(frames) - 1, T).round().astype(int)
    clip = np.stack([letterbox(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), SIZE) for i in idx])
    tmp = OUT_DIR / (path.stem + ".npy.part")
    with open(tmp, "wb") as f:
        np.save(f, clip)
    os.replace(tmp, out)
    return path.stem, "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(6, cpu_count() - 2)))
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for left in OUT_DIR.glob("*.part"):
        left.unlink()
    paths = sorted(p for p in VIDEO_DIR.iterdir() if p.suffix.lower() == ".mp4")
    if a.limit:
        paths = paths[:a.limit]
    print(f"{len(paths)} videos -> {OUT_DIR} | {T} frames x {SIZE}² | workers {a.workers}", flush=True)
    failed = []
    with Pool(a.workers) as pool:
        for i, (name, st) in enumerate(pool.imap_unordered(extract, paths, chunksize=4), 1):
            if st not in ("ok", "skip"):
                failed.append((name, st))
            if i % 200 == 0 or i == len(paths):
                print(f"{i}/{len(paths)}", flush=True)
    print(f"เสร็จ: {len(list(OUT_DIR.glob('*.npy')))} ไฟล์ | ล้มเหลว {len(failed)} {failed[:5]}")

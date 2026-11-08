"""สกัด pose landmark (MediaPipe Holistic) จากคลิป mp4 -> .npy ขนาด (เฟรม, 150)
150 = 75 จุด x (x, y) = pose 33 + มือซ้าย 21 + มือขวา 21

ใช้ได้ทั้ง Windows และ macOS / Linux  (ต้องใช้ mediapipe==0.10.14 เพราะรุ่นใหม่ไม่มี mp.solutions แล้ว)

    python tools/extract_pose.py --limit 5      # ทดสอบ 5 คลิปแรก
    python tools/extract_pose.py                # ทั้งชุด (รันซ้ำได้ ข้ามไฟล์ที่ทำแล้ว)
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_DIR = BASE / "data" / "videos"
DEFAULT_OUT_DIR = BASE / "data" / "pose_npy"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}   # เทียบแบบไม่สนตัวพิมพ์ (.MP4 บน macOS/Linux)


def lm_to_array(lms, n):
    if lms is None:                      # จับไม่ได้ -> ศูนย์
        return np.zeros((n, 2))
    return np.array([[p.x, p.y] for p in lms.landmark])


def extract(path, out_dir):
    out = out_dir / (path.stem + ".npy")
    if out.exists():                     # ข้ามไฟล์ที่ทำแล้ว (รันต่อได้ถ้าโดนหยุดกลางคัน)
        return path.stem, "skip"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return path.stem, "เปิดวิดีโอไม่ได้"
    frames = []
    with mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5) as h:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            res = h.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frame = np.concatenate([
                lm_to_array(res.pose_landmarks, 33),
                lm_to_array(res.left_hand_landmarks, 21),
                lm_to_array(res.right_hand_landmarks, 21)])
            frames.append(frame.reshape(-1))
    cap.release()
    if not frames:
        return path.stem, "ไม่มีเฟรม"
    # ไฟล์ชั่วคราวลงท้าย .part (ไม่ใช่ .npy) จะได้ไม่ถูกนับปนถ้าโดนปิดกลางคัน
    tmp = out_dir / (path.stem + ".npy.part")
    with open(tmp, "wb") as f:
        np.save(f, np.array(frames))
    os.replace(tmp, out)                 # กันไฟล์เสียถ้าถูกปิดกลางคัน
    return path.stem, "ok"


if __name__ == "__main__":               # จำเป็นบน Windows และ macOS (multiprocessing แบบ spawn)
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="ทดสอบกับ N คลิปแรก")
    ap.add_argument("--workers", type=int,
                    default=max(1, min(6, cpu_count() - 2)))
    ap.add_argument("--video_dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for leftover in args.out_dir.glob("*.part"):    # ไฟล์ชั่วคราวที่ค้างจากรอบก่อน
        leftover.unlink()

    paths = sorted(p for p in args.video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS) \
        if args.video_dir.exists() else []
    if not paths:
        raise SystemExit(f"ไม่พบวิดีโอใน {args.video_dir} — ใส่ --video_dir ให้ถูก")
    if args.limit:
        paths = paths[:args.limit]
    print(f"{len(paths)} videos | {args.video_dir} -> {args.out_dir} | workers = {args.workers}",
          flush=True)

    # ส่ง out_dir เป็นอาร์กิวเมนต์ (worker บน Windows/macOS ไม่เห็นค่าที่ตั้งใน __main__)
    job = partial(extract, out_dir=args.out_dir)
    failed = []
    with Pool(args.workers) as p:
        for i, (name, status) in enumerate(p.imap_unordered(job, paths), 1):
            if status not in ("ok", "skip"):
                failed.append((name, status))
            if i % 20 == 0 or i == len(paths):
                print(f"{i}/{len(paths)}", name, flush=True)

    n_npy = len(list(args.out_dir.glob("*.npy")))
    print(f"\nเสร็จ: มี .npy {n_npy} ไฟล์ใน {args.out_dir}")
    if failed:
        print(f"ล้มเหลว {len(failed)} คลิป:", failed[:10])

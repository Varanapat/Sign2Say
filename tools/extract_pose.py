"""สกัด pose landmark (MediaPipe Holistic) จากคลิป mp4 -> .npy ขนาด (เฟรม, 150)
150 = 75 จุด x (x, y) = pose 33 + มือซ้าย 21 + มือขวา 21
"""
import argparse
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BASE = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE / "TSL-ONE-S"
OUT_DIR = BASE / "pose_npy"


def lm_to_array(lms, n):
    if lms is None:                      # จับไม่ได้ -> ศูนย์
        return np.zeros((n, 2))
    return np.array([[p.x, p.y] for p in lms.landmark])


def extract(path):
    out = OUT_DIR / (path.stem + ".npy")
    if out.exists():                     # ข้ามไฟล์ที่ทำแล้ว (รันต่อได้ถ้าโดนหยุดกลางคัน)
        return path.stem
    cap = cv2.VideoCapture(str(path))
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
    if frames:
        tmp = OUT_DIR / (path.stem + ".tmp.npy")
        np.save(tmp, np.array(frames))
        os.replace(tmp, out)             # กันไฟล์เสียถ้าถูกปิดกลางคัน
    return path.stem


if __name__ == "__main__":               # จำเป็นบน Windows
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="ทดสอบกับ N คลิปแรก")
    ap.add_argument("--workers", type=int,
                    default=max(1, min(6, cpu_count() - 2)))
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    paths = sorted(VIDEO_DIR.glob("*.mp4"))
    if args.limit:
        paths = paths[:args.limit]
    print(len(paths), "videos, workers =", args.workers, flush=True)

    with Pool(args.workers) as p:
        for i, name in enumerate(p.imap_unordered(extract, paths), 1):
            if i % 20 == 0 or i == len(paths):
                print(f"{i}/{len(paths)}", name, flush=True)
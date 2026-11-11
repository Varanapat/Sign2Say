"""ใช้งานจริง: วิดีโอภาษามือ (1 คำ) -> top-k gloss + probability (สำหรับกลุ่ม NLP)

pipeline ที่ดีที่สุดจากการทดลอง (EXPERIMENTS.md): ST-GCN (B3) + hand-local, ensemble 3 seed
  วิดีโอ -> MediaPipe Holistic (C54, ค่าตั้งเดียวกับ cache ของทีม) -> preprocess ของทีม
  (เติมช่องว่าง ≤5 เฟรม, normalize ด้วยไหล่, resample 64 เฟรม) -> x, y, mask + hand-local
  -> ST-GCN 3 ตัว (seed 0, 1, 2) -> เฉลี่ย probability -> top-k

    python -m experiments.predict data/videos/03_01_0001.mp4
    python -m experiments.predict a.mp4 b.mp4 --topk 5 --out preds.json
    python -m experiments.predict data/videos/*.mp4 --device cpu

ผลลัพธ์ (JSON ต่อวิดีโอ):
  {"video": "a.mp4", "top": [{"gloss": "Thailand", "label": 1, "prob": 0.83}, ...], "frames": 58}
หมายเหตุ: โมเดลรู้จักเฉพาะ 184 คำของ TSL-ONE-S (data/label_map.json) และรับวิดีโอที่มี 1 คำต่อคลิป
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.augment import select_features
from experiments.c54_landmarks import preprocess
from experiments.data import CHANNELS, NUM_CLASSES, load_label_map
from experiments.models import build
from experiments.train import get_device

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPTS = [ROOT / "results" / f"B3_hl_fold4_seed{s}" / "best.pth" for s in (0, 1, 2)]
POSE_IDX = [11, 12, 13, 14, 15, 16]          # เหมือน my_work/scripts/extract_tsl1s.py
FACE_IDX = [105, 70, 300, 334, 61, 291]


def _xyz(lms, idx, W, H):
    out = np.full((len(idx), 3), np.nan, np.float32)
    if lms is not None:
        for k, i in enumerate(idx):
            p = lms.landmark[i]
            out[k] = (p.x * W, p.y * H, p.z * W)
    return out


def extract_c54(video_path):
    """วิดีโอ -> [T, 54, 3] พิกัด pixel (NaN = หาย) ด้วยค่าตั้ง MediaPipe เดียวกับ cache ของทีม"""
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"เปิดวิดีโอไม่ได้: {video_path}")
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = []
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            r = hol.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frames.append(np.concatenate([
                _xyz(r.pose_landmarks, POSE_IDX, W, H), _xyz(r.face_landmarks, FACE_IDX, W, H),
                _xyz(r.left_hand_landmarks, range(21), W, H), _xyz(r.right_hand_landmarks, range(21), W, H)]))
    cap.release()
    if not frames:
        raise ValueError(f"ไม่มีเฟรม: {video_path}")
    return np.stack(frames)


def load_models(ckpts, device):
    models = []
    for c in ckpts:
        m, _, _ = build("B3", (64, 54, 5), NUM_CLASSES)
        m.load_state_dict(torch.load(c, map_location="cpu", weights_only=True))
        models.append(m.to(device).eval())
    return models


@torch.no_grad()
def predict_features(models, feats, device):
    """feats [N, 64, 54, 5] -> prob เฉลี่ยของทุกโมเดล [N, 184]"""
    x = torch.from_numpy(feats).to(device)
    return torch.stack([torch.softmax(m(x), 1) for m in models]).mean(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ckpt", nargs="*", default=[str(c) for c in DEFAULT_CKPTS])
    ap.add_argument("--device", default="cpu", help="cpu พอสำหรับ inference (ST-GCN ~ไม่กี่ ms ต่อคลิป)")
    ap.add_argument("--out", help="บันทึกผลเป็นไฟล์ JSON")
    a = ap.parse_args()

    device = get_device(a.device)
    models = load_models(a.ckpt, device)
    gloss_of = {v: k for k, v in load_label_map().items()}
    ch = CHANNELS["xy_mask"]
    results = []
    for v in a.videos:
        try:
            raw = extract_c54(v)
            x = preprocess(raw)                          # [64, 54, 7] หรือ None ถ้าไม่เห็นไหล่เลย
            if x is None:
                raise ValueError("ไม่พบไหล่ในวิดีโอ (normalize ไม่ได้)")
            p = predict_features(models, select_features(x, ch, True)[None], device)[0]
            top = np.argsort(-p)[:a.topk]
            r = {"video": str(v), "frames": int(len(raw)),
                 "top": [{"gloss": gloss_of[int(i)], "label": int(i), "prob": round(float(p[i]), 4)} for i in top]}
        except Exception as e:                           # วิดีโอเสียไม่ควรทำให้ทั้ง batch ล้ม
            r = {"video": str(v), "error": str(e)}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()

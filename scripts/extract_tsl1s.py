"""Extract MediaPipe Holistic landmarks from TSL-ONE-S videos and build the ISLR cache.

Same MediaPipe settings as TSL-51 (det 0.5, track 0.5, model_complexity 1).
Per clip it saves dataset/features/landmarks/tsl1s/{video_id}.npz with
  c54   [T, 54, 3] pixel coords, NaN = missing (TSL-51 schema order, scripts/landmarks.py)
  extra [T, 17, 3] lips 10, eyes 4, nose 1, hips 2  (for the X71 face ablation)
  fps, width, height
Then writes dataset/cache/tsl1s_c54_T64.npz (X, label, signer, group, clip_id).

Face/pose/hand order matches TSL-51 exactly: verified on 15 TSL-51 user_sign clips (1,356
frames) by re-running MediaPipe 0.10.21 on the videos and matching every CSV point to the
nearest landmark. lbrow_outer, lbrow_inner, rbrow_inner, rbrow_outer, mouth_right, mouth_left
= face 105, 70, 300, 334, 61, 291 (>99.7% of frames); pose = 11..16; lh/rh = MediaPipe
left/right hand. Median distance 1e-4 (face) and 1e-3 (wrist) in normalized units, so the
TSL-51 CSVs come from an equivalent MediaPipe build (Data Pipeline D2).

Usage:
  .venv/bin/python scripts/extract_tsl1s.py --workers 8          # resumable
  .venv/bin/python scripts/extract_tsl1s.py --limit 20           # quick test
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmarks import preprocess  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "TSL-ONE-S"
FEAT = ROOT / "dataset" / "features" / "landmarks" / "tsl1s"
CACHE = ROOT / "dataset" / "cache" / "tsl1s_c54_T64.npz"
CACHE_X71 = ROOT / "dataset" / "cache" / "tsl1s_x71_T64.npz"

POSE_IDX = [11, 12, 13, 14, 15, 16]
FACE_IDX = [105, 70, 300, 334, 61, 291]
LIPS = [0, 17, 13, 14, 37, 267, 84, 314, 78, 308]
EYES = [159, 145, 386, 374]
EXTRA_FACE = LIPS + EYES + [1]
EXTRA_POSE = [23, 24]


def _xyz(lms, idx, W, H):
    out = np.full((len(idx), 3), np.nan, np.float32)
    if lms is not None:
        for k, i in enumerate(idx):
            p = lms.landmark[i]
            out[k] = (p.x * W, p.y * H, p.z * W)
    return out


def extract(path):
    import cv2
    import mediapipe as mp

    out_path = FEAT / (path.stem + ".npz")
    if out_path.exists():
        return path.stem, "skip"
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    c54, extra = [], []
    # a fresh Holistic per clip so tracking state never carries over between clips
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as hol:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose = _xyz(r.pose_landmarks, POSE_IDX, W, H)
            face = _xyz(r.face_landmarks, FACE_IDX, W, H)
            lh = _xyz(r.left_hand_landmarks, range(21), W, H)
            rh = _xyz(r.right_hand_landmarks, range(21), W, H)
            c54.append(np.concatenate([pose, face, lh, rh]))
            extra.append(np.concatenate([_xyz(r.face_landmarks, EXTRA_FACE, W, H),
                                         _xyz(r.pose_landmarks, EXTRA_POSE, W, H)]))
    cap.release()
    if not c54:
        return path.stem, "no frames"
    np.savez_compressed(out_path, c54=np.stack(c54), extra=np.stack(extra),
                        fps=fps, width=W, height=H)
    return path.stem, "ok"


def build_cache(schema="c54"):
    labels = {}
    for entry in json.loads((ROOT / "tsl_one_s_dataset_i3d.json").read_text(encoding="utf-8")):
        for inst in entry["instances"]:
            labels[inst["video_id"]] = entry["gloss"]
    X, meta, bad = [], [], []
    for p in sorted(FEAT.glob("*.npz")):
        vid = p.stem
        if vid not in labels:
            # 4 files on disk are not in the JSON, and 4 JSON ids have no file
            # (likely renamed; same finding as the friend's Sign2Say-base_line README 4.6)
            bad.append({"video_id": vid, "reason": "not in split JSON"})
            continue
        z = np.load(p)
        pts = z["c54"] if schema == "c54" else np.concatenate([z["c54"], z["extra"]], 1)  # [T,71,3]
        x = preprocess(pts)
        if x is None:
            bad.append({"video_id": vid, "reason": "no shoulders"})
            continue
        X.append(x)
        meta.append((labels[vid], vid.split("_")[0], vid, "tsl1s:" + vid))
    lab, sig, grp, cid = map(np.array, zip(*meta))
    out = CACHE if schema == "c54" else CACHE_X71
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.stack(X), label=lab, signer=sig, group=grp, clip_id=cid)
    (out.parent / "tsl1s_quarantine.json").write_text(json.dumps(bad, indent=1))
    print(f"cache: {len(X)} clips, {len(set(lab))} glosses, quarantined {len(bad)} -> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--schema", choices=["c54", "x71"], default="c54")
    args = ap.parse_args()
    FEAT.mkdir(parents=True, exist_ok=True)
    if not args.cache_only:
        vids = sorted(VIDEOS.glob("*.mp4"))
        if args.limit:
            vids = vids[:args.limit]
        t0, done, status = time.time(), 0, {}
        with Pool(args.workers) as pool:
            for vid, st in pool.imap_unordered(extract, vids, chunksize=4):
                done += 1
                status[st] = status.get(st, 0) + 1
                if done % 100 == 0 or done == len(vids):
                    el = time.time() - t0
                    print(f"{done}/{len(vids)} {status} {el/60:.1f} min, "
                          f"eta {el/done*(len(vids)-done)/60:.1f} min", flush=True)
    if not args.limit:
        build_cache(args.schema)


if __name__ == "__main__":
    main()

"""Robustness tests (Experiment Design section 6) for R (S2S-LTR) and X1 (BiGRU) on TSL-ONE-S
signer-independent fold 4 test (416 clips, signers 03 / 08 / 22). Models are the ones trained with
--save-models (results/models/tsl1s_si_fold4_{S2S,BiGRU}_seed0.pt); nothing is retrained.

A. Landmark-level perturbations of the raw MediaPipe output, then the normal preprocessing:
   rotation (in-plane, about the image centre), frame rate (keep every k-th frame), uniform
   speed, random time warp, Gaussian jitter (x shoulder width), random per-frame hand
   detection loss, contiguous occlusion of the right / both hands, and a horizontal flip
   (left-handed signer; left/right labels swapped as MediaPipe would).
V. Video-level perturbations with MediaPipe re-run on the altered frames: brightness,
   lower resolution, Gaussian blur. Also reports the hand detection rate per condition.
B. Natural slices: per-signer accuracy over all 10 folds from the saved prediction files.

Writes results/robustness.json and prints tables.
Usage: .venv/bin/python scripts/run_robustness.py [--skip-video] [--workers 6]
"""

import argparse
import csv
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmarks import preprocess  # noqa: E402
from run_baselines import PARTS_C54, RESULTS, ROOT, build_model, metrics  # noqa: E402

RAW = ROOT / "dataset" / "features" / "landmarks" / "tsl1s"
VID = ROOT / "TSL-ONE-S"
ROB = ROOT / "dataset" / "features" / "landmarks" / "tsl1s_robust"
FOLD = 4
HANDS = (slice(12, 33), slice(33, 54))
SWAP = [1, 0, 3, 2, 5, 4, 9, 8, 7, 6, 11, 10] + list(range(33, 54)) + list(range(12, 33))


def test_clips():
    m = json.loads((ROOT / "dataset" / "annotations" / "label_map_tsl1s.json").read_text(encoding="utf-8"))
    out = []
    for e in json.loads((ROOT / "splits" / f"tsl_one_s_signer_indep_fold{FOLD}.json").read_text(encoding="utf-8")):
        for i in e["instances"]:
            if i["split"] == "test" and (RAW / f"{i['video_id']}.npz").exists():
                out.append((i["video_id"], m[e["gloss"]]))
    return out


# ---------------------------------------------------------------- A: landmark perturbations

def rotate(x, W, H, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    y = x.copy()
    dx, dy = x[..., 0] - W / 2, x[..., 1] - H / 2
    y[..., 0], y[..., 1] = W / 2 + c * dx - s * dy, H / 2 + s * dx + c * dy
    return y


def time_warp(x, strength, rng):
    T = len(x)
    knots = np.cumsum(1 + strength * rng.uniform(-1, 1, 5))
    t = np.interp(np.linspace(0, 4, T), np.arange(5), knots - knots[0])
    t = t / t[-1] * (T - 1)
    lo = np.clip(np.floor(t).astype(int), 0, T - 1)
    return x[lo]


def speed(x, f):
    T = len(x)
    n = max(4, int(round(T / f)))
    return x[np.round(np.linspace(0, T - 1, n)).astype(int)]


def occlude(x, frac, which, rng):
    y = x.copy()
    T = len(x)
    n = int(round(frac * T))
    s = rng.integers(0, max(1, T - n + 1))
    for h in which:
        y[s:s + n, h] = np.nan
    return y


def drop_hands(x, p, rng):
    y = x.copy()
    for h in HANDS:
        y[rng.random(len(x)) < p, h] = np.nan
    return y


def jitter(x, sigma, rng):
    sw = np.nanmedian(np.linalg.norm(x[:, 0, :2] - x[:, 1, :2], axis=-1))
    return x + rng.normal(0, sigma * sw, x.shape).astype(np.float32)


def flip(x, W):
    y = x[:, SWAP].copy()
    y[..., 0] = W - y[..., 0]
    return y


CONDITIONS = (
    [("clean", "none", 0, lambda x, W, H, r: x)] +
    [("rotation", f"{d:+d}°", d, (lambda d: lambda x, W, H, r: rotate(x, W, H, d))(d)) for d in (-30, -15, 15, 30)] +
    [("frame rate", f"1/{k}", k, (lambda k: lambda x, W, H, r: x[::k])(k)) for k in (2, 3, 4)] +
    [("speed", f"{f}x", f, (lambda f: lambda x, W, H, r: speed(x, f))(f)) for f in (0.5, 0.75, 1.25, 1.5, 2.0)] +
    [("time warp", f"{s}", s, (lambda s: lambda x, W, H, r: time_warp(x, s, r))(s)) for s in (0.2, 0.4, 0.6)] +
    [("jitter", f"σ={s}", s, (lambda s: lambda x, W, H, r: jitter(x, s, r))(s)) for s in (0.01, 0.02, 0.05, 0.1)] +
    [("hand drop", f"p={p}", p, (lambda p: lambda x, W, H, r: drop_hands(x, p, r))(p)) for p in (0.1, 0.2, 0.3, 0.5)] +
    [("occlude right hand", f"{int(f*100)}%", f, (lambda f: lambda x, W, H, r: occlude(x, f, [HANDS[1]], r))(f))
     for f in (0.1, 0.2, 0.3, 0.4)] +
    [("occlude both hands", f"{int(f*100)}%", f, (lambda f: lambda x, W, H, r: occlude(x, f, HANDS, r))(f))
     for f in (0.1, 0.2, 0.3, 0.4)] +
    [("flip (left-handed)", "mirrored", 1, lambda x, W, H, r: flip(x, W))]
)


# ---------------------------------------------------------------- V: video perturbations

VIDEO_CONDS = [("brightness", "0.4x", ("bright", 0.4)), ("brightness", "0.6x", ("bright", 0.6)),
               ("brightness", "1.6x", ("bright", 1.6)), ("resolution", "0.5x", ("scale", 0.5)),
               ("resolution", "0.33x", ("scale", 1 / 3)), ("blur", "k=9", ("blur", 9))]


def _extract(job):
    import cv2
    import mediapipe as mp
    from extract_tsl1s import FACE_IDX, POSE_IDX, _xyz
    vid, kind, val, out_dir = job
    out = Path(out_dir) / f"{vid}.npz"
    if out.exists():
        return
    cap = cv2.VideoCapture(str(VID / f"{vid}.mp4"))
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = []
    with mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as hol:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if kind == "bright":
                f = cv2.convertScaleAbs(f, alpha=val, beta=0)
            elif kind == "scale":
                f = cv2.resize(f, None, fx=val, fy=val, interpolation=cv2.INTER_AREA)
            elif kind == "blur":
                f = cv2.GaussianBlur(f, (val, val), 0)
            r = hol.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            frames.append(np.concatenate([_xyz(r.pose_landmarks, POSE_IDX, W, H), _xyz(r.face_landmarks, FACE_IDX, W, H),
                                          _xyz(r.left_hand_landmarks, range(21), W, H),
                                          _xyz(r.right_hand_landmarks, range(21), W, H)]))
    cap.release()
    np.savez_compressed(out, c54=np.stack(frames), width=W, height=H)


# ---------------------------------------------------------------- evaluation

def load_models(device):
    import torch
    models = {}
    for kind in ("S2S", "BiGRU"):
        m = build_model(kind, 184, PARTS_C54).to(device)
        m.load_state_dict(torch.load(RESULTS / "models" / f"tsl1s_si_fold{FOLD}_{kind}_seed0.pt", map_location=device))
        models["R S2S-LTR" if kind == "S2S" else "X1 BiGRU"] = m.eval()
    return models


def evaluate(models, X, y, device):
    import torch
    out = {}
    Xt = torch.tensor(X, device=device)
    for name, m in models.items():
        with torch.no_grad():
            p = np.concatenate([torch.softmax(m(Xt[i:i + 256]), -1).cpu().numpy() for i in range(0, len(X), 256)])
        out[name] = metrics(y, p, 184)
    return out


def hand_rate(raws):
    lh = np.mean([(~np.isnan(x[:, 12, 0])).mean() for x in raws])
    rh = np.mean([(~np.isnan(x[:, 33, 0])).mean() for x in raws])
    return float(lh), float(rh)


def prep(raw):
    x = preprocess(raw)
    return x if x is not None else np.zeros((64, 54, 7), np.float32)   # no shoulders: counts as an error


def per_signer():
    """Top-1 per test signer over all 10 folds (B: natural slices)."""
    res = {}
    for model in ("S2S", "BiGRU"):
        acc = {}
        for k in range(10):
            p = RESULTS / "preds" / f"tsl1s_si_fold{k}_{model}_seed0_test.csv"
            for row in list(csv.reader(open(p)))[1:]:
                acc.setdefault(row[0][:2], []).append(int(row[1]) == int(np.argmax(np.array(row[2:], float))))
        res[model] = {s: float(np.mean(v)) * 100 for s, v in sorted(acc.items())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clips = test_clips()
    y = np.array([c[1] for c in clips])
    raw = [np.load(RAW / f"{v}.npz") for v, _ in clips]
    models = load_models(device)
    rows = []

    print(f"fold {FOLD} test: {len(clips)} clips")
    for factor, level, value, fn in CONDITIONS:
        rng = np.random.default_rng(0)
        P = [fn(z["c54"].astype(np.float32), float(z["width"]), float(z["height"]), rng) for z in raw]
        res = evaluate(models, np.stack([prep(p) for p in P]), y, device)
        lh, rh = hand_rate(P)
        rows.append({"type": "landmark", "factor": factor, "level": level, "value": value,
                     "hand_rate_left": lh, "hand_rate_right": rh,
                     **{k: v["top1"] * 100 for k, v in res.items()}})
        print(f"  {factor:20s} {level:9s} " + "  ".join(f"{k} {v['top1']*100:5.1f}" for k, v in res.items()), flush=True)

    if not args.skip_video:
        for factor, level, (kind, val) in VIDEO_CONDS:
            d = ROB / f"{kind}_{val:.2f}"
            d.mkdir(parents=True, exist_ok=True)
            with Pool(args.workers) as pool:
                list(pool.imap_unordered(_extract, [(v, kind, val, str(d)) for v, _ in clips], chunksize=4))
            P = []
            for v, _ in clips:
                z = np.load(d / f"{v}.npz")
                P.append(z["c54"].astype(np.float32))
            res = evaluate(models, np.stack([prep(p) for p in P]), y, device)
            lh, rh = hand_rate(P)
            rows.append({"type": "video", "factor": factor, "level": level, "value": val,
                         "hand_rate_left": lh, "hand_rate_right": rh,
                         **{k: v["top1"] * 100 for k, v in res.items()}})
            print(f"  {factor:20s} {level:9s} " + "  ".join(f"{k} {v['top1']*100:5.1f}" for k, v in res.items())
                  + f"  hands L {lh:.2f} R {rh:.2f}", flush=True)

    ps = per_signer()
    for m, d in ps.items():
        v = np.array(list(d.values()))
        print(f"  per-signer {m}: min {v.min():.1f} ({min(d, key=d.get)}) median {np.median(v):.1f} "
              f"max {v.max():.1f} ({max(d, key=d.get)}) std {v.std():.1f} over {len(v)} signers")
    (RESULTS / "robustness.json").write_text(json.dumps({"fold": FOLD, "n_test": len(clips), "rows": rows,
                                                         "per_signer": ps}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

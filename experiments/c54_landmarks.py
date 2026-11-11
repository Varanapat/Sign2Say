"""คัดลอกจาก my_work/scripts/landmarks.py ของทีม (ไม่แก้ logic) เพื่อให้ inference ใช้ preprocessing เดียวกับ cache
(my_work/ ไม่อยู่ใน git) — ต้นฉบับ docstring ด้านล่าง
"""
"""Shared landmark schema and preprocessing for Sign2Say.

C54 = the TSL-51 schema (Data_Pipeline_Sign2Say.md section 3.4):
  pose 6 (shoulders, elbows, wrists = MediaPipe pose 11-16),
  face 6 (4 brow points + 2 mouth corners), left hand 21, right hand 21.

All arrays here are [T, N, 3] in *pixel* units, with NaN for missing points.
"""

import csv
import io

import numpy as np

POSE6 = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]
FACE6 = ["lbrow_outer", "lbrow_inner", "rbrow_inner", "rbrow_outer", "mouth_right", "mouth_left"]
C54_NAMES = POSE6 + FACE6 + [f"lh{i}" for i in range(21)] + [f"rh{i}" for i in range(21)]
N54 = len(C54_NAMES)

# index ranges inside C54, used by part-aware models and ablations
PARTS = {"body": slice(0, 6), "face": slice(6, 12), "lhand": slice(12, 33), "rhand": slice(33, 54)}
L_SH, R_SH = 0, 1


def _csv_columns():
    cols = []
    for n in POSE6 + FACE6:
        cols.append([f"{n}_x", f"{n}_y", f"{n}_z"])
    for side in ("lh", "rh"):
        for i in range(21):
            cols.append([f"{side}_x{i}", f"{side}_y{i}", f"{side}_z{i}"])
    return cols


_COLS = _csv_columns()


def read_tsl51_csv(text_or_file, width, height):
    """Read a TSL-51 landmark CSV by column name -> [T, 54, 3] pixel coords (NaN = missing).

    Missing values are empty cells in the real files (the README says NaN).
    The t_ms column is processing time, not video time, so it is ignored.
    """
    f = io.StringIO(text_or_file) if isinstance(text_or_file, str) else text_or_file
    rows = list(csv.DictReader(f))
    out = np.full((len(rows), N54, 3), np.nan, dtype=np.float32)
    for t, r in enumerate(rows):
        for j, (cx, cy, cz) in enumerate(_COLS):
            vx = r.get(cx, "")
            if vx not in ("", None) and vx.lower() != "nan":
                out[t, j, 0] = float(vx)
                out[t, j, 1] = float(r[cy])
                out[t, j, 2] = float(r[cz])
    out[..., 0] *= width   # normalized [0,1] -> pixels, so different aspect ratios agree
    out[..., 1] *= height
    out[..., 2] *= width   # MediaPipe z is in the same scale as x
    return out


def fill_short_gaps(xyz, max_gap=5):
    """Linearly interpolate per point across gaps of <= max_gap frames."""
    xyz = xyz.copy()
    T = xyz.shape[0]
    valid = ~np.isnan(xyz[..., 0])
    for j in range(xyz.shape[1]):
        idx = np.flatnonzero(valid[:, j])
        if len(idx) < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= max_gap + 1:
                w = (np.arange(a + 1, b) - a) / (b - a)
                xyz[a + 1:b, j] = xyz[a, j] * (1 - w)[:, None] + xyz[b, j] * w[:, None]
    return xyz


def normalize(xyz):
    """Centre on the shoulder midpoint and divide by the clip's median shoulder width."""
    mid = (xyz[:, L_SH] + xyz[:, R_SH]) / 2                       # [T,3]
    width = np.linalg.norm(xyz[:, L_SH, :2] - xyz[:, R_SH, :2], axis=-1)
    ok = ~np.isnan(width)
    if not ok.any():
        return None                                               # cannot normalize: quarantine
    scale = float(np.median(width[ok]))
    if scale <= 1e-6:
        return None
    # frames without shoulders borrow the nearest frame's centre
    idx = np.flatnonzero(ok)
    nearest = idx[np.abs(np.arange(len(mid))[:, None] - idx[None]).argmin(1)]
    mid = mid[nearest]
    return (xyz - mid[:, None, :]) / scale


def resample(xyz, T_out=64):
    """Resample a [T,N,3] clip to T_out frames (linear where both neighbours exist)."""
    T = xyz.shape[0]
    if T == 1:
        return np.repeat(xyz, T_out, axis=0)
    pos = np.linspace(0, T - 1, T_out)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo)[:, None, None]
    a, b = xyz[lo], xyz[hi]
    out = a * (1 - w) + b * w
    # where one neighbour is missing, take the other (nearest-valid) instead of NaN
    out = np.where(np.isnan(a), b, out)
    out = np.where(np.isnan(b), a, out)
    return out.astype(np.float32)


def features(xyz_norm):
    """[T,N,3] normalized (NaN = missing) -> [T,N,7] = xyz, velocity, valid."""
    valid = ~np.isnan(xyz_norm[..., :1])
    x = np.nan_to_num(xyz_norm, nan=0.0)
    vel = np.zeros_like(x)
    vel[1:] = x[1:] - x[:-1]
    both = valid[1:] & valid[:-1]
    vel[1:] *= both
    return np.concatenate([x, vel, valid.astype(np.float32)], axis=-1).astype(np.float32)


def preprocess(xyz_px, T_out=64):
    """Full ISLR preprocessing used by the baselines. Returns [T_out, 54, 7] or None."""
    x = fill_short_gaps(xyz_px)
    x = normalize(x)
    if x is None:
        return None
    return features(resample(x, T_out))

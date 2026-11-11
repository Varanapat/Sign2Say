"""Landmark augmentation ของ Part B: ใช้ 4 เทคนิคและค่าพารามิเตอร์เดียวกับ SPOTER (spoter/augmentations/)

ใช้กับ cache C54 ([T, 54, 7] = xyz 0-2, velocity 3-5, valid 6) ที่ normalize แล้ว
(จุดกลางไหล่ = 0, หน่วยเป็นความกว้างไหล่) จึงเขียนใหม่ แทนการเรียกโค้ดของ SPOTER ที่อ้างจุดตามชื่อ
(nose, neck, ...) และทำงานบนพิกัดดิบ [0, 1]

เหมือน SPOTER:
  - มีโอกาส p = 0.5 ที่จะ aug แล้วสุ่ม 1 ใน 4 แบบ
      rotate ±13° | perspective 0–0.1 | squeeze 0–0.15 | arm-joint rotate (แต่ละข้อ 30%, ±4°)
  - perspective/squeeze ใช้ homography จากสี่เหลี่ยม [0,1]² ไปยังมุมปลายทางแบบเดียวกับ SPOTER
    โดยใช้กรอบของคลิป (bounding box ของจุดที่มีอยู่จริง) แทน "กรอบภาพ"
  - gaussian noise หลัง aug

ต่างจาก SPOTER (ตั้งใจแก้):
  - จุดที่หาย (valid = 0) ยังคงเป็น 0 หลัง aug  (SPOTER หมุนจุด (0,0) ไปด้วย)
  - perspective/squeeze ทำกับทุกจุดรวมมือ  (ใน C54 มือใช้พิกัดเดียวกับลำตัว ถ้าไม่แปลงมือจะหลุดจากข้อมือ)
  - arm-joint rotate ต่อห่วงโซ่ถึงมือทั้ง 21 จุด และใช้จุดกลางไหล่ (= 0) แทนคอ (C54 ไม่มีจุดคอ)
  - หมุนเฉพาะระนาบ x–y (z คงเดิม) แล้วคำนวณ velocity ใหม่จากพิกัดหลัง aug
  - noise std ตั้งในหน่วยความกว้างไหล่ (default 0.006 ≈ 0.001 ของกรอบ 6 ความกว้างไหล่ใน SPOTER)
"""
import math
import warnings

import numpy as np

XYZ, VEL, VALID = slice(0, 3), slice(3, 6), 6
# ห่วงโซ่แขนใน C54: ไหล่ -> ศอก -> ข้อมือ -> มือ 21 จุด
ARM_CHAINS = {
    "left": [0, 2, 4, list(range(12, 33))],
    "right": [1, 3, 5, list(range(33, 54))],
}
SRC = np.array([(0, 1), (1, 1), (0, 0), (1, 0)], dtype=np.float64)


def _homography(src, dst):
    """3x3 H ที่ map 4 จุด src -> dst (เหมือน cv2.getPerspectiveTransform)"""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    h = np.linalg.solve(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64))
    return np.append(h, 1.0).reshape(3, 3)


def _apply_in_clip_frame(xy, valid, H):
    """แปลง xy [T, N, 2] ด้วย H ในกรอบ [0,1]² ของคลิป (bbox ของจุดที่มีจริง) แล้วแปลงกลับ"""
    pts = xy[valid]
    if len(pts) == 0:
        return xy
    lo, hi = pts.min(0), pts.max(0)
    span = np.where(hi - lo > 1e-6, hi - lo, 1.0)
    u = (xy - lo) / span
    uh = np.concatenate([u, np.ones(u.shape[:-1] + (1,))], axis=-1) @ H.T
    u2 = uh[..., :2] / uh[..., 2:3]
    return u2 * span + lo


def _rotate(xy, origin, angle):
    c, s = math.cos(angle), math.sin(angle)
    d = xy - origin
    return np.stack([origin[..., 0] + c * d[..., 0] - s * d[..., 1],
                     origin[..., 1] + s * d[..., 0] + c * d[..., 1]], axis=-1)


def aug_rotate(xy, valid, rng, angle_range=(-13, 13)):
    return _rotate(xy, np.zeros(2), math.radians(rng.uniform(*angle_range)))


def aug_perspective(xy, valid, rng, ratio=(0, 0.1)):
    m = rng.uniform(*ratio)
    if rng.random() < 0.5:
        dst = [(0 + m, 1 - m), (1, 1), (0 + m, 0 + m), (1, 0)]
    else:
        dst = [(0, 1), (1 - m, 1 - m), (0, 0), (1 - m, 0 + m)]
    return _apply_in_clip_frame(xy, valid, _homography(SRC, np.array(dst, dtype=np.float64)))


def aug_squeeze(xy, valid, rng, ratio=(0, 0.15)):
    ml, mr = rng.uniform(*ratio), rng.uniform(*ratio)
    dst = [(0 + ml, 1), (1 - mr, 1), (0 + ml, 0), (1 - mr, 0)]
    return _apply_in_clip_frame(xy, valid, _homography(SRC, np.array(dst, dtype=np.float64)))


def aug_arm_joint_rotate(xy, valid, rng, prob=0.3, angle_range=(-4, 4)):
    xy = xy.copy()
    for chain in ARM_CHAINS.values():
        origins = [None] + chain[:3]            # None = จุดกลางไหล่ (แทนคอ) | ไหล่ | ศอก | ข้อมือ
        for k, origin in enumerate(origins):
            if rng.random() >= prob:
                continue
            angle = math.radians(rng.uniform(*angle_range))
            o = np.zeros((xy.shape[0], 1, 2)) if origin is None else xy[:, [origin]]
            for nxt in chain[k:]:               # ทุกจุดถัดจาก origin ในห่วงโซ่
                idx = nxt if isinstance(nxt, list) else [nxt]
                xy[:, idx] = _rotate(xy[:, idx], o, angle)
    return xy


AUGS = [aug_rotate, aug_perspective, aug_squeeze, aug_arm_joint_rotate]


def recompute_velocity(x):
    """velocity แบบเดียวกับ my_work/scripts/landmarks.py: ต่างจากเฟรมก่อน, 0 ถ้าเฟรมใดเฟรมหนึ่งหาย"""
    valid = x[..., VALID:VALID + 1] > 0.5
    vel = np.zeros_like(x[..., XYZ])
    vel[1:] = (x[1:, :, XYZ] - x[:-1, :, XYZ]) * (valid[1:] & valid[:-1])
    x[..., VEL] = vel
    return x


def augment(x, rng, prob=0.5, noise_std=0.006):
    """x [T, 54, 7] (ทั้ง 7 channel) -> สำเนาที่ aug แล้ว"""
    x = x.copy()
    valid = x[..., VALID] > 0.5                          # [T, N]
    xy = x[..., :2].astype(np.float64)
    changed = False
    if rng.random() < prob:
        xy = AUGS[rng.integers(len(AUGS))](xy, valid, rng)
        changed = True
    if noise_std > 0:
        xy = xy + rng.normal(0.0, noise_std, xy.shape)
        x[..., 2] += rng.normal(0.0, noise_std, x.shape[:2]).astype(x.dtype)
        changed = True
    if changed:
        x[..., :2] = xy.astype(x.dtype)
        x[..., XYZ] *= valid[..., None]                  # จุดที่หายต้องเป็น 0 เหมือนเดิม
        x = recompute_velocity(x)
    return x


HANDS = (slice(12, 33), slice(33, 54))


def hand_local(x):
    """พิกัดมือแบบ "ในกรอบของมือเอง" (คล้าย hand normalization ของ SPOTER) -> [..., T, 54, 2]

    x [..., T, 54, 7] (7 channel เต็ม) -> ต่อมือแต่ละข้างแต่ละเฟรม:
      (xy - ข้อมือของมือ) / ขนาดมือ (ด้านที่ยาวกว่าของ bounding box)
      ถ้าข้อมือหายใช้ค่าเฉลี่ยของจุดที่มี; จุดที่หาย และจุดที่ไม่ใช่มือ = 0
    ตำแหน่งมือเทียบลำตัวยังอยู่ใน channel x, y เดิม ส่วนนี้เพิ่มรายละเอียดรูปมือให้ชัดขึ้น
    """
    xy = x[..., :2].astype(np.float64)
    valid = x[..., 6] > 0.5
    out = np.zeros(xy.shape, dtype=np.float32)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)       # มือหายทั้งเฟรม -> nan -> 0
        for sl in HANDS:
            h, v = xy[..., sl, :], valid[..., sl]
            pts = np.where(v[..., None], h, np.nan)
            origin = np.where(v[..., :1, None], h[..., :1, :],
                              np.nanmean(pts, axis=-2, keepdims=True))
            size = np.nanmax(np.nanmax(pts, axis=-2) - np.nanmin(pts, axis=-2), axis=-1)  # [..., T]
            size = np.where(np.isfinite(size) & (size > 1e-3), size, 1.0)
            local = (h - origin) / size[..., None, None]
            out[..., sl, :] = np.nan_to_num(np.where(v[..., None], local, 0.0)).astype(np.float32)
    return out


def select_features(x, channels, use_hand_local=False):
    """เลือก channel ที่โมเดลใช้ (+ hand-local 2 channel ถ้าเปิด) จาก 7 channel เต็ม"""
    feats = x[..., channels]
    if use_hand_local:
        feats = np.concatenate([feats, hand_local(x)], axis=-1)
    return np.ascontiguousarray(feats, dtype=np.float32)


class AugmentedLandmarks:
    """Dataset ของ train: aug บน 7 channel เต็ม แล้วค่อยเลือก channel ที่โมเดลใช้"""

    def __init__(self, X_full, y, channels, aug=True, seed=0, prob=0.5, noise_std=0.006,
                 use_hand_local=False):
        self.X, self.y, self.ch = X_full, y, channels
        self.aug, self.prob, self.noise_std = aug, prob, noise_std
        self.use_hand_local = use_hand_local
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.aug:
            x = augment(x, self.rng, self.prob, self.noise_std)
        return select_features(x, self.ch, self.use_hand_local), self.y[i]

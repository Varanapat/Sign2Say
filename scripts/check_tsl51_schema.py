"""For TSL-51 user_sign clips: re-run MediaPipe, match each CSV point to the nearest face-mesh / pose / hand index."""
import csv, sys, random
from collections import Counter
from pathlib import Path
import numpy as np, cv2, mediapipe as mp
D = Path("/Users/thiya/Documents/Project/Deep Learning/Source(Paper)/tsl51")
FACE6 = ["lbrow_outer", "lbrow_inner", "rbrow_inner", "rbrow_outer", "mouth_right", "mouth_left"]
POSE6 = None
rows = list(csv.DictReader(open(D/"metadata/user_sign_metadata.csv", encoding="utf-8-sig")))
rows = [r for r in rows if r["extraction_status"] == "ok"]
random.Random(0).shuffle(rows)
hdr = next(csv.reader(open(D/rows[0]["landmark_path"], encoding="utf-8-sig")))
pose_names = [h[:-2] for h in hdr if h.endswith("_x") and h[:-2] not in FACE6 and not h.startswith(("lh","rh"))]
print("pose cols:", pose_names)
votes = {n: Counter() for n in FACE6}; pvotes = {n: Counter() for n in pose_names}
dists = {n: [] for n in FACE6}; hand = Counter(); hand_d=[]
for r in rows[:15]:
    lm = list(csv.DictReader(open(D/r["landmark_path"], encoding="utf-8-sig")))
    cap = cv2.VideoCapture(str(D/r["video_path"]))
    with mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:
        t = 0
        while True:
            ok, fr = cap.read()
            if not ok or t >= len(lm): break
            res = hol.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); row = lm[t]; t += 1
            if res.face_landmarks:
                F = np.array([[p.x, p.y] for p in res.face_landmarks.landmark])
                for n in FACE6:
                    if row.get(n+"_x"):
                        q = np.array([float(row[n+"_x"]), float(row[n+"_y"])])
                        dd = np.linalg.norm(F - q, axis=1); k = int(dd.argmin())
                        votes[n][k] += 1; dists[n].append(dd[k])
            if res.pose_landmarks:
                P = np.array([[p.x, p.y] for p in res.pose_landmarks.landmark])
                for n in pose_names:
                    if row.get(n+"_x"):
                        q = np.array([float(row[n+"_x"]), float(row[n+"_y"])])
                        pvotes[n][int(np.linalg.norm(P-q,axis=1).argmin())] += 1
            for side, h in (("lh", res.left_hand_landmarks), ("rh", res.right_hand_landmarks)):
                if h and row.get(side+"_x0"):
                    q = np.array([float(row[side+"_x0"]), float(row[side+"_y0"])])
                    hand[side] += 1; hand_d.append(np.hypot(h.landmark[0].x-q[0], h.landmark[0].y-q[1]))
    cap.release()
for n in FACE6:
    print(n, votes[n].most_common(3), "median dist", np.median(dists[n]) if dists[n] else None)
for n in pose_names: print(n, pvotes[n].most_common(2))
print("hand wrist matches", dict(hand), "median dist", np.median(hand_d))

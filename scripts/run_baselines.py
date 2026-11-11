"""Run the Sign2Say ISLR models (Experiment_Design_Sign2Say.md sections 2-3).

  LR / RF : X0, logistic regression / random forest on per-clip landmark statistics
  BiGRU   : X1, per-frame Linear -> 2-layer BiGRU -> mean pool        (spatial -> temporal)
  S2S     : R,  S2S-LTR = part-aware MLPs -> Transformer x4 -> [CLS]  (TDD section 6.2)

Protocols (Data_Pipeline_Sign2Say.md section 5.2):
  tsl51_ext   train = TSL-51 expert originals, test = TSL-51 user_sign (new signer + camera)
  tsl51_loso  train = expert_02 + web, test = expert_01 (leave-one-signer-out)
  tsl1s_si    TSL-ONE-S signer-independent folds (splits/tsl_one_s_signer_indep_fold*.json)
  tsl1s_orig  TSL-ONE-S original split (signer-dependent, for comparison with the paper)

TSL-ONE-S class indices follow dataset/annotations/label_map_tsl1s.json (the label map
shared with the friend's Sign2Say-base_line), so prediction files line up across teams.

Validation is a group-held-out slice of train (TSL-51) or the fold's val signers
(TSL-ONE-S); torch models pick their best epoch on validation macro-F1.
Outputs: results/<protocol>.json (metrics) and
         results/preds/<split>_<model>_seed<s>_{val,test}.csv (video_id,label,p0..pK-1).

Usage:
  .venv/bin/python scripts/run_baselines.py --protocol tsl51_ext
  .venv/bin/python scripts/run_baselines.py --protocol tsl1s_si --folds 4 --models S2S BiGRU
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "dataset" / "cache"
RESULTS = ROOT / "results"
LABEL_MAP_TSL1S = ROOT / "dataset" / "annotations" / "label_map_tsl1s.json"
DEFAULT_EPOCHS = {"BiGRU": 100, "S2S": 150, "S2S-HL": 150, "STGCN": 150, "STGCN-HL": 150}
TORCH_MODELS = set(DEFAULT_EPOCHS)

# Ablation variants (Experiment_Design_Sign2Say.md section 5). Each changes one thing vs "base".
#   noface : zero the 6 face points (input + valid flag)          -> does the face help?
#   x71    : C54 + lips 10, eyes 4, nose 1, hips 2 (needs tsl1s_x71_T64.npz) -> more face
#   novel  : zero the velocity channels                           -> does explicit motion help?
#   nomask : zero the valid flag                                  -> does telling missing points help?
#   flat   : one MLP over all points instead of part-aware MLPs   -> does part-aware help?
VARIANTS = ["base", "noface", "x71", "novel", "nomask", "flat"]
R6, R21 = list(range(6)), list(range(21))
PARTS_C54 = {"body": (R6, 32), "face": ([6 + i for i in R6], 32),
             "lhand": ([12 + i for i in R21], 96), "rhand": ([33 + i for i in R21], 96)}
# X71 order = C54 (0-53) + lips 54-63 + eyes 64-67 + nose 68 + hips 69-70 (extract_tsl1s.py)
PARTS_X71 = {"body": (R6 + [69, 70], 32), "face": ([6 + i for i in R6] + list(range(54, 69)), 32),
             "lhand": PARTS_C54["lhand"], "rhand": PARTS_C54["rhand"]}


def apply_variant(X, variant):
    X = X.copy() if variant in ("noface", "novel", "nomask") else X
    if variant == "noface":
        X[:, :, 6:12, :] = 0
    elif variant == "novel":
        X[..., 3:6] = 0
    elif variant == "nomask":
        X[..., 6] = 0
    return X


# ---------------------------------------------------------------- data / splits

def load(name):
    d = np.load(CACHE / name, allow_pickle=False)
    return {k: d[k] for k in d.files}


def group_val_split(groups, labels, frac, seed):
    """Hold out ~frac of *groups* (stratified by label) as validation."""
    rng = np.random.default_rng(seed)
    val = np.zeros(len(groups), bool)
    for lab in np.unique(labels):
        g = np.unique(groups[labels == lab])
        rng.shuffle(g)
        k = max(1, int(round(len(g) * frac))) if len(g) > 2 else 0
        val |= np.isin(groups, g[:k]) & (labels == lab)
    return val


def splits_for(protocol, folds, seed, cache="tsl1s_c54_T64.npz"):
    """Yield (name, data, train_idx, val_idx, test_idx)."""
    if protocol.startswith("tsl51"):
        d = load("tsl51_c54_T64.npz")
        y, part, signer, grp = d["sign_id"], d["part"], d["signer"], d["group"]
        if protocol == "tsl51_ext":
            pool = part == "expert"
            test = part == "user"
        else:  # tsl51_loso: test on expert_01 originals
            pool = (part == "expert") & (signer != "expert_01")
            test = (part == "expert") & (signer == "expert_01")
        val = pool & group_val_split(grp, y, 0.15, seed)
        train = pool & ~val
        yield protocol, d, np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test)
        return

    d = load(cache)
    pos = {c: i for i, c in enumerate(d["clip_id"])}

    def from_json(path):
        idx = {"train": [], "val": [], "test": []}
        for entry in json.loads(Path(path).read_text(encoding="utf-8")):
            for inst in entry["instances"]:
                c = "tsl1s:" + inst["video_id"]
                if c in pos:                      # 4 JSON ids have no video file
                    idx[inst["split"]].append(pos[c])
        return {k: np.array(v) for k, v in idx.items()}

    if protocol == "tsl1s_orig":
        s = from_json(ROOT / "tsl_one_s_dataset_i3d.json")
        yield "tsl1s_orig", d, s["train"], s["val"], s["test"]
        return
    for f in folds:
        if protocol == "tsl1s_rand":   # same sizes as SI fold f, clips drawn at random (per gloss)
            s = from_json(ROOT / "splits" / "random_matched" / f"tsl_one_s_random_fold{f}.json")
            yield f"tsl1s_rand_fold{f}", d, s["train"], s["val"], s["test"]
        else:
            s = from_json(ROOT / "splits" / f"tsl_one_s_signer_indep_fold{f}.json")
            yield f"tsl1s_si_fold{f}", d, s["train"], s["val"], s["test"]


def class_index(d, protocol):
    """Return (labels as int, class names in index order)."""
    if protocol.startswith("tsl1s"):
        m = json.loads(LABEL_MAP_TSL1S.read_text(encoding="utf-8"))
        names = sorted(m, key=m.get)
        return np.array([m[g] for g in d["label"]]), names
    names = sorted(np.unique(d["sign_id"]))
    cmap = {c: i for i, c in enumerate(names)}
    return np.array([cmap[c] for c in d["sign_id"]]), names


# ---------------------------------------------------------------- metrics / outputs

def metrics(y_true, scores, n_classes):
    from sklearn.metrics import f1_score
    pred = scores.argmax(1)
    top5 = np.argsort(-scores, 1)[:, :5]
    return {"top1": float((pred == y_true).mean()),
            "top5": float((top5 == y_true[:, None]).any(1).mean()),
            "macro_f1": float(f1_score(y_true, pred, labels=np.arange(n_classes), average="macro",
                                       zero_division=0)),
            "n": int(len(y_true))}


def save_preds(path, clip_ids, y, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label"] + [f"p{k}" for k in range(scores.shape[1])])
        for c, lab, s in zip(clip_ids, y, scores):
            w.writerow([c.split(":", 1)[1], int(lab)] + [f"{v:.6g}" for v in s])


# ---------------------------------------------------------------- X0: statistics + sklearn

def stat_features(X):
    """[N,T,54,7] -> per-clip mean/std/min/max of xyz + mean|velocity| + valid rate."""
    xyz, vel, valid = X[..., :3], X[..., 3:6], X[..., 6:7]
    big = np.where(valid.astype(bool), xyz, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f = [np.nanmean(big, 1), np.nanstd(big, 1), np.nanmin(big, 1), np.nanmax(big, 1),
                 np.abs(vel).mean(1), valid.mean(1)]
    F = np.concatenate([np.nan_to_num(a, nan=0.0).reshape(len(X), -1) for a in f], 1)
    return F.astype(np.float32)


def full_scores(model, F, n_classes):
    p = model.predict_proba(F)
    out = np.zeros((len(F), n_classes), np.float32)
    out[:, model.classes_] = p
    return out


def run_sklearn(kind, X, y, tr, va, te, n_classes, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    F = stat_features(X)
    if kind == "LR":
        best = None
        for C in (0.01, 0.1, 1.0):           # small, fixed tuning budget, chosen on val
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            m.fit(F[tr], y[tr])
            mf = metrics(y[va], full_scores(m, F[va], n_classes), n_classes)["macro_f1"]
            if best is None or mf > best[0]:
                best = (mf, C, m)
        _, C, m = best
        cfg = {"C": C}
    else:
        m = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
        m.fit(F[tr], y[tr])
        cfg = {"n_estimators": 500}
    return full_scores(m, F[va], n_classes), full_scores(m, F[te], n_classes), cfg


# ---------------------------------------------------------------- torch models

def build_model(kind, n_classes, parts=None):
    import torch
    import torch.nn as nn

    class BiGRU(nn.Module):
        """X1: spatial (one Linear over all 54 points) -> temporal (BiGRU) -> mean pool."""
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(54 * 7, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1))
            self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n_classes))

        def forward(self, x):                        # x [B,T,54,7]
            h, _ = self.gru(self.inp(x.flatten(2)))  # [B,T,512]
            return self.head(h.mean(1))

    class S2SLTR(nn.Module):
        """R: S2S-LTR (TDD section 6.2).

        Part-aware spatial encoder: left hand 21x7 -> 96, right hand 21x7 -> 96,
        body 6x7 -> 32, face 6x7 -> 32; concat -> 256 + LayerNorm.
        Temporal: [CLS] + learned positional encoding (T+1 positions) ->
        Transformer encoder x4 (d=256, 4 heads, ff=1024, dropout 0.1, pre-norm) -> [CLS] -> classes.
        """
        def __init__(self, parts, T=64, d=256, C=7):
            super().__init__()

            def mlp(n_in, n_out):
                return nn.Sequential(nn.Linear(n_in, n_out), nn.GELU(), nn.Linear(n_out, n_out))

            self.parts = list(parts)
            for k, (idx, _) in parts.items():
                self.register_buffer(f"idx_{k}", torch.tensor(idx), persistent=False)
            self.enc = nn.ModuleDict({k: mlp(len(idx) * C, dim) for k, (idx, dim) in parts.items()})
            self.norm = nn.LayerNorm(d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.pos = nn.Parameter(torch.randn(1, T + 1, d) * 0.02)   # one vector per time step
            self.drop = nn.Dropout(0.1)
            layer = nn.TransformerEncoderLayer(d, 4, 1024, 0.1, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.1), nn.Linear(d, n_classes))

        def forward(self, x):                        # x [B,T,54,7]
            B, T = x.shape[:2]
            h = torch.cat([self.enc[k](x[:, :, getattr(self, f"idx_{k}")].flatten(2))
                           for k in self.parts], -1)
            h = self.norm(h)                         # [B,T,256]
            h = torch.cat([self.cls.expand(B, -1, -1), h], 1) + self.pos[:, :T + 1]
            h = self.tf(self.drop(h))                # [B,T+1,256]
            return self.head(h[:, 0])

    if kind == "BiGRU":
        return BiGRU()
    from stgcn import STGCN, WithHandLocal   # friend's RQ-S findings, re-implemented (stgcn.py)
    if kind == "S2S-HL":
        return WithHandLocal(S2SLTR(parts or PARTS_C54, C=9))
    if kind == "STGCN":
        return STGCN(n_classes, C=7)
    if kind == "STGCN-HL":
        return WithHandLocal(STGCN(n_classes, C=9))
    return S2SLTR(parts or PARTS_C54)


def augment(xb):
    """Online augmentation on normalized coords (train only): rotation, scale, shift, noise."""
    import torch
    B = xb.shape[0]
    ang = (torch.rand(B, device=xb.device) - 0.5) * 2 * math.radians(10)
    sc = 0.9 + 0.2 * torch.rand(B, device=xb.device)
    c, s = torch.cos(ang) * sc, torch.sin(ang) * sc
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)   # [B,2,2]
    # missing points are exactly 0 in xyz; derive the mask from that (identical to channel 6 on
    # our caches) so the "nomask" ablation, which zeroes channel 6, does not also zero every input
    valid = (xb[..., :3] != 0).any(-1, keepdim=True).to(xb.dtype)
    xy = torch.einsum("btnd,bed->btne", xb[..., :2], R)
    xy = xy + (torch.rand(B, 1, 1, 2, device=xb.device) - 0.5) * 0.2
    xy = xy + torch.randn_like(xy) * 0.01
    vxy = torch.einsum("btnd,bed->btne", xb[..., 3:5], R)
    out = torch.cat([xy, xb[..., 2:3], vxy, xb[..., 5:7]], -1)
    return out * torch.cat([valid.expand(-1, -1, -1, 6), torch.ones_like(valid)], -1)


def run_torch(kind, X, y, tr, va, te, n_classes, seed, epochs, device, parts=None, save_to=None):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(kind, n_classes, parts).to(device)
    if kind == "BiGRU":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        patience_max = 20
    else:  # S2S (and S2S-HL / STGCN / STGCN-HL, same recipe): AdamW 3e-4, wd 0.05, 5 warm-up epochs then cosine (Experiment Design section 2)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        warm = 5
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda e: (e + 1) / warm if e < warm
            else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm))))
        patience_max = 30
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)

    def predict(idx):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                outs.append(torch.softmax(model(Xt[idx[i:i + 256]]), -1).cpu().numpy())
        return np.concatenate(outs)

    best, best_state, best_ep, patience = -1, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            loss = lossf(model(augment(Xt[b])), yt[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        mf = metrics(y[va], predict(va), n_classes)["macro_f1"]
        if mf > best:
            best, best_ep, patience = mf, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= patience_max:
                break
    model.load_state_dict(best_state)
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_to)
    n_params = sum(p.numel() for p in model.parameters())
    return predict(va), predict(te), {"best_epoch": best_ep, "epochs_run": ep + 1,
                                      "epochs_max": epochs, "params": n_params}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True,
                    choices=["tsl51_ext", "tsl51_loso", "tsl1s_si", "tsl1s_orig", "tsl1s_rand"])
    ap.add_argument("--models", nargs="+", default=["LR", "RF", "BiGRU"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=0, help="0 = per-model default (BiGRU 100, S2S 150)")
    ap.add_argument("--variant", default="base", choices=VARIANTS, help="ablation variant (S2S only)")
    ap.add_argument("--save-models", action="store_true", help="save best weights to results/models/")
    ap.add_argument("--tag", default="", help="suffix for the results file, so parallel runs never overwrite each other")
    args = ap.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{args.protocol}{args.tag}.json"
    log = json.loads(out_path.read_text()) if out_path.exists() else []

    for seed in args.seeds:
        cache = "tsl1s_x71_T64.npz" if args.variant == "x71" else "tsl1s_c54_T64.npz"
        for name, d, tr, va, te in splits_for(args.protocol, args.folds, seed, cache):
            X = apply_variant(d["X"], args.variant)
            parts = {"x71": PARTS_X71,
                     "flat": {"all": (list(range(X.shape[2])), 256)}}.get(args.variant, PARTS_C54)
            y, names = class_index(d, args.protocol)
            K = len(names)
            labels = d["sign_id"] if "sign_id" in d else d["label"]
            unseen = sorted(set(labels[te]) - set(labels[tr]))
            print(f"\n== {name} seed {seed}: train {len(tr)} val {len(va)} test {len(te)} "
                  f"classes {K} (test classes unseen in train: {len(unseen)})", flush=True)
            for m in args.models:
                t0 = time.time()
                if m in TORCH_MODELS:
                    ep = args.epochs or DEFAULT_EPOCHS[m]
                    m_tag = m if args.variant == "base" else f"{m}-{args.variant}"
                    save_to = RESULTS / "models" / f"{name}_{m_tag}_seed{seed}.pt" if args.save_models else None
                    sv, st, cfg = run_torch(m, X, y, tr, va, te, K, seed, ep, device, parts, save_to)
                else:
                    sv, st, cfg = run_sklearn(m, X, y, tr, va, te, K, seed)
                v, t = metrics(y[va], sv, K), metrics(y[te], st, K)
                dt = time.time() - t0
                m_name = m if args.variant == "base" else f"{m}-{args.variant}"
                stem = RESULTS / "preds" / f"{name}_{m_name}_seed{seed}"
                save_preds(stem.with_name(stem.name + "_val.csv"), d["clip_id"][va], y[va], sv)
                save_preds(stem.with_name(stem.name + "_test.csv"), d["clip_id"][te], y[te], st)
                print(f"  {m_name:6s} val top1 {v['top1']:.3f} f1 {v['macro_f1']:.3f} | "
                      f"TEST top1 {t['top1']:.3f} top5 {t['top5']:.3f} macroF1 {t['macro_f1']:.3f} "
                      f"({dt:.0f}s) {cfg}", flush=True)
                log.append({"protocol": args.protocol, "split": name, "seed": seed, "model": m_name, "variant": args.variant,
                            "val": v, "test": t, "config": cfg, "seconds": round(dt, 1),
                            "device": device if m in TORCH_MODELS else "cpu",
                            "label_map": "label_map_tsl1s.json" if args.protocol.startswith("tsl1s") else "sorted sign_id",
                            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                            "n_classes": K, "test_classes_unseen_in_train": unseen})
                out_path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(f"\nresults -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""X16 / RQ9: train one S2S-LTR encoder on TSL-ONE-S + TSL-51 together, with two heads.

Shared encoder (part-aware MLPs + Transformer, same as R), one output layer of 184 + K51
units that is used as two separate heads: the TSL-ONE-S head is logits[:, :184] and the
TSL-51 head is logits[:, 184:]. Each clip's cross-entropy uses only its own dataset's head
(label space stays separate, softmax never mixes the two vocabularies).

Splits are exactly those of the single-dataset runs, so results compare paired:
  TSL-ONE-S: signer-independent fold k (splits/), test = held-out signers
  TSL-51:    tsl51_ext (train/val = expert originals + their copies, test = user_sign)
Model selection: mean of the two validation macro-F1s.

Results go to results/joint{tag}.json with two rows per fold:
  split tsl1s_si_fold{k}, model S2S-joint   (compare with S2S on the same fold)
  split tsl51_ext,        model S2S-joint-f{k} (compare with S2S on tsl51_ext)

Known caveat (Data Pipeline D2 / extract_tsl1s.py): the 4 brow points of TSL-ONE-S follow
the README index order mapped onto TSL-51's column order; the pairing is unverified.

Usage:
  .venv/bin/python scripts/run_joint.py --folds 0 2 4 6 8
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines import (PARTS_C54, RESULTS, ROOT, augment, build_model, class_index,  # noqa: E402
                           metrics, save_preds, splits_for)


def run_joint(X, y, ds, tr, va1, va2, te1, te2, K1, K2, seed, epochs, device, save_to=None):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model("S2S", K1 + K2, PARTS_C54).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    warm = 5
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm
        else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm))))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="sum")
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)        # local label inside each dataset's head
    dt = torch.tensor(ds, device=device)

    def predict(idx, head):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                lo = model(Xt[idx[i:i + 256]])
                lo = lo[:, :K1] if head == 0 else lo[:, K1:]
                outs.append(torch.softmax(lo, -1).cpu().numpy())
        return np.concatenate(outs)

    best, best_state, best_ep, patience = -1, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            lo = model(augment(Xt[b]))
            m1, m2 = dt[b] == 0, dt[b] == 1
            loss = 0
            if m1.any():
                loss = loss + lossf(lo[m1][:, :K1], yt[b][m1])
            if m2.any():
                loss = loss + lossf(lo[m2][:, K1:], yt[b][m2])
            loss = loss / len(b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        f1 = metrics(y[va1], predict(va1, 0), K1)["macro_f1"]
        f2 = metrics(y[va2], predict(va2, 1), K2)["macro_f1"]
        score = (f1 + f2) / 2
        if score > best:
            best, best_ep, patience = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 30:
                break
    model.load_state_dict(best_state)
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_to)
    cfg = {"best_epoch": best_ep, "epochs_run": ep + 1, "epochs_max": epochs,
           "params": sum(p.numel() for p in model.parameters())}
    return (predict(va1, 0), predict(te1, 0), predict(va2, 1), predict(te2, 1), cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_path = RESULTS / f"joint{args.tag}.json"
    log = json.loads(out_path.read_text()) if out_path.exists() else []

    for seed in args.seeds:
        _, d2, tr2, va2, te2 = next(splits_for("tsl51_ext", None, seed))
        y2, names2 = class_index(d2, "tsl51_ext")
        K2 = len(names2)
        for name, d1, tr1, va1, te1 in splits_for("tsl1s_si", args.folds, seed):
            y1, names1 = class_index(d1, "tsl1s_si")
            K1 = len(names1)
            n1 = len(d1["X"])
            X = np.concatenate([d1["X"], d2["X"]])
            y = np.concatenate([y1, y2])
            ds = np.concatenate([np.zeros(n1, np.int64), np.ones(len(d2["X"]), np.int64)])
            tr = np.concatenate([tr1, tr2 + n1])
            print(f"\n== joint {name} + tsl51_ext seed {seed}: train {len(tr1)} + {len(tr2)}, "
                  f"val {len(va1)} + {len(va2)}, test {len(te1)} + {len(te2)}, heads {K1} + {K2}", flush=True)
            t0 = time.time()
            sv1, st1, sv2, st2, cfg = run_joint(X, y, ds, tr, va1, va2 + n1, te1, te2 + n1,
                                                K1, K2, seed, args.epochs, device,
                                                RESULTS / "models" / f"joint_{name}_seed{seed}.pt" if args.save_models else None)
            sec = round(time.time() - t0, 1)
            rows = [(name, "S2S-joint", d1, y1, va1, te1, sv1, st1, K1),
                    ("tsl51_ext", f"S2S-joint-f{name.split('fold')[-1]}", d2, y2, va2, te2, sv2, st2, K2)]
            for split, m_name, d, yy, va, te, sv, st, K in rows:
                v, t = metrics(yy[va], sv, K), metrics(yy[te], st, K)
                stem = RESULTS / "preds" / f"{split}_{m_name}_seed{seed}"
                save_preds(stem.with_name(stem.name + "_val.csv"), d["clip_id"][va], yy[va], sv)
                save_preds(stem.with_name(stem.name + "_test.csv"), d["clip_id"][te], yy[te], st)
                print(f"  {split:16s} {m_name:14s} val top1 {v['top1']:.3f} f1 {v['macro_f1']:.3f} | "
                      f"TEST top1 {t['top1']:.3f} top5 {t['top5']:.3f} macroF1 {t['macro_f1']:.3f}", flush=True)
                log.append({"protocol": "joint", "split": split, "seed": seed, "model": m_name,
                            "variant": "joint", "val": v, "test": t, "config": cfg, "seconds": sec,
                            "device": device, "n_train": int(len(tr)), "n_val": int(len(va)),
                            "n_test": int(len(te)), "n_classes": K})
            print(f"  ({sec:.0f}s) {cfg}", flush=True)
            out_path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(f"\nresults -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

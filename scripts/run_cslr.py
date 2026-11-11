"""Continuous recognition on TSL-51 user_sentence (Experiment Design C1, C2; RQ7).

C1  CTC: part-aware spatial MLPs (same as S2S-LTR) -> 2x Conv1d stride 2 (30 -> 7.5 steps/s)
    -> sinusoidal positions -> Transformer x4 -> CTC over 31 sign_ids + blank. Greedy decode.
    --init x16     : spatial MLPs + Transformer start from the X16 joint model (never saw user01)
    --init scratch : random init
    --synth N      : N synthetic sentences per epoch, made by concatenating user_sign clips
                     (same signer + camera as the sentences) in random order, random end trims,
                     optional rest (null_act) gaps and 6-frame blends between clips
C2  segment -> classify: slide a window over the sentence, classify each window with an
    isolated-sign model (TSL-51 head of X16, restricted to the 31 sentence sign_ids + null),
    threshold, drop short runs, merge repeats. Window / threshold / min-run are chosen on val.
    --c2-model x16    : X16 as is (signer-independent: user01 never seen)
    --c2-model x16ft  : X16 fine-tuned on all TSL-51 isolated clips (expert + user_sign)

Protocols (splits/): rep (seen sentences, new repetitions) and template (unseen gloss sequences).
Metrics: WER on sign_id, WER on words (variants merged), sentence error rate (SER), and WER
split by no_space / with_space.

Usage:
  .venv/bin/python scripts/run_cslr.py c1 --protocol rep --init x16 --synth 300 --seeds 0 1 2
  .venv/bin/python scripts/run_cslr.py c2 --c2-model x16
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmarks import features, resample  # noqa: E402
from run_baselines import PARTS_C54, RESULTS, ROOT, augment, build_model  # noqa: E402

CACHE = ROOT / "dataset" / "cache" / "tsl51_sentences.npz"
X16 = RESULTS / "models" / "joint_tsl1s_si_fold0_seed0.pt"
K1 = 184     # TSL-ONE-S head size inside the X16 model


# ---------------------------------------------------------------- data

def load_data():
    d = np.load(CACHE)
    cut = lambda X, o: [X[o[i]:o[i + 1]] for i in range(len(o) - 1)]  # noqa: E731
    D = {k: d[k] for k in d.files}
    D["sents"] = cut(d["sent_xyz"], d["sent_off"])
    D["signs"] = cut(d["sign_xyz"], d["sign_off"])
    D["vocab"] = list(d["vocab"])
    return D


def split_idx(D, protocol):
    sp = json.loads((ROOT / "splits" / f"tsl51_sentence_{protocol}.json").read_text(encoding="utf-8"))
    s = np.array([sp[v] for v in D["video_id"]])
    return {k: np.flatnonzero(s == k) for k in ("train", "val", "test")}


def word(t):
    return t.rsplit("_var_", 1)[0]


def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def score(refs, hyps, variation):
    out = {}
    for name, f in (("wer", lambda t: t), ("wer_word", word)):
        e = n = 0
        for r, h in zip(refs, hyps):
            e += edit([f(t) for t in r], [f(t) for t in h])
            n += len(r)
        out[name] = e / n
    out["ser"] = float(np.mean([r != h for r, h in zip(refs, hyps)]))
    for v in ("no_space", "with_space"):
        idx = [i for i, x in enumerate(variation) if x == v]
        if idx:
            out[f"wer_{v}"] = (sum(edit(refs[i], hyps[i]) for i in idx) / sum(len(refs[i]) for i in idx))
    out["n"] = len(refs)
    return out


def synth_sentence(D, rng, by_sign, nulls, tokens=None):
    if tokens is None:
        tokens = list(rng.choice(D["vocab"], size=rng.integers(3, 7), replace=False))
    parts = []
    for k, t in enumerate(tokens):
        x = D["signs"][rng.choice(by_sign[t])]
        a, b = int(len(x) * rng.uniform(0, 0.2)), int(len(x) * rng.uniform(0, 0.2))
        x = x[a:len(x) - b]
        if parts:
            w = np.linspace(0, 1, 8)[1:-1][:, None, None]
            parts.append(parts[-1][-1:] * (1 - w) + x[:1] * w)     # 6-frame blend
        parts.append(x)
        if k < len(tokens) - 1 and rng.random() < 0.25:            # occasional rest gap
            z = D["signs"][rng.choice(nulls)]
            s = rng.integers(0, max(1, len(z) - 45))
            parts.append(z[s:s + rng.integers(10, 45)])
    return np.concatenate(parts).astype(np.float32), tokens


def to_feat(xyz, rng=None):
    if rng is not None:                                            # speed augmentation 0.8-1.25x
        xyz = resample(xyz, max(16, int(round(len(xyz) / rng.uniform(0.8, 1.25)))))
    return features(xyz)


# ---------------------------------------------------------------- C1 model

def build_cslr(n_out):
    import torch
    import torch.nn as nn

    class CSLR(nn.Module):
        def __init__(self, parts=PARTS_C54, d=256):
            super().__init__()

            def mlp(n_in, n_o):
                return nn.Sequential(nn.Linear(n_in, n_o), nn.GELU(), nn.Linear(n_o, n_o))

            self.parts = list(parts)
            for k, (idx, _) in parts.items():
                self.register_buffer(f"idx_{k}", torch.tensor(idx), persistent=False)
            self.enc = nn.ModuleDict({k: mlp(len(idx) * 7, dim) for k, (idx, dim) in parts.items()})
            self.norm = nn.LayerNorm(d)
            self.down = nn.Sequential(nn.Conv1d(d, d, 5, 2, 2), nn.GELU(), nn.Conv1d(d, d, 5, 2, 2), nn.GELU())
            self.drop = nn.Dropout(0.1)
            layer = nn.TransformerEncoderLayer(d, 4, 1024, 0.1, activation="gelu", batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))
            self.d = d

        def forward(self, x, lengths):                 # x [B,T,54,7]
            h = torch.cat([self.enc[k](x[:, :, getattr(self, f"idx_{k}")].flatten(2)) for k in self.parts], -1)
            h = self.down(self.norm(h).transpose(1, 2)).transpose(1, 2)        # [B,T/4,d]
            L = (((lengths + 1) // 2) + 1) // 2
            T = h.shape[1]
            pos = torch.arange(T, device=h.device)[:, None]
            div = torch.exp(torch.arange(0, self.d, 2, device=h.device) * (-math.log(10000.0) / self.d))
            pe = torch.zeros(T, self.d, device=h.device)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            pad = torch.arange(T, device=h.device)[None] >= L[:, None]
            h = self.tf(self.drop(h + pe), src_key_padding_mask=pad)
            return self.out(h), L

    return CSLR()


def pad_batch(feats, device):
    import torch
    T = max(len(f) for f in feats)
    X = np.zeros((len(feats), T) + feats[0].shape[1:], np.float32)
    for i, f in enumerate(feats):
        X[i, :len(f)] = f
    return torch.tensor(X, device=device), torch.tensor([len(f) for f in feats], device=device)


def greedy(logp, L, inv):
    out = []
    for p, n in zip(logp.argmax(-1).cpu().numpy(), L.cpu().numpy()):
        seq, prev = [], 0
        for k in p[:n]:
            if k != prev and k != 0:
                seq.append(inv[k])
            prev = k
        out.append(seq)
    return out


def run_c1(args, D, sp, seed, device):
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    vocab = D["vocab"]
    tok = {t: i + 1 for i, t in enumerate(vocab)}
    inv = {i: t for t, i in tok.items()}
    model = build_cslr(len(vocab) + 1).to(device)
    if args.init == "x16":
        st = torch.load(X16, map_location=device)
        own = model.state_dict()
        keep = {k: v for k, v in st.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
    by_sign = {t: [i for i, s in enumerate(D["sign_id"]) if s == t] for t in vocab}
    nulls = [i for i, s in enumerate(D["sign_id"]) if s.startswith("null")]
    held = {tuple(D["gloss"][i].split()) for i in np.concatenate([sp["val"], sp["test"]])}
    real = [(D["sents"][i], D["gloss"][i].split()) for i in sp["train"]]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    E, warm = args.epochs, 3
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, E - warm))))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def decode(idx):
        model.eval()
        hyps = []
        with torch.no_grad():
            for i in range(0, len(idx), 16):
                X, L = pad_batch([features(D["sents"][j]) for j in idx[i:i + 16]], device)
                lo, Lo = model(X, L)
                hyps += greedy(lo, Lo, inv)
        return hyps

    def evaluate(idx):
        return score([D["gloss"][i].split() for i in idx], decode(idx), [D["variation"][i] for i in idx])

    best, best_state, best_ep = 9e9, None, 0
    for ep in range(E):
        model.train()
        items = list(real)
        while len(items) < len(real) + args.synth:
            x, t = synth_sentence(D, rng, by_sign, nulls)
            if tuple(t) not in held:
                items.append((x, t))
        order = rng.permutation(len(items))
        for b in range(0, len(order), 8):
            batch = [items[k] for k in order[b:b + 8]]
            X, L = pad_batch([to_feat(x, rng) for x, _ in batch], device)
            lo, Lo = model(augment(X), L)
            tgt = torch.tensor([tok[t] for _, ts in batch for t in ts])
            tl = torch.tensor([len(ts) for _, ts in batch])
            loss = ctc(lo.log_softmax(-1).transpose(0, 1).cpu(), tgt, Lo.cpu(), tl)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 2 == 1 or ep == E - 1:
            v = evaluate(sp["val"])["wer"]
            if v < best:
                best, best_ep = v, ep
                best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    hyps = decode(sp["test"])
    return evaluate(sp["val"]), evaluate(sp["test"]), hyps, {"best_epoch": best_ep, "epochs": E,
                                                              "params": sum(p.numel() for p in model.parameters())}


# ---------------------------------------------------------------- C2

def tsl51_names():
    d = np.load(ROOT / "dataset" / "cache" / "tsl51_c54_T64.npz")
    return sorted(np.unique(d["sign_id"]))


def finetune_x16(device, epochs=30, seed=0):
    """X16 encoder + TSL-51 head fine-tuned on all TSL-51 isolated clips (expert + user_sign)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    d = np.load(ROOT / "dataset" / "cache" / "tsl51_c54_T64.npz")
    names = tsl51_names()
    y = np.array([names.index(s) for s in d["sign_id"]]) + K1
    model = build_model("S2S", K1 + len(names), PARTS_C54).to(device)
    model.load_state_dict(torch.load(X16, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xt, yt = torch.tensor(d["X"], device=device), torch.tensor(y - K1, device=device)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(len(y))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            loss = lossf(model(augment(Xt[b]))[:, K1:], yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def window_probs(model, xyz, W, stride, keep, device):
    import torch
    starts = list(range(0, max(1, len(xyz) - W + 1), stride))
    F = np.stack([features(resample(xyz[s:s + W], 64)) for s in starts])
    with torch.no_grad():
        lo = model(torch.tensor(F, device=device))[:, K1:][:, keep]
        return torch.softmax(lo, -1).cpu().numpy()


def decode_windows(P, labels, tau, m):
    lab = [labels[k] if p >= tau else None for k, p in zip(P.argmax(1), P.max(1))]
    lab = [None if (x is None or x.startswith("null")) else x for x in lab]
    runs, out = [], []
    for x in lab:
        if runs and runs[-1][0] == x:
            runs[-1][1] += 1
        else:
            runs.append([x, 1])
    for x, n in runs:
        if x is not None and n >= m and (not out or out[-1] != x):
            out.append(x)
    return out


def run_c2(args, D, device):
    import torch
    names = tsl51_names()
    keep = [i for i, n in enumerate(names) if n in D["vocab"] or n.startswith("null")]
    labels = [names[i] for i in keep]
    if args.c2_model == "x16ft":
        model = finetune_x16(device)
    else:
        model = build_model("S2S", K1 + len(names), PARTS_C54).to(device)
        model.load_state_dict(torch.load(X16, map_location=device))
    model.eval()
    Ws, stride, taus, ms = [30, 45, 60, 75, 90], 3, [0.2, 0.35, 0.5, 0.65, 0.8], [1, 2, 3, 5, 8]
    probs = {W: [window_probs(model, s, W, stride, keep, device) for s in D["sents"]] for W in Ws}
    results = []
    for protocol in ("rep", "template"):
        sp = split_idx(D, protocol)
        refs = lambda idx: [D["gloss"][i].split() for i in idx]  # noqa: E731
        best = None
        for W in Ws:
            for tau in taus:
                for m in ms:
                    hy = [decode_windows(probs[W][i], labels, tau, m) for i in sp["val"]]
                    w = score(refs(sp["val"]), hy, [D["variation"][i] for i in sp["val"]])["wer"]
                    if best is None or w < best[0]:
                        best = (w, W, tau, m)
        _, W, tau, m = best
        res = {}
        for part in ("val", "test"):
            idx = sp[part]
            hy = [decode_windows(probs[W][i], labels, tau, m) for i in idx]
            res[part] = score(refs(idx), hy, [D["variation"][i] for i in idx])
        results.append((protocol, res, {"window": W, "stride": stride, "tau": tau, "min_run": m},
                        [decode_windows(probs[W][i], labels, tau, m) for i in sp["test"]], sp["test"]))
    return results


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["c1", "c2"])
    ap.add_argument("--protocol", default="rep", choices=["rep", "template"])
    ap.add_argument("--init", default="x16", choices=["x16", "scratch"])
    ap.add_argument("--synth", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--c2-model", default="x16", choices=["x16", "x16ft"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    D = load_data()
    out_path = RESULTS / f"cslr{args.tag}.json"
    log = json.loads(out_path.read_text()) if out_path.exists() else []
    hyp_dir = RESULTS / "preds" / "cslr"
    hyp_dir.mkdir(parents=True, exist_ok=True)

    def save(protocol, model_name, seed, v, t, cfg, hyps, idx, sec):
        rows = [{"video_id": D["video_id"][i], "ref": D["gloss"][i], "hyp": " ".join(h),
                 "variation": D["variation"][i]} for i, h in zip(idx, hyps)]
        (hyp_dir / f"{protocol}_{model_name}_seed{seed}_test.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        log.append({"protocol": protocol, "model": model_name, "seed": seed, "val": v, "test": t,
                    "config": cfg, "seconds": sec, "device": device})
        out_path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
        print(f"  {protocol:8s} {model_name:22s} seed {seed} | val WER {v['wer']:.3f} | TEST WER {t['wer']:.3f} "
              f"word {t['wer_word']:.3f} SER {t['ser']:.3f} no_space {t.get('wer_no_space', float('nan')):.3f} "
              f"with_space {t.get('wer_with_space', float('nan')):.3f} ({sec:.0f}s) {cfg}", flush=True)

    if args.method == "c1":
        sp = split_idx(D, args.protocol)
        name = f"C1-{args.init}" + ("" if args.synth else "-nosynth")
        print(f"== C1 {args.protocol}: train {len(sp['train'])} val {len(sp['val'])} test {len(sp['test'])} "
              f"init {args.init} synth {args.synth}/epoch", flush=True)
        for seed in args.seeds:
            t0 = time.time()
            v, t, hyps, cfg = run_c1(args, D, sp, seed, device)
            save(args.protocol, name, seed, v, t, cfg, hyps, sp["test"], round(time.time() - t0, 1))
    else:
        t0 = time.time()
        for protocol, res, cfg, hyps, idx in run_c2(args, D, device):
            save(protocol, f"C2-{args.c2_model}", 0, res["val"], res["test"], cfg, hyps, idx,
                 round(time.time() - t0, 1))


if __name__ == "__main__":
    main()

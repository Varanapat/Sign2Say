"""Efficiency (Experiment Design section 7). Run on an otherwise idle machine.

- parameters per part, state-dict size fp32 / fp16
- model latency per clip, batch 1, warm-up 20, 200 timed runs, median + p95, on CPU and MPS
- preprocessing latency per clip (gap fill + normalize + resample + features)
- MediaPipe Holistic per-frame latency and FPS on CPU (native clip size and 640x480)
- training memory for one batch of 32 on MPS, and training time from the 10-fold logs
Writes results/efficiency.json.
"""

import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmarks import preprocess  # noqa: E402
from run_baselines import PARTS_C54, RESULTS, ROOT, build_model  # noqa: E402
from run_cslr import build_cslr  # noqa: E402


def timeit(fn, warm=20, n=200):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        a = time.perf_counter()
        fn()
        t.append((time.perf_counter() - a) * 1000)
    return float(np.median(t)), float(np.percentile(t, 95))


def size_mb(model, half=False):
    import torch
    buf = io.BytesIO()
    sd = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save(sd, buf)
    return buf.tell() / 2 ** 20


def main():
    import torch
    out = {"machine": f"{platform.machine()} {platform.platform()}", "torch": torch.__version__,
           "cpu_threads": torch.get_num_threads(), "models": {}}
    models = {"R S2S-LTR": build_model("S2S", 184, PARTS_C54), "X1 BiGRU": build_model("BiGRU", 184),
              "X16 joint (2 heads)": build_model("S2S", 236, PARTS_C54), "C1 CTC (CSLR)": build_cslr(32)}
    for name, m in models.items():
        m.eval()
        parts = {}
        for k, mod in m.named_children():
            parts[k] = sum(p.numel() for p in mod.parameters())
        parts["other"] = sum(p.numel() for p in m.parameters()) - sum(parts.values())
        r = {"params": sum(p.numel() for p in m.parameters()), "params_by_part": parts,
             "size_fp32_mb": size_mb(m), "size_fp16_mb": size_mb(m, True)}
        for dev in ("cpu", "mps"):
            mm = m.to(dev)
            if name.startswith("C1"):          # a 10-second sentence at 30 fps
                x = torch.randn(1, 300, 54, 7, device=dev)
                L = torch.tensor([300], device=dev)
                fn = lambda: mm(x, L)  # noqa: E731
            else:
                x = torch.randn(1, 64, 54, 7, device=dev)
                fn = lambda: mm(x)  # noqa: E731
            with torch.no_grad():
                if dev == "mps":
                    def f():
                        fn()
                        torch.mps.synchronize()
                    med, p95 = timeit(f)
                else:
                    med, p95 = timeit(fn)
            r[f"latency_{dev}_ms"] = {"median": med, "p95": p95}
        out["models"][name] = r
        print(f"{name:22s} params {r['params']/1e6:.2f}M size {r['size_fp32_mb']:.1f}/{r['size_fp16_mb']:.1f} MB "
              f"CPU {r['latency_cpu_ms']['median']:.2f} (p95 {r['latency_cpu_ms']['p95']:.2f}) ms  "
              f"MPS {r['latency_mps_ms']['median']:.2f} (p95 {r['latency_mps_ms']['p95']:.2f}) ms", flush=True)

    # training memory, one batch of 32 (MPS)
    for name, kind in (("R S2S-LTR", "S2S"), ("X1 BiGRU", "BiGRU")):
        torch.mps.empty_cache()
        m = build_model(kind, 184, PARTS_C54).to("mps").train()
        opt = torch.optim.AdamW(m.parameters())
        base = torch.mps.driver_allocated_memory()
        x = torch.randn(32, 64, 54, 7, device="mps")
        y = torch.randint(0, 184, (32,), device="mps")
        for _ in range(3):
            loss = torch.nn.functional.cross_entropy(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.mps.synchronize()
        out["models"][name]["train_mem_batch32_mb"] = (torch.mps.driver_allocated_memory() - base) / 2 ** 20
        print(f"{name} training memory (batch 32, MPS driver): {out['models'][name]['train_mem_batch32_mb']:.0f} MB")

    # preprocessing latency on a real clip
    z = np.load(sorted((ROOT / "dataset" / "features" / "landmarks" / "tsl1s").glob("*.npz"))[0])
    raw = z["c54"].astype(np.float32)
    med, p95 = timeit(lambda: preprocess(raw), 5, 100)
    out["preprocess_ms"] = {"median": med, "p95": p95, "frames": int(len(raw))}
    print(f"preprocess ({len(raw)} frames): {med:.2f} ms (p95 {p95:.2f})")

    # MediaPipe per frame, CPU
    import cv2
    import mediapipe as mp
    vids = sorted((ROOT / "TSL-ONE-S").glob("*.mp4"))[:5]
    for label, size in (("native", None), ("640x480", (640, 480))):
        frames = []
        for v in vids:
            cap = cv2.VideoCapture(str(v))
            while len(frames) < 60 * (vids.index(v) + 1):
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(cv2.resize(f, size) if size else f, cv2.COLOR_BGR2RGB))
            cap.release()
        with mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5) as hol:
            for f in frames[:10]:
                hol.process(f)
            t = []
            for f in frames:
                a = time.perf_counter()
                hol.process(f)
                t.append((time.perf_counter() - a) * 1000)
        out[f"mediapipe_{label}"] = {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
                                     "fps": 1000 / float(np.median(t)), "frames": len(t),
                                     "frame_size": f"{frames[0].shape[1]}x{frames[0].shape[0]}"}
        print(f"MediaPipe Holistic ({label}, {frames[0].shape[1]}x{frames[0].shape[0]}): "
              f"{np.median(t):.1f} ms/frame (p95 {np.percentile(t, 95):.1f}) = {1000/np.median(t):.1f} fps")

    # training time from the 10-fold logs
    tt = {}
    for p, key in ((RESULTS / "tsl1s_si_s2s.json", "S2S"), (RESULTS / "tsl1s_si_bigru.json", "BiGRU")):
        if p.exists():
            rs = [r for r in json.loads(p.read_text()) if r["model"] == key]
            tt[key] = {"median_s": float(np.median([r["seconds"] for r in rs])),
                       "median_epochs": float(np.median([r["config"]["epochs_run"] for r in rs])),
                       "runs": len(rs)}
    out["train_time_10fold"] = tt
    print("training time per fold (median, MPS, sometimes 2 jobs in parallel):", tt)
    (RESULTS / "efficiency.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

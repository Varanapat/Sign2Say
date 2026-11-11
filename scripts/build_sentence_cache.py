"""Cache for continuous recognition (C1/C2) and the sentence splits (Data Pipeline section 5).

Keeps every sequence at its native 30 fps (no resampling), as shoulder-normalized xyz with
NaN for missing points (gap-filled <= 5 frames), so CTC sees real timing and the sliding
window decoder can cut and resample windows itself.

Outputs
  dataset/cache/tsl51_sentences.npz
      sent_xyz [sum T, 54, 3], sent_off [N+1], video_id, sentence_id, gloss (space-separated
      sign_ids), words (space-separated, variants removed), variation (no_space / with_space)
      sign_xyz [sum T, 54, 3], sign_off [M+1], sign_id     (user_sign clips, same signer + camera,
                                                             used to synthesize training sentences)
  splits/tsl51_sentence_rep.json       last recording of each (template, variation) = test,
                                       second-to-last = val, rest = train  (seen sentences)
  splits/tsl51_sentence_template.json  whole gloss sequences held out (~20% test, ~10% val);
                                       every test/val sign_id also occurs in train sentences
"""

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmarks import fill_short_gaps, normalize, read_tsl51_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TSL51 = ROOT / "Source(Paper)" / "tsl51"
OUT = ROOT / "dataset" / "cache" / "tsl51_sentences.npz"


def rows(name):
    return list(csv.DictReader(open(TSL51 / "metadata" / name, encoding="utf-8-sig")))


def load(r):
    xyz = read_tsl51_csv((TSL51 / r["landmark_path"]).read_text(encoding="utf-8"),
                         float(r["width_extracted"]), float(r["height_extracted"]))
    return normalize(fill_short_gaps(xyz))


def word(sign_id):
    return sign_id.rsplit("_var_", 1)[0]


def main():
    sents = rows("sentence_metadata.csv")
    S, meta = [], []
    for r in sents:
        x = load(r)
        assert x is not None, r["video_id"]
        gl = r["sentence_id"].split("__")
        S.append(x.astype(np.float32))
        meta.append((r["video_id"], r["sentence_id"], " ".join(gl), " ".join(word(g) for g in gl),
                     r["recording_variation"]))
    vocab = sorted({g for m in meta for g in m[2].split()})

    U, uid = [], []
    for r in rows("user_sign_metadata.csv"):
        if r["sign_id"] in vocab or r["sign_id"].startswith("null"):
            x = load(r)
            if x is not None:
                U.append(x.astype(np.float32))
                uid.append(r["sign_id"])
    off = lambda L: np.cumsum([0] + [len(a) for a in L])  # noqa: E731
    vid, sid, gloss, words, var = map(np.array, zip(*meta))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, sent_xyz=np.concatenate(S), sent_off=off(S), video_id=vid, sentence_id=sid,
                        gloss=gloss, words=words, variation=var, sign_xyz=np.concatenate(U),
                        sign_off=off(U), sign_id=np.array(uid), vocab=np.array(vocab))
    print(f"{len(S)} sentences ({Counter(var)}), frames {min(map(len, S))}-{max(map(len, S))}; "
          f"vocab {len(vocab)} sign_ids; user_sign clips for synthesis {len(U)} "
          f"(covering {len(set(uid) & set(vocab))}/{len(vocab)} vocab + null {sum(u.startswith('null') for u in uid)})")

    # --- S-rep: repetitions of the same template
    by = defaultdict(list)
    for i, m in enumerate(meta):
        by[(m[1], m[4])].append(i)
    rep = {}
    for k, idx in by.items():
        idx = sorted(idx, key=lambda i: int(meta[i][0].rsplit("_", 1)[1]))
        for j, i in enumerate(idx):
            rep[meta[i][0]] = "test" if j == len(idx) - 1 else "val" if j == len(idx) - 2 and len(idx) > 2 else "train"
    # --- S-template: hold out whole gloss sequences (variant-free word sequence as the group)
    groups = sorted({m[3] for m in meta})
    for seed in range(1000):
        rng = random.Random(seed)
        g = groups[:]
        rng.shuffle(g)
        n_te, n_va = round(0.2 * len(g)), round(0.1 * len(g))
        te, va = set(g[:n_te]), set(g[n_te:n_te + n_va])
        tr_tok = {t for m in meta if m[3] not in te | va for t in m[2].split()}
        if all(t in tr_tok for m in meta if m[3] in te | va for t in m[2].split()):
            break
    tmpl = {m[0]: "test" if m[3] in te else "val" if m[3] in va else "train" for m in meta}
    for name, sp in (("rep", rep), ("template", tmpl)):
        p = ROOT / "splits" / f"tsl51_sentence_{name}.json"
        p.write_text(json.dumps(sp, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"S-{name}: {Counter(sp.values())} -> {p.relative_to(ROOT)}"
              + (f" (seed {seed}, {len(te)} test / {len(va)} val sequences)" if name == "template" else ""))


if __name__ == "__main__":
    main()

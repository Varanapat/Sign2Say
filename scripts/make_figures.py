"""Architecture diagrams and result charts for the presentation (Knowledge Base/figures/).

Diagrams are drawn from the actual model definitions (run_baselines.py, stgcn.py, run_cslr.py,
run_joint.py); result charts read the numbers from results/*.json, nothing is typed in by hand
except the friend's fold-4 numbers (EXPERIMENTS.md of branch base_line).

    .venv/bin/python scripts/make_figures.py          # writes PNG (200 dpi) + SVG
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Knowledge Base" / "figures"
R = ROOT / "results"

# component colours (fill, edge) — by role, same in every diagram
ROLE = {
    "data":     ("#eeede8", "#8a8983"),
    "spatial":  ("#dce9f9", "#2a78d6"),
    "temporal": ("#fde6db", "#d95926"),
    "st":       ("#efe9fb", "#6250d6"),
    "head":     ("#d9f2e8", "#138a5f"),
    "loss":     ("#fbf1d4", "#b07800"),
    "plain":    ("#ffffff", "#8a8983"),
}
FIX = "#e34948"           # outline + badge for "what we changed / fixed"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURFACE = "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})


def canvas(w, h, scale=0.13):
    fig = plt.figure(figsize=(w * scale, h * scale), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, sub=None, role="plain", fix=False, fs=10.5, sfs=8.5, bold=True):
    """Box centred at (x, y). fix=True draws the red dashed outline used for 'what we changed'."""
    fc, ec = ROLE[role]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fc, ec=FIX if fix else ec, lw=2.0 if fix else 1.3,
                                ls=(0, (4, 2)) if fix else "-", zorder=2))
    if sub:
        ax.text(x, y + h * 0.13, title, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", zorder=3)
        ax.text(x, y - h * 0.2, sub, ha="center", va="center", fontsize=sfs, color=INK2, zorder=3,
                linespacing=1.25)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p, q, label=None, dx=0, dy=1.6, color=INK2, fs=8, style="-|>", ls="-"):
    ax.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.3, shrinkA=0, shrinkB=0, ls=ls,
                                mutation_scale=11))
    if label:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label, ha="center", va="bottom",
                fontsize=fs, color=MUTED, family="DejaVu Sans Mono")


def badge(ax, x, y, text):
    """Red problem badge, e.g. 'P3' — the number matches the problem in the presentation script."""
    ax.add_patch(Circle((x, y), 1.9, fc=FIX, ec="white", lw=1.2, zorder=5))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=6)


def title(ax, x, y, text, sub=None):
    ax.text(x, y, text, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    if sub:
        ax.text(x, y - 3.6, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def legend_fix(ax, x, y, text="red dashed = what we changed  ·  Pn = problem n in the script"):
    ax.add_patch(FancyBboxPatch((x, y - 1), 4, 2, boxstyle="round,pad=0,rounding_size=0.5",
                                fc="white", ec=FIX, lw=1.6, ls=(0, (4, 2))))
    ax.text(x + 5, y, text, ha="left", va="center", fontsize=8, color=INK2)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=SURFACE)
    fig.savefig(OUT / f"{name}.svg", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- 1. data pipeline
def fig_pipeline():
    fig, ax = canvas(120, 46)
    title(ax, 3, 44.5, "Data pipeline: video → landmark tensor",
          "same pipeline for every model; TSL-51 CSVs enter at step 3 (same C54 points)")
    y1, y2, w, h = 31, 11, 22, 11
    xs = [14, 43, 72, 101]
    row1 = [("Video", "RGB, 30 fps\n[T × H × W × 3]", "data", False),
            ("MediaPipe Holistic", "v0.10.21, CPU\n543 landmarks / frame", "spatial", True),
            ("Select C54", "arms 6 · brows+mouth 6\nhands 21 × 2", "spatial", True),
            ("Fill gaps", "linear, ≤ 5 frames\nlonger gaps stay 0", "plain", True)]
    row2 = [("Normalize", "origin = mid-shoulder\n÷ shoulder width", "plain", True),
            ("Resample", "T → 64 frames\n(linear)", "plain", True),
            ("7 features / point", "x, y, z, Δx, Δy, Δz\nvalid (0/1)", "plain", True),
            ("Tensor", "[64 × 54 × 7]\n→ ISLR models", "head", False)]
    for i, (t, s, r, f) in enumerate(row1):
        box(ax, xs[i], y1, w, h, t, s, r, fix=f)
        if i:
            arrow(ax, (xs[i - 1] + w / 2, y1), (xs[i] - w / 2, y1))
    for i, (t, s, r, f) in enumerate(row2):
        box(ax, xs[i], y2, w, h, t, s, r, fix=f)
        if i:
            arrow(ax, (xs[i - 1] + w / 2, y2), (xs[i] - w / 2, y2))
    # bend from end of row 1 to start of row 2
    ax.plot([xs[3], xs[3], xs[0], xs[0]], [y1 - h / 2, 21, 21, y2 + h / 2 + 0.2], color=INK2, lw=1.3, zorder=1)
    arrow(ax, (xs[0], 21.5), (xs[0], y2 + h / 2))
    for (x, y), t in zip([(xs[1] - 9.5, y1 + 5), (xs[2] - 9.5, y1 + 5), (xs[3] - 9.5, y1 + 5),
                          (xs[0] - 9.5, y2 + 5), (xs[1] - 9.5, y2 + 5), (xs[2] - 9.5, y2 + 5)],
                         ["P1", "P2", "P3", "P4", "P4", "P3"]):
        badge(ax, x, y, t)
    ax.text(xs[3], y2 - 8.2, "CSLR: no resample (keeps all T frames)", ha="center", fontsize=8, color=MUTED)
    legend_fix(ax, 3, 2)
    save(fig, "fig1_pipeline")


# ---------------------------------------------------------------- 2. baseline vs S2S-LTR
def fig_s2s():
    fig, ax = canvas(130, 72)
    title(ax, 3, 70, "From baseline to S2S-LTR (main model, 3.28M params)",
          "B = batch, T = 64 frames · shapes per clip")
    # --- baseline strip
    yb = 54
    ax.text(3, yb + 6.5, "Before — X1 BiGRU (2.16M): 77.6%", fontsize=10.5, fontweight="bold", color=INK2)
    items = [(13, "Input", "[64 × 54 × 7]", "data"), (36, "Flatten + Linear", "378 → 256", "spatial"),
             (61, "BiGRU × 2", "256 → 512, step by step", "temporal"), (86, "Mean over T", "[512]", "plain"),
             (108, "Linear", "→ 184 words", "head")]
    for i, (x, t, s, r) in enumerate(items):
        box(ax, x, yb, 19, 8, t, s, r, fs=9.5, sfs=8)
        if i:
            arrow(ax, (items[i - 1][0] + 9.5, yb), (x - 9.5, yb))
    badge(ax, 28, yb + 4.6, "P6")
    ax.text(36, yb - 6.2, "one Linear mixes hands, arms, face", ha="center", fontsize=8, color=FIX)
    ax.text(61, yb - 6.2, "sequential → slow on GPU", ha="center", fontsize=8, color=FIX)
    ax.plot([3, 127], [45, 45], color="#d8d6d0", lw=1)

    # --- S2S-LTR
    ax.text(3, 42, "After — S2S-LTR: 84.8% (10 folds), wins 10/10 folds", fontsize=10.5, fontweight="bold", color=INK)
    box(ax, 9, 22, 11, 30, "Input\n\n[64 × 54 × 7]", None, "data", fs=9.5)
    parts = [(34, "Left hand", "21 pts × 7 = 147", "96"), (27, "Right hand", "21 pts × 7 = 147", "96"),
             (20, "Arms", "6 pts × 7 = 42", "32"), (13, "Brows + mouth", "6 pts × 7 = 42", "32")]
    for y, name, dim, out in parts:
        arrow(ax, (14.5, y), (19, y))
        box(ax, 28, y, 17, 5.6, name, dim, "data", fs=8.5, sfs=7.5)
        arrow(ax, (36.5, y), (40, y))
        box(ax, 47, y, 13, 5.6, "MLP", f"Lin-GELU-Lin → {out}", "spatial", fix=True, fs=9, sfs=7.2)
        arrow(ax, (53.5, y), (58, 23.5))
    ax.text(47, 39, "part-aware spatial encoder\n(per frame, shared over T)", ha="center", fontsize=8, color=INK2)
    box(ax, 63, 23.5, 10, 14, "Concat", "+ LayerNorm\n[64 × 256]", "spatial", fs=9.5, sfs=8)
    arrow(ax, (68, 23.5), (72, 23.5))
    box(ax, 78, 23.5, 11, 14, "[CLS] +", "learned PE\n[65 × 256]", "temporal", fs=9.5, sfs=8)
    arrow(ax, (83.5, 23.5), (87, 23.5))
    # transformer stack
    ax.add_patch(FancyBboxPatch((87, 9), 21, 30, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=ROLE["temporal"][0], ec=FIX, lw=2.0, ls=(0, (4, 2)), zorder=2))
    ax.text(97.5, 36.5, "Transformer encoder", ha="center", fontsize=9.5, fontweight="bold", zorder=3)
    for i, (t, r) in enumerate([("LayerNorm", "plain"), ("Multi-head attention (4)", "temporal"), ("⊕ residual", None),
                                ("LayerNorm", "plain"), ("FFN 256→1024→256", "temporal"), ("⊕ residual", None)]):
        yy = 32.5 - i * 4
        if r:
            box(ax, 97.5, yy, 17, 3.1, t, None, r, fs=7.6, bold=False)
        else:
            ax.text(97.5, yy, t, ha="center", va="center", fontsize=7.6, color=INK2, zorder=3)
    ax.text(97.5, 10.5, "× 4 layers · d 256 · pre-norm", ha="center", fontsize=8, color=INK2, zorder=3)
    arrow(ax, (108, 23.5), (112, 23.5), "[CLS]", dy=1.0)
    box(ax, 120, 23.5, 13, 14, "Head", "LayerNorm\n→ Linear\n→ 184 words", "head", fs=9.5, sfs=8)
    badge(ax, 54, 36.6, "P6")
    badge(ax, 88.5, 38, "P6")
    ax.text(97.5, 5.5, "every frame attends to every frame → parallel over T", ha="center", fontsize=8, color=FIX)
    ax.text(47, 7.2, "hands get most capacity (96 + 96 of 256)", ha="center", fontsize=8, color=FIX)
    legend_fix(ax, 3, 1.6)
    save(fig, "fig2_s2s_ltr")


# ---------------------------------------------------------------- 3. RQ-S levels (friend)
def fig_levels():
    fig, ax = canvas(130, 50)
    title(ax, 3, 48, "Friend's question (RQ-S): learn space and time separately or together?",
          "same input, same training recipe (friend's), fold 4, 3 seeds — test top-1")
    cols = [(23, "L0  spatial only", "no time order"), (66, "L1  spatial → temporal", "space first, then time"),
            (109, "L2  space + time together", "local in joints and frames")]
    for x, t, s in cols:
        ax.text(x, 38, t, ha="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(x, 35, s, ha="center", fontsize=8.5, color=INK2)
    # L0
    for i in range(4):
        box(ax, 9 + i * 7, 27, 5.6, 4, f"t{i + 1}", None, "data", fs=8, bold=False)
        box(ax, 9 + i * 7, 20, 5.6, 4, "MLP", None, "spatial", fs=7.5, bold=False)
        arrow(ax, (9 + i * 7, 25), (9 + i * 7, 22))
        arrow(ax, (9 + i * 7, 18), (23, 14.5))
    box(ax, 23, 12, 18, 4.4, "mean over frames", None, "plain", fs=8.5, bold=False)
    ax.text(23, 6.5, "B0 MLP 67.6 · A1 SPOTER 70.6", ha="center", fontsize=9, color=INK)
    # L1
    for i in range(4):
        box(ax, 52 + i * 7, 27, 5.6, 4, f"t{i + 1}", None, "data", fs=8, bold=False)
        box(ax, 52 + i * 7, 20, 5.6, 4, "MLP", None, "spatial", fs=7.5, bold=False)
        arrow(ax, (52 + i * 7, 25), (52 + i * 7, 22))
        arrow(ax, (52 + i * 7, 18), (52 + i * 7, 14.4))
    box(ax, 62.5, 12, 27, 4.4, "GRU / Transformer over t", None, "temporal", fs=8.5, bold=False)
    ax.text(66, 6.5, "B1 GRU 69.0 · B2 Transformer 69.7", ha="center", fontsize=9, color=INK)
    # L2
    for j in range(3):
        box(ax, 109, 27 - j * 6.2, 26, 4.4, f"ST block {j + 1}: graph conv + temporal conv", None, "st", fs=7.8, bold=False)
        if j:
            arrow(ax, (109, 27 - (j - 1) * 6.2 - 2.2), (109, 27 - j * 6.2 + 2.2))
    ax.text(109, 6.5, "B3 ST-GCN 75.0  (B4 attention 61.5)", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.text(65, 1.8, "L1 − L0 ≈ +1–2 (n.s.) · L2 (ST-GCN) − L1 = +5.3, p = 3e-4 · graph alone does not help (B6 GCN→GRU 69.1)",
            ha="center", fontsize=8.5, color=INK2)
    badge(ax, 122.5, 40, "P7")
    save(fig, "fig3_rqs_levels")


# ---------------------------------------------------------------- 4. ST-GCN
def fig_stgcn():
    fig, ax = canvas(130, 62)
    title(ax, 3, 60, "ST-GCN (L2) — re-implemented in scripts/stgcn.py, 3.14M params",
          "Yan et al. 2018 · graph = C54 skeleton (arms, brows/mouth, two 21-point hands)")
    y = 38
    box(ax, 9, y, 12, 12, "Input", "[C × 64 × 54]\nC = 7 (+2 HL)", "data", fs=9.5, sfs=7.8)
    arrow(ax, (15, y), (18.5, y))
    box(ax, 23, y, 8, 12, "Data", "BN per\njoint×ch", "plain", fs=8.5, sfs=7.5)
    arrow(ax, (27, y), (30, y))
    chans = [(64, 1), (64, 1), (64, 1), (128, 2), (128, 1), (128, 1), (256, 2), (256, 1), (256, 1)]
    x, T = 31, 64
    for i, (c, s) in enumerate(chans):
        T = T // s
        hgt = 6 + c / 256 * 12
        ax.add_patch(FancyBboxPatch((x, y - hgt / 2), 6.4, hgt, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=ROLE["st"][0], ec=ROLE["st"][1], lw=1.2, zorder=2))
        ax.text(x + 3.2, y, str(c), ha="center", va="center", fontsize=8, zorder=3, rotation=90)
        ax.text(x + 3.2, y - hgt / 2 - 1.6, f"T{T}" + ("↓" if s == 2 else ""), ha="center", fontsize=7, color=MUTED)
        x += 7.4
    ax.text(31 + 7.4 * 4.5 - 0.5, y + 10.6, "9 ST-GCN blocks (channels; ↓ = temporal stride 2)", ha="center",
            fontsize=8.5, color=INK2)
    arrow(ax, (x, y), (x + 3.5, y))
    box(ax, x + 9, y, 11, 12, "Global", "avg pool\nover T, V", "plain", fs=9, sfs=7.8)
    arrow(ax, (x + 14.5, y), (x + 18, y))
    box(ax, x + 23.5, y, 11, 12, "Linear", "→ 184\nwords", "head", fs=9.5, sfs=7.8)
    # block zoom
    yz = 13
    ax.text(3, 24.5, "One block", fontsize=10, fontweight="bold", color=INK)
    seq = [(10, "x", "[C × T × V]", "data", 9), (27, "1×1 conv", "C → 3·C'", "st", 12),
           (47, "Σ_k (A_k ⊙ M_k)", "graph conv, K = 3\nself / toward / away\nfrom shoulders", "st", 18),
           (67, "BN · ReLU", None, "plain", 10), (84, "9 × 1 conv", "temporal, stride s", "st", 14),
           (101, "BN · Dropout", None, "plain", 12), (117, "⊕ · ReLU", "residual", "plain", 11)]
    for i, (xx, t, s, r, w) in enumerate(seq):
        box(ax, xx, yz, w, 9, t, s, r, fs=8.8, sfs=7.2)
        if i:
            arrow(ax, (seq[i - 1][0] + seq[i - 1][4] / 2, yz), (xx - w / 2, yz))
    ax.plot([10, 10, 117, 117], [yz - 4.5, 4.2, 4.2, yz - 4.5], color=MUTED, lw=1.1, ls="--", zorder=1)
    ax.text(63, 2.0, "residual (1×1 conv if shape changes)", ha="center", fontsize=7.5, color=MUTED)
    ax.text(47, 20.3, "A = fixed skeleton adjacency · M = learned edge importance", ha="center", fontsize=7.5, color=MUTED)
    save(fig, "fig4_stgcn")


# ---------------------------------------------------------------- 5. hand-local
HAND = [(0, 0), (-0.35, 0.25), (-0.6, 0.5), (-0.8, 0.7), (-0.95, 0.85),
        (-0.35, 0.95), (-0.42, 1.35), (-0.46, 1.6), (-0.5, 1.82),
        (-0.1, 1.0), (-0.1, 1.45), (-0.1, 1.73), (-0.1, 1.97),
        (0.15, 0.95), (0.2, 1.35), (0.22, 1.6), (0.24, 1.8),
        (0.38, 0.85), (0.48, 1.15), (0.53, 1.35), (0.57, 1.52)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def draw_hand(ax, ox, oy, s, color, lw=1.2, ms=2.2):
    pts = [(ox + x * s, oy + y * s) for x, y in HAND]
    for a, b in HAND_EDGES:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=color, lw=lw, zorder=3)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=ms, color=color, zorder=4)
    return pts


def fig_handlocal():
    fig, ax = canvas(130, 58)
    title(ax, 3, 56.5, "Hand-local feature (friend's finding) added to S2S-LTR = S2S-HL",
          "computed inside the model after augmentation · +2 channels · params unchanged (3.28M)")
    # panel A: shoulder-normalised body
    cx, cy, S = 22, 26, 13          # 1 shoulder width = 15 drawing units
    ax.add_patch(Rectangle((4, 9), 36, 38, fc="white", ec="#d8d6d0", lw=1))
    ax.text(22, 44.5, "A. shoulder coordinates (what the model saw)", ha="center", fontsize=8.6, color=INK, fontweight="bold")
    ax.add_patch(Circle((cx, cy + 0.85 * S), 0.32 * S, fc="none", ec="#d8d6d0", lw=1.2))
    body = {"ls": (cx + 0.5 * S, cy), "rs": (cx - 0.5 * S, cy), "le": (cx + 0.72 * S, cy - 0.75 * S),
            "re": (cx - 0.72 * S, cy - 0.75 * S), "lw": (cx + 0.3 * S, cy - 0.2 * S), "rw": (cx - 0.28 * S, cy + 0.12 * S)}
    for a, b in [("ls", "rs"), ("ls", "le"), ("le", "lw"), ("rs", "re"), ("re", "rw")]:
        ax.plot([body[a][0], body[b][0]], [body[a][1], body[b][1]], color=MUTED, lw=1.6)
    for x, y in [(-0.2, 1.02), (-0.08, 1.05), (0.08, 1.05), (0.2, 1.02), (-0.1, 0.7), (0.1, 0.7)]:
        ax.scatter([cx + x * S], [cy + y * S], s=6, color=MUTED, zorder=4)
    hs = 0.17 * S
    draw_hand(ax, *body["lw"], hs, "#2a78d6")
    rp = draw_hand(ax, *body["rw"], hs, "#2a78d6")
    xs_, ys_ = [p[0] for p in rp], [p[1] for p in rp]
    ax.add_patch(Rectangle((min(xs_) - 0.5, min(ys_) - 0.5), max(xs_) - min(xs_) + 1, max(ys_) - min(ys_) + 1,
                           fc="none", ec=FIX, lw=1.2, ls="--"))
    ax.text(22, 11.5, "a hand spans ≈ 0.3 shoulder widths\n→ finger detail ≈ 0.02 units, mixed with hand position",
            ha="center", fontsize=7.8, color=FIX)
    badge(ax, 6.5, 44.6, "P8")
    arrow(ax, (max(xs_) + 0.5, (min(ys_) + max(ys_)) / 2), (47, 26), ls="--", color=FIX)
    # panel B: hand-local
    ax.add_patch(Rectangle((46, 9), 32, 38, fc="white", ec="#d8d6d0", lw=1))
    ax.text(62, 44.5, "B. hand-local (per hand, per frame)", ha="center", fontsize=8.6, color=INK, fontweight="bold")
    hp = draw_hand(ax, 63, 14.5, 9, "#2a78d6", lw=1.8, ms=8)
    ax.plot([63], [14.5], "o", ms=7, mfc="white", mec=FIX, mew=1.6, zorder=6)
    ax.text(64.5, 12.2, "wrist = (0, 0)", fontsize=7.5, color=FIX)
    ax.text(62, 9.9 + 0.3, "", fontsize=1)
    ax.text(62, 39.5, "h = (p − p_wrist) / max(bbox w, h)", ha="center", fontsize=8.6, color=INK, family="DejaVu Sans Mono")
    ax.text(62, 36.6, "values in [−1, 1] · non-hand points = 0", ha="center", fontsize=7.6, color=INK2)
    # panel C: model
    yx = 28
    box(ax, 91, 40, 18, 6, "x  [64 × 54 × 7]", "after augmentation", "data", fs=8.6, sfs=7.2)
    arrow(ax, (91, 37), (91, 33))
    box(ax, 91, 30, 18, 6, "HandLocal()", "torch, no parameters", "spatial", fix=True, fs=9, sfs=7.2)
    arrow(ax, (100, 40), (112, 40))
    arrow(ax, (100, 30), (112, 26.5), "+2 ch", dy=0.6)
    box(ax, 119, 33, 13, 16, "Concat", "[64 × 54 × 9]", "data", fs=9, sfs=7.8)
    arrow(ax, (119, 25), (119, 21.5))
    box(ax, 112, 17, 30, 8, "S2S-LTR (unchanged)", "hand MLP 21×9 = 189 → 96", "spatial", fs=9, sfs=7.6)
    arrow(ax, (112, 13), (112, 9.6))
    box(ax, 112, 6.8, 30, 5, "184 words", None, "head", fs=9)
    ax.text(62, 4.0, "5 folds: 84.2 → 87.1% (+2.9 pp, better on 5/5 folds, McNemar p = 1.8e-5)",
            ha="center", fontsize=9.5, color=INK, fontweight="bold")
    legend_fix(ax, 3, 1.2, "red dashed = new module")
    save(fig, "fig5_hand_local")


# ---------------------------------------------------------------- 6. X16 joint encoder
def fig_joint():
    fig, ax = canvas(120, 44)
    title(ax, 3, 42, "X16: one encoder, two datasets, two heads",
          "the vocabularies barely overlap → share the encoder, not the classifier")
    box(ax, 15, 27, 22, 8, "TSL-ONE-S clip", "184 words, 29 signers", "data", fs=9.5, sfs=8)
    box(ax, 15, 12, 22, 8, "TSL-51 clip", "51 signs + 1, 3 people", "data", fs=9.5, sfs=8)
    arrow(ax, (26, 27), (40, 21))
    arrow(ax, (26, 12), (40, 18))
    box(ax, 55, 19.5, 28, 14, "Shared S2S-LTR encoder", "part-aware MLP + Transformer ×4\n→ [CLS] 256", "spatial",
        fix=True, fs=9.5, sfs=8)
    arrow(ax, (69, 22), (82, 27))
    arrow(ax, (69, 17), (82, 12))
    box(ax, 94, 27, 22, 8, "Head A → 184", "loss only on TSL-ONE-S clips", "head", fix=True, fs=9.5, sfs=7.6)
    box(ax, 94, 12, 22, 8, "Head B → 52", "loss only on TSL-51 clips", "head", fix=True, fs=9.5, sfs=7.6)
    ax.text(60, 3.2, "TSL-51 new signer: 76.2 → 80.9% (+4.7) · TSL-ONE-S unchanged (83.7 vs 84.2, p = 0.38)",
            ha="center", fontsize=9.5, color=INK, fontweight="bold")
    badge(ax, 42.5, 26.8, "P10")
    save(fig, "fig6_joint_x16")


# ---------------------------------------------------------------- 7. C1 CTC
def fig_ctc():
    fig, ax = canvas(130, 50)
    title(ax, 3, 48.5, "C1: continuous sentences with CTC (3.88M params)",
          "no word boundaries are given — CTC learns the alignment from the word sequence alone")
    y = 32
    seq = [(10, "Sentence", "[T × 54 × 7]\nT varies, 30 fps", "data", 15, False),
           (30, "Part-aware", "encoder per frame\n(init from X16)\n[T × 256]", "spatial", 17, False),
           (51, "Conv1d k5 s2", "+ GELU, × 2\n[T/4 × 256]", "temporal", 17, True),
           (71, "Transformer", "× 4, sinusoidal PE\npadding mask", "temporal", 17, False),
           (90, "Linear", "LayerNorm →\n31 signs + blank", "head", 14, False),
           (111, "CTC loss", "train\n(greedy decode at test)", "loss", 18, False)]
    for i, (x, t, s, r, w, f) in enumerate(seq):
        box(ax, x, y, w, 13, t, s, r, fix=f, fs=9.5, sfs=7.6)
        if i:
            arrow(ax, (seq[i - 1][0] + seq[i - 1][4] / 2, y), (x - w / 2, y))
    ax.text(51, y + 9, "4× shorter → faster, smoother", ha="center", fontsize=7.6, color=FIX)
    # output example
    ax.text(111, y - 10, "→  I  LIKE  TRAVEL", ha="center", fontsize=10, color=INK, family="DejaVu Sans Mono",
            fontweight="bold")
    # synthetic sentence generator
    yg = 9
    box(ax, 12, yg, 18, 9, "user_sign clips", "same signer,\nsingle words", "data", fs=8.8, sfs=7.4)
    arrow(ax, (21, yg), (26, yg))
    box(ax, 44, yg, 34, 9, "Synthetic sentence", "3–6 words · trim 0–20% · 6-frame blend\nrandom pauses · not val/test orders",
        "loss", fix=True, fs=9, sfs=7.4)
    arrow(ax, (61, yg), (66, yg))
    box(ax, 77, yg, 20, 9, "+300 / epoch", "mixed with the real\ntrain sentences", "plain", fs=9, sfs=7.4)
    ax.plot([77, 77, 10], [yg + 4.5, 19, 19], color=INK2, lw=1.2, zorder=1)
    arrow(ax, (10, 18.8), (10, y - 6.5))
    ax.text(44, 19.8, "training data", ha="center", fontsize=7.8, color=MUTED)
    badge(ax, 28.5, yg + 4.8, "P11")
    ax.text(100, 6.2, "new word orders: WER 36.4 → 11.8%\nseen sentences: WER 5.1%", ha="left", fontsize=9,
            color=INK, fontweight="bold", linespacing=1.4)
    legend_fix(ax, 3, 1.2, "red dashed = what we added")
    save(fig, "fig7_cslr_ctc")


# ---------------------------------------------------------------- 8. full system
def fig_system():
    fig, ax = canvas(130, 46)
    title(ax, 3, 44, "Sign2Say: final system", "who contributed what · latency on Apple M5 (CPU)")
    y = 22
    items = [(10, "Camera", "webcam", "data", 13), (29, "MediaPipe", "23.7 ms / frame\n= 42 fps", "spatial", 17),
             (48, "Preprocess", "C54 · fill · norm\n0.8 ms / clip", "plain", 16),
             (68, "HandLocal", "+2 channels", "spatial", 15),
             (89, "S2S-LTR", "part-aware +\nTransformer\n1.3 ms / clip", "temporal", 18),
             (113, "Word", "184 signs\n(87.1%, 5 folds)", "head", 18)]
    for i, (x, t, s, r, w) in enumerate(items):
        box(ax, x, y, w, 12, t, s, r, fix=(t == "HandLocal"), fs=9.5, sfs=7.6)
        if i:
            arrow(ax, (items[i - 1][0] + items[i - 1][4] / 2, y), (x - w / 2, y))
    box(ax, 101, 7, 22, 7, "C1 CTC → sentence", "WER 5–12% (1 signer)", "loss", fs=8.8, sfs=7.4)
    ax.annotate("", xy=(90, 9), xytext=(48, 16), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2,
                                                                   connectionstyle="arc3,rad=0.25"))
    for x0, x1, t, c in [(21, 57, "core-pipeline (us): data, protocol, recipe", "#2a78d6"),
                         (61, 76, "base_line (friend)", FIX), (80, 98, "core-pipeline", "#2a78d6")]:
        ax.plot([x0, x1], [34, 34], color=c, lw=3, solid_capstyle="round")
        ax.text((x0 + x1) / 2, 35.5, t, ha="center", fontsize=8, color=c)
    ax.text(65, 2.0, "camera → word in ≈ 2 ms after the sign ends (MediaPipe runs while signing)",
            ha="center", fontsize=8.6, color=INK2)
    save(fig, "fig8_system")


# ---------------------------------------------------------------- charts
SER = {"S2S": ("#2a78d6", "S2S-LTR"), "S2S-HL": ("#eb6834", "S2S-LTR + hand-local"),
       "STGCN-HL": ("#1baf7a", "ST-GCN + hand-local"), "BiGRU": ("#8a8983", "BiGRU (baseline)")}


def load_top1(models):
    out = {}
    for p in sorted(R.glob("tsl1s_si*.json"), key=lambda p: p.stat().st_mtime):
        for r in json.loads(p.read_text()):
            if r["model"] in models and r["seed"] == 0 and r.get("variant", "base") == "base":
                out[(r["model"], int(r["split"].rsplit("fold", 1)[1]))] = r["test"]["top1"] * 100
    return out


def dot_chart(models, folds, name, ttl, sub, xlim):
    d = load_top1(models)
    rows = [f"fold {f}" for f in folds] + ["mean"]
    fig = plt.figure(figsize=(9.5, 0.42 * len(rows) + 1.9), facecolor=SURFACE)
    ax = fig.add_axes([0.11, 0.13, 0.86, 0.62])
    ax.set_facecolor(SURFACE)
    for i, f in enumerate(folds + ["mean"]):
        yy = len(rows) - 1 - i
        vals = []
        for m in models:
            if f == "mean":
                v = sum(d[(m, k)] for k in folds) / len(folds)
            else:
                v = d.get((m, f))
            vals.append(v)
        vv = [v for v in vals if v is not None]
        ax.plot([min(vv), max(vv)], [yy, yy], color="#d8d6d0", lw=2, zorder=1)
        for j, (m, v) in enumerate(zip(models, vals)):
            if v is not None:
                off = (j - (len(models) - 1) / 2) * 0.16      # small vertical offset so equal values stay visible
                ax.scatter([v], [yy + off], s=90 if f == "mean" else 64, color=SER[m][0], edgecolor=SURFACE,
                           linewidth=2, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows[::-1], fontsize=9.5, color=INK2)
    ax.axhline(0.5, color="#d8d6d0", lw=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("test top-1 accuracy (%) — signers never seen in training", fontsize=9.5, color=INK2)
    ax.grid(axis="x", color="#ebe9e4", lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bdbbb5")
    ax.tick_params(colors=INK2, length=0)
    fig.text(0.02, 0.95, ttl, fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.95 - 0.33 / (0.42 * len(rows) + 1.9), sub, fontsize=9, color=INK2, va="top")
    for j, m in enumerate(models):
        mean = sum(d[(m, k)] for k in folds) / len(folds)
        x = 0.11 + j * 0.29
        fig.text(x, 0.80, "●", color=SER[m][0], fontsize=12, va="center")
        fig.text(x + 0.022, 0.80, f"{SER[m][1]}  (mean {mean:.1f})", color=INK, fontsize=9.5, va="center")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=SURFACE)
    fig.savefig(OUT / f"{name}.svg", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_s2s()
    fig_levels()
    fig_stgcn()
    fig_handlocal()
    fig_joint()
    fig_ctc()
    fig_system()
    dot_chart(["BiGRU", "S2S"], list(range(10)), "chart1_10fold",
              "S2S-LTR beats the BiGRU baseline on every fold", "10 signer-independent folds, seed 0 · Wilcoxon p = 0.002",
              (70, 95))
    dot_chart(["S2S", "STGCN-HL", "S2S-HL"], [0, 2, 4, 6, 8], "chart2_combined",
              "Adding the friend's hand-local feature: 84.2 → 87.1%",
              "same recipe and folds · S2S-HL better on 5/5 folds (p = 1.8e-5) · ST-GCN-HL below S2S-HL on 5/5",
              (78, 95))

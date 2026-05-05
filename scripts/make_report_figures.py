"""
Generate report figures from outputs/eval/*.json and the per-epoch logs.

Run: python scripts/make_report_figures.py
Output: figures/*.png
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
EVAL = ROOT / "outputs" / "eval"
LOGS = ROOT / "logs"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load_eval(name):
    with open(EVAL / f"{name}.json") as f:
        return json.load(f)


# ---- Figure 1: Cross-domain Pareto frontier (KD sweep) ----
def fig_pareto():
    points = [
        ("Zero-shot",          load_eval("v4_remap_zeroshot")["datasets"]),
        ("No-KD\n(v3 headline)", load_eval("v4_remap_v3_no_h36m_rank2")["datasets"]),
        ("KD λ=1",             load_eval("v4_kd1")["datasets"]),
        ("KD λ=10",            load_eval("v4_kd10")["datasets"]),
        ("KD λ=100",           load_eval("v4_kd100")["datasets"]),
        ("KD λ=1000",          load_eval("v4_kd1000")["datasets"]),
    ]
    fit3d = [p[1]["Fit3D"]["overall"]["mpjpe"] for p in points]
    h36m  = [p[1]["H36M"]["overall"]["p_mpjpe"] for p in points]
    labels = [p[0] for p in points]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    colors = ["#cccccc", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728"]
    for x, y, lbl, c in zip(fit3d, h36m, labels, colors):
        ax.scatter(x, y, s=120, c=c, edgecolors="black", linewidth=0.8, zorder=3)
        # Label offsets
        dx, dy = 3, 2
        if lbl == "KD λ=1":     dx, dy = -3, 4
        if lbl == "Zero-shot":  dx, dy = 3, 2
        if lbl == "No-KD\n(v3 headline)": dx, dy = 3, -8
        ax.annotate(lbl, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, ha="left")

    # Connect with a smooth line through the KD sweep points (sorted by Fit3D MPJPE)
    sweep = sorted(zip(fit3d[2:], h36m[2:]))
    sx = [s[0] for s in sweep]
    sy = [s[1] for s in sweep]
    ax.plot(sx, sy, "--", c="gray", alpha=0.5, zorder=2, label="KD-weight sweep")

    ax.set_xlabel("Fit3D MPJPE (mm) — out-of-domain target  ←  better")
    ax.set_ylabel("H36M P-MPJPE (mm) — in-domain  ←  better")
    ax.set_title("Cross-domain Pareto frontier: KD weight controls the trade-off")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = FIGS / "fig_pareto.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---- Figure 2: Ablation bar chart (Fit3D MPJPE + P-MPJPE) ----
def fig_ablation():
    methods = [
        ("Zero-shot",                "v4_remap_zeroshot"),
        ("low LR",                   "v3_low_lr_15ep"),
        ("biomech × 2",              "v3_biomech2_15ep"),
        ("rank-2 (with H36M)",       "v3_lora_rank2_15ep"),
        ("no H36M (rank 8)",         "v3_no_h36m_15ep"),
        ("no H36M + rank-2",         "v4_remap_v3_no_h36m_rank2"),
        ("KD λ=1 (best)",            "v4_kd1"),
    ]
    labels = [m[0] for m in methods]
    mpjpe   = [load_eval(m[1])["datasets"]["Fit3D"]["overall"]["mpjpe"]   for m in methods]
    p_mpjpe = [load_eval(m[1])["datasets"]["Fit3D"]["overall"]["p_mpjpe"] for m in methods]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(labels))
    w = 0.38
    bars1 = ax.bar(x - w/2, mpjpe,   w, label="MPJPE",   color="#1f77b4", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, p_mpjpe, w, label="P-MPJPE", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    # highlight best
    bars1[-1].set_color("#2ca02c"); bars1[-1].set_edgecolor("black")
    bars2[-1].set_color("#5fb35f"); bars2[-1].set_edgecolor("black")

    # Zero-shot reference lines
    ax.axhline(mpjpe[0],   color="#1f77b4", ls=":", alpha=0.6)
    ax.axhline(p_mpjpe[0], color="#ff7f0e", ls=":", alpha=0.6)

    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                f"{b.get_height():.0f}", ha="center", fontsize=8)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                f"{b.get_height():.0f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Fit3D s11 error (mm) ← lower is better")
    ax.set_title("Ablation results on Fit3D s11 (out-of-domain). Dotted lines = zero-shot.")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(mpjpe) * 1.15)
    fig.tight_layout()
    out = FIGS / "fig_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---- Figure 3: Per-action P-MPJPE breakdown for headline run ----
def fig_per_action():
    d = load_eval("v4_kd1")["datasets"]["Fit3D"]["per_action"]
    actions = sorted(d.items(), key=lambda kv: -kv[1]["p_mpjpe"])
    names    = [a[0] for a in actions]
    p_mpjpe  = [a[1]["p_mpjpe"] for a in actions]
    n_wins   = [a[1]["n"]       for a in actions]

    # Manual category coloring
    def category(name):
        floor = ("pushup", "burpees", "diamond_pushup", "mule_kick", "man_maker")
        warmup = name.startswith("warmup_")
        barbell = "barbell" in name or "deadlift" in name or "clean" in name
        dumbbell = "dumbbell" in name or "shoulder_press" in name
        if any(f in name for f in floor):     return "Floor / inverted",   "#d62728"
        if barbell:                            return "Barbell / heavy",     "#9467bd"
        if dumbbell:                           return "Dumbbell standing",  "#2ca02c"
        if warmup:                             return "Warmup",             "#ff7f0e"
        return "Other / mixed", "#7f7f7f"
    cats = [category(n) for n in names]
    colors = [c[1] for c in cats]
    cat_names = [c[0] for c in cats]

    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    y = np.arange(len(names))
    bars = ax.barh(y, p_mpjpe, color=colors, edgecolor="black", linewidth=0.4)
    for i, (bar, n) in enumerate(zip(bars, n_wins)):
        ax.text(bar.get_width() + 3, i, f"n={n}", va="center", fontsize=7)

    ax.axvline(163.3, color="black", ls="--", alpha=0.5, label="Zero-shot Fit3D P-MPJPE (163.3 mm)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("P-MPJPE (mm) ← lower is better")
    ax.set_title("Per-action P-MPJPE on Fit3D s11 (headline: no H36M + rank-2 LoRA + KD λ=1)")
    ax.grid(True, axis="x", alpha=0.3)

    # Custom legend by category
    seen = set()
    handles = []
    for n, c in zip(cat_names, colors):
        if n in seen: continue
        seen.add(n)
        handles.append(plt.Rectangle((0,0),1,1, color=c, ec="black", lw=0.4, label=n))
    handles.append(plt.Line2D([0],[0], color="black", ls="--", label="Zero-shot baseline"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIGS / "fig_per_action.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---- Figure 4: Training trajectories (val MPJPE per epoch) ----
def parse_epochs(log_path):
    """Returns list of (epoch, train_loss, val_mpjpe, val_p_mpjpe)."""
    if not log_path.exists(): return []
    rows = []
    pat = re.compile(
        r"^EPOCH (\d+)/\d+ \(\d+s\) \| train: .*?loss=([\d.]+).*?\| val: mpjpe=([\d.]+) p_mpjpe=([\d.]+)"
    )
    with open(log_path) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                rows.append((int(m.group(1)), float(m.group(2)),
                             float(m.group(3)), float(m.group(4))))
    return rows


def fig_trajectories():
    runs = [
        ("v3 + low LR (null)",         "train_5250734.out", "#888888"),
        ("v3 + biomech×2 (null)",      "train_5250736.out", "#aaaaaa"),
        ("v3 + no H36M",               "train_5250683.out", "#9467bd"),
        ("v3 + rank-2 (with H36M)",    "train_5250735.out", "#ff7f0e"),
        ("v3 + no H36M + rank-2",      "train_5253991.out", "#2ca02c"),
        ("v4 + KD λ=1 (BEST)",         "train_5254266.out", "#d62728"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for label, log, color in runs:
        rows = parse_epochs(LOGS / log)
        if not rows: continue
        ep = [r[0] for r in rows]
        tl = [r[1] for r in rows]
        vp = [r[3] * 1000 for r in rows]  # to mm

        lw = 2.4 if "BEST" in label else 1.5
        axes[0].plot(ep, tl, marker="o", markersize=3, label=label, color=color, linewidth=lw)
        axes[1].plot(ep, vp, marker="o", markersize=3, label=label, color=color, linewidth=lw)

    axes[0].set_title("Training loss over epochs")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total training loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_yscale("log")

    axes[1].axhline(280, color="black", ls=":", alpha=0.5,
                    label="Zero-shot val P-MPJPE")
    axes[1].set_title("Validation P-MPJPE on Fit3D s11 (trainer's metric, mm)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val P-MPJPE (mm)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIGS / "fig_trajectories.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---- Figure 5: Architecture diagram (matplotlib blocks) ----
def fig_arch():
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_xlim(0, 17); ax.set_ylim(0, 8); ax.axis("off")

    def box(x, y, w, h, text, fc="#cce5ff", ec="black", fontsize=9, weight="normal"):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    # Inputs
    box(0.2, 6.0, 3.0, 1.2, "H36M\n2D + 3D",  fc="#fff4e6")
    box(0.2, 4.2, 3.0, 1.2, "Fit3D-train\n2D + cam_root\n+ intrinsics", fc="#fff4e6")

    # Mixed batch (wider, shorter text)
    box(3.6, 5.1, 3.6, 1.2,
        "Mixed batch\nhas_3d / has_reproj flags",
        fc="#e6f7ff", fontsize=8.5)
    arrow(3.2, 6.6, 3.6, 5.85)
    arrow(3.2, 4.8, 3.6, 5.55)

    # Student model
    box(7.6, 5.1, 3.6, 2.2,
        "DSTformer\n(MB_ft_h36m init,\nfrozen backbone)\n+ LoRA rank 2",
        fc="#cce5ff", weight="bold")
    arrow(7.2, 5.7, 7.6, 6.2)

    # Teacher (shortened text)
    box(7.6, 2.4, 3.6, 1.4,
        "Teacher DSTformer\n(frozen, no LoRA)",
        fc="#f0f0f0", fontsize=9)
    arrow(5.4, 5.1, 7.6, 3.4, color="gray")

    # Predictions
    box(11.6, 6.0, 3.4, 1.2, "ŷ_student", fc="#d4f4dd")
    arrow(11.2, 6.5, 11.6, 6.6)
    box(11.6, 2.4, 3.4, 1.0, "ŷ_teacher (no_grad)", fc="#dddddd")
    arrow(11.2, 3.1, 11.6, 2.9, color="gray")

    # Loss (4 lines, comfortable spacing)
    box(11.6, 4.0, 3.4, 1.5,
        "L_total = λ_3D L_3D\n+ λ_rep L_reproj\n+ λ_bm L_biomech\n+ λ_kd L_kd",
        fc="#fff5cc", fontsize=8.5)
    arrow(13.3, 5.95, 13.3, 5.55)
    arrow(13.3, 3.4, 13.3, 4.0, color="gray")

    ax.text(8.5, 7.7, "Hybrid 2.5D Training Pipeline", ha="center",
            fontsize=13, fontweight="bold")
    ax.text(0.2, 0.5,
            "Notes: L_3D and L_kd gated to has_3d=True samples.  "
            "L_reproj recovers absolute 3D via pred + cam_root, projects "
            "with per-sample (fx, fy, cx, cy).  L_biomech (symmetry + hinge) "
            "always active.",
            fontsize=8, style="italic", color="#444")

    out = FIGS / "fig_architecture.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig_pareto()
    fig_ablation()
    fig_per_action()
    fig_trajectories()
    fig_arch()
    print("\nDone. Figures in figures/ — embed in report with e.g. `\\includegraphics{figures/fig_pareto.png}`.")

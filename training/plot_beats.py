"""Plot 12 new beat recordings: each figure shows sample[i] from each of the 3 classes."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "realtime" / "data"
OUT_DIR = ROOT / "docs"
OUT_DIR.mkdir(exist_ok=True)

CLASSES = [
    ("Class A", ["beat_001", "beat_002", "beat_003", "beat_004"], "#ff5252"),
    ("Class B", ["beat_005", "beat_006", "beat_007", "beat_008"], "#44bbff"),
    ("Class C", ["beat_009", "beat_010", "beat_011", "beat_012"], "#ffaa44"),
]
DT = 0.02
BASELINE_FRAMES = 5
VEL_DECAY = 0.93
JERK_ALPHA = 0.6           # low-pass for jerk magnitude (matches dashboard)
BEAT_THRESHOLD = 12000     # matches dashboard default
BEAT_COOLDOWN_S = 0.18


def process(csv_path):
    df = pd.read_csv(csv_path)
    a = df[["ax_a", "ay_a", "az_a"]].values.astype(np.float32)
    t = np.arange(len(a)) * DT

    baseline = a[:BASELINE_FRAMES].mean(axis=0)
    dev = a - baseline  # (T, 3) signed accel deviation

    # Signed jerk vector (first difference of raw accel)
    jerk_vec = np.diff(a, axis=0)
    jerk_vec = np.concatenate([np.zeros((1, 3)), jerk_vec], axis=0)  # (T, 3)

    # Integrated velocity (same decay as dashboard)
    v = np.zeros_like(dev)
    cur = np.zeros(3, dtype=np.float32)
    for i, d in enumerate(dev):
        cur = cur * VEL_DECAY + d * DT
        v[i] = cur  # (T, 3) signed velocity

    # Signed "down" component: project onto -gravity unit vector.
    # Baseline points in the +up device direction (accelerometer reads +1g up at rest).
    g_mag = np.linalg.norm(baseline) + 1e-9
    g_hat = baseline / g_mag
    a_down = -(dev @ g_hat)
    v_down = -(v @ g_hat)
    j_down = -(jerk_vec @ g_hat)

    # Beat detection (rising-edge on smoothed jerk magnitude, same logic as dashboard)
    j_mag = np.linalg.norm(jerk_vec, axis=1)
    j_smooth = np.zeros_like(j_mag)
    cur = 0.0
    for i, jm in enumerate(j_mag):
        cur = cur * (1 - JERK_ALPHA) + jm * JERK_ALPHA
        j_smooth[i] = cur
    beats = []
    last = -1e9
    for i in range(1, len(j_smooth)):
        if j_smooth[i] >= BEAT_THRESHOLD and j_smooth[i - 1] < BEAT_THRESHOLD \
           and (t[i] - last) > BEAT_COOLDOWN_S:
            beats.append(t[i])
            last = t[i]

    return t, dev, jerk_vec, v, a_down, v_down, j_down, beats


def plot_one_figure(sample_idx):
    """One figure with 4 rows (metrics) × 3 columns (classes), showing sample_idx from each class."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        6, 3, figsize=(13, 14), sharex=False,
        gridspec_kw={"hspace": 0.42, "wspace": 0.22},
    )
    fig.suptitle(f"Beat sample #{sample_idx + 1} from each class",
                 fontsize=13, fontweight="bold", y=0.995)

    row_titles = [
        "Acceleration deviation (signed, x/y/z)",
        "Jerk vector (signed Δa, x/y/z)",
        "Integrated velocity (signed, x/y/z)",
        "Accel projected on −gravity (↑ = accelerating down)",
        "Jerk projected on −gravity (↑ = Δa points down)",
        "Velocity projected on −gravity (↑ = moving down)",
    ]

    axis_colors = ["#ff5252", "#44ff88", "#448aff"]
    axis_labels = ["x", "y", "z"]

    for col, (name, files, color) in enumerate(CLASSES):
        fname = files[sample_idx]
        path = DATA_DIR / f"{fname}.csv"
        if not path.exists():
            continue
        t, dev, jerk_vec, v, a_down, v_down, j_down, beats = process(path)

        # Row 0 — signed acceleration deviation (3 axes)
        for ax_i in range(3):
            axes[0, col].plot(t, dev[:, ax_i], color=axis_colors[ax_i],
                              linewidth=1.3, label=axis_labels[ax_i])
        axes[0, col].axhline(0, color="#444", linewidth=0.6)
        axes[0, col].set_title(f"{name}  —  {fname}",
                               fontsize=11, color=color, fontweight="bold")

        # Row 1 — signed jerk (3 axes)
        for ax_i in range(3):
            axes[1, col].plot(t, jerk_vec[:, ax_i], color=axis_colors[ax_i], linewidth=1.2)
        axes[1, col].axhline(0, color="#444", linewidth=0.6)

        # Row 2 — signed velocity (3 axes)
        for ax_i in range(3):
            axes[2, col].plot(t, v[:, ax_i], color=axis_colors[ax_i], linewidth=1.3)
        axes[2, col].axhline(0, color="#444", linewidth=0.6)

        # Row 3 — accel deviation down-projection
        axes[3, col].plot(t, a_down, color=color, linewidth=1.8)
        axes[3, col].fill_between(t, a_down, 0, where=(a_down > 0),
                                  color="#44ff88", alpha=0.12, interpolate=True)
        axes[3, col].fill_between(t, a_down, 0, where=(a_down < 0),
                                  color="#ff5252", alpha=0.12, interpolate=True)
        axes[3, col].axhline(0, color="#555", linewidth=0.8)

        # Row 4 — jerk down-projection
        axes[4, col].plot(t, j_down, color=color, linewidth=1.5)
        axes[4, col].fill_between(t, j_down, 0, where=(j_down > 0),
                                  color="#44ff88", alpha=0.12, interpolate=True)
        axes[4, col].fill_between(t, j_down, 0, where=(j_down < 0),
                                  color="#ff5252", alpha=0.12, interpolate=True)
        axes[4, col].axhline(0, color="#555", linewidth=0.8)

        # Row 5 — velocity down-projection
        axes[5, col].plot(t, v_down, color=color, linewidth=1.8)
        axes[5, col].fill_between(t, v_down, 0, where=(v_down > 0),
                                  color="#44ff88", alpha=0.12, interpolate=True)
        axes[5, col].fill_between(t, v_down, 0, where=(v_down < 0),
                                  color="#ff5252", alpha=0.12, interpolate=True)
        axes[5, col].axhline(0, color="#555", linewidth=0.8)

        # Beat markers — vertical red lines on every row
        for r in range(6):
            for bt in beats:
                axes[r, col].axvline(bt, color="#ff2040", linewidth=1.2,
                                     alpha=0.75, linestyle="--")

    for r in range(6):
        axes[r, 0].set_ylabel(row_titles[r], fontsize=9)
        for c in range(3):
            axes[r, c].grid(True, alpha=0.15, linewidth=0.5)
            axes[r, c].tick_params(labelsize=8)
        if r == 5:
            for c in range(3):
                axes[r, c].set_xlabel("time (s)", fontsize=9)

    axes[0, 2].legend(loc="upper right", fontsize=8, framealpha=0.3)

    out = OUT_DIR / f"beat_sample_{sample_idx + 1}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#08080c")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    for i in range(4):
        plot_one_figure(i)


if __name__ == "__main__":
    main()

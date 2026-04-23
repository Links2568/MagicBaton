"""Detailed explanation diagram for Bidirectional GRU."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

C_BG = "#FAFBFC"
C_FWD = "#2ECC71"    # forward GRU
C_BWD = "#E74C3C"    # backward GRU
C_INPUT = "#4488FF"
C_HIDDEN = "#F39C12"
C_CONCAT = "#9B59B6"
C_OUT = "#E74C3C"
C_CELL = "#ECF0F1"

def draw_box(ax, x, y, w, h, text, color, tc="white", fs=9, fw="bold"):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle="round,pad=0.06", facecolor=color,
                          edgecolor="#333", linewidth=0.7)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight=fw, color=tc, family="monospace")


def main():
    fig = plt.figure(figsize=(13, 14))
    fig.patch.set_facecolor(C_BG)

    # ============================================================
    # Panel 1: What is Bidirectional?
    # ============================================================
    ax1 = fig.add_axes([0.05, 0.62, 0.9, 0.35])
    ax1.set_xlim(-0.5, 12)
    ax1.set_ylim(-3.5, 3)
    ax1.axis("off")

    ax1.text(6, 2.7, "Bidirectional GRU - How It Works", fontsize=15,
             fontweight="bold", ha="center", family="sans-serif")
    ax1.text(6, 2.2, "Read the sequence both forward and backward, then combine",
             fontsize=9, ha="center", color="#666")

    # Time steps
    T = 6
    xs = [1.5 + i * 1.7 for i in range(T)]
    labels = ["$t_1$", "$t_2$", "$t_3$", "$t_4$", "...", "$t_T$"]

    # Input row
    for i, (x, lab) in enumerate(zip(xs, labels)):
        draw_box(ax1, x, -2.5, 1.0, 0.5, lab, C_INPUT, fs=10)
        ax1.text(x, -3.2, f"$x_{{{i+1}}}$" if i < 4 else ("" if i == 4 else "$x_T$"),
                 ha="center", fontsize=8, color="#666")

    ax1.text(-0.3, -2.5, "Input\nSequence", ha="center", va="center",
             fontsize=8, color="#444", family="sans-serif")

    # Forward GRU row
    for i, x in enumerate(xs):
        draw_box(ax1, x, -0.5, 1.0, 0.6, "", C_FWD, fs=8)
        ax1.text(x, -0.5, r"$\vec{h}$" + f"$_{{{i+1}}}$" if i < 4 else
                 ("..." if i == 4 else r"$\vec{h}$" + "$_T$"),
                 ha="center", fontsize=9, color="white", fontweight="bold")
        # Arrow from input
        ax1.annotate("", xy=(x, -0.82), xytext=(x, -2.23),
                     arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.6))

    # Forward arrows between cells
    for i in range(T - 1):
        ax1.annotate("", xy=(xs[i+1]-0.5, -0.5), xytext=(xs[i]+0.5, -0.5),
                     arrowprops=dict(arrowstyle="-|>", color=C_FWD, lw=1.5))

    ax1.text(-0.3, -0.5, "Forward\nGRU", ha="center", va="center",
             fontsize=8, color=C_FWD, fontweight="bold", family="sans-serif")
    ax1.annotate("", xy=(xs[0]-0.5, -0.5), xytext=(-0.3+0.5, -0.5),
                 arrowprops=dict(arrowstyle="-|>", color=C_FWD, lw=1, ls="--"))

    # Backward GRU row
    for i, x in enumerate(xs):
        draw_box(ax1, x, 1, 1.0, 0.6, "", C_BWD, fs=8)
        ax1.text(x, 1, r"$\overleftarrow{h}$" + f"$_{{{i+1}}}$" if i < 4 else
                 ("..." if i == 4 else r"$\overleftarrow{h}$" + "$_T$"),
                 ha="center", fontsize=9, color="white", fontweight="bold")
        # Arrow from input
        ax1.annotate("", xy=(x, 0.68), xytext=(x, -2.23),
                     arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.6, ls=":"))

    # Backward arrows (right to left)
    for i in range(T - 1):
        ax1.annotate("", xy=(xs[i]+0.5, 1), xytext=(xs[i+1]-0.5, 1),
                     arrowprops=dict(arrowstyle="-|>", color=C_BWD, lw=1.5))

    ax1.text(-0.3, 1, "Backward\nGRU", ha="center", va="center",
             fontsize=8, color=C_BWD, fontweight="bold", family="sans-serif")

    # Direction labels
    ax1.text(9.8, -0.15, "reads left-to-right", fontsize=7, color=C_FWD,
             style="italic", ha="center")
    ax1.text(9.8, 1.4, "reads right-to-left", fontsize=7, color=C_BWD,
             style="italic", ha="center")

    # Concatenation annotation
    ax1.annotate("concatenate", xy=(6, 1.5), xytext=(10.5, 1.8),
                 fontsize=8, color=C_CONCAT, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=C_CONCAT, lw=0.8))

    # Output explanation
    ax1.text(6, -3.4, r"Output at each step: $h_t = [\vec{h}_t \| \overleftarrow{h}_t]$"
             "    (64 + 64 = 128 dims)",
             ha="center", fontsize=9, color="#333",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4FD", edgecolor="#B3D9F2"))

    # ============================================================
    # Panel 2: Why Bidirectional?
    # ============================================================
    ax2 = fig.add_axes([0.05, 0.35, 0.9, 0.25])
    ax2.set_xlim(-0.5, 12)
    ax2.set_ylim(-2.5, 2.5)
    ax2.axis("off")

    ax2.text(6, 2.2, "Why Bidirectional?", fontsize=13,
             fontweight="bold", ha="center", family="sans-serif")

    # Example: beat gesture
    ax2.text(1, 1.3, "Example: 'beat' gesture (downward strike)",
             fontsize=9, fontweight="bold", color="#333")

    # Timeline
    ax2.plot([1, 10], [0.5, 0.5], color="#ccc", lw=1)
    phases = [(1.5, "arm\nraises"), (4, "wrist\naccelerates"), (6.5, "IMPACT"), (9, "recovery\n(decelerate)")]
    for x, label in phases:
        ax2.plot(x, 0.5, "o", color="#333", ms=5)
        ax2.text(x, 0.1, label, ha="center", fontsize=7.5, color="#555",
                 family="sans-serif", linespacing=1.3)

    # Forward sees
    ax2.annotate("", xy=(6.5, 0.5), xytext=(1.5, 0.5),
                 arrowprops=dict(arrowstyle="-|>", color=C_FWD, lw=2, ls="-"))
    ax2.text(4, 0.8, "Forward: sees buildup before impact", fontsize=7.5,
             color=C_FWD, style="italic")

    # Backward sees
    ax2.annotate("", xy=(6.5, 0.5), xytext=(9, 0.5),
                 arrowprops=dict(arrowstyle="-|>", color=C_BWD, lw=2, ls="-"))
    ax2.text(7.8, 0.8, "Backward: sees recovery after impact", fontsize=7.5,
             color=C_BWD, style="italic")

    # Comparison boxes
    y_comp = -1.3
    # Forward only
    draw_box(ax2, 2.5, y_comp, 4.5, 1.2, "", "#F5F5F5", fs=7)
    ax2.text(2.5, y_comp + 0.35, "Unidirectional (forward only)", fontsize=8,
             fontweight="bold", ha="center", color="#555")
    ax2.text(2.5, y_comp - 0.1, "At time t, only knows past: $x_1, ..., x_t$",
             fontsize=7, ha="center", color="#777")
    ax2.text(2.5, y_comp - 0.4, "Cannot use future context for classification",
             fontsize=7, ha="center", color="#999")

    # Bidirectional
    draw_box(ax2, 8.5, y_comp, 4.5, 1.2, "", "#E8F8E8", fs=7)
    ax2.text(8.5, y_comp + 0.35, "Bidirectional (our approach)", fontsize=8,
             fontweight="bold", ha="center", color="#27AE60")
    ax2.text(8.5, y_comp - 0.1, "At time t, knows both past AND future",
             fontsize=7, ha="center", color="#555")
    ax2.text(8.5, y_comp - 0.4, "Full context for each timestep's representation",
             fontsize=7, ha="center", color="#555")

    # ============================================================
    # Panel 3: GRU Cell Detail
    # ============================================================
    ax3 = fig.add_axes([0.05, 0.02, 0.9, 0.31])
    ax3.set_xlim(-0.5, 12)
    ax3.set_ylim(-4, 3.5)
    ax3.axis("off")

    ax3.text(6, 3.2, "GRU Cell Internal Structure", fontsize=13,
             fontweight="bold", ha="center", family="sans-serif")
    ax3.text(6, 2.7, "Gated Recurrent Unit — controls how much to remember vs. update",
             fontsize=8.5, ha="center", color="#666")

    # Gate boxes
    draw_box(ax3, 2.5, 1.5, 3, 0.7, "Update Gate  z_t", "#3498DB", fs=9)
    draw_box(ax3, 7.5, 1.5, 3, 0.7, "Reset Gate  r_t", "#E67E22", fs=9)
    draw_box(ax3, 5, -0.2, 3.5, 0.7, "Candidate  h_t~", C_FWD, fs=9)
    draw_box(ax3, 5, -1.8, 4, 0.7, "Output  h_t", C_CONCAT, fs=9)

    # Equations
    ax3.text(2.5, 0.8, r"$z_t = \sigma(W_z x_t + U_z h_{t-1})$",
             ha="center", fontsize=9, color="#333")
    ax3.text(7.5, 0.8, r"$r_t = \sigma(W_r x_t + U_r h_{t-1})$",
             ha="center", fontsize=9, color="#333")
    ax3.text(5, -0.9, r"$\tilde{h}_t = \tanh(W x_t + U (r_t \odot h_{t-1}))$",
             ha="center", fontsize=9, color="#333")
    ax3.text(5, -2.5, r"$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$",
             ha="center", fontsize=9, color="#333")

    # Annotations
    annots = [
        (2.5, 2.1, '"How much of the new info\n should I let in?"', "#3498DB"),
        (7.5, 2.1, '"How much of the old memory\n should I look at?"', "#E67E22"),
    ]
    for x, y, txt, c in annots:
        ax3.text(x, y, txt, ha="center", fontsize=7, color=c,
                 style="italic", linespacing=1.3)

    # Bottom summary
    summary = (
        "z_t close to 0:  keep old memory h_{t-1} (ignore new input)\n"
        "z_t close to 1:  replace with new candidate h_t~ (use new input)\n"
        "r_t close to 0:  forget previous hidden state when computing candidate\n"
        "r_t close to 1:  fully use previous hidden state"
    )
    ax3.text(6, -3.5, summary, ha="center", fontsize=7.5, color="#555",
             family="monospace", linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ddd"))

    fig.savefig(str(RESULTS_DIR / "arch_bigru_explained.png"), dpi=180, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/arch_bigru_explained.png")


if __name__ == "__main__":
    main()

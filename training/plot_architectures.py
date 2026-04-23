"""Draw LinkNet model architecture diagrams for presentation."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Colors
C_INPUT  = "#4488FF"
C_CONV   = "#FF6B35"
C_BN     = "#FFB347"
C_POOL   = "#87CEEB"
C_GAP    = "#9B59B6"
C_DROP   = "#95A5A6"
C_FC     = "#E74C3C"
C_GRU    = "#2ECC71"
C_FEAT   = "#F39C12"
C_SVM    = "#8E44AD"
C_RF     = "#27AE60"
C_SCALE  = "#3498DB"
C_BG     = "#FAFBFC"
C_DIM    = "#888888"

def draw_block(ax, x, y, w, h, text, color, text_color="white", fontsize=9):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.05", facecolor=color,
                          edgecolor="#333", linewidth=0.6)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, family="monospace")
    return y

def draw_dim(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="center", fontsize=7,
            color=C_DIM, family="monospace")

def draw_arrow(ax, x, y1, y2):
    ax.annotate("", xy=(x, y2 + 0.15), xytext=(x, y1 - 0.15),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=0.8))

def draw_harrow(ax, x1, y, x2):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=0.8))


# ============================================================
# Figure 1: CNN Architecture
# ============================================================
def plot_cnn():
    fig, ax = plt.subplots(figsize=(5, 10))
    ax.set_xlim(-3, 5)
    ax.set_ylim(-16, 1.5)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    ax.text(0, 1.2, "LinkNet-CNN", fontsize=14, fontweight="bold",
            ha="center", family="sans-serif")
    ax.text(0, 0.7, "1D Convolutional Neural Network  |  37,735 params",
            fontsize=8, ha="center", color="#666", family="sans-serif")

    W, H = 3.2, 0.5
    layers = [
        (0,    "Input",                  C_INPUT, "(B, 6, 128)"),
        (-1.2, "Conv1d  6→32, k=7",     C_CONV,  "(B, 32, 128)"),
        (-2.2, "BatchNorm + ReLU",       C_BN,    ""),
        (-3.0, "MaxPool1d(2)",           C_POOL,  "(B, 32, 64)"),
        (-4.2, "Conv1d  32→64, k=5",    C_CONV,  "(B, 64, 64)"),
        (-5.2, "BatchNorm + ReLU",       C_BN,    ""),
        (-6.0, "MaxPool1d(2)",           C_POOL,  "(B, 64, 32)"),
        (-7.2, "Conv1d  64→128, k=3",   C_CONV,  "(B, 128, 32)"),
        (-8.2, "BatchNorm + ReLU",       C_BN,    ""),
        (-9.4, "Global Avg Pool",        C_GAP,   "(B, 128)"),
        (-10.4,"Dropout(0.4)",           C_DROP,  ""),
        (-11.4,"Linear 128→7",           C_FC,    "(B, 7)"),
    ]

    for y, text, color, dim in layers:
        tc = "white" if color not in [C_BN, C_POOL] else "black"
        draw_block(ax, 0, y, W, H, text, color, tc)
        if dim:
            draw_dim(ax, 1.8, y, dim)

    # Arrows
    positions = [l[0] for l in layers]
    for i in range(len(positions)-1):
        draw_arrow(ax, 0, positions[i], positions[i+1])

    # Block labels
    for label, y1, y2 in [("Block 1", -0.9, -3.3), ("Block 2", -3.9, -6.3), ("Block 3", -6.9, -8.5)]:
        ax.annotate("", xy=(-2.3, y2), xytext=(-2.3, y1),
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
        ax.plot([-2.3, -2.1], [y1, y1], color="#999", lw=0.8)
        ax.plot([-2.3, -2.1], [y2, y2], color="#999", lw=0.8)
        ax.text(-2.5, (y1+y2)/2, label, ha="right", va="center",
                fontsize=8, color="#666", style="italic", family="sans-serif")

    # Design notes
    ax.text(0, -12.5, "Kernel sizes: 7→5→3  (coarse to fine)\n"
            "Channels: 6→32→64→128  (expand capacity)\n"
            "Time: 128→64→32→1  (progressive downsampling via pool + GAP)",
            ha="center", va="top", fontsize=7, color="#777",
            family="sans-serif", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ddd"))

    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "arch_cnn.png"), dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/arch_cnn.png")


# ============================================================
# Figure 2: RNN Architecture
# ============================================================
def plot_rnn():
    fig, ax = plt.subplots(figsize=(5, 9))
    ax.set_xlim(-3, 5.5)
    ax.set_ylim(-13, 1.5)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    ax.text(0, 1.2, "LinkNet-RNN", fontsize=14, fontweight="bold",
            ha="center", family="sans-serif")
    ax.text(0, 0.7, "Bidirectional GRU  |  103,047 params",
            fontsize=8, ha="center", color="#666", family="sans-serif")

    W, H = 3.5, 0.5

    layers = [
        (0,    "Input",                   C_INPUT, "(B, 6, 128)"),
        (-1.2, "Permute (0,2,1)",         C_DROP,  "(B, 128, 6)"),
        (-2.8, "BiGRU Layer 1",           C_GRU,   "(B, 128, 128)"),
        (-4.0, "Dropout(0.3)",            C_DROP,  ""),
        (-5.2, "BiGRU Layer 2",           C_GRU,   "(B, 128, 128)"),
        (-6.6, "Mean Pool (time axis)",   C_GAP,   "(B, 128)"),
        (-7.6, "Dropout(0.4)",            C_DROP,  ""),
        (-8.6, "Linear 128→7",            C_FC,    "(B, 7)"),
    ]

    for y, text, color, dim in layers:
        tc = "white"
        h = 0.8 if "BiGRU" in text else H
        draw_block(ax, 0, y, W, h, text, color, tc, fontsize=9)
        if dim:
            draw_dim(ax, 2.0, y, dim)

    positions = [l[0] for l in layers]
    for i in range(len(positions)-1):
        h1 = 0.4 if "BiGRU" in layers[i][1] else 0.25
        h2 = 0.4 if "BiGRU" in layers[i+1][1] else 0.25
        draw_arrow(ax, 0, positions[i] - h1, positions[i+1] + h2)

    # BiGRU detail labels
    ax.text(2.0, -2.4, "→ GRU(6→64)", fontsize=7, color=C_GRU, family="monospace")
    ax.text(2.0, -3.0, "← GRU(6→64)", fontsize=7, color=C_GRU, family="monospace")
    ax.text(2.0, -4.8, "→ GRU(128→64)", fontsize=7, color=C_GRU, family="monospace")
    ax.text(2.0, -5.4, "← GRU(128→64)", fontsize=7, color=C_GRU, family="monospace")

    # GRU equations
    ax.text(0, -10, "GRU Cell Equations:", fontsize=8, fontweight="bold",
            ha="center", color="#333", family="sans-serif")
    eqs = [
        r"$z_t = \sigma(W_z x_t + U_z h_{t-1})$        (update gate)",
        r"$r_t = \sigma(W_r x_t + U_r h_{t-1})$         (reset gate)",
        r"$\tilde{h}_t = \tanh(W x_t + U(r_t \odot h_{t-1}))$",
        r"$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$",
    ]
    for i, eq in enumerate(eqs):
        ax.text(0, -10.7 - i*0.55, eq, fontsize=7.5, ha="center",
                color="#555", family="serif")

    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "arch_rnn.png"), dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/arch_rnn.png")


# ============================================================
# Figure 3: Feature Extraction + SVM + RF
# ============================================================
def plot_features_svm_rf():
    fig, axes = plt.subplots(3, 1, figsize=(10, 14),
                              gridspec_kw={"height_ratios": [4, 3.5, 2.5], "hspace": 0.35})
    for ax in axes:
        ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    # --- Panel 1: Feature Extraction ---
    ax = axes[0]
    ax.set_xlim(-1, 11)
    ax.set_ylim(-6.5, 2)

    ax.text(5, 1.6, "Hand-Crafted Feature Extraction", fontsize=13,
            fontweight="bold", ha="center", family="sans-serif")
    ax.text(5, 1.1, "82 features per sample  (shared by SVM and RF)",
            fontsize=8, ha="center", color="#666", family="sans-serif")

    # Input
    draw_block(ax, 5, 0.3, 3.5, 0.5, "Diff Signal  (T, 6)", C_INPUT)

    # 6 channels
    ch_names = ["da_x", "da_y", "da_z", "dg_x", "dg_y", "dg_z"]
    ch_xs = [0.5, 2.3, 4.1, 5.9, 7.7, 9.5]
    for x, name in zip(ch_xs, ch_names):
        draw_block(ax, x, -1, 1.5, 0.45, name, C_FEAT, "black", fontsize=8)
        ax.annotate("", xy=(x, -0.75), xytext=(x - 0.0 + (5-x)*0.15, 0.05),
                    arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=0.6))

    # Stats box
    box = FancyBboxPatch((0.1, -4.8), 9.8, 2.8,
                          boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="#ccc", linewidth=0.8)
    ax.add_patch(box)
    ax.text(5, -2.2, "13 Statistics per Channel  ×  6 channels  =  78 features",
            fontsize=9, fontweight="bold", ha="center", color="#333", family="sans-serif")

    stats_text = (
        "Central:     mean (μ),  median,  Q₂₅,  Q₇₅\n"
        "Dispersion: std (σ),  min,  max,  range,  IQR\n"
        "Shape:       skewness,  kurtosis,  energy (Σx²/T)\n"
        "Frequency: zero-crossing rate"
    )
    ax.text(1, -2.7, stats_text, fontsize=7.5, va="top", color="#444",
            family="monospace", linespacing=1.7)

    # Formulas
    formulas = (
        "skew = E[(x−μ)³/σ³]     kurt = E[(x−μ)⁴/σ⁴] − 3\n"
        "ZCR = Σ|sign(xᵢ) ≠ sign(xᵢ₋₁)| / (T−1)"
    )
    ax.text(6, -2.7, formulas, fontsize=7, va="top", color="#666",
            family="monospace", linespacing=1.7)

    for x in ch_xs:
        ax.annotate("", xy=(x, -2.0), xytext=(x, -1.25),
                    arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=0.5))

    # Global features
    ax.text(5, -5.2, "+  4 Global Features:   ‖accel‖ mean/std,  ‖gyro‖ mean/std",
            fontsize=8, ha="center", color="#333", family="sans-serif",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4FD", edgecolor="#B3D9F2"))

    # Output
    draw_block(ax, 5, -6, 3.5, 0.5, "82-dim Feature Vector", C_SCALE)
    draw_arrow(ax, 5, -5.5, -5.7)

    # --- Panel 2: SVM ---
    ax = axes[1]
    ax.set_xlim(-1, 11)
    ax.set_ylim(-6, 2)

    ax.text(5, 1.5, "LinkNet-SVM  (RBF Kernel, C=10)", fontsize=13,
            fontweight="bold", ha="center", family="sans-serif")

    # Pipeline
    draw_block(ax, 1.5, 0.3, 2.5, 0.5, "82-dim features", C_INPUT)
    draw_block(ax, 4.5, 0.3, 2.2, 0.5, "StandardScaler", C_SCALE)
    draw_block(ax, 7.5, 0.3, 2.5, 0.5, "SVM (RBF)", C_SVM)
    draw_block(ax, 10, 0.3, 1.5, 0.5, "7 classes", C_FC)

    draw_harrow(ax, 2.8, 0.3, 3.4)
    draw_harrow(ax, 5.6, 0.3, 6.2)
    draw_harrow(ax, 8.8, 0.3, 9.2)

    # RBF box
    box = FancyBboxPatch((0.3, -4.7), 9.4, 4.2,
                          boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="#ccc", linewidth=0.8)
    ax.add_patch(box)

    ax.text(5, -0.8, "RBF (Gaussian) Kernel", fontsize=10,
            fontweight="bold", ha="center", color="#333")

    rbf_text = (
        "Decision:   f(x) = sign( Σᵢ αᵢ yᵢ K(xᵢ, x) + b )\n\n"
        "RBF Kernel: K(xᵢ, xⱼ) = exp( −γ ‖xᵢ − xⱼ‖² )\n\n"
        "γ = scale = 1 / (n_features × Var(X)) = 1 / (82 × Var(X))\n"
        "C = 10  (higher → harder margin, less misclassification tolerance)"
    )
    ax.text(1, -1.3, rbf_text, fontsize=7.5, va="top", color="#444",
            family="monospace", linespacing=1.8)

    # Kernel comparison
    ax.text(5, -3.8, "Kernel Comparison", fontsize=9, fontweight="bold",
            ha="center", color="#333")

    headers = ["Linear", "Polynomial", "RBF (chosen)"]
    descs = ["K = xᵢᵀxⱼ\nSingle hyperplane\n✗ Too simple",
             "K = (γxᵢᵀxⱼ + r)ᵈ\nInteraction features\n~ Overfit risk",
             "K = exp(−γ‖x−x'‖²)\n∞-dim feature space\n✓ Best for gestures"]
    colors_k = ["#E74C3C", "#F39C12", "#27AE60"]

    for i, (h, d, ck) in enumerate(zip(headers, descs, colors_k)):
        x = 1.8 + i * 3.2
        ax.text(x, -4.15, h, fontsize=8, fontweight="bold", ha="center", color=ck)
        ax.text(x, -4.5, d, fontsize=6.5, ha="center", va="top", color="#555",
                family="monospace", linespacing=1.5)

    # --- Panel 3: RF ---
    ax = axes[2]
    ax.set_xlim(-1, 11)
    ax.set_ylim(-4.5, 2)

    ax.text(5, 1.5, "LinkNet-RF  (200 Decision Trees)", fontsize=13,
            fontweight="bold", ha="center", family="sans-serif")

    # Pipeline
    draw_block(ax, 1.2, 0.3, 2.2, 0.5, "82-dim features", C_INPUT)
    draw_block(ax, 3.8, 0.3, 2, 0.5, "StandardScaler", C_SCALE)
    draw_harrow(ax, 2.35, 0.3, 2.8)

    # Trees
    tree_ys = [0.9, 0.3, -0.3, -1.1]
    tree_labels = ["Tree 1", "Tree 2", "Tree 3", "Tree 200"]
    for y, label in zip(tree_ys, tree_labels):
        draw_block(ax, 6.5, y, 1.5, 0.4, label, C_RF, fontsize=7)
        draw_harrow(ax, 4.8, 0.3, 5.75)

    ax.text(6.5, -0.65, "⋮", fontsize=14, ha="center", color="#999")

    # Arrows to trees
    for y in tree_ys:
        ax.annotate("", xy=(5.75, y), xytext=(4.85, 0.3),
                    arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=0.5))

    # Vote
    draw_block(ax, 8.5, 0, 1.8, 0.5, "Majority Vote", C_FC, fontsize=8)
    for y in tree_ys:
        ax.annotate("", xy=(7.6, 0), xytext=(7.25, y),
                    arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=0.5))

    draw_block(ax, 10.2, 0, 1.2, 0.5, "7 classes", C_FC, fontsize=8)
    draw_harrow(ax, 9.4, 0, 9.6)

    # Explanation
    rf_text = (
        "Each tree:  bootstrap sample  →  at each split, randomly pick √82 ≈ 9 features\n"
        "            →  split on best Gini impurity reduction  →  grow to full depth\n"
        "Ensemble:   200 independent trees  →  majority vote  (reduces variance, prevents overfitting)"
    )
    ax.text(5, -2, rf_text, fontsize=7, ha="center", va="top", color="#444",
            family="monospace", linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ddd"))

    fig.savefig(str(RESULTS_DIR / "arch_features_svm_rf.png"), dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/arch_features_svm_rf.png")


# ============================================================
# Figure 4: Full Pipeline Overview
# ============================================================
def plot_pipeline():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-4, 2.5)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    ax.text(6, 2.2, "LinkNet Training Pipeline", fontsize=14,
            fontweight="bold", ha="center", family="sans-serif")

    # Data
    draw_block(ax, 1, 0.8, 2.2, 0.7, "Dual IMU\n12-axis @ 50Hz", C_INPUT, fontsize=8)
    draw_block(ax, 3.8, 0.8, 2, 0.7, "Diff Signal\n(A−B)/2/32768", C_SCALE, fontsize=8)
    draw_harrow(ax, 2.1, 0.8, 2.8)

    # Split
    ax.annotate("", xy=(5.5, 1.5), xytext=(4.85, 0.8),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=0.8))
    ax.annotate("", xy=(5.5, 0.1), xytext=(4.85, 0.8),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=0.8))

    # Neural branch
    draw_block(ax, 6.5, 1.5, 1.8, 0.5, "Raw Sequence", C_INPUT, fontsize=7)
    draw_block(ax, 8.5, 1.8, 1.2, 0.4, "CNN", C_CONV, fontsize=8)
    draw_block(ax, 8.5, 1.2, 1.2, 0.4, "RNN", C_GRU, fontsize=8)
    draw_harrow(ax, 7.4, 1.5, 7.9)
    ax.annotate("", xy=(7.9, 1.2), xytext=(7.4, 1.5),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=0.5))

    # Feature branch
    draw_block(ax, 6.5, 0.1, 1.8, 0.5, "82 Features", C_FEAT, "black", fontsize=7)
    draw_block(ax, 8.5, 0.4, 1.2, 0.4, "SVM", C_SVM, fontsize=8)
    draw_block(ax, 8.5, -0.2, 1.2, 0.4, "RF", C_RF, fontsize=8)
    draw_harrow(ax, 7.4, 0.1, 7.9)
    ax.annotate("", xy=(7.9, -0.2), xytext=(7.4, 0.1),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=0.5))

    # Output
    draw_block(ax, 10.5, 0.8, 1.8, 0.7, "7 Gestures", C_FC, fontsize=9)
    for y in [1.8, 1.2, 0.4, -0.2]:
        ax.annotate("", xy=(9.6, 0.8), xytext=(9.1, y),
                    arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=0.5))

    # Evaluation
    draw_block(ax, 3, -1.5, 2.5, 0.6, "Single-Person\n5-Fold CV", "#5DADE2", fontsize=7)
    draw_block(ax, 6, -1.5, 2.5, 0.6, "Multi-Person\n5-Fold CV", "#5DADE2", fontsize=7)
    draw_block(ax, 9, -1.5, 2.5, 0.6, "Multi-Person\nLOPO CV", "#E74C3C", fontsize=7)

    ax.text(3, -2.3, "98.2%", fontsize=12, fontweight="bold", ha="center", color="#27AE60")
    ax.text(3, -2.7, "CNN best", fontsize=7, ha="center", color="#666")

    ax.text(6, -2.3, "99.1%", fontsize=12, fontweight="bold", ha="center", color="#27AE60")
    ax.text(6, -2.7, "CNN best", fontsize=7, ha="center", color="#666")

    ax.text(9, -2.3, "70.6%", fontsize=12, fontweight="bold", ha="center", color="#E74C3C")
    ax.text(9, -2.7, "CNN best", fontsize=7, ha="center", color="#666")

    ax.text(6, -3.3, "5-Fold mixes people → inflated accuracy  |  LOPO = true cross-person generalization",
            fontsize=7.5, ha="center", color="#888", style="italic", family="sans-serif")

    fig.savefig(str(RESULTS_DIR / "arch_pipeline.png"), dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/arch_pipeline.png")


if __name__ == "__main__":
    plot_cnn()
    plot_rnn()
    plot_features_svm_rf()
    plot_pipeline()
    print("Done.")

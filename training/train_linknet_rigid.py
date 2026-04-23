#!/usr/bin/env python3
"""
LinkNet Rigid Body — Compare input representations:
  1. diff only (6ch)          — current baseline: (A-B)/2
  2. rigid body (12ch)        — a_trans(3) + a_rot(3) + omega(3) + alpha(3)
  3. raw 12-axis (12ch)       — all channels as-is

Rigid body decomposition:
  a_trans = (accel_A + accel_B) / 2   — translational acceleration (diff discards this!)
  a_rot   = (accel_A - accel_B) / 2   — rotational acceleration component
  omega   = (gyro_A + gyro_B) / 2     — angular velocity
  alpha   = d(omega)/dt               — angular acceleration (temporal derivative)

Evaluation: LOPO + 5-Fold on 4-person data (7 gestures).
Model: CNN only (best performer from prior experiments).
"""

import json
import logging
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GESTURES = ["beat", "stab", "spin", "slash", "shake", "flick", "wing"]
GESTURE_TO_IDX = {g: i for i, g in enumerate(GESTURES)}
NUM_CLASSES = len(GESTURES)

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "dataset"
RESULTS_DIR = ROOT / "results"

PERSONS = {
    "zuchen":  BASE / "zuchen",
    "tiffany": BASE / "tiffany",
    "xiaolan": BASE / "xiaolan",
    "yoyo":    BASE / "yoyo",
}

SEQ_LEN = 128
NORM_SCALE = 32768.0
DT = 0.02  # ~50Hz sampling

BATCH_SIZE = 32
MAX_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 30
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data loading — three representations
# ---------------------------------------------------------------------------
def load_csv_all(path):
    """Load CSV, return (accel_A, accel_B, gyro_A, gyro_B) each (T, 3)."""
    df = pd.read_csv(path)
    aa = df[["ax_a", "ay_a", "az_a"]].values.astype(np.float32)
    ga = df[["gx_a", "gy_a", "gz_a"]].values.astype(np.float32)
    ab = df[["ax_b", "ay_b", "az_b"]].values.astype(np.float32)
    gb = df[["gx_b", "gy_b", "gz_b"]].values.astype(np.float32)
    return aa, ga, ab, gb


def make_diff(aa, ga, ab, gb):
    """(A-B)/2, 6 channels. Current baseline."""
    a = np.concatenate([aa, ga], axis=1)  # (T, 6)
    b = np.concatenate([ab, gb], axis=1)
    return (a - b) / 2.0 / NORM_SCALE


def make_rigid(aa, ga, ab, gb):
    """Reconstruct virtual IMU at baton tip using rigid body kinematics.
    Returns 6 channels: [accel_tip(3), gyro_tip(3)].

    Rigid body equation:
      a_tip = a_A + alpha x r_{A->tip} + omega x (omega x r_{A->tip})

    Assumptions:
      - IMU_A is near the tip, IMU_B is near the handle
      - Baton axis is estimated from (a_A - a_B) direction at rest (gravity)
      - r_{A->tip} ≈ r_{A->B} * extension_ratio along baton axis
      - Angular velocity is same everywhere on rigid body: gyro_tip = gyro_avg
    """
    # Angular velocity: same everywhere on rigid body, average for denoising
    omega = (ga + gb) / 2.0  # (T, 3) raw units

    # Angular acceleration: finite difference of omega
    alpha = np.zeros_like(omega)
    alpha[1:] = np.diff(omega, axis=0) / DT

    # Estimate baton axis from accelerometer difference
    # At rest, a_A - a_B is dominated by the gravity projection difference,
    # which aligns with the baton axis. Use first 5 frames average.
    diff_baseline = (aa[:5] - ab[:5]).mean(axis=0)
    baton_len = np.linalg.norm(diff_baseline) + 1e-9
    baton_axis = diff_baseline / baton_len  # unit vector along baton

    # r_{A->B} is along baton axis. We extrapolate tip as 0.5x beyond A.
    # (tip extends past IMU_A by half the A-B distance)
    r_a_to_tip = baton_axis * baton_len * 0.5  # (3,)

    # Reconstruct tip acceleration using rigid body formula
    # a_tip = a_A + alpha x r + omega x (omega x r)
    accel_tip = np.zeros_like(aa)
    for t in range(len(aa)):
        w = omega[t]
        al = alpha[t]
        r = r_a_to_tip
        centripetal = np.cross(w, np.cross(w, r))
        tangential = np.cross(al, r)
        accel_tip[t] = aa[t] + tangential + centripetal

    # Gyro at tip = same as anywhere on rigid body
    gyro_tip = omega

    # Normalize
    return np.concatenate([accel_tip, gyro_tip], axis=1) / NORM_SCALE  # (T, 6)


def make_raw12(aa, ga, ab, gb):
    """Raw 12 channels: [accel_A(3), gyro_A(3), accel_B(3), gyro_B(3)]."""
    return np.concatenate([aa, ga, ab, gb], axis=1) / NORM_SCALE  # (T, 12)


def pad_or_truncate(seq, length=SEQ_LEN):
    t, c = seq.shape
    if t >= length:
        return seq[:length]
    return np.concatenate([seq, np.zeros((length - t, c), dtype=seq.dtype)], axis=0)


def load_all(transform_fn):
    """Load all data with given transform. Returns X, y, person_ids, person_names."""
    samples, labels, pids = [], [], []
    person_names = list(PERSONS.keys())
    for pid, (name, data_dir) in enumerate(PERSONS.items()):
        for csv_path in sorted(data_dir.glob("*.csv")):
            gesture = csv_path.stem.rsplit("_", 1)[0]
            if gesture not in GESTURE_TO_IDX:
                continue
            aa, ga, ab, gb = load_csv_all(csv_path)
            seq = transform_fn(aa, ga, ab, gb)
            seq = pad_or_truncate(seq, SEQ_LEN)
            samples.append(seq)
            labels.append(GESTURE_TO_IDX[gesture])
            pids.append(pid)
    X = np.stack(samples)
    y = np.array(labels, dtype=np.int64)
    p = np.array(pids, dtype=np.int64)
    return X, y, p, person_names


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GestureDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            x = self._augment(x)
        return x.permute(1, 0), self.y[idx]

    @staticmethod
    def _augment(x):
        T, C = x.shape
        speed = random.uniform(0.8, 1.2)
        new_len = int(round(T / speed))
        if new_len > 1:
            xt = x.permute(1, 0).unsqueeze(0)
            xt = F.interpolate(xt, size=new_len, mode="linear", align_corners=False)
            xt = xt.squeeze(0).permute(1, 0)
            x = xt[:T] if xt.shape[0] >= T else torch.cat([xt, torch.zeros(T - xt.shape[0], C)], dim=0)
        x = x * random.uniform(0.8, 1.2)
        x = x + torch.randn_like(x) * 0.02
        shift = random.randint(-10, 10)
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=0)
            if shift > 0: x[:shift] = 0.0
            else: x[shift:] = 0.0
        if random.random() < 0.1:
            x[:, random.randint(0, C - 1)] = 0.0
        return x


# ---------------------------------------------------------------------------
# Model — CNN with configurable input channels
# ---------------------------------------------------------------------------
class LinkNetCNN(nn.Module):
    def __init__(self, in_ch=6, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def predict_nn(model, loader):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def train_nn(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    best_val_acc, best_state, wait = 0.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch % 20 == 0:
            logging.info("    epoch %3d | train=%.3f val=%.3f (best=%.3f)",
                         epoch, train_acc, val_acc, best_val_acc)
        if wait >= EARLY_STOP_PATIENCE:
            logging.info("    early stop at epoch %d", epoch)
            break
    return best_state, best_val_acc


# ---------------------------------------------------------------------------
# LOPO + 5-Fold runners
# ---------------------------------------------------------------------------
def run_lopo(X, y, pids, pnames, in_ch, label):
    logging.info("=" * 60)
    logging.info("[%s] LOPO CV (in_ch=%d)", label, in_ch)
    logging.info("=" * 60)
    fold_accs = []
    all_preds = np.full(len(y), -1, dtype=np.int64)
    for pid, pname in enumerate(pnames):
        test_mask = pids == pid
        train_mask = ~test_mask
        if test_mask.sum() == 0:
            continue
        set_seed(SEED + pid)
        train_ds = GestureDataset(X[train_mask], y[train_mask], augment=True)
        test_ds = GestureDataset(X[test_mask], y[test_mask], augment=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        model = LinkNetCNN(in_ch=in_ch).to(DEVICE)
        best_state, _ = train_nn(model, train_loader, test_loader)
        model.load_state_dict(best_state)
        preds, labels = predict_nn(model, test_loader)
        acc = (preds == labels).mean()
        fold_accs.append(acc)
        all_preds[test_mask] = preds
        logging.info("  %s: %.4f", pname, acc)
    m, s = np.mean(fold_accs), np.std(fold_accs)
    logging.info("[%s] LOPO: %.4f +/- %.4f", label, m, s)
    return all_preds, fold_accs, m, s


def run_kfold(X, y, skf, in_ch, label):
    logging.info("=" * 60)
    logging.info("[%s] 5-Fold CV (in_ch=%d)", label, in_ch)
    logging.info("=" * 60)
    fold_accs = []
    all_preds = np.zeros(len(y), dtype=np.int64)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        set_seed(SEED + fold)
        train_ds = GestureDataset(X[train_idx], y[train_idx], augment=True)
        val_ds = GestureDataset(X[val_idx], y[val_idx], augment=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        model = LinkNetCNN(in_ch=in_ch).to(DEVICE)
        best_state, _ = train_nn(model, train_loader, val_loader)
        model.load_state_dict(best_state)
        preds, labels = predict_nn(model, val_loader)
        acc = (preds == labels).mean()
        fold_accs.append(acc)
        all_preds[val_idx] = preds
        logging.info("  Fold %d: %.4f", fold, acc)
    m, s = np.mean(fold_accs), np.std(fold_accs)
    logging.info("[%s] 5-Fold: %.4f +/- %.4f", label, m, s)
    return all_preds, fold_accs, m, s


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(results, save_path):
    """Grouped bar chart: LOPO vs 5-Fold for each representation."""
    names = [r["name"] for r in results]
    lopo_m = [r["lopo_mean"] for r in results]
    lopo_s = [r["lopo_std"] for r in results]
    kf_m = [r["kfold_mean"] for r in results]
    kf_s = [r["kfold_std"] for r in results]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, lopo_m, w, yerr=lopo_s, capsize=5,
                label="LOPO (cross-person)", color="#ff5252", alpha=0.85)
    b2 = ax.bar(x + w/2, kf_m, w, yerr=kf_s, capsize=5,
                label="5-Fold (mixed)", color="#4488ff", alpha=0.85)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Input Representation Comparison (CNN)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bars, means, stds in [(b1, lopo_m, lopo_s), (b2, kf_m, kf_s)]:
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.012,
                    f"{m:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info("Saved %s", save_path)


def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=GESTURES)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_person(results, person_names, save_path):
    """Heatmap: representation × person LOPO accuracy."""
    names = [r["name"] for r in results]
    data = np.array([r["lopo_per_person"] for r in results])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(person_names)))
    ax.set_xticklabels(person_names, fontsize=11)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Test Person", fontsize=12)
    ax.set_title("LOPO Accuracy: Input Representation x Person", fontsize=12, fontweight="bold")
    for i in range(len(names)):
        for j in range(len(person_names)):
            ax.text(j, i, f"{data[i,j]:.1%}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if data[i,j] < 0.65 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info("Saved %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTS_DIR / "train_linknet_rigid.log", mode="w"),
        ],
    )
    logging.info("Device: %s", DEVICE)

    representations = [
        ("Diff (6ch)", make_diff, 6),
        ("Tip IMU (6ch)", make_rigid, 6),
        ("Raw 12-axis (12ch)", make_raw12, 12),
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_results = []

    for rep_name, transform_fn, in_ch in representations:
        logging.info("\n" + "#" * 60)
        logging.info("# Representation: %s", rep_name)
        logging.info("#" * 60)

        X, y, pids, pnames = load_all(transform_fn)
        logging.info("Loaded: %d samples, %d channels", len(y), X.shape[2])

        # LOPO
        preds_lopo, accs_lopo, m_lopo, s_lopo = run_lopo(X, y, pids, pnames, in_ch, rep_name)
        plot_confusion(y, preds_lopo,
                       f"{rep_name} LOPO (acc={m_lopo:.3f})",
                       RESULTS_DIR / f"cm_rigid_{rep_name.split()[0].lower()}_lopo.png")
        rep_lopo = classification_report(y, preds_lopo, target_names=GESTURES, digits=4)
        logging.info("\n%s", rep_lopo)

        # 5-Fold
        preds_kf, accs_kf, m_kf, s_kf = run_kfold(X, y, skf, in_ch, rep_name)
        plot_confusion(y, preds_kf,
                       f"{rep_name} 5-Fold (acc={m_kf:.3f})",
                       RESULTS_DIR / f"cm_rigid_{rep_name.split()[0].lower()}_5fold.png")
        rep_kf = classification_report(y, preds_kf, target_names=GESTURES, digits=4)
        logging.info("\n%s", rep_kf)

        all_results.append({
            "name": rep_name,
            "lopo_mean": m_lopo, "lopo_std": s_lopo,
            "kfold_mean": m_kf, "kfold_std": s_kf,
            "lopo_per_person": accs_lopo,
        })

    # Summary
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info("%-25s | %-22s | %-22s", "Representation", "LOPO", "5-Fold")
    logging.info("-" * 73)
    for r in all_results:
        logging.info("%-25s | %.4f +/- %.4f      | %.4f +/- %.4f",
                     r["name"], r["lopo_mean"], r["lopo_std"],
                     r["kfold_mean"], r["kfold_std"])

    plot_comparison(all_results, RESULTS_DIR / "rigid_comparison.png")
    plot_per_person(all_results, list(PERSONS.keys()), RESULTS_DIR / "rigid_per_person.png")

    # Save report
    report_path = RESULTS_DIR / "rigid_report.txt"
    with open(report_path, "w") as f:
        f.write("Input Representation Comparison (CNN)\n")
        f.write("=" * 60 + "\n\n")
        for r in all_results:
            f.write(f"{r['name']}\n")
            f.write(f"  LOPO:  {r['lopo_mean']:.4f} +/- {r['lopo_std']:.4f}\n")
            f.write(f"  5-Fold: {r['kfold_mean']:.4f} +/- {r['kfold_std']:.4f}\n\n")
    logging.info("Saved %s", report_path)
    logging.info("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare LOPO with different numbers of training people:
  - Train on 2, test on 1 (all combos of 2 from remaining 3)
  - Train on 3, test on 1 (standard LOPO)
Uses CNN (best model) with Diff (6ch) input.
"""

import itertools
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
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

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
# Data
# ---------------------------------------------------------------------------
def load_csv_diff(path):
    df = pd.read_csv(path)
    a = df[["ax_a", "ay_a", "az_a", "gx_a", "gy_a", "gz_a"]].values.astype(np.float32)
    b = df[["ax_b", "ay_b", "az_b", "gx_b", "gy_b", "gz_b"]].values.astype(np.float32)
    return (a - b) / 2.0 / NORM_SCALE


def pad_or_truncate(seq, length=SEQ_LEN):
    t, c = seq.shape
    if t >= length:
        return seq[:length]
    return np.concatenate([seq, np.zeros((length - t, c), dtype=seq.dtype)], axis=0)


def load_all():
    samples, labels, pids = [], [], []
    person_names = list(PERSONS.keys())
    for pid, (name, data_dir) in enumerate(PERSONS.items()):
        for csv_path in sorted(data_dir.glob("*.csv")):
            gesture = csv_path.stem.rsplit("_", 1)[0]
            if gesture not in GESTURE_TO_IDX:
                continue
            seq = load_csv_diff(csv_path)
            seq = pad_or_truncate(seq, SEQ_LEN)
            samples.append(seq)
            labels.append(GESTURE_TO_IDX[gesture])
            pids.append(pid)
    return np.stack(samples), np.array(labels, dtype=np.int64), np.array(pids, dtype=np.int64), person_names


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
            T, C = x.shape
            x = x * random.uniform(0.8, 1.2)
            x = x + torch.randn_like(x) * 0.02
            shift = random.randint(-10, 10)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)
                if shift > 0: x[:shift] = 0.0
                else: x[shift:] = 0.0
        return x.permute(1, 0), self.y[idx]


# ---------------------------------------------------------------------------
# Model
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
        train_one_epoch(model, train_loader, criterion, optimizer)
        _, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= EARLY_STOP_PATIENCE:
            break
    return best_state, best_val_acc


def run_experiment(X, y, pids, train_pids, test_pid, person_names, seed_offset=0):
    """Train on train_pids, test on test_pid. Returns accuracy."""
    train_mask = np.isin(pids, train_pids)
    test_mask = pids == test_pid
    set_seed(SEED + seed_offset)

    train_ds = GestureDataset(X[train_mask], y[train_mask], augment=True)
    test_ds = GestureDataset(X[test_mask], y[test_mask], augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LinkNetCNN().to(DEVICE)
    best_state, _ = train_nn(model, train_loader, test_loader)
    model.load_state_dict(best_state)
    preds, labels = predict_nn(model, test_loader)
    acc = (preds == labels).mean()

    train_names = [person_names[p] for p in train_pids]
    test_name = person_names[test_pid]
    logging.info("  train=[%s] -> test=%s: %.4f (%d->%d samples)",
                 ",".join(train_names), test_name, acc,
                 train_mask.sum(), test_mask.sum())
    return acc


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
            logging.FileHandler(RESULTS_DIR / "train_lopo_compare.log", mode="w"),
        ],
    )
    logging.info("Device: %s", DEVICE)

    X, y, pids, pnames = load_all()
    n_persons = len(pnames)
    logging.info("Loaded: %d samples, %d people", len(y), n_persons)
    for pid, pname in enumerate(pnames):
        logging.info("  %s: %d samples", pname, (pids == pid).sum())

    # ============================================================
    # Train on 3, test on 1 (standard LOPO)
    # ============================================================
    logging.info("=" * 60)
    logging.info("LOPO-3: Train on 3 people, test on 1")
    logging.info("=" * 60)

    lopo3_results = {}  # test_person -> acc
    for test_pid in range(n_persons):
        train_pids = [p for p in range(n_persons) if p != test_pid]
        acc = run_experiment(X, y, pids, train_pids, test_pid, pnames, seed_offset=test_pid)
        lopo3_results[test_pid] = acc

    lopo3_accs = list(lopo3_results.values())
    logging.info("LOPO-3 mean: %.4f +/- %.4f", np.mean(lopo3_accs), np.std(lopo3_accs))

    # ============================================================
    # Train on 2, test on 1 (all combos)
    # ============================================================
    logging.info("=" * 60)
    logging.info("LOPO-2: Train on 2 people, test on 1")
    logging.info("=" * 60)

    lopo2_all = []  # list of (train_pair, test_pid, acc)
    lopo2_by_test = {pid: [] for pid in range(n_persons)}  # test_person -> [accs]
    seed_counter = 100

    for test_pid in range(n_persons):
        others = [p for p in range(n_persons) if p != test_pid]
        # All combinations of 2 from the other 3
        for train_pair in itertools.combinations(others, 2):
            acc = run_experiment(X, y, pids, list(train_pair), test_pid, pnames,
                                seed_offset=seed_counter)
            lopo2_all.append((train_pair, test_pid, acc))
            lopo2_by_test[test_pid].append(acc)
            seed_counter += 1

    lopo2_accs = [a for _, _, a in lopo2_all]
    logging.info("LOPO-2 mean: %.4f +/- %.4f", np.mean(lopo2_accs), np.std(lopo2_accs))

    # ============================================================
    # Summary
    # ============================================================
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info("")
    logging.info("Per test person:")
    logging.info("%-10s | %-20s | %-20s", "Test", "Train-on-3", "Train-on-2 (avg)")
    logging.info("-" * 55)
    for pid in range(n_persons):
        a3 = lopo3_results[pid]
        a2_list = lopo2_by_test[pid]
        a2_mean = np.mean(a2_list)
        a2_std = np.std(a2_list)
        logging.info("%-10s | %.4f              | %.4f +/- %.4f",
                     pnames[pid], a3, a2_mean, a2_std)

    logging.info("")
    logging.info("Overall:")
    logging.info("  Train-on-3 (LOPO): %.4f +/- %.4f", np.mean(lopo3_accs), np.std(lopo3_accs))
    logging.info("  Train-on-2:        %.4f +/- %.4f", np.mean(lopo2_accs), np.std(lopo2_accs))

    # ============================================================
    # Plot
    # ============================================================

    # 1. Per-person grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_persons)
    w = 0.35
    vals3 = [lopo3_results[pid] for pid in range(n_persons)]
    vals2_mean = [np.mean(lopo2_by_test[pid]) for pid in range(n_persons)]
    vals2_std = [np.std(lopo2_by_test[pid]) for pid in range(n_persons)]

    b1 = ax.bar(x - w/2, vals3, w, label="Train on 3", color="#4488ff", alpha=0.85)
    b2 = ax.bar(x + w/2, vals2_mean, w, yerr=vals2_std, capsize=5,
                label="Train on 2 (avg)", color="#ff5252", alpha=0.85)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Cross-Person Generalization: Train on 3 vs 2 People", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pnames, fontsize=11)
    ax.set_xlabel("Test Person", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(b1, vals3):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val, std in zip(b2, vals2_mean, vals2_std):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.015,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "lopo_2v3_comparison.png", dpi=150)
    plt.close(fig)
    logging.info("Saved lopo_2v3_comparison.png")

    # 2. Detailed heatmap: train-on-2 all combos
    fig, ax = plt.subplots(figsize=(10, 5))
    # Build matrix: rows = train pairs, cols = test person
    all_combos = []
    matrix = []
    for test_pid in range(n_persons):
        others = [p for p in range(n_persons) if p != test_pid]
        for train_pair in itertools.combinations(others, 2):
            combo_label = f"{pnames[train_pair[0]]}+{pnames[train_pair[1]]}"
            if combo_label not in [c[0] for c in all_combos]:
                all_combos.append((combo_label, {}))
            for cl, d in all_combos:
                if cl == combo_label:
                    d[test_pid] = [a for tp, tp_id, a in lopo2_all
                                   if tp == train_pair and tp_id == test_pid]

    # Reorganize: unique train pairs as rows
    train_pairs_unique = []
    for test_pid in range(n_persons):
        others = [p for p in range(n_persons) if p != test_pid]
        for train_pair in itertools.combinations(others, 2):
            if train_pair not in train_pairs_unique:
                train_pairs_unique.append(train_pair)

    row_labels = []
    data = []
    for train_pair in train_pairs_unique:
        row_label = f"{pnames[train_pair[0]]} + {pnames[train_pair[1]]}"
        row_labels.append(row_label)
        row = []
        for test_pid in range(n_persons):
            if test_pid in train_pair:
                row.append(np.nan)  # can't test on training person
            else:
                acc = [a for tp, tp_id, a in lopo2_all
                       if tp == train_pair and tp_id == test_pid]
                row.append(acc[0] if acc else np.nan)
        data.append(row)

    data = np.array(data)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n_persons))
    ax.set_xticklabels(pnames, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Test Person", fontsize=11)
    ax.set_ylabel("Training Pair", fontsize=11)
    ax.set_title("Train-on-2: All Combinations", fontsize=12, fontweight="bold")

    for i in range(len(row_labels)):
        for j in range(n_persons):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "-", ha="center", va="center", fontsize=10, color="#999")
            else:
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val < 0.55 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "lopo_2v3_heatmap.png", dpi=150)
    plt.close(fig)
    logging.info("Saved lopo_2v3_heatmap.png")
    logging.info("Done.")


if __name__ == "__main__":
    main()

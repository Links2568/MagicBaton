#!/usr/bin/env python3
"""
LinkNet Cross-Person — 4-model comparison with Leave-One-Person-Out CV.
Models: 1D CNN, SVM, Random Forest, RNN (GRU).
Evaluation: LOPO CV (train on 3 people, test on 1) + standard 5-fold CV.
Data: 4 people × 7 shared gestures (beat, stab, spin, slash, shake, flick, wing).
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
MODELS_DIR = ROOT / "models"

PERSONS = {
    "zuchen":  BASE / "zuchen",
    "tiffany": BASE / "tiffany",
    "xiaolan": BASE / "xiaolan",
    "yoyo":    BASE / "yoyo",
}

SEQ_LEN = 128
IN_CHANNELS = 6
NORM_SCALE = 32768.0

BATCH_SIZE = 32
MAX_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 30

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data loading
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
    """Returns X (N, 128, 6), y (N,), person_ids (N,), person_names list."""
    samples, labels, pids = [], [], []
    person_names = list(PERSONS.keys())

    for pid, (name, data_dir) in enumerate(PERSONS.items()):
        count = 0
        for csv_path in sorted(data_dir.glob("*.csv")):
            gesture = csv_path.stem.rsplit("_", 1)[0]
            if gesture not in GESTURE_TO_IDX:
                continue
            seq = load_csv_diff(csv_path)
            seq = pad_or_truncate(seq, SEQ_LEN)
            samples.append(seq)
            labels.append(GESTURE_TO_IDX[gesture])
            pids.append(pid)
            count += 1
        logging.info("  %-10s: %d samples from %s", name, count, data_dir)

    X = np.stack(samples)
    y = np.array(labels, dtype=np.int64)
    p = np.array(pids, dtype=np.int64)
    return X, y, p, person_names


# ---------------------------------------------------------------------------
# Feature extraction (SVM / RF)
# ---------------------------------------------------------------------------
def extract_features(seq):
    T, C = seq.shape
    feats = []
    for c in range(C):
        ch = seq[:, c]
        m, s = np.mean(ch), np.std(ch)
        feats.extend([
            m, s, np.min(ch), np.max(ch), np.max(ch) - np.min(ch),
            np.median(ch),
            np.mean(((ch - m) / (s + 1e-10)) ** 3),
            np.mean(((ch - m) / (s + 1e-10)) ** 4) - 3,
            np.sum(ch ** 2) / T,
            np.sum(np.diff(np.sign(ch)) != 0) / max(T - 1, 1),
            np.percentile(ch, 25), np.percentile(ch, 75),
            np.percentile(ch, 75) - np.percentile(ch, 25),
        ])
    accel_mag = np.sqrt(np.sum(seq[:, :3] ** 2, axis=1))
    gyro_mag = np.sqrt(np.sum(seq[:, 3:] ** 2, axis=1))
    feats.extend([np.mean(accel_mag), np.std(accel_mag),
                  np.mean(gyro_mag), np.std(gyro_mag)])
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# PyTorch dataset
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
            if xt.shape[0] >= T:
                x = xt[:T]
            else:
                x = torch.cat([xt, torch.zeros(T - xt.shape[0], C)], dim=0)
        x = x * random.uniform(0.8, 1.2)
        x = x + torch.randn_like(x) * 0.02
        shift = random.randint(-10, 10)
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=0)
            if shift > 0:
                x[:shift] = 0.0
            else:
                x[shift:] = 0.0
        if random.random() < 0.1:
            x[:, random.randint(0, C - 1)] = 0.0
        return x


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LinkNetCNN(nn.Module):
    def __init__(self, in_ch=IN_CHANNELS, num_classes=NUM_CLASSES):
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


class LinkNetRNN(nn.Module):
    def __init__(self, in_ch=IN_CHANNELS, hidden=64, num_layers=2,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, num_layers=num_layers,
                          batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.fc(out)


# ---------------------------------------------------------------------------
# NN training helpers
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


def train_nn(model, train_loader, val_loader, max_epochs=MAX_EPOCHS,
             patience=EARLY_STOP_PATIENCE):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_state = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
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
        if wait >= patience:
            logging.info("    early stop at epoch %d", epoch)
            break

    return best_state, best_val_acc


# ---------------------------------------------------------------------------
# LOPO CV for NN
# ---------------------------------------------------------------------------
def lopo_nn(model_cls, X, y, person_ids, person_names, model_name):
    logging.info("=" * 60)
    logging.info("[%s] Leave-One-Person-Out CV", model_name)
    logging.info("=" * 60)

    fold_accs = []
    all_preds = np.full(len(y), -1, dtype=np.int64)

    for pid, pname in enumerate(person_names):
        test_mask = person_ids == pid
        train_mask = ~test_mask
        n_test = test_mask.sum()
        if n_test == 0:
            continue
        logging.info("--- Leave out: %s (%d samples) ---", pname, n_test)
        set_seed(SEED + pid)

        train_ds = GestureDataset(X[train_mask], y[train_mask], augment=True)
        test_ds = GestureDataset(X[test_mask], y[test_mask], augment=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = model_cls().to(DEVICE)
        best_state, _ = train_nn(model, train_loader, test_loader)
        model.load_state_dict(best_state)

        preds, labels = predict_nn(model, test_loader)
        acc = (preds == labels).mean()
        fold_accs.append(acc)
        all_preds[test_mask] = preds
        logging.info("  %s acc: %.4f", pname, acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    logging.info("[%s] LOPO Accuracy: %.4f +/- %.4f", model_name, mean_acc, std_acc)
    return all_preds, fold_accs, mean_acc, std_acc


# ---------------------------------------------------------------------------
# LOPO CV for sklearn
# ---------------------------------------------------------------------------
def lopo_sklearn(clf_factory, X_feat, y, person_ids, person_names, model_name):
    logging.info("=" * 60)
    logging.info("[%s] Leave-One-Person-Out CV", model_name)
    logging.info("=" * 60)

    fold_accs = []
    all_preds = np.full(len(y), -1, dtype=np.int64)

    for pid, pname in enumerate(person_names):
        test_mask = person_ids == pid
        train_mask = ~test_mask
        n_test = test_mask.sum()
        if n_test == 0:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_feat[train_mask])
        X_test = scaler.transform(X_feat[test_mask])

        clf = clf_factory()
        clf.fit(X_train, y[train_mask])
        preds = clf.predict(X_test)
        acc = (preds == y[test_mask]).mean()
        fold_accs.append(acc)
        all_preds[test_mask] = preds
        logging.info("  %s acc: %.4f (%d samples)", pname, acc, n_test)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    logging.info("[%s] LOPO Accuracy: %.4f +/- %.4f", model_name, mean_acc, std_acc)
    return all_preds, fold_accs, mean_acc, std_acc


# ---------------------------------------------------------------------------
# Standard 5-fold CV (mixed people) for comparison
# ---------------------------------------------------------------------------
def kfold_nn(model_cls, X, y, skf, model_name):
    logging.info("=" * 60)
    logging.info("[%s] 5-Fold CV (mixed)", model_name)
    logging.info("=" * 60)

    fold_accs = []
    all_preds = np.zeros(len(y), dtype=np.int64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info("--- Fold %d/5 ---", fold)
        set_seed(SEED + fold)
        train_ds = GestureDataset(X[train_idx], y[train_idx], augment=True)
        val_ds = GestureDataset(X[val_idx], y[val_idx], augment=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = model_cls().to(DEVICE)
        best_state, _ = train_nn(model, train_loader, val_loader)
        model.load_state_dict(best_state)
        preds, labels = predict_nn(model, val_loader)
        acc = (preds == labels).mean()
        fold_accs.append(acc)
        all_preds[val_idx] = preds
        logging.info("  Fold %d acc: %.4f", fold, acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    logging.info("[%s] 5-Fold Accuracy: %.4f +/- %.4f", model_name, mean_acc, std_acc)
    return all_preds, fold_accs, mean_acc, std_acc


def kfold_sklearn(clf_factory, X_feat, y, skf, model_name):
    logging.info("=" * 60)
    logging.info("[%s] 5-Fold CV (mixed)", model_name)
    logging.info("=" * 60)

    fold_accs = []
    all_preds = np.zeros(len(y), dtype=np.int64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, y), 1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_feat[train_idx])
        X_val = scaler.transform(X_feat[val_idx])

        clf = clf_factory()
        clf.fit(X_train, y[train_idx])
        preds = clf.predict(X_val)
        acc = (preds == y[val_idx]).mean()
        fold_accs.append(acc)
        all_preds[val_idx] = preds
        logging.info("  Fold %d acc: %.4f", fold, acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    logging.info("[%s] 5-Fold Accuracy: %.4f +/- %.4f", model_name, mean_acc, std_acc)
    return all_preds, fold_accs, mean_acc, std_acc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=GESTURES)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info("Saved %s", save_path)


def plot_comparison(lopo_results, kfold_results, save_path):
    """Grouped bar chart: LOPO vs 5-Fold for each model."""
    names = [r[0] for r in lopo_results]
    lopo_means = [r[1] for r in lopo_results]
    lopo_stds = [r[2] for r in lopo_results]
    kfold_means = [r[1] for r in kfold_results]
    kfold_stds = [r[2] for r in kfold_results]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - w / 2, lopo_means, w, yerr=lopo_stds, capsize=5,
                label="LOPO (cross-person)", color="#ff5252", alpha=0.85)
    b2 = ax.bar(x + w / 2, kfold_means, w, yerr=kfold_stds, capsize=5,
                label="5-Fold (mixed)", color="#4488ff", alpha=0.85)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("LinkNet Cross-Person — LOPO vs 5-Fold CV", fontsize=13,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bars, means, stds in [(b1, lopo_means, lopo_stds),
                               (b2, kfold_means, kfold_stds)]:
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.012,
                    f"{m:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info("Saved %s", save_path)


def plot_per_person(fold_accs_dict, person_names, save_path):
    """Heatmap: model × person accuracy."""
    model_names = list(fold_accs_dict.keys())
    data = np.array([fold_accs_dict[m] for m in model_names])  # (models, persons)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(person_names)))
    ax.set_xticklabels(person_names, fontsize=11)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=11)
    ax.set_xlabel("Test Person", fontsize=12)
    ax.set_title("LOPO Accuracy per Person", fontsize=13, fontweight="bold")

    for i in range(len(model_names)):
        for j in range(len(person_names)):
            ax.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if data[i, j] < 0.65 else "black")

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
            logging.FileHandler(RESULTS_DIR / "train_linknet_cross.log", mode="w"),
        ],
    )
    logging.info("Device: %s", DEVICE)

    # ---- Load ----
    logging.info("Loading data from 4 people ...")
    X_seq, y, person_ids, person_names = load_all()
    logging.info("Total: %d samples, %d classes, %d people",
                 len(y), NUM_CLASSES, len(person_names))
    for i, g in enumerate(GESTURES):
        logging.info("  %-8s: %d", g, (y == i).sum())
    for pid, pname in enumerate(person_names):
        logging.info("  person %-10s: %d samples", pname, (person_ids == pid).sum())

    # Features for SVM/RF
    logging.info("Extracting features ...")
    X_feat = np.stack([extract_features(X_seq[i]) for i in range(len(X_seq))])
    logging.info("Feature matrix: %s", X_feat.shape)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    model_configs = [
        ("CNN", LinkNetCNN, None),
        ("SVM", None, lambda: SVC(kernel="rbf", C=10, gamma="scale")),
        ("RF",  None, lambda: RandomForestClassifier(n_estimators=200, random_state=SEED)),
        ("RNN", LinkNetRNN, None),
    ]

    lopo_summary = []       # (name, mean, std)
    kfold_summary = []      # (name, mean, std)
    lopo_per_person = {}    # name -> [acc_per_person]
    report_lines = []

    for name, nn_cls, sk_factory in model_configs:
        # --- LOPO ---
        if nn_cls is not None:
            preds_lopo, accs_lopo, m_lopo, s_lopo = lopo_nn(
                nn_cls, X_seq, y, person_ids, person_names, f"{name}")
        else:
            preds_lopo, accs_lopo, m_lopo, s_lopo = lopo_sklearn(
                sk_factory, X_feat, y, person_ids, person_names, f"{name}")

        lopo_summary.append((name, m_lopo, s_lopo))
        lopo_per_person[name] = accs_lopo

        plot_confusion(y, preds_lopo,
                       f"{name} LOPO (acc={m_lopo:.3f}±{s_lopo:.3f})",
                       RESULTS_DIR / f"cm_cross_{name.lower()}_lopo.png")
        rep = classification_report(y, preds_lopo, target_names=GESTURES, digits=4)
        logging.info("\n%s", rep)
        report_lines.append(f"{'='*60}\n{name} — LOPO\n{'='*60}\n{rep}\n")

        # --- 5-Fold mixed ---
        if nn_cls is not None:
            preds_kf, accs_kf, m_kf, s_kf = kfold_nn(
                nn_cls, X_seq, y, skf, f"{name}")
        else:
            preds_kf, accs_kf, m_kf, s_kf = kfold_sklearn(
                sk_factory, X_feat, y, skf, f"{name}")

        kfold_summary.append((name, m_kf, s_kf))

        plot_confusion(y, preds_kf,
                       f"{name} 5-Fold (acc={m_kf:.3f}±{s_kf:.3f})",
                       RESULTS_DIR / f"cm_cross_{name.lower()}_5fold.png")
        rep = classification_report(y, preds_kf, target_names=GESTURES, digits=4)
        logging.info("\n%s", rep)
        report_lines.append(f"{'='*60}\n{name} — 5-Fold\n{'='*60}\n{rep}\n")

    # ---- Summary ----
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info("%-6s | %-22s | %-22s", "Model", "LOPO (cross-person)", "5-Fold (mixed)")
    logging.info("-" * 56)
    for (n1, m1, s1), (n2, m2, s2) in zip(lopo_summary, kfold_summary):
        logging.info("%-6s | %.4f +/- %.4f      | %.4f +/- %.4f", n1, m1, s1, m2, s2)

    plot_comparison(lopo_summary, kfold_summary,
                    RESULTS_DIR / "linknet_cross_comparison.png")
    plot_per_person(lopo_per_person, person_names,
                    RESULTS_DIR / "linknet_cross_per_person.png")

    # Save reports
    report_path = RESULTS_DIR / "linknet_cross_reports.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        f.write(f"\n{'='*60}\nSUMMARY\n{'='*60}\n")
        f.write(f"{'Model':<6s} | {'LOPO (cross-person)':<22s} | {'5-Fold (mixed)':<22s}\n")
        f.write("-" * 56 + "\n")
        for (n1, m1, s1), (n2, m2, s2) in zip(lopo_summary, kfold_summary):
            f.write(f"{n1:<6s} | {m1:.4f} +/- {s1:.4f}      | {m2:.4f} +/- {s2:.4f}\n")
    logging.info("Saved %s", report_path)

    # Export best CNN on all data
    logging.info("Training final CNN on all data for export ...")
    set_seed(SEED)
    full_ds = GestureDataset(X_seq, y, augment=True)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = GestureDataset(X_seq, y, augment=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    final_model = LinkNetCNN().to(DEVICE)
    best_state, _ = train_nn(final_model, full_loader, val_loader)
    final_model.load_state_dict(best_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = MODELS_DIR / "linknet_cross.pt"
    torch.save(final_model.state_dict(), pt_path)
    logging.info("Saved %s (%.1f KB)", pt_path, pt_path.stat().st_size / 1024)

    params = {
        "norm_scale": NORM_SCALE, "seq_len": SEQ_LEN, "in_channels": IN_CHANNELS,
        "class_names": GESTURES,
        "transform": "diff = (IMU_A - IMU_B) / 2 / 32768.0",
        "persons": person_names,
    }
    (RESULTS_DIR / "linknet_cross_params.json").write_text(json.dumps(params, indent=2))
    logging.info("Done.")


if __name__ == "__main__":
    main()

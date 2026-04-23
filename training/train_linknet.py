#!/usr/bin/env python3
"""
LinkNet — 4-model comparison for MagicBaton gesture classification.
Models: 1D CNN, SVM, Random Forest, RNN (GRU).
Evaluation: 5-fold stratified cross-validation, no held-out test set.
Data: zuchen (443 samples, 8 classes).
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
GESTURES = ["idle", "beat", "stab", "spin", "slash", "shake", "flick", "wing"]
GESTURE_TO_IDX = {g: i for i, g in enumerate(GESTURES)}
NUM_CLASSES = len(GESTURES)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "zuchen"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

SEQ_LEN = 128
IN_CHANNELS = 6          # diff signal: (A-B)/2
NORM_SCALE = 32768.0

# CNN / RNN training
BATCH_SIZE = 32
MAX_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 30

NUM_FOLDS = 5
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
    pad = np.zeros((length - t, c), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)


def load_dataset():
    """Returns (N, SEQ_LEN, 6) array and (N,) labels."""
    samples, labels = [], []
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        gesture = csv_path.stem.rsplit("_", 1)[0]
        if gesture == "metadata" or gesture not in GESTURE_TO_IDX:
            continue
        seq = load_csv_diff(csv_path)
        seq = pad_or_truncate(seq, SEQ_LEN)
        samples.append(seq)
        labels.append(GESTURE_TO_IDX[gesture])
    X = np.stack(samples)       # (N, 128, 6)
    y = np.array(labels, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Feature extraction (for SVM / RF)
# ---------------------------------------------------------------------------
def extract_features(seq):
    """(T, C) -> flat feature vector. 13 stats * 6 ch + 4 global = 82."""
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
# PyTorch dataset with augmentation
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
        return x.permute(1, 0), self.y[idx]   # (C, T)

    @staticmethod
    def _augment(x):
        T, C = x.shape
        # Time stretch [0.8, 1.2]
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
        # Amplitude [0.8, 1.2]
        x = x * random.uniform(0.8, 1.2)
        # Noise
        x = x + torch.randn_like(x) * 0.02
        # Time shift ±10
        shift = random.randint(-10, 10)
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=0)
            if shift > 0:
                x[:shift] = 0.0
            else:
                x[shift:] = 0.0
        # Channel dropout p=0.1
        if random.random() < 0.1:
            x[:, random.randint(0, C - 1)] = 0.0
        return x


# ---------------------------------------------------------------------------
# Model 1: LinkNet-CNN (1D Conv)
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
        x = x.mean(dim=2)   # GAP
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Model 4: LinkNet-RNN (GRU)
# ---------------------------------------------------------------------------
class LinkNetRNN(nn.Module):
    def __init__(self, in_ch=IN_CHANNELS, hidden=64, num_layers=2,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, num_layers=num_layers,
                          batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden * 2, num_classes)   # *2 for bidirectional

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)          # (B, T, hidden*2)
        out = out.mean(dim=1)         # average pooling over time
        out = self.dropout(out)
        return self.fc(out)


# ---------------------------------------------------------------------------
# PyTorch training helpers
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
        xb = xb.to(DEVICE)
        preds.append(model(xb).argmax(1).cpu().numpy())
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
            logging.info("  epoch %3d | train_acc=%.3f val_acc=%.3f (best=%.3f)",
                         epoch, train_acc, val_acc, best_val_acc)

        if wait >= patience:
            logging.info("  early stop at epoch %d", epoch)
            break

    return best_state, best_val_acc


# ---------------------------------------------------------------------------
# Cross-validation runner for NN models
# ---------------------------------------------------------------------------
def cv_nn(model_cls, X, y, skf, model_name):
    logging.info("=" * 60)
    logging.info("[%s] 5-Fold Cross-Validation", model_name)
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
        best_state, best_acc = train_nn(model, train_loader, val_loader)
        model.load_state_dict(best_state)

        preds, labels = predict_nn(model, val_loader)
        fold_acc = (preds == labels).mean()
        fold_accs.append(fold_acc)
        all_preds[val_idx] = preds
        logging.info("  Fold %d acc: %.4f", fold, fold_acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    logging.info("[%s] CV Accuracy: %.4f +/- %.4f", model_name, mean_acc, std_acc)
    return all_preds, fold_accs, mean_acc, std_acc


# ---------------------------------------------------------------------------
# Cross-validation runner for sklearn models
# ---------------------------------------------------------------------------
def cv_sklearn(clf_factory, X_feat, y, skf, model_name):
    logging.info("=" * 60)
    logging.info("[%s] 5-Fold Cross-Validation", model_name)
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
        fold_acc = (preds == y[val_idx]).mean()
        fold_accs.append(fold_acc)
        all_preds[val_idx] = preds
        logging.info("  Fold %d acc: %.4f", fold, fold_acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    logging.info("[%s] CV Accuracy: %.4f +/- %.4f", model_name, mean_acc, std_acc)
    return all_preds, fold_accs, mean_acc, std_acc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=GESTURES)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info("Saved %s", save_path)


def plot_comparison(results, save_path):
    """Bar chart comparing all models."""
    names = [r[0] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4488ff", "#ff5252", "#44ff88", "#ffaa44"]
    bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors[:len(names)],
                  edgecolor="#222", linewidth=0.8, alpha=0.9)
    ax.set_ylabel("5-Fold CV Accuracy", fontsize=12)
    ax.set_title("LinkNet — Model Comparison (zuchen)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.015,
                f"{m:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

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
            logging.FileHandler(RESULTS_DIR / "train_linknet.log", mode="w"),
        ],
    )
    logging.info("Device: %s", DEVICE)

    # ---- Load data ----
    logging.info("Loading data from %s", DATA_DIR)
    X_seq, y = load_dataset()
    logging.info("Dataset: %d samples, seq_len=%d, channels=%d",
                 X_seq.shape[0], X_seq.shape[1], X_seq.shape[2])
    for i, g in enumerate(GESTURES):
        logging.info("  class %d %-8s: %d samples", i, g, (y == i).sum())

    # Feature matrix for SVM / RF
    logging.info("Extracting features for SVM/RF ...")
    X_feat = np.stack([extract_features(X_seq[i]) for i in range(len(X_seq))])
    logging.info("Feature matrix: %s", X_feat.shape)

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    results = []  # (name, mean_acc, std_acc)

    # ---- 1. CNN ----
    preds_cnn, accs_cnn, mean_cnn, std_cnn = cv_nn(
        LinkNetCNN, X_seq, y, skf, "LinkNet-CNN")
    results.append(("CNN", mean_cnn, std_cnn))
    plot_confusion(y, preds_cnn,
                   f"LinkNet-CNN  5-Fold CV  (acc={mean_cnn:.3f}±{std_cnn:.3f})",
                   RESULTS_DIR / "cm_linknet_cnn.png")
    report_cnn = classification_report(y, preds_cnn, target_names=GESTURES, digits=4)
    logging.info("\n%s", report_cnn)

    # ---- 2. SVM ----
    preds_svm, accs_svm, mean_svm, std_svm = cv_sklearn(
        lambda: SVC(kernel="rbf", C=10, gamma="scale"),
        X_feat, y, skf, "LinkNet-SVM")
    results.append(("SVM", mean_svm, std_svm))
    plot_confusion(y, preds_svm,
                   f"LinkNet-SVM  5-Fold CV  (acc={mean_svm:.3f}±{std_svm:.3f})",
                   RESULTS_DIR / "cm_linknet_svm.png")
    report_svm = classification_report(y, preds_svm, target_names=GESTURES, digits=4)
    logging.info("\n%s", report_svm)

    # ---- 3. Random Forest ----
    preds_rf, accs_rf, mean_rf, std_rf = cv_sklearn(
        lambda: RandomForestClassifier(n_estimators=200, max_depth=None,
                                       random_state=SEED),
        X_feat, y, skf, "LinkNet-RF")
    results.append(("RF", mean_rf, std_rf))
    plot_confusion(y, preds_rf,
                   f"LinkNet-RF  5-Fold CV  (acc={mean_rf:.3f}±{std_rf:.3f})",
                   RESULTS_DIR / "cm_linknet_rf.png")
    report_rf = classification_report(y, preds_rf, target_names=GESTURES, digits=4)
    logging.info("\n%s", report_rf)

    # ---- 4. RNN (GRU) ----
    preds_rnn, accs_rnn, mean_rnn, std_rnn = cv_nn(
        LinkNetRNN, X_seq, y, skf, "LinkNet-RNN")
    results.append(("RNN", mean_rnn, std_rnn))
    plot_confusion(y, preds_rnn,
                   f"LinkNet-RNN  5-Fold CV  (acc={mean_rnn:.3f}±{std_rnn:.3f})",
                   RESULTS_DIR / "cm_linknet_rnn.png")
    report_rnn = classification_report(y, preds_rnn, target_names=GESTURES, digits=4)
    logging.info("\n%s", report_rnn)

    # ---- Comparison ----
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    for name, mean, std in results:
        logging.info("  %-10s  %.4f +/- %.4f", name, mean, std)

    plot_comparison(results, RESULTS_DIR / "linknet_comparison.png")

    # Save all reports
    report_path = RESULTS_DIR / "linknet_reports.txt"
    with open(report_path, "w") as f:
        for name, report in [("CNN", report_cnn), ("SVM", report_svm),
                              ("RF", report_rf), ("RNN", report_rnn)]:
            f.write(f"{'='*60}\n{name}\n{'='*60}\n{report}\n\n")
        f.write(f"{'='*60}\nSUMMARY\n{'='*60}\n")
        for name, mean, std in results:
            f.write(f"  {name:<10s}  {mean:.4f} +/- {std:.4f}\n")
    logging.info("Saved %s", report_path)

    # Export best CNN model for detect.py
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
    pt_path = MODELS_DIR / "linknet.pt"
    torch.save(final_model.state_dict(), pt_path)
    logging.info("Saved %s (%.1f KB)", pt_path, pt_path.stat().st_size / 1024)

    params = {
        "norm_scale": NORM_SCALE,
        "seq_len": SEQ_LEN,
        "in_channels": IN_CHANNELS,
        "class_names": GESTURES,
        "transform": "diff = (IMU_A - IMU_B) / 2 / 32768.0",
    }
    params_path = RESULTS_DIR / "linknet_params.json"
    params_path.write_text(json.dumps(params, indent=2))
    logging.info("Saved %s", params_path)
    logging.info("Done.")


if __name__ == "__main__":
    main()

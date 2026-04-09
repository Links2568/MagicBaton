#!/usr/bin/env python3
"""MagicBaton ML Training Pipeline — 10-class gesture classification for ESP32."""

import json
import logging
import os
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GESTURES = [
    "idle", "beat", "stab", "spin", "infinity",
    "slash", "shake", "flick", "wing", "slope",
]
GESTURE_TO_IDX = {g: i for i, g in enumerate(GESTURES)}
NUM_CLASSES = len(GESTURES)

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SEQ_LEN = 128
IN_CHANNELS = 6
NORM_SCALE = 32768.0

BATCH_SIZE = 32
MAX_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 30
NUM_FOLDS = 10
SEED = 42

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> np.ndarray:
    """Load a single CSV and return diff-transformed, normalized array (T, 6)."""
    df = pd.read_csv(path)
    a_cols = ["ax_a", "ay_a", "az_a", "gx_a", "gy_a", "gz_a"]
    b_cols = ["ax_b", "ay_b", "az_b", "gx_b", "gy_b", "gz_b"]
    imu_a = df[a_cols].values.astype(np.float32)
    imu_b = df[b_cols].values.astype(np.float32)
    diff = (imu_a - imu_b) / 2.0
    diff /= NORM_SCALE  # map to approx [-1, 1]
    return diff


def pad_or_truncate(seq: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Pad with zeros or truncate at end to fixed length. Input shape (T, C)."""
    t, c = seq.shape
    if t >= length:
        return seq[:length]
    pad = np.zeros((length - t, c), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load all CSVs → (N, SEQ_LEN, 6), (N,) int labels."""
    samples, labels = [], []
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        gesture_name = csv_path.stem.rsplit("_", 1)[0]
        if gesture_name not in GESTURE_TO_IDX:
            logging.warning("Skipping unknown gesture file: %s", csv_path.name)
            continue
        seq = load_csv(csv_path)
        seq = pad_or_truncate(seq, SEQ_LEN)
        samples.append(seq)
        labels.append(GESTURE_TO_IDX[gesture_name])
    X = np.stack(samples, axis=0)  # (N, 128, 6)
    y = np.array(labels, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Dataset & augmentation
# ---------------------------------------------------------------------------

class GestureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(X).float()  # (N, T, C)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # (T, C)
        if self.augment:
            x = self._augment(x)
        # Conv1d expects (C, T)
        return x.permute(1, 0), self.y[idx]

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        T, C = x.shape

        # Time stretch: speed ∈ [0.8, 1.2]
        speed = random.uniform(0.8, 1.2)
        new_len = int(round(T / speed))
        if new_len > 1:
            x_t = x.permute(1, 0).unsqueeze(0)  # (1, C, T)
            x_t = F.interpolate(x_t, size=new_len, mode="linear", align_corners=False)
            x_t = x_t.squeeze(0).permute(1, 0)  # (new_len, C)
            # re-pad/truncate to T
            if x_t.shape[0] >= T:
                x = x_t[:T]
            else:
                pad = torch.zeros(T - x_t.shape[0], C)
                x = torch.cat([x_t, pad], dim=0)

        # Amplitude scaling: [0.8, 1.2] uniform across all channels
        scale = random.uniform(0.8, 1.2)
        x = x * scale

        # Gaussian noise: std=0.02
        x = x + torch.randn_like(x) * 0.02

        # Time shift: ±10 samples
        shift = random.randint(-10, 10)
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=0)
            if shift > 0:
                x[:shift] = 0.0
            else:
                x[shift:] = 0.0

        # Channel dropout: p=0.1, zero out one channel
        if random.random() < 0.1:
            ch = random.randint(0, C - 1)
            x[:, ch] = 0.0

        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BatonNet(nn.Module):
    def __init__(self, in_ch: int = IN_CHANNELS, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 24, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(24)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(24, 48, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(48)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(48, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)  # GAP
        x = self.dropout(x)
        return self.fc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        preds = model(xb).argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def train_model(
    model, train_loader, val_loader, max_epochs=MAX_EPOCHS, patience=EARLY_STOP_PATIENCE
):
    """Train with early stopping. Returns history dict and best state_dict."""
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )

    best_val_acc = 0.0
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or wait == 0:
            logging.info(
                "Epoch %3d | train_loss=%.4f train_acc=%.3f | "
                "val_loss=%.4f val_acc=%.3f | best=%.3f",
                epoch, train_loss, train_acc, val_loss, val_acc, best_val_acc,
            )

        if wait >= patience:
            logging.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    return history, best_state


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=GESTURES)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info("Saved %s", save_path)


def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info("Saved %s", save_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTS_DIR / "train.log", mode="w"),
        ],
    )
    logging.info("Device: %s", DEVICE)

    # ---- 1. Load data ----
    logging.info("Loading dataset from %s ...", DATA_DIR)
    X, y = load_dataset()
    logging.info("Dataset: %d samples, seq_len=%d, channels=%d", X.shape[0], X.shape[1], X.shape[2])
    for i, g in enumerate(GESTURES):
        logging.info("  class %d %-10s : %d samples", i, g, (y == i).sum())

    # ---- 2. 10-Fold Cross-Validation ----
    logging.info("=" * 60)
    logging.info("10-Fold Stratified Cross-Validation")
    logging.info("=" * 60)

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_accs = []
    cv_preds_all = np.zeros(len(y), dtype=np.int64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info("--- Fold %d/%d ---", fold, NUM_FOLDS)
        train_ds = GestureDataset(X[train_idx], y[train_idx], augment=True)
        val_ds = GestureDataset(X[val_idx], y[val_idx], augment=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = BatonNet().to(DEVICE)
        _, best_state = train_model(model, train_loader, val_loader)
        model.load_state_dict(best_state)

        preds, labels = predict(model, val_loader)
        fold_acc = (preds == labels).mean()
        fold_accs.append(fold_acc)
        cv_preds_all[val_idx] = preds
        logging.info("Fold %d accuracy: %.4f", fold, fold_acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    logging.info("=" * 60)
    logging.info("10-Fold CV Accuracy: %.4f ± %.4f", mean_acc, std_acc)
    logging.info("=" * 60)

    plot_confusion_matrix(
        y, cv_preds_all,
        f"10-Fold CV Confusion Matrix (acc={mean_acc:.3f}±{std_acc:.3f})",
        RESULTS_DIR / "confusion_matrix_cv.png",
    )

    # ---- 3. Final model: 80/20 split ----
    logging.info("=" * 60)
    logging.info("Training Final Model (80/20 split)")
    logging.info("=" * 60)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=SEED
    )
    logging.info("Train: %d, Val: %d, Test: %d", len(y_train), len(y_val), len(y_test))

    train_ds = GestureDataset(X_train, y_train, augment=True)
    val_ds = GestureDataset(X_val, y_val, augment=False)
    test_ds = GestureDataset(X_test, y_test, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BatonNet().to(DEVICE)
    logging.info("BatonNet parameters: %d", count_parameters(model))

    history, best_state = train_model(model, train_loader, val_loader)
    model.load_state_dict(best_state)

    plot_training_curves(history, RESULTS_DIR / "training_curves.png")

    # ---- 4. Test evaluation ----
    preds, labels = predict(model, test_loader)
    test_acc = (preds == labels).mean()
    logging.info("Final model test accuracy: %.4f", test_acc)

    plot_confusion_matrix(
        labels, preds,
        f"Final Model Test Confusion Matrix (acc={test_acc:.3f})",
        RESULTS_DIR / "confusion_matrix_final.png",
    )

    report = classification_report(labels, preds, target_names=GESTURES, digits=4)
    logging.info("\n%s", report)
    report_path = RESULTS_DIR / "classification_report.txt"
    report_path.write_text(report)
    logging.info("Saved %s", report_path)

    # ---- 5. Export ----
    # PyTorch weights
    pt_path = RESULTS_DIR / "batonnet.pt"
    torch.save(model.state_dict(), pt_path)
    logging.info("Saved %s (%.1f KB)", pt_path, pt_path.stat().st_size / 1024)

    # ONNX (optional)
    try:
        import onnx as _onnx_check  # noqa: F401
        onnx_path = RESULTS_DIR / "batonnet.onnx"
        dummy = torch.randn(1, IN_CHANNELS, SEQ_LEN, device=DEVICE)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=13,
        )
        logging.info("Saved %s", onnx_path)
    except ImportError:
        logging.warning("onnx package not installed — skipping ONNX export")

    # Normalization params for deployment
    params = {
        "norm_scale": NORM_SCALE,
        "seq_len": SEQ_LEN,
        "in_channels": IN_CHANNELS,
        "class_names": GESTURES,
        "transform": "diff = (IMU_A - IMU_B) / 2 / 32768.0",
    }
    params_path = RESULTS_DIR / "normalization_params.json"
    params_path.write_text(json.dumps(params, indent=2))
    logging.info("Saved %s", params_path)

    logging.info("Done. All outputs in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()

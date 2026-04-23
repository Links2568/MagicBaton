#!/usr/bin/env python3
"""Train and export best SVM model for beat detection (zuchen data, RBF C=10)."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "dataset" / "zuchen"
MODELS_DIR = ROOT / "models"
NORM_SCALE = 32768.0


def load_csv_diff(path):
    df = pd.read_csv(path)
    a = df[["ax_a","ay_a","az_a","gx_a","gy_a","gz_a"]].values.astype(np.float32)
    b = df[["ax_b","ay_b","az_b","gx_b","gy_b","gz_b"]].values.astype(np.float32)
    return (a - b) / 2.0 / NORM_SCALE


def extract_features(seq):
    T, C = seq.shape
    feats = []
    for c in range(C):
        ch = seq[:, c]
        m, s = np.mean(ch), np.std(ch)
        feats.extend([
            m, s, np.min(ch), np.max(ch), np.max(ch)-np.min(ch),
            np.median(ch),
            np.mean(((ch-m)/(s+1e-10))**3),
            np.mean(((ch-m)/(s+1e-10))**4)-3,
            np.sum(ch**2)/T,
            np.sum(np.diff(np.sign(ch))!=0)/max(T-1,1),
            np.percentile(ch,25), np.percentile(ch,75),
            np.percentile(ch,75)-np.percentile(ch,25),
        ])
    accel_mag = np.sqrt(np.sum(seq[:,:3]**2, axis=1))
    gyro_mag = np.sqrt(np.sum(seq[:,3:]**2, axis=1))
    feats.extend([np.mean(accel_mag), np.std(accel_mag), np.mean(gyro_mag), np.std(gyro_mag)])
    return np.array(feats, dtype=np.float32)


def main():
    # Load data
    feats, labels = [], []
    for csv_path in sorted(BASE_DIR.glob("*.csv")):
        gesture = csv_path.stem.rsplit("_", 1)[0]
        if gesture == "metadata":
            continue
        seq = load_csv_diff(csv_path)
        feats.append(extract_features(seq))
        labels.append(1 if gesture == "beat" else 0)

    X = np.stack(feats)
    y = np.array(labels)
    print(f"Training on {len(y)} samples (beat={sum(y==1)}, other={sum(y==0)})")

    # Scale + train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    clf.fit(X_scaled, y)
    print(f"Training accuracy: {clf.score(X_scaled, y):.4f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "beat_svm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"scaler": scaler, "model": clf}, f)
    print(f"Saved {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

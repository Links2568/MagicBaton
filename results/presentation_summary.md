# MagicBaton Gesture Recognition — ML Results Summary

## 1. Data Collection

### Hardware
- ESP32 + 2x MPU6050 IMUs (addresses 0x68 and 0x69) mounted on a conducting baton
- 12-axis raw data at ~50Hz: accelerometer (3-axis) + gyroscope (3-axis) per IMU
- Communication: BLE (device name "MagicBaton")
- Data format per sample: `timestamp, ax_a, ay_a, az_a, gx_a, gy_a, gz_a, ax_b, ay_b, az_b, gx_b, gy_b, gz_b`

### Recording Process
- Custom Tkinter GUI recorder (`record.py`) connects to baton via BLE
- Operator selects gesture class, presses Space to start/stop recording
- Each recording captures one clean gesture execution (no idle padding)
- Saved as individual CSV files with gesture label in filename (e.g., `beat_001.csv`)

### Preprocessing
- Diff signal: `(IMU_A - IMU_B) / 2 / 32768.0` — 6 channels, normalized to [-1, 1]
- The difference cancels common-mode motion and isolates relative baton dynamics
- Pad or truncate each sample to 128 time steps (SEQ_LEN)

### 7 Gesture Classes

| Gesture | Motion Description |
|---------|-------------------|
| **beat** | Conductor's downbeat — quick downward strike |
| **stab** | Forward thrust along baton axis (fencing lunge) |
| **spin** | Draw a full circle in the air |
| **slash** | Diagonal cut (upper-left to lower-right) |
| **shake** | Rapid side-to-side tremolo for 1-2 seconds |
| **flick** | Single sharp wrist snap |
| **wing** | Trace the letter "W" in the air |

---

## 2. Datasets

### Single-Person Dataset (data_zuchen)
- **1 person** (zuchen), **8 classes** (7 gestures + idle)
- **443 samples** total

| Class | Samples |
|-------|---------|
| idle | 22 |
| beat | 61 |
| stab | 60 |
| spin | 60 |
| slash | 60 |
| shake | 60 |
| flick | 60 |
| wing | 60 |

### Multi-Person Dataset (4 people)
- **4 people** (zuchen, tiffany, xiaolan, yoyo), **7 classes** (gesture intersection, no idle)
- **842 samples** total

| Person | Samples |
|--------|---------|
| zuchen | 421 |
| tiffany | 141 |
| xiaolan | 140 |
| yoyo | 140 |

Per-gesture breakdown (all 7 gestures): beat=121, stab=120, spin=120, slash=121, shake=120, flick=120, wing=120

---

## 3. Model Architectures

### 3.1 LinkNet-CNN (1D Convolutional Neural Network)
- Input: (batch, 6, 128) — 6-channel diff signal, 128 time steps
- Conv1d(6→32, k=7) → BN → ReLU → MaxPool(2)
- Conv1d(32→64, k=5) → BN → ReLU → MaxPool(2)
- Conv1d(64→128, k=3) → BN → ReLU → Global Average Pooling
- Dropout(0.4) → Linear(128→7)
- Training: AdamW (lr=1e-3, wd=1e-2), CosineAnnealingLR, label smoothing=0.1
- Data augmentation: time stretch, amplitude scaling, Gaussian noise, time shift, channel dropout
- Early stopping: patience=30 epochs

### 3.2 LinkNet-SVM (Support Vector Machine)
- 82 hand-crafted features per sample:
  - 13 statistics per channel x 6 channels = 78 (mean, std, min, max, range, median, skewness, kurtosis, energy, zero-crossing rate, Q25, Q75, IQR)
  - 4 global features (accel magnitude mean/std, gyro magnitude mean/std)
- StandardScaler normalization per fold
- RBF kernel, C=10, gamma="scale"

### 3.3 LinkNet-RF (Random Forest)
- Same 82 features as SVM
- 200 estimators, no max depth limit
- StandardScaler normalization per fold

### 3.4 LinkNet-RNN (Bidirectional GRU)
- Input: (batch, 128, 6) — 128 time steps, 6 features
- 2-layer Bidirectional GRU, hidden=64 (→128 with bidir)
- Average pooling over time → Dropout(0.4) → Linear(128→7)
- Same training setup as CNN

---

## 4. Experiment 1: Single-Person (data_zuchen, 5-Fold CV)

8 classes including idle. 5-fold stratified cross-validation.

### Results

| Model | Accuracy |
|-------|----------|
| **CNN** | **98.20% +/- 1.53%** |
| SVM | 97.52% +/- 1.80% |
| RF | 97.52% +/- 1.10% |
| RNN | 96.85% +/- 3.29% |

### Per-Class F1 Scores (CNN)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|------|
| idle | 1.000 | 0.955 | 0.977 |
| beat | 0.966 | 0.934 | 0.950 |
| stab | 1.000 | 1.000 | 1.000 |
| spin | 0.984 | 1.000 | 0.992 |
| slash | 0.937 | 0.983 | 0.959 |
| shake | 1.000 | 1.000 | 1.000 |
| flick | 0.984 | 1.000 | 0.992 |
| wing | 1.000 | 0.967 | 0.983 |

**Key findings:**
- All 4 models achieve >96% accuracy on single-person data
- CNN is the best (98.2%), with lowest error variance
- stab, shake are perfectly classified by all models
- Most confusion occurs between beat/slash (similar downward motions)

Confusion matrices: `cm_linknet_cnn.png`, `cm_linknet_svm.png`, `cm_linknet_rf.png`, `cm_linknet_rnn.png`
Comparison chart: `linknet_comparison.png`

---

## 5. Experiment 2: Cross-Person (4 people, LOPO + 5-Fold)

7 classes (no idle). Two evaluation strategies compared:

### 5-Fold CV (Mixed People)
Randomly split all 842 samples into 5 folds — same person's data appears in both train and test.

| Model | Accuracy |
|-------|----------|
| **CNN** | **99.05% +/- 0.60%** |
| SVM | 97.51% +/- 1.02% |
| RNN | 97.51% +/- 1.47% |
| RF | 96.79% +/- 1.33% |

### LOPO (Leave-One-Person-Out)
Train on 3 people, test on the 4th — the test person is completely unseen during training.

| Model | Accuracy |
|-------|----------|
| **CNN** | **70.62% +/- 7.24%** |
| RNN | 70.21% +/- 9.81% |
| RF | 63.18% +/- 19.08% |
| SVM | 61.76% +/- 18.46% |

### LOPO Per-Person Accuracy

| Model | zuchen | tiffany | xiaolan | yoyo |
|-------|--------|---------|---------|------|
| CNN | 70.1% | 60.3% | 80.7% | 71.4% |
| SVM | 73.9% | 53.9% | 83.6% | 35.7% |
| RF | 67.2% | 31.9% | 83.6% | 70.0% |
| RNN | 66.3% | 58.9% | 70.0% | 85.7% |

### LOPO Per-Class F1 Scores (CNN)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|------|
| beat | 0.608 | 0.628 | 0.618 |
| stab | 0.818 | 0.825 | 0.822 |
| spin | 0.726 | 0.617 | 0.667 |
| slash | 0.547 | 0.678 | 0.605 |
| shake | 0.727 | 0.533 | 0.615 |
| flick | 0.891 | 0.950 | 0.919 |
| wing | 0.656 | 0.700 | 0.677 |

### Key Findings

1. **Massive gap between 5-Fold and LOPO**: 99% vs 70% (CNN). 5-Fold overfits to individual users' styles when data from the same person appears in both train and test splits.

2. **LOPO reveals the real challenge**: Cross-person generalization is fundamentally harder — each person has unique grip, wrist mechanics, force patterns, and timing.

3. **CNN and RNN generalize best** (~70%), while traditional ML (SVM/RF) struggle more (~62%) and show extremely high variance (std ~19%), meaning some people are nearly unrecognizable.

4. **flick is the most person-invariant gesture** (F1=0.92 LOPO) — likely because the sharp wrist snap produces a distinctive, universal signal pattern.

5. **slash is the hardest to generalize** (F1=0.61 LOPO) — diagonal motions vary most between individuals in angle and speed.

6. **Person-specific difficulty**: xiaolan is easiest to generalize to (80.7% CNN), tiffany is hardest (60.3% CNN). This suggests tiffany's gesture style is most unique/different from the other 3 people.

7. **Implications**: For production deployment, either (a) collect calibration data per user, (b) use domain adaptation / fine-tuning, or (c) collect data from many more people to learn person-invariant features.

Confusion matrices: `cm_cross_cnn_lopo.png`, `cm_cross_cnn_5fold.png`, etc.
Comparison chart: `linknet_cross_comparison.png`
Per-person heatmap: `linknet_cross_per_person.png`

---

## 6. Summary Table

| Experiment | Model | Accuracy |
|------------|-------|----------|
| Single-person, 5-Fold | CNN | 98.2% |
| Single-person, 5-Fold | SVM | 97.5% |
| Single-person, 5-Fold | RF | 97.5% |
| Single-person, 5-Fold | RNN | 96.9% |
| Multi-person, 5-Fold (mixed) | CNN | 99.1% |
| Multi-person, 5-Fold (mixed) | SVM | 97.5% |
| Multi-person, 5-Fold (mixed) | RF | 96.8% |
| Multi-person, 5-Fold (mixed) | RNN | 97.5% |
| Multi-person, LOPO | CNN | 70.6% |
| Multi-person, LOPO | SVM | 61.8% |
| Multi-person, LOPO | RF | 63.2% |
| Multi-person, LOPO | RNN | 70.2% |

---

## 7. Output Files

### Models
- `results/linknet.pt` — CNN trained on single-person data (8 classes)
- `results/linknet_cross.pt` — CNN trained on all 4 people (7 classes, deployed in server.py)

### Figures
- `results/linknet_comparison.png` — Single-person 4-model comparison bar chart
- `results/linknet_cross_comparison.png` — Cross-person LOPO vs 5-Fold comparison
- `results/linknet_cross_per_person.png` — Per-person LOPO accuracy heatmap
- `results/cm_linknet_*.png` — Single-person confusion matrices (4 models)
- `results/cm_cross_*_lopo.png` — Cross-person LOPO confusion matrices (4 models)
- `results/cm_cross_*_5fold.png` — Cross-person 5-Fold confusion matrices (4 models)

### Reports
- `results/linknet_reports.txt` — Single-person classification reports
- `results/linknet_cross_reports.txt` — Cross-person classification reports

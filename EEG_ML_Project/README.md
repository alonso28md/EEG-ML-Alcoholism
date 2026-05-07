# 🧠 EEG & Machine Learning — Alcoholism Predisposition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**Detecting genetic predisposition to alcoholism through EEG signal analysis and machine learning**

*Biomedical Engineering Portfolio · Alonso Martín Díez · Universidad Europea de Madrid 2026*

</div>

---

## 📋 Overview

Complete **end-to-end machine learning pipeline** for classifying EEG signals from alcoholic vs. control subjects. Using the [UCI EEG Database](https://doi.org/10.24432/C5TS3D) (Henri Begleiter, 1995), 46 features are engineered from raw brainwave recordings across 122 subjects and 7 ML classifiers are trained to detect the neurological signature of alcohol dependency.

### Results — Real Data (122 subjects, 97 train / 25 test)

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| 🏆 XGBoost | **0.76** | **0.758** | **0.788** |
| Decision Tree | 0.72 | 0.719 | **0.865** |
| Random Forest | 0.72 | 0.720 | 0.782 |
| Logistic Regression | 0.68 | 0.677 | 0.821 |
| MLP Neural Net | 0.64 | 0.633 | 0.699 |
| SVM (RBF) | 0.64 | 0.633 | 0.635 |
| KNN | 0.56 | 0.556 | 0.750 |

> Decision Tree obtiene el mayor ROC AUC (0.865), XGBoost la mayor accuracy (0.76).

---

## 🧬 Scientific Background

### Why EEG for Alcoholism Detection?

Chronic alcohol exposure produces measurable, lasting changes in brain oscillatory activity:

```
Normal Brain                     Alcoholic Brain
────────────────                 ────────────────
Alpha (8-13 Hz) ████████         Alpha (8-13 Hz) ████
Beta  (13-30 Hz) ████            Beta  (13-30 Hz) ████████
→ Strong inhibitory control      → Cortical hyperexcitability
```

The **alpha/beta power ratio** is the primary biomarker — reduced ratio consistently distinguishes alcoholic from control subjects even in abstinent individuals, suggesting a **genetic predisposition marker**.

### Brain Region Significance

| Region | Electrodes | Finding |
|--------|-----------|---------|
| Prefrontal | FP1, FP2 | Executive dysfunction, impulse control loss |
| Frontal | F3, F4, FZ | Decision-making, inhibitory control |
| Occipital | O1, O2, OZ | P300 ERP abnormalities (visual processing) |
| Central | CZ, C3, C4 | Motor control alterations |

---

## 📁 Repository Structure

```
EEG_ML_Project/
│
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb     ← Data loading, EEG signals, topographic maps
│   ├── 02_preprocessing.ipynb        ← Feature engineering (46 features), normalisation
│   ├── 03_models.ipynb               ← 7 ML classifiers, cross-validation, ROC curves
│   └── 04_results.ipynb              ← Grand dashboard, clinical conclusions
│
├── 🐍 src/
│   └── utils.py                      ← Shared utilities: theme, features, evaluation
│
├── 📊 data/
│   ├── raw/                          ← CSVs generated from UCI dataset (11,057 files)
│   ├── eeg_cleaned.parquet           ← Output of Notebook 01
│   ├── X_train_scaled.parquet        ← Output of Notebook 02
│   ├── X_test_scaled.parquet
│   ├── y_train.parquet / y_test.parquet
│   ├── scaler.pkl                    ← Fitted StandardScaler
│   ├── model_*.pkl                   ← 7 trained models
│   ├── model_results.csv             ← Full metrics table
│   └── project_summary.json         ← Project summary
│
├── 📄 docs/
│   ├── EEG_ML_Technical_Documentation.docx
│   └── fig_01 … fig_17.png          ← All generated figures
│
├── 🔧 convert_eeg_to_csv.py         ← Converts UCI .tar.gz to CSV format
└── 📝 README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/alonsomartindiez/eeg-ml-alcoholism.git
cd EEG_ML_Project
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost scipy joblib pyarrow jupyter ipykernel
```

### 3. Prepare the data

**Option A — Use the UCI dataset (recommended):**
1. Download from https://archive.ics.uci.edu/dataset/121/eeg+database
2. Extract to a folder and run:
```bash
python convert_eeg_to_csv.py
```
This converts the `.tar.gz` files to CSV format in `data/raw/`.

**Option B — Run without data (demo mode):**
No setup needed. The notebooks auto-generate synthetic EEG data.

### 4. Run the pipeline
```bash
jupyter lab
```
Execute notebooks in order: **01 → 02 → 03 → 04**

| Notebook | Runtime |
|----------|---------|
| 01 — Data Exploration | ~5 min |
| 02 — Preprocessing | ~20 min |
| 03 — Models | ~10 min |
| 04 — Results | ~2 min |

---

## 🔬 Feature Engineering (46 features)

Features extracted per (subject, electrode) pair, then aggregated as mean + std across all 64 electrodes:

| Category | Features | Count |
|----------|---------|-------|
| Statistical | mean, std, variance, skewness, kurtosis, p2p, rms, iqr, pcm, pcsd | 10 × 2 |
| Band power (absolute) | delta, theta, **alpha ⭐**, **beta ⭐**, gamma | 5 × 2 |
| Band power (relative) | rel_delta, rel_theta, rel_alpha, rel_beta, rel_gamma | 5 × 2 |
| Ratio features | **alpha/beta ⭐**, theta/alpha, delta/alpha | 3 × 2 |
| **Total** | | **46** |

⭐ = Primary biomarkers for alcoholism detection

---

## 📊 Visual Outputs (17 figures)

| Figure | Description |
|--------|-------------|
| fig_01 | Dataset overview: class balance, counts, conditions |
| fig_02 | Raw EEG signals: alcoholic vs control, 4 electrodes |
| fig_03 | Power Spectral Density with band annotations |
| fig_04 | Topographic brain maps: voltage by electrode |
| fig_05 | Amplitude distributions by brain region |
| fig_06 | Frequency band power distributions |
| fig_07 | Alpha power topomaps: spatial comparison |
| fig_08 | Feature correlation matrix + target correlation |
| fig_09 | Top 8 discriminative features |
| fig_10 | All 7 confusion matrices |
| fig_11 | ROC curves — all models |
| fig_12 | Feature importance — tree models |
| fig_13 | Accuracy · F1 · AUC comparison |
| fig_14 | **Grand results dashboard** |
| fig_15 | Learning curves — bias/variance analysis |
| fig_16 | Electrode predictive power topomaps |
| fig_17 | **One-page project summary** |

---

## 🏥 Clinical Implications

> **Disclaimer**: Academic portfolio project. Not for clinical use without validation.

- **Early detection**: EEG biomarkers could identify at-risk individuals before clinical presentation
- **Treatment monitoring**: Track neurological recovery during abstinence
- **Research tool**: Stratify patients in clinical trials using objective neural markers

---

## 📚 References

1. Begleiter, H. (1995). *EEG Database*. UCI ML Repository. https://doi.org/10.24432/C5TS3D
2. Zhang, X.L. et al. (1995). *Event related potentials during object recognition tasks*. Brain Research Bulletin, 38(6), 531-538.
3. Porjesz, B. et al. (2005). *The utility of neurophysiological markers in the study of alcoholism*. Clinical Neurophysiology, 116(5), 993-1018.
4. Rangaswamy, M. & Porjesz, B. (2014). *Understanding alcohol use disorders with neuroelectrophysiology*. Handbook of Clinical Neurology, 125, 383-414.

---

## 👤 Author

**Alonso Martín Díez** · Biomedical Engineering · Universidad Europea de Madrid · 2026

"""
EEG Alcoholism Analysis — Shared Utilities
===========================================
Shared functions used across all notebooks in the EEG + Machine Learning
alcoholism predisposition analysis pipeline.

Author: Alonso Martín Díez
Institution: Universidad Europea de Madrid
Stack: Python · scikit-learn · XGBoost · pandas · matplotlib · seaborn · Jupyter
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as pe
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  COLOUR PALETTE  (purple / scientific theme)
# ─────────────────────────────────────────────
PALETTE = {
    'bg':         '#0D0D1A',   # deep space background
    'bg_card':    '#13132B',   # card / panel background
    'purple1':    '#7B2FBE',   # vivid purple
    'purple2':    '#9D4EDD',   # medium purple
    'purple3':    '#C77DFF',   # light purple
    'teal':       '#00F5D4',   # accent teal
    'gold':       '#FFD60A',   # accent gold
    'coral':      '#FF6B6B',   # alcoholic group
    'blue':       '#4CC9F0',   # control group
    'white':      '#F0E6FF',   # text
    'grey':       '#6B6B8A',   # muted text
    'grid':       '#1E1E3F',   # grid lines
}

MODEL_COLORS = {
    'Logistic Regression': '#4CC9F0',
    'Decision Tree':       '#FF6B6B',
    'Random Forest':       '#FFD60A',
    'KNN':                 '#C77DFF',
    'XGBoost':             '#00F5D4',
    'SVM (RBF)':           '#F72585',
    'MLP Neural Net':      '#7B2FBE',
}

ALCOHOLIC_COLOR = PALETTE['coral']
CONTROL_COLOR   = PALETTE['blue']

# EEG electrode positions (standard 10-20 system, simplified 2D)
ELECTRODE_POSITIONS = {
    'FP1': (-0.18, 0.90), 'FP2': (0.18, 0.90),
    'AF1': (-0.15, 0.78), 'AF2': (0.15, 0.78),
    'AF7': (-0.40, 0.75), 'AF8': (0.40, 0.75),
    'F7':  (-0.62, 0.55), 'F8':  (0.62, 0.55),
    'F5':  (-0.47, 0.60), 'F6':  (0.47, 0.60),
    'F3':  (-0.30, 0.62), 'F4':  (0.30, 0.62),
    'F1':  (-0.15, 0.65), 'F2':  (0.15, 0.65),
    'FZ':  (0.00, 0.65),
    'FC5': (-0.55, 0.38), 'FC6': (0.55, 0.38),
    'FC3': (-0.33, 0.40), 'FC4': (0.33, 0.40),
    'FC1': (-0.16, 0.42), 'FC2': (0.16, 0.42),
    'FCZ': (0.00, 0.42),
    'T7':  (-0.80, 0.00), 'T8':  (0.80, 0.00),
    'C5':  (-0.60, 0.05), 'C6':  (0.60, 0.05),
    'C3':  (-0.38, 0.05), 'C4':  (0.38, 0.05),
    'C1':  (-0.19, 0.05), 'C2':  (0.19, 0.05),
    'CZ':  (0.00, 0.05),
    'CP5': (-0.55,-0.28), 'CP6': (0.55,-0.28),
    'CP3': (-0.33,-0.30), 'CP4': (0.33,-0.30),
    'CP1': (-0.16,-0.32), 'CP2': (0.16,-0.32),
    'CPZ': (0.00,-0.32),
    'P7':  (-0.62,-0.55), 'P8':  (0.62,-0.55),
    'P5':  (-0.47,-0.60), 'P6':  (0.47,-0.60),
    'P3':  (-0.30,-0.62), 'P4':  (0.30,-0.62),
    'P1':  (-0.15,-0.65), 'P2':  (0.15,-0.65),
    'PZ':  (0.00,-0.65),
    'PO7': (-0.40,-0.75), 'PO8': (0.40,-0.75),
    'PO3': (-0.25,-0.78), 'PO4': (0.25,-0.78),
    'POZ': (0.00,-0.78),
    'O1':  (-0.18,-0.90), 'O2':  (0.18,-0.90),
    'OZ':  (0.00,-0.88),
    'CB1': (-0.25,-0.97), 'CB2': (0.25,-0.97),
}

# EEG frequency bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50),
}

BAND_COLORS = {
    'delta': PALETTE['purple1'],
    'theta': PALETTE['teal'],
    'alpha': PALETTE['gold'],
    'beta':  PALETTE['coral'],
    'gamma': PALETTE['blue'],
}

FS = 256  # Sampling frequency (Hz)

# ─────────────────────────────────────────────
#  STYLE & THEMING
# ─────────────────────────────────────────────

def apply_dark_theme():
    """Apply the consistent dark purple scientific theme to all matplotlib figures."""
    plt.rcParams.update({
        'figure.facecolor':  PALETTE['bg'],
        'axes.facecolor':    PALETTE['bg_card'],
        'axes.edgecolor':    PALETTE['purple2'],
        'axes.labelcolor':   PALETTE['white'],
        'axes.titlecolor':   PALETTE['white'],
        'axes.grid':         True,
        'grid.color':        PALETTE['grid'],
        'grid.linewidth':    0.5,
        'grid.alpha':        0.7,
        'text.color':        PALETTE['white'],
        'xtick.color':       PALETTE['grey'],
        'ytick.color':       PALETTE['grey'],
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'legend.facecolor':  PALETTE['bg_card'],
        'legend.edgecolor':  PALETTE['purple2'],
        'legend.labelcolor': PALETTE['white'],
        'font.family':       'DejaVu Sans',
        'figure.dpi':        120,
        'savefig.dpi':       150,
        'savefig.bbox':      'tight',
        'savefig.facecolor': PALETTE['bg'],
    })

def add_background_shapes(ax, n_circles=6, alpha=0.04):
    """Add subtle decorative background circles for visual depth."""
    rng = np.random.default_rng(42)
    for _ in range(n_circles):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        r = rng.uniform(0.05, 0.20)
        circle = Circle((x, y), r, transform=ax.transAxes,
                         color=PALETTE['purple2'], alpha=alpha,
                         zorder=0, clip_on=False)
        ax.add_patch(circle)

def styled_title(fig, title, subtitle=None, y=0.98):
    """Add a styled title with optional subtitle to a figure."""
    fig.text(0.5, y, title,
             fontsize=20, fontweight='bold', color=PALETTE['purple3'],
             ha='center', va='top',
             path_effects=[pe.withStroke(linewidth=3, foreground=PALETTE['bg'])])
    if subtitle:
        fig.text(0.5, y - 0.04, subtitle,
                 fontsize=11, color=PALETTE['grey'],
                 ha='center', va='top', style='italic')

def add_purple_spine(ax, sides=('bottom', 'left')):
    """Highlight specific axis spines in purple."""
    for side in sides:
        ax.spines[side].set_color(PALETTE['purple2'])
        ax.spines[side].set_linewidth(1.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_eeg_data(data_folder, n_files=122, seed=6942069, verbose=True):
    """
    Load n_files random CSV files from data_folder into a combined DataFrame.

    Parameters
    ----------
    data_folder : str  — path to folder containing Data1.csv … Data480.csv
    n_files     : int  — number of files to sample (default 122)
    seed        : int  — random seed for reproducibility
    verbose     : bool — print progress

    Returns
    -------
    pd.DataFrame with combined EEG data, cleaned column names
    """
    random.seed(seed)
    np.random.seed(seed)

    all_files = [f for f in os.listdir(data_folder)
                 if f.endswith('.csv') and f.startswith('Data')]
    total = len(all_files)

    if verbose:
        print(f"  Found {total} CSV files in '{data_folder}'")

    indices = random.sample(range(1, total + 1), min(n_files, total))
    file_names = [f'Data{i}.csv' for i in indices]

    frames = []
    loaded = 0
    for fname in file_names:
        fpath = os.path.join(data_folder, fname)
        if os.path.exists(fpath):
            frames.append(pd.read_csv(fpath, header=0))
            loaded += 1

    if verbose:
        print(f"  Loaded {loaded} files → {sum(len(f) for f in frames):,} rows")

    df = pd.concat(frames, ignore_index=True)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop unnamed index column if present
    unnamed = [c for c in df.columns if 'Unnamed' in c]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)

    return df


def load_processed_data(path='../data/eeg_processed.parquet'):
    """Load the processed feature dataset saved by notebook 02."""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

# ─────────────────────────────────────────────
#  DATA CLEANING
# ─────────────────────────────────────────────

def clean_eeg_dataframe(df, verbose=True):
    """
    Full cleaning pipeline for raw EEG data.
    - Remove duplicates
    - Fix data types
    - Handle missing values
    - Add derived columns
    """
    n0 = len(df)

    # Remove exact duplicates
    df = df.drop_duplicates()
    if verbose:
        print(f"  Duplicates removed: {n0 - len(df):,}")

    # Fix dtypes
    str_cols = ['sensor position', 'subject identifier', 'matching condition', 'name']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype('string')

    numeric_cols = ['sensor value', 'sample num', 'trial number', 'channel']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Encode target: 'a' = alcoholic (1), 'c' = control (0)
    if 'subject identifier' in df.columns:
        df['is_alcoholic'] = (df['subject identifier'] == 'a').astype(int)

    if verbose:
        nan_total = df.isna().sum().sum()
        print(f"  NaN cells after type fix: {nan_total:,}")

    # Fill sample num (cyclical 0-255)
    if 'sample num' in df.columns:
        df['sample num'] = df['sample num'].fillna(pd.Series(df.index % 256, index=df.index))

    # Fill categorical columns with forward-fill then mode
    cat_fill = ['trial number', 'sensor position', 'subject identifier',
                'matching condition', 'channel', 'name']
    for col in cat_fill:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].ffill().bfill()

    # Fill sensor value with group median
    if 'sensor value' in df.columns and df['sensor value'].isna().any():
        df['sensor value'] = df.groupby('subject identifier')['sensor value'].transform(
            lambda x: x.fillna(x.median())
        )

    if verbose:
        print(f"  NaN cells after imputation: {df.isna().sum().sum():,}")
        print(f"  Final shape: {df.shape}")

    return df

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────

def compute_band_power(signal_values, fs=FS):
    """
    Compute power in each EEG frequency band using Welch's method.

    Returns dict: {band_name: power_value}
    """
    if len(signal_values) < 4:
        return {b: np.nan for b in FREQ_BANDS}

    nperseg = min(256, len(signal_values))
    freqs, psd = signal.welch(signal_values, fs=fs, nperseg=nperseg)

    band_powers = {}
    for band, (flo, fhi) in FREQ_BANDS.items():
        idx = np.logical_and(freqs >= flo, freqs <= fhi)
        band_powers[band] = np.trapezoid(psd[idx], freqs[idx]) if idx.any() else np.nan

    return band_powers


def extract_features_per_subject_electrode(df, fs=FS, verbose=True):
    """
    Extract rich features per (subject, electrode) pair.
    
    Features extracted:
    - Statistical: mean, std, variance, skewness, kurtosis, p2p amplitude
    - Frequency: delta/theta/alpha/beta/gamma band power, dominant frequency
    - Ratio features: alpha/beta ratio (key for alcoholism detection)
    
    Returns
    -------
    pd.DataFrame — one row per (subject, electrode) with all features
    """
    if verbose:
        print("  Extracting features per subject-electrode pair...")

    records = []
    grouped = df.groupby(['name', 'sensor position'])

    total = len(grouped)
    for i, ((name, electrode), grp) in enumerate(grouped):
        if i % 500 == 0 and verbose:
            print(f"    Progress: {i}/{total} ({100*i/total:.1f}%)")

        vals = grp['sensor value'].dropna().values
        if len(vals) < 10:
            continue

        # Subject metadata
        row = {
            'subject_id':  name,
            'electrode':   electrode,
            'is_alcoholic': int(grp['is_alcoholic'].mode()[0]),
        }

        # ── Statistical features
        row['mean']      = np.mean(vals)
        row['std']       = np.std(vals)
        row['variance']  = np.var(vals)
        row['skewness']  = float(skew(vals))
        row['kurt']      = float(kurtosis(vals))
        row['p2p']       = np.ptp(vals)   # peak-to-peak amplitude
        row['rms']       = np.sqrt(np.mean(vals**2))
        row['iqr']       = float(np.percentile(vals, 75) - np.percentile(vals, 25))

        # ── PCM (position condition mean) — from original analysis
        row['pcm'] = np.mean(vals)
        row['pcsd'] = np.std(vals)

        # ── Frequency band powers
        bp = compute_band_power(vals, fs=fs)
        total_power = sum(v for v in bp.values() if not np.isnan(v)) or 1e-10
        for band, power in bp.items():
            row[f'power_{band}'] = power
            row[f'rel_power_{band}'] = (power / total_power) if not np.isnan(power) else np.nan

        # ── Derived ratio features (clinically meaningful)
        alpha = bp.get('alpha', np.nan)
        beta  = bp.get('beta',  np.nan)
        theta = bp.get('theta', np.nan)
        delta = bp.get('delta', np.nan)

        row['alpha_beta_ratio'] = alpha / beta   if (beta  and beta > 0)  else np.nan
        row['theta_alpha_ratio'] = theta / alpha if (alpha and alpha > 0) else np.nan
        row['delta_alpha_ratio'] = delta / alpha if (alpha and alpha > 0) else np.nan

        # ── Matching condition
        cond = grp['matching condition'].mode()
        row['condition'] = str(cond.iloc[0]) if len(cond) > 0 else 'unknown'

        records.append(row)

    feat_df = pd.DataFrame(records)
    if verbose:
        print(f"  Feature matrix: {feat_df.shape[0]:,} rows × {feat_df.shape[1]} columns")

    return feat_df


def aggregate_features_per_subject(feat_df, verbose=True):
    """
    Aggregate electrode-level features into one row per subject.
    Uses mean across all electrodes + electrode-specific features for key channels.
    """
    if verbose:
        print("  Aggregating features per subject...")

    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['is_alcoholic']]

    agg = feat_df.groupby('subject_id').agg(
        is_alcoholic=('is_alcoholic', 'first'),
        **{f'{col}_mean': (col, 'mean') for col in numeric_cols},
        **{f'{col}_std':  (col, 'std')  for col in numeric_cols},
    ).reset_index()

    if verbose:
        alcoholic = agg['is_alcoholic'].sum()
        control   = len(agg) - alcoholic
        print(f"  Subjects: {len(agg)} total ({alcoholic} alcoholic, {control} control)")
        print(f"  Feature matrix: {agg.shape[0]} × {agg.shape[1]}")

    return agg

# ─────────────────────────────────────────────
#  TOPOGRAPHIC MAP
# ─────────────────────────────────────────────

def plot_topomap(values_dict, title='EEG Topomap', cmap='RdPu',
                 colorbar_label='Value', ax=None):
    """
    Plot a simple topographic heatmap of EEG electrode values.

    Parameters
    ----------
    values_dict : dict {electrode_name: value}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    # Draw head outline
    head = plt.Circle((0, 0), 1.0, fill=False,
                       color=PALETTE['purple3'], linewidth=2.5)
    ax.add_patch(head)

    # Nose
    nose_x = [-.08, 0, .08]
    nose_y = [0.97, 1.10, 0.97]
    ax.plot(nose_x, nose_y, color=PALETTE['purple3'], linewidth=2)

    # Ears
    for sign in [-1, 1]:
        ear_x = [sign*0.98, sign*1.05, sign*1.05, sign*0.98]
        ear_y = [0.10, 0.05, -0.05, -0.10]
        ax.plot(ear_x, ear_y, color=PALETTE['purple3'], linewidth=2)

    # Plot electrodes
    vmin = min(values_dict.values())
    vmax = max(values_dict.values())
    cm = plt.get_cmap(cmap)

    for elec, val in values_dict.items():
        pos = ELECTRODE_POSITIONS.get(elec.upper())
        if pos is None:
            continue
        norm_val = (val - vmin) / (vmax - vmin + 1e-10)
        color = cm(norm_val)
        circle = plt.Circle(pos, 0.055, color=color, zorder=5, alpha=0.9)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], elec.upper(),
                ha='center', va='center', fontsize=4.5,
                color='white', fontweight='bold', zorder=6)

    # Colorbar-like gradient
    sm = plt.cm.ScalarMappable(cmap=cm,
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label(colorbar_label, color=PALETTE['white'], fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=PALETTE['grey'])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE['grey'])

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, color=PALETTE['purple3'], fontweight='bold', pad=12)

    return ax

# ─────────────────────────────────────────────
#  EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate_model(name, model, X_test, y_test, results_list=None):
    """
    Evaluate a trained model and return a metrics dict.
    Optionally appends to results_list for dashboard building.
    """
    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)[:, 1]
              if hasattr(model, 'predict_proba') else None)

    metrics = {
        'Model':    name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1':       f1_score(y_test, y_pred, average='weighted'),
        'ROC_AUC':  roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        'y_pred':   y_pred,
        'y_prob':   y_prob,
    }

    if results_list is not None:
        results_list.append(metrics)

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, ax=None):
    """Plot a styled confusion matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap='Purples',
                linecolor=PALETTE['bg'], linewidths=1,
                cbar=False,
                annot_kws={'fontsize': 14, 'fontweight': 'bold',
                           'color': PALETTE['white']})
    ax.set_xlabel('Predicted', color=PALETTE['white'])
    ax.set_ylabel('Actual', color=PALETTE['white'])
    ax.set_title(f'{model_name}\nConfusion Matrix',
                 color=PALETTE['purple3'], fontweight='bold')
    ax.set_xticklabels(['Control', 'Alcoholic'], color=PALETTE['white'])
    ax.set_yticklabels(['Control', 'Alcoholic'], color=PALETTE['white'], rotation=0)
    return ax


def print_model_report(name, metrics):
    """Pretty-print model evaluation report."""
    sep = '─' * 50
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)
    print(f"  Accuracy : {metrics['Accuracy']:.4f}  ({metrics['Accuracy']*100:.2f}%)")
    print(f"  F1 Score : {metrics['F1']:.4f}")
    if not np.isnan(metrics['ROC_AUC']):
        print(f"  ROC AUC  : {metrics['ROC_AUC']:.4f}")
    print(sep)

# ─────────────────────────────────────────────
#  SIGNAL VISUALISATION
# ─────────────────────────────────────────────

def plot_eeg_signal_comparison(df, electrode='FZ', n_samples=256,
                                subject_a=None, subject_c=None):
    """
    Plot raw EEG signal comparison: alcoholic vs control for a given electrode.
    Returns fig, axes
    """
    apply_dark_theme()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), facecolor=PALETTE['bg'])
    fig.subplots_adjust(hspace=0.45)

    elec_data = df[df['sensor position'].str.upper() == electrode.upper()]

    for idx, (group, color, label) in enumerate([
        ('a', ALCOHOLIC_COLOR, 'Alcoholic'),
        ('c', CONTROL_COLOR,   'Control'),
    ]):
        grp_data = elec_data[elec_data['subject identifier'] == group]

        # Pick representative subject
        if group == 'a' and subject_a:
            subj_data = grp_data[grp_data['name'] == subject_a]
        elif group == 'c' and subject_c:
            subj_data = grp_data[grp_data['name'] == subject_c]
        else:
            subjects = grp_data['name'].unique()
            if len(subjects) == 0:
                continue
            subj_data = grp_data[grp_data['name'] == subjects[0]]

        vals = subj_data['sensor value'].values[:n_samples]
        t = np.arange(len(vals)) / FS

        ax = axes[idx]
        add_background_shapes(ax, n_circles=4, alpha=0.03)
        ax.plot(t, vals, color=color, linewidth=1.2, alpha=0.9, zorder=3)
        ax.fill_between(t, vals, alpha=0.15, color=color, zorder=2)
        ax.set_title(f'{label} Group — Electrode {electrode.upper()}',
                     color=color, fontweight='bold', fontsize=12)
        ax.set_ylabel('Amplitude (µV)', color=PALETTE['white'])
        add_purple_spine(ax)

    # Difference / overlay
    ax = axes[2]
    add_background_shapes(ax, n_circles=4, alpha=0.03)
    a_vals = elec_data[elec_data['subject identifier'] == 'a']['sensor value'].values[:n_samples]
    c_vals = elec_data[elec_data['subject identifier'] == 'c']['sensor value'].values[:n_samples]
    min_len = min(len(a_vals), len(c_vals))
    if min_len > 0:
        t = np.arange(min_len) / FS
        ax.plot(t, a_vals[:min_len], color=ALCOHOLIC_COLOR, linewidth=1.0,
                alpha=0.8, label='Alcoholic', zorder=3)
        ax.plot(t, c_vals[:min_len], color=CONTROL_COLOR,   linewidth=1.0,
                alpha=0.8, label='Control',   zorder=3)
        ax.fill_between(t, a_vals[:min_len], c_vals[:min_len],
                        alpha=0.12, color=PALETTE['purple3'], zorder=2,
                        label='Difference region')
    ax.set_title(f'Overlay Comparison — Electrode {electrode.upper()}',
                 color=PALETTE['purple3'], fontweight='bold', fontsize=12)
    ax.set_xlabel('Time (seconds)', color=PALETTE['white'])
    ax.set_ylabel('Amplitude (µV)', color=PALETTE['white'])
    ax.legend(loc='upper right', fontsize=9)
    add_purple_spine(ax)

    styled_title(fig,
                 f'EEG Signal Analysis — Electrode {electrode.upper()}',
                 'Alcoholic vs Control Group Comparison')
    return fig, axes

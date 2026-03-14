import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_welch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import os
import sys
import shutil
import zipfile
from pathlib import Path
import requests

# CONFIG
RANDOM_STATE = 42
LEAPD_BANDS = {
    "delta_leapd": (0.5, 4),
    "theta_leapd": (5, 8),
    "l_alpha_leapd": (9, 12)
}
#Inclusion of a "sham band" for specificity control
SHAM_BANDS = {
  "delta":(0.5, 4),
  "theta":(5, 8),
  "alpha":(9, 12),
  "beta":(13, 30), # Additoinal band for checking
}
SELECTED_CHANNELS = ['P8', 'PO7', 'CP1', 'CP2', 'P6', 'O2', 'P4', 'F4']

# 0. ENVIRONMENT SETUP

def setup_environment():
    """Set up project paths and download EEG dataset if needed."""

    IN_COLAB = "google.colab" in sys.modules

    if IN_COLAB:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=True)

        SHARED_FOLDER = Path("/content/drive/MyDrive/ML_RS_EEG")
        PROJECT_ROOT = Path("/content")

        # Copy required files to runtime
        for file in ["utils.py", "participants_clin_cog.csv"]:
            src = SHARED_FOLDER / file
            dst = PROJECT_ROOT / file
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)

    else:
        PROJECT_ROOT = Path.cwd()

    sys.path.append(str(PROJECT_ROOT))

    # Data paths
    DATA_ROOT = PROJECT_ROOT / "data" / "ds004584"
    DATA_ZIP = PROJECT_ROOT / "ds004584.zip"

    DATA_URL = (
        "https://nemar.org/dataexplorer/download?"
        "filepath=/data/nemar/openneuro//zip_files/ds004584.zip"
    )

    (PROJECT_ROOT / "data").mkdir(exist_ok=True)

    if not DATA_ROOT.exists():

        if not DATA_ZIP.exists():
            print("Downloading EEG dataset...")

            r = requests.get(DATA_URL, stream=True)
            with open(DATA_ZIP, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("Extracting EEG dataset...")

        with zipfile.ZipFile(DATA_ZIP, "r") as z:
            z.extractall(PROJECT_ROOT / "data")

    csv_path = PROJECT_ROOT / "participants_clin_cog.csv"

    return PROJECT_ROOT, DATA_ROOT, csv_path

# 1. DATASET DISCOVERY & LABEL ENGINEERING
# PARTICIPANTS

def sync_eeg_and_metadata(participants_df, data_root):
    """
    Only subjects with cognitive scores and .set files are kept.
    """
    # Drop rows missing any of the 5 key cognitive domains
    cognitive_cols = ['NIH FICAT', 'NIH DCCST', 'NIH PCPST', 'NIH PSMT', 'NIH PVT']
    df = participants_df.dropna(subset=cognitive_cols).copy()

    # Find all available .set files on Drive
    all_set_files = sorted(data_root.glob("sub-*/eeg/*.set"))

    file_map = {f.parts[f.parts.index("eeg") - 1]: f for f in all_set_files}

    # Find the intersection
    valid_ids = [sid for sid in df['participant_id'] if sid in file_map]

    # Filter the dataframe and generate the final file list
    df_filtered = df[df['participant_id'].isin(valid_ids)].sort_values('participant_id').reset_index(drop=True)
    final_files = [file_map[sid] for sid in df_filtered['participant_id']]

    # Sanity Check for alignment
    if len(df_filtered) != len(final_files):
        raise ValueError(f"Sync failed: {len(df_filtered)} subjects vs {len(final_files)} files")

    print(f"Sync Complete: {len(df_filtered)} subjects with clinical scores and EEG files.")
    return df_filtered, final_files, valid_ids


def get_processed_participants(df):
    """Calculates Executive Composite and Z-score Tiers using controls as baseline."""
    core_tests = ['NIH FICAT', 'NIH DCCST', 'NIH PCPST']
    controls = df[df['GROUP'] == 'Control']

    # Baseline from Healthy Controls only
    ctrl_means = controls[core_tests].mean()
    ctrl_stds  = controls[core_tests].std()

    # Z-scores for everyone based on HC norms
    z_scores = (df[core_tests] - ctrl_means) / ctrl_stds
    df['Executive_Composite'] = z_scores.mean(axis=1)

    def normative_map(z):
        if z < -1.0: return 2 # Impaired
        if z < 0.0:  return 1 # Standard
        return 0              # Preserved

    df['cog_tier'] = df['Executive_Composite'].apply(normative_map)
    df['cog_binary'] = (df['cog_tier'] == 2).astype(int)
    return df


#  CHANNEL DISCOVERY UTILITIES

def get_common_channels(set_files):
    """
    Finds the intersection of all available channels across all subjects.
    """
    channel_sets = []
    for file in set_files:
        raw = mne.io.read_raw_eeglab(file, preload=False, verbose=False)
        channel_sets.append(set(raw.ch_names))

    intersected = sorted(set.intersection(*channel_sets))
    return intersected


def validate_selected_channels(set_files, target_channels=SELECTED_CHANNELS):
    """
    Raises an error if any subject is missing one of the selected channels.
    """
    common = get_common_channels(set_files)
    missing = [ch for ch in target_channels if ch not in common]

    if missing:
        raise ValueError(f"CRITICAL ERROR: The following channels are missing from the dataset: {missing}")

    print(f"Channel Validation Passed: All {len(target_channels)} channels present across subjects.")
    return True


def get_feature_names():
    """
    Names for the low_freq feature set
    """
    bands = ["Delta", "Theta", "L-Alpha"]
    profile_cols = []

    # Channel-specific band powers (24 features)
    for ch in SELECTED_CHANNELS:
        for b in bands:
            profile_cols.append(f"{ch}_{b}")

    # Channel-specific TAR (8 features)
    for ch in SELECTED_CHANNELS:
        profile_cols.append(f"{ch}_TAR")

    # Global Summary (1 feature)
    profile_cols.append("Global_TAR")
    
    return profile_cols
    


# 2. CORE PROCESSING

"""Multiple feature representations are constructed to probe EEG structure at different levels of abstraction. This includes high-dimensional spectral representations as well as biologically motivated low-frequency summaries. Importantly, features are constructed **per channel × band**, preserving spatial information and enabling interpretability of latent axes.

If we focus on known PD-sensitive frequencies, does diagnostic separation improve?
Prior work identified the 0.5–12 Hz range as maximally discriminative between PD and control subjects, spanning delta, theta, and low-alpha bands known to be abnormal in PD.

LEAPD bandpower features were included as a hypothesis-driven, disease-optimized representation, complementary to generic spectral features. This allows us to assess whether unsupervised structure aligns more strongly with PD-specific spectral abnormalities or emerges independently of disease-informed feature design.
"""


def compute_psd(raw, fmin=0.5, fmax=40.0):
  data = raw.get_data()  # shape (n_channels, n_times)
  sfreq = raw.info["sfreq"] # Sampling freq (Hz) is required to map samples to physical frequencies

  # PSD using Welch’s method - avg periodograms over overlapping windows, reducing variance and making it well-suited for resting-state EEG
  psd, freqs = psd_array_welch(data, sfreq=sfreq, fmin=fmin, fmax=fmax,
                                n_fft=2048, average="mean", verbose=False)
  return psd, freqs

def get_band_features(psd, freqs, channels, bands):
  values, names = [], []
  for ch_idx, ch_name in enumerate(channels):
    for b_name, (fmin, fmax) in bands.items():
      idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
      val = psd[ch_idx, idx].mean()
      values.append(val)
      names.append(f"{ch_name}_{b_name}")
  return np.array(values), names

def construct_master_features(set_files, channel_list):
  # (one per subject)
  X_full_list = []
  X_low_list = []
  X_sham_list = []

  print(f"Starting processing for {len(set_files)} files...")

  for file in tqdm(set_files):
      try:
        raw = mne.io.read_raw_eeglab(file, preload=True, verbose=False)
        raw.pick(channel_list)

        # PSD once (0.5 to 40 Hz)
        psd, freqs = compute_psd(raw, fmin=0.5, fmax=40.0)

        # FULL PSD
        X_full_list.append(np.log10(psd.flatten() + 1e-12))

        # LOW FREQ (0.5-12Hz) + RATIOS
        # Band powers per channel
        bp_leapd, _ = get_band_features(psd, freqs, channel_list, LEAPD_BANDS)

        # Reshape to (Channels, Bands)
        bp_reshaped = bp_leapd.reshape(len(channel_list), -1)
        theta = bp_reshaped[:, 1] # Index 1 is Theta (5-8Hz)
        alpha = bp_reshaped[:, 2] # Index 2 is Alpha (9-12Hz)

        # TAR per channel
        tar_vals = theta / (alpha + 1e-12)
        # Global TAR (One value per subject)
        global_tar = np.mean(theta) / (np.mean(alpha) + 1e-12)

        # [24 bandpowers, 8 TARs, 1 Global TAR]
        low_freq_combined = np.hstack([bp_leapd, tar_vals, [global_tar]])
        X_low_list.append(low_freq_combined)

        # SHAM BAND
        bp_sham, _ = get_band_features(psd, freqs, channel_list, SHAM_BANDS)
        X_sham_list.append(bp_sham)

      except Exception as e:
        print(f"Error processing {file.name}: {e}")

  # Lists to final 2D NumPy arrays (Rows = Subjects, Cols = Features)
  return {
    "full_psd": np.array(X_full_list),
    "low_freq": np.array(X_low_list),
    "sham_band": np.array(X_sham_list)
  }


# 3. CLEANING & SCALING

def clean_outliers(features_dict, initial_subs, participants_df, random_state=RANDOM_STATE):

    # Isolation Forest to find subjects whose EEG 'shape' is weird
    # It returns -1 for outliers, 1 dlc
    iso = IsolationForest(contamination=0.05, random_state=random_state)
    outlier_preds = iso.fit_predict(features_dict["low_freq"])
    
    # Identify who they are
    outlier_indices = np.where(outlier_preds == -1)[0]
    outlier_ids = [initial_subs[i] for i in outlier_indices]

    keep_indices = [i for i, sub_id in enumerate(initial_subs) if sub_id not in outlier_ids]
    
    # Update  participants DataFrame
    # This ensures our metadata stays in sync
    participants_df = participants_df[participants_df['participant_id'].isin([initial_subs[i] for i in keep_indices])].reset_index(drop=True)
    
    # Update all feature matrices in dict
    # We apply the keep_indices filter to every matrix (low_freq, sham, full)
    for key in features_dict:
        features_dict[key] = features_dict[key][keep_indices]
    
    # Update master ID list to reflect the new set
    subs_list = [initial_subs[i] for i in keep_indices]

    return subs_list, outlier_ids, participants_df
    
def prepare_staged_data(features_dict, participants_df, subs_list):
    """
    Synchronizes feature matrices with clinical labels and applies scaling.
    Ensures that labels are pulled in the exact order of the subs_list.
    """

    # Align Metadata
    # We set the index to participant_id to safely pull labels by ID order
    participants_indexed = participants_df.set_index('participant_id')
    
    # Extract labels in the specific order of our 'subs_list'
    y_binary = participants_indexed.loc[subs_list, 'cog_binary'].values
    y_tiered = participants_indexed.loc[subs_list, 'cog_tier'].values
    y_diag = participants_indexed.loc[subs_list, 'GROUP'].values
    
    # Scale Features
    # We create a new dictionary to avoid modifying the raw FEATURES global
    features_scaled = {}
    scaler = StandardScaler()

    for key, matrix in features_dict.items():
        # Fit and transform ONLY on the cleaned subjects present in the dict
        features_scaled[key] = scaler.fit_transform(matrix)
        
    return features_scaled, y_binary, y_tiered, y_diag
import pandas as pd
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier

# --------------- STEP 1: File Read ---------------
file_name = "0_0.csv"   # Tumhari EEG file

# Try to read with comma separator
df = pd.read_csv(file_name, sep=",")

# Check columns
print("Columns in file:", df.columns)
print(df.head())

# Channels (ignore timestamp & AUX)
channels = ["TP9", "AF7", "AF8", "TP10"]

fs = 256  # Sampling frequency (assumption)

# --------------- STEP 2: Bandpower Function ---------------
def bandpower(data, fs, band):
    f, Pxx = welch(data, fs=fs, nperseg=fs)
    freq_res = f[1] - f[0]
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx_band], dx=freq_res)

# Frequency bands
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40)
}

# --------------- STEP 3: Feature Extraction ---------------
features = []
for ch in channels:
    if ch not in df.columns:
        print(f"⚠️ Column '{ch}' not found in file!")
        continue
    ch_data = df[ch].values
    ch_features = []
    for b in bands:
        bp = bandpower(ch_data, fs, bands[b])
        ch_features.append(bp)
    features.extend(ch_features)

print("Extracted Features (bandpowers):")
for i, b in enumerate(bands.keys()):
    print(f"{b}: {features[i::len(bands)]}")

# --------------- STEP 4: Classifier (example) ---------------
# Normally yahan multiple trials honge, har trial ke labels ke sath
X = [features]  # ek trial ke features
y = [0]         # dummy label (jaise left-hand = 0)

clf = RandomForestClassifier()
clf.fit(X, y)
print("✅ Classifier trained successfully")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa 
import librosa.display

import IPython.display as ipd


# amplitude envelope (ae)
# this feature is the max value from the sample 

DATASET_DIR = "Data/genres_original"   # GTZAN folder: genres_original/<genre>/<genre>.000xx.wav
SR = 22050
SEG_SEC = 5
N_FFT = 2048
HOP = 512
N_MFCC = 20

def frame_features(y, sr):
    """Frame-level features -> dict of arrays shaped (n_features, n_frames)."""
    kw = dict(n_fft=N_FFT, hop_length=HOP)
    feats = {
        "chroma":    librosa.feature.chroma_stft(y=y, sr=sr, **kw),              # 12
        "rms":       librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP),
        "centroid":  librosa.feature.spectral_centroid(y=y, sr=sr, **kw),
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr, **kw),
        "rolloff":   librosa.feature.spectral_rolloff(y=y, sr=sr, **kw),
        "zcr":       librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP),
        "contrast":  librosa.feature.spectral_contrast(y=y, sr=sr, **kw),        # 7
        "mfcc":      librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, **kw),
    }
    return feats

def summarize(feats):
    """Collapse frames -> mean & variance per coefficient."""
    row = {}
    for name, mat in feats.items():
        for i in range(mat.shape[0]):
            row[f"{name}{i+1}_mean"] = mat[i].mean()
            row[f"{name}{i+1}_var"]  = mat[i].var()
    return row

rows = []
seg_len = SEG_SEC * SR

for genre in sorted(os.listdir(DATASET_DIR)):
    gdir = os.path.join(DATASET_DIR, genre)
    if not os.path.isdir(gdir):
        continue
    for fname in sorted(os.listdir(gdir)):
        if not fname.endswith(".wav"):
            continue
        path = os.path.join(gdir, fname)
        try:
            y, sr = librosa.load(path, sr=SR, mono=True)
        except Exception as e:
            print(f"Skipping {path}: {e}")   # jazz.00054.wav is known to be corrupted
            continue

        n_segs = len(y) // seg_len            # drops any leftover tail < 5 s
        for s in range(n_segs):
            seg = y[s * seg_len:(s + 1) * seg_len]
            row = {"file": fname, "genre": genre, "segment": s,
                   "start_sec": s * SEG_SEC}
            row.update(summarize(frame_features(seg, sr)))
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("gtzan_5s_features.csv", index=False)
print(df.shape)   # expect about (5994, ...) because of the corrupted file
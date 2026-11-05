# make_features.py
import numpy as np
import pandas as pd
import librosa

SR, SEG_SEC, N_FFT, HOP, N_MFCC = 22050, 5, 2048, 512, 20

def band_energy_ratio(y, sr, split_hz=2000, n_fft=N_FFT, hop=HOP, eps=1e-10):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    split_bin = np.searchsorted(freqs, split_hz)
    low, high = S[:split_bin].sum(axis=0), S[split_bin:].sum(axis=0)
    return np.log10((low + eps) / (high + eps))[np.newaxis, :]

def frame_features(y, sr):
    kw = dict(n_fft=N_FFT, hop_length=HOP)
    return {
        "chroma": librosa.feature.chroma_stft(y=y, sr=sr, **kw),
        "rms": librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP),
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr, **kw),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr, **kw),
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr, **kw),
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP),
        "contrast": librosa.feature.spectral_contrast(y=y, sr=sr, **kw),
        "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, **kw),
        "ber": band_energy_ratio(y, sr),
    }

def summarize(feats):
    row = {}
    for name, mat in feats.items():
        for i in range(mat.shape[0]):
            row[f"{name}{i+1}_mean"] = mat[i].mean()
            row[f"{name}{i+1}_var"] = mat[i].var()
    return row

def song_to_features(path, max_sec=30):
    """Load a song, take the middle `max_sec` seconds, cut into 5 s segments,
    return a DataFrame with one row of features per segment."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    max_len = max_sec * SR
    if len(y) > max_len:                       # long song: use the middle 30 s
        start = (len(y) - max_len) // 2
        y = y[start:start + max_len]

    seg_len = SEG_SEC * SR
    n_segs = len(y) // seg_len
    if n_segs == 0:
        raise ValueError(f"Audio shorter than {SEG_SEC} s")

    rows = [summarize(frame_features(y[s*seg_len:(s+1)*seg_len], sr))
            for s in range(n_segs)]
    return pd.DataFrame(rows)
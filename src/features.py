
# src/features.py
import librosa
import numpy as np

def extract_features(path, n_mfcc=40):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    feat = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
    return feat

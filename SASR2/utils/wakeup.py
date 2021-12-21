import librosa
import numpy as np
from sklearn import preprocessing


def wavfile_to_mfccs(wavfile):
    """
    Returns a matrix of shape (nframes, 39), since there are 39 MFCCs (deltas
    included for each 20ms frame in the wavfile).
    """
    x, sampling_rate = librosa.load(wavfile)
    window_duration_ms = 20
    n_fft = int((window_duration_ms / 1000.) * sampling_rate)

    hop_duration_ms = 10
    hop_length = int((hop_duration_ms / 1000.) * sampling_rate)  # hop_length is 220
    mfcc_count = 13
    mfcc = librosa.feature.mfcc(x, sampling_rate, hop_length=hop_length, n_fft=n_fft, n_mfcc=mfcc_count)
    mfcc_delta = librosa.feature.delta(mfcc, mfcc_count)
    mfcc_delta2 = librosa.feature.delta(mfcc, mfcc_count, order=2)
    mfccs_and_deltas = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
    mfccs_and_deltas = mfccs_and_deltas.transpose()

    return mfccs_and_deltas, hop_length, n_fft


def normalize_mean(X):
    """
    Using scikit learn preprocessing to transform feature matrix
    using StandardScaler with mean and standard deviation
    """
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def apply_normalize_mean(X, scaler):
    """
    Apply normalizaton to a testing dataset that have been fit using training dataset.

    @arguments:
    X: #frames, #features (in case we use mfcc, #features is 39)
    scaler_mean: mean of fitted StandardScaler that you used in normalize_mean function.

    @returns:
    X: normalized matrix
    """
    X = scaler.transform(X)
    return X

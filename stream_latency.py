import librosa
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt

def get_transient(y, sr, height=0.5, prominence=1, width=10000):
    peaks, _ = find_peaks(y, height=height, prominence=prominence, distance=sr*0.5, width=width)
    T_coef = np.arange(y.shape[0])[0]
    peaks_samples = T_coef[peaks] if T_coef.any() else np.argmax(y)
    return peaks_samples / sr

def get_onset_times(y, sr):
    """
    Function to get approximate time(s) of onsets (transients). Adapted from librosa
    official API documentation [1].

    Parameters
    ----------
    y : np.array
        Audio time series.

    sr : int > 0
        Sampling rate of y.

    Returns
    -------
    np.array
        An array of time estimations of detected onsets.

    [1] https://librosa.org/doc/latest/generated/librosa.onset.onset_detect.html#librosa.onset.onset_detect
    """

    o_env = librosa.onset.onset_strength(y, sr=sr)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr,
            pre_max=1, post_max=1, pre_avg=50, post_avg=50, delta=0.75, wait=10)
    if len(onset_frames) > 0:
        return times[onset_frames]
    else:
        return []


def get_mean_latency(fname1, fname2, amount=3, duration=None, offset=None):
    """
    Function to determine the approximate latency between two audio files. It is assumed
    that both files are recorded concurrently.

    Parameters
    ----------
    fname1: str
        Name of the first audio file.

    fname2: str
        Name of the second audio file

    Returns
    -------
    latency: np.float64
        The mean latency between both audio files, in seconds.
    """

    y1, sr1 = librosa.load(fname1, sr=None, duration=duration, offset=offset)
    y2, sr2 = librosa.load(fname2, sr=None, duration=duration, offset=offset)
    y1_norm = librosa.util.normalize(y1)
    y2_norm = librosa.util.normalize(y2)
    onset_times1 = get_onset_times(y1_norm, sr1)
    onset_times2 = get_onset_times(y2_norm, sr2)
    latency = np.mean(np.diff([onset_times1[:amount], onset_times2[:amount]], axis=0))
    return latency

import numpy as np
import math
from librosa import util

def clean_speech_VAD(speech_t,
                     fs=16e3,
                     wlen_sec=50e-3,
                     hop_percent=0.25,
                     center=True,
                     pad_mode='reflect',
                     pad_at_end=True,
                     vad_threshold=1.70):
    """ Computes VAD based on threshold in the time domain

    Args:
        speech_t ([type]): [description]
        fs ([type]): [description]
        wlen_sec ([type]): [description]
        hop_percent ([type]): [description]
        center ([type]): [description]
        pad_mode ([type]): [description]
        pad_at_end ([type]): [description]
        eps ([type], optional): [description]. Defaults to 1e-8.

    Returns:
        ndarray: [description]
    """
    nfft = int(wlen_sec * fs) # STFT window length in samples
    hopsamp = int(hop_percent * nfft) # hop size in samples
    # Sometimes stft / istft shortens the ouput due to window size
    
    # so you need to pad the end with hopsamp zeros
    if pad_at_end:
        utt_len = len(speech_t) / fs
        if math.ceil(utt_len / wlen_sec / hop_percent) != int(utt_len / wlen_sec / hop_percent):
            y = np.pad(speech_t, (0,hopsamp), mode='constant')
        else:
            y = speech_t.copy()
    else:
        y = speech_t.copy()

    if center:
        y = np.pad(y, int(nfft // 2), mode=pad_mode)

    y_frames = util.frame(y, frame_length=nfft, hop_length=hopsamp)
    
    # power = (10 * np.log10(np.power(y_frames,2).sum(axis=0)))
    # vad = power > np.min(power) + 11
    # vad = power > np.min(power) - np.min(power)*0.41
    
    power = np.power(y_frames,2).sum(axis=0)
    # vad = power > np.power(10, 1.20) * np.min(power)
    vad = power > np.power(10, vad_threshold) * np.min(power)
    vad = np.float32(vad) # convert to float32
    vad = vad[None]
    return vad

def clean_speech_IBM(speech_tf,
                     eps=1e-8,
                     ibm_threshold=50):
    """ Calculate softened mask
    """
    # power = abs(observations * observations.conj())
    # power_db = 10 * np.log10(power + eps) # Smoother mask with log
    mag = abs(speech_tf)
    power_db = 20 * np.log10(mag + eps) # Smoother mask with log 
    # mask = power_db > np.max(power_db) - 65
    mask = power_db > np.max(power_db) - ibm_threshold
    mask = np.float32(mask) # convert to float32
    return mask

def noise_robust_clean_speech_IBM(speech_t,
                                  speech_tf,
                                  fs=16e3,
                                  wlen_sec=50e-3,
                                  hop_percent=0.25,
                                  center=True,
                                  pad_mode='reflect',
                                  pad_at_end=True,
                                  vad_threshold=1.70,
                                  eps=1e-8,
                                  ibm_threshold=50):
    """
    Create IBM labels robust to noisy speech recordings using noise-robst VAD.
    In particular, the labels are robust to noise occuring before / after speech.
    """
    # Compute vad
    vad = clean_speech_VAD(speech_t,
                           fs=fs,
                           wlen_sec=wlen_sec,
                           hop_percent=hop_percent,
                           center=center,
                           pad_mode=pad_mode,
                           pad_at_end=pad_at_end,
                           vad_threshold=vad_threshold)

    # binary mask
    ibm = clean_speech_IBM(speech_tf,
                           eps=eps,
                           ibm_threshold=ibm_threshold)
    
    # Noise-robust binary mask
    ibm = ibm * vad
    return ibm

########################
#### Threshold-based IBM
########################

def _voiced_unvoiced_split_characteristic(number_of_frequency_bins):
    split_bin = 200
    transition_width = 99
    fast_transition_width = 5
    low_bin = 4
    high_bin = 500

    a = np.arange(0, transition_width)
    a = np.pi / (transition_width - 1) * a
    transition = 0.5 * (1 + np.cos(a))

    b = np.arange(0, fast_transition_width)
    b = np.pi / (fast_transition_width - 1) * b
    fast_transition = (np.cos(b) + 1) / 2

    transition_voiced_start = int(split_bin - transition_width / 2)
    voiced = np.ones(number_of_frequency_bins)

    # High Edge
    voiced[transition_voiced_start - 1: (
        transition_voiced_start + transition_width - 1)] = transition
    voiced[transition_voiced_start - 1 + transition_width: len(voiced)] = 0

    # Low Edge
    voiced[0: low_bin] = 0
    voiced[low_bin - 1: (low_bin + fast_transition_width - 1)] = \
        1 - fast_transition

    # Low Edge
    unvoiced = np.ones(number_of_frequency_bins)
    unvoiced[transition_voiced_start - 1: (
        transition_voiced_start + transition_width - 1)] = 1 - transition
    unvoiced[0: (transition_voiced_start)] = 0

    # High Edge
    unvoiced[high_bin - 1: (len(unvoiced))] = 0
    unvoiced[
    high_bin - 1: (high_bin + fast_transition_width - 1)] = fast_transition

    return (voiced, unvoiced)

def noise_aware_IBM(X, N,
                 threshold_unvoiced_speech=5,
                 threshold_voiced_speech=0,
                 threshold_unvoiced_noise=-10,
                 threshold_voiced_noise=-10,
                 low_cut=5,
                 high_cut=500):
    """Estimate an ideal binary mask given the speech and noise spectrum.
    :param X: speech signal in STFT domain with shape (frames, frequency-bins)
    :param N: noise signal in STFT domain with shape (frames, frequency-bins)
    :param threshold_unvoiced_speech:
    :param threshold_voiced_speech:
    :param threshold_unvoiced_noise:
    :param threshold_voiced_noise:
    :param low_cut: all values with frequency<low_cut are set to 0 in the
        speech mask ans set to 1 in the noise mask
    :param high_cut: all values with frequency>high_cut are set to 0 in the
        speech mask ans set to 1 in the noise mask
    :return: (speech mask, noise mask): tuple containing the two arrays,
        which are the masks for X and N
    """
    (voiced, unvoiced) = _voiced_unvoiced_split_characteristic(X.shape[-1])

    # calculate the thresholds
    threshold = threshold_voiced_speech * voiced + \
                threshold_unvoiced_speech * unvoiced
    threshold_new = threshold_unvoiced_noise * voiced + \
                    threshold_voiced_noise * unvoiced

    xPSD = X * X.conjugate()  # |X|^2 = Power-Spectral-Density

    # each frequency is multiplied with another threshold
    c = np.power(10, (threshold / 10))
    xPSD_threshold = xPSD / c
    c_new = np.power(10, (threshold_new / 10))
    xPSD_threshold_new = xPSD / c_new

    nPSD = N * N.conjugate()

    speechMask = (xPSD_threshold > nPSD)

    speechMask = np.logical_and(speechMask, (xPSD_threshold > 0.005))
    speechMask[..., 0:low_cut - 1] = 0
    speechMask[..., high_cut:len(speechMask[0])] = 0

    noiseMask = (xPSD_threshold_new < nPSD)

    noiseMask = np.logical_or(noiseMask, (xPSD_threshold_new < 0.005))
    noiseMask[..., 0: low_cut - 1] = 1
    noiseMask[..., high_cut: len(noiseMask[0])] = 1

    return (speechMask, noiseMask)


def threshold_IBM(X,
                 threshold_unvoiced_speech=5,
                 threshold_voiced_speech=0,
                 threshold_unvoiced_noise=-10,
                 threshold_voiced_noise=-10,
                 low_cut=5,
                 high_cut=500):
    """Estimate an ideal binary mask given the speech and noise spectrum.
    :param X: speech signal in STFT domain with shape (frames, frequency-bins)
    :param N: noise signal in STFT domain with shape (frames, frequency-bins)
    :param threshold_unvoiced_speech:
    :param threshold_voiced_speech:
    :param threshold_unvoiced_noise:
    :param threshold_voiced_noise:
    :param low_cut: all values with frequency<low_cut are set to 0 in the
        speech mask ans set to 1 in the noise mask
    :param high_cut: all values with frequency>high_cut are set to 0 in the
        speech mask ans set to 1 in the noise mask
    :return: (speech mask, noise mask): tuple containing the two arrays,
        which are the masks for X and N
    """
    (voiced, unvoiced) = _voiced_unvoiced_split_characteristic(X.shape[-1])

    # calculate the thresholds
    threshold = threshold_voiced_speech * voiced + \
                threshold_unvoiced_speech * unvoiced
    threshold_new = threshold_unvoiced_noise * voiced + \
                    threshold_voiced_noise * unvoiced

    xPSD = X * X.conjugate()  # |X|^2 = Power-Spectral-Density

    # each frequency is multiplied with another threshold
    c = np.power(10, (threshold / 10))
    xPSD_threshold = xPSD / c
    c_new = np.power(10, (threshold_new / 10))
    xPSD_threshold_new = xPSD / c_new

    # nPSD = N * N.conjugate()
    nPSD = 10

    speechMask = (xPSD_threshold > nPSD)

    speechMask = np.logical_and(speechMask, (xPSD_threshold > 0.005))
    speechMask[..., 0:low_cut - 1] = 0
    speechMask[..., high_cut:len(speechMask[0])] = 0

    return speechMask
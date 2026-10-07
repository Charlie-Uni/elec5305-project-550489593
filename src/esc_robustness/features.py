"""92-dimensional baseline: six descriptors, 20 MFCCs and 20 deltas, mean/std."""

import numpy as np

from .dsp import (
    FeatureConfig,
    frame_audio,
    log_mel_from_stft,
    mfcc_from_log_mel,
    periodic_hann,
    temporal_delta,
)


def descriptor_names(config=FeatureConfig()):
    return (
        ["rms", "zcr", "spectral_centroid", "spectral_spread", "spectral_rolloff", "spectral_flux"]
        + [f"mfcc_{i:02d}" for i in range(config.n_mfcc)]
        + [f"delta_mfcc_{i:02d}" for i in range(config.n_mfcc)]
    )


def feature_names(config=FeatureConfig()):
    return [f"{name}_{stat}" for stat in ("mean", "std") for name in descriptor_names(config)]


def frame_features(audio, config=FeatureConfig()):
    frames = frame_audio(audio, config)
    spectrum = np.fft.rfft(frames * periodic_hann(config.frame_length), n=config.n_fft, axis=1)
    magnitude = np.abs(spectrum)
    power = magnitude**2 / config.n_fft
    frequencies = np.fft.rfftfreq(config.n_fft, 1 / config.sample_rate)
    total = power.sum(axis=1)
    weights = np.divide(power, total[:, None], out=np.zeros_like(power), where=total[:, None] > 0)
    centroid = weights @ frequencies
    spread = np.sqrt(np.sum(weights * (frequencies - centroid[:, None]) ** 2, axis=1))
    rolloff_bins = np.argmax(
        np.cumsum(power, axis=1) >= config.rolloff_fraction * total[:, None], axis=1
    )
    rolloff = frequencies[rolloff_bins]
    norm = magnitude.sum(axis=1, keepdims=True)
    normalised = np.divide(magnitude, norm, out=np.zeros_like(magnitude), where=norm > 0)
    flux = np.zeros(len(frames))
    flux[1:] = np.linalg.norm(np.diff(normalised, axis=0), axis=1)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    # Exact zeros are treated as nonnegative; crossings are adjacent sign changes.
    zcr = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)
    mfcc = mfcc_from_log_mel(log_mel_from_stft(spectrum, config), config)
    return np.column_stack(
        (rms, zcr, centroid, spread, rolloff, flux, mfcc, temporal_delta(mfcc, config.delta_width))
    )


def extract_features(audio, config=FeatureConfig()):
    values = frame_features(audio, config)
    pooled = np.concatenate((values.mean(axis=0), values.std(axis=0, ddof=0)))
    if not np.isfinite(pooled).all():
        raise ValueError("Nonfinite features")
    return pooled

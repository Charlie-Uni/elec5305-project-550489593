"""Explicit DSP conventions shared by the handcrafted and future CNN frontends.

STFT is an unnormalised, one-sided FFT, shaped (frames, frequency bins).
Frames start at sample zero. Only the last incomplete frame is zero padded.
No pre-emphasis, centre padding, amplitude normalisation or per-clip scaling is used.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.fft import dct


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 44100
    frame_length: int = 1024
    hop_length: int = 512
    n_fft: int = 1024
    n_mels: int = 64
    n_mfcc: int = 20
    rolloff_fraction: float = 0.85
    delta_width: int = 2
    log_floor: float = 1e-12

    def __post_init__(self):
        for key in (
            "sample_rate",
            "frame_length",
            "hop_length",
            "n_fft",
            "n_mels",
            "n_mfcc",
            "delta_width",
        ):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.frame_length < 2 or self.hop_length > self.frame_length:
            raise ValueError("Require frame_length >= 2 and hop_length <= frame_length")
        if self.n_fft < self.frame_length or self.n_mfcc > self.n_mels:
            raise ValueError("Require n_fft >= frame_length and n_mfcc <= n_mels")
        if not 0 < self.rolloff_fraction <= 1 or not 0 < self.log_floor < 1:
            raise ValueError("Invalid rolloff_fraction or log_floor")

    def as_dict(self):
        return asdict(self)


def validate_waveform(audio):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must be a finite, nonempty mono waveform")
    return audio


def frame_audio(audio, config=FeatureConfig()):
    """Return independent frames, including a zero-padded final frame if needed."""
    audio = validate_waveform(audio)
    n_frames = 1 + max(
        0, (audio.size - config.frame_length + config.hop_length - 1) // config.hop_length
    )
    length = config.frame_length + (n_frames - 1) * config.hop_length
    padded = np.pad(audio, (0, length - audio.size))
    return np.lib.stride_tricks.sliding_window_view(padded, config.frame_length)[
        :: config.hop_length
    ].copy()


def periodic_hann(length):
    if length < 2:
        raise ValueError("Hann window length must be >= 2")
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(length) / length)


def stft(audio, config=FeatureConfig()):
    return np.fft.rfft(
        frame_audio(audio, config) * periodic_hann(config.frame_length),
        n=config.n_fft,
        axis=1,
    )


@lru_cache(maxsize=16)
def mel_filterbank(config=FeatureConfig()):
    """HTK frequency scale, continuous triangular weights, no area normalisation."""
    nyquist_mel = 2595 * np.log10(1 + (config.sample_rate / 2) / 700)
    edges = 700 * (10 ** (np.linspace(0, nyquist_mel, config.n_mels + 2) / 2595) - 1)
    frequencies = np.fft.rfftfreq(config.n_fft, 1 / config.sample_rate)
    left, centre, right = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    bank = np.maximum(
        0,
        np.minimum(
            (frequencies - left) / (centre - left), (right - frequencies) / (right - centre)
        ),
    )
    if np.any(bank.sum(axis=1) <= 0):
        raise ValueError("Empty Mel filter: increase n_fft or reduce n_mels")
    bank.setflags(write=False)
    return bank


def log_mel_from_stft(spectrum, config=FeatureConfig()):
    power = np.abs(spectrum) ** 2 / config.n_fft
    return np.log(np.maximum(power @ mel_filterbank(config).T, config.log_floor))


def log_mel(audio, config=FeatureConfig()):
    """Natural-log Mel power, shape (frames, Mel bands)."""
    return log_mel_from_stft(stft(audio, config), config)


def mfcc_from_log_mel(log_mel_frames, config=FeatureConfig()):
    return dct(log_mel_frames, type=2, norm="ortho", axis=1)[:, : config.n_mfcc]


def temporal_delta(values, width=2):
    """Regression delta per frame, using edge replication and +/- width frames."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or width < 1:
        raise ValueError("Expected nonempty frame-by-feature matrix and positive width")
    padded = np.pad(values, ((width, width), (0, 0)), mode="edge")
    result = np.zeros_like(values)
    for offset in range(1, width + 1):
        result += offset * (
            padded[width + offset : width + offset + len(values)]
            - padded[width - offset : width - offset + len(values)]
        )
    return result / (2 * sum(i * i for i in range(1, width + 1)))

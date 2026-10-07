"""Numerical comparisons of the custom STFT against SciPy's independent routine."""

import numpy as np
from scipy import signal

from .dsp import FeatureConfig, frame_audio, stft


def validate_stft(config=FeatureConfig()):
    rng = np.random.default_rng(5305)
    cases = {
        "short_impulse": np.r_[1.0, np.zeros(30)],
        "silence": np.zeros(config.frame_length * 2),
        "random_non_hop_aligned": rng.normal(size=10007),
        "bin_centred_sine": np.sin(2 * np.pi * 17 * np.arange(8192) / config.n_fft),
    }
    results = []
    for name, audio in cases.items():
        n_frames = len(frame_audio(audio, config))
        padded_length = config.frame_length + (n_frames - 1) * config.hop_length
        padded = np.pad(audio, (0, padded_length - len(audio)))
        _, _, reference = signal.stft(
            padded,
            fs=config.sample_rate,
            window="hann",
            nperseg=config.frame_length,
            noverlap=config.frame_length - config.hop_length,
            nfft=config.n_fft,
            detrend=False,
            return_onesided=True,
            boundary=None,
            padded=False,
            scaling="spectrum",
        )
        # SciPy's spectrum scaling divides the FFT by sum(window).
        reference = reference.T * signal.get_window("hann", config.frame_length, fftbins=True).sum()
        actual = stft(audio, config)
        np.testing.assert_allclose(actual, reference, rtol=1e-11, atol=1e-10)
        results.append(
            {
                "case": name,
                "frames": n_frames,
                "max_absolute_error": float(np.max(np.abs(actual - reference))),
            }
        )
    return {
        "reference": "scipy.signal.stft",
        "passed": True,
        "cases": results,
        "feature_config": config.as_dict(),
        "course_MyStft_m": "not_checked_source_not_available",
    }

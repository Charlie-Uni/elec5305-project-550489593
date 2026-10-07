import numpy as np
import pytest
from scipy import signal

from esc_robustness.dsp import (
    FeatureConfig,
    frame_audio,
    log_mel,
    mel_filterbank,
    mfcc_from_log_mel,
    periodic_hann,
    stft,
    temporal_delta,
)
from esc_robustness.features import extract_features, feature_names, frame_features
from esc_robustness.validation import validate_stft


@pytest.mark.parametrize("length,expected", [(1, 1), (1024, 1), (1025, 2), (1536, 2), (1537, 3)])
def test_framing_preserves_samples_and_zero_pads_only_tail(length, expected):
    x = np.arange(length, dtype=float)
    frames = frame_audio(x)
    assert frames.shape == (expected, 1024)
    for index, frame in enumerate(frames):
        valid = x[index * 512 : index * 512 + 1024]
        np.testing.assert_array_equal(frame[: len(valid)], valid)
        assert np.all(frame[len(valid) :] == 0)


def test_periodic_window_matches_scipy():
    np.testing.assert_allclose(periodic_hann(1024), signal.get_window("hann", 1024), atol=1e-15)


def test_stft_matches_independent_reference():
    report = validate_stft()
    assert report["passed"]
    assert max(row["max_absolute_error"] for row in report["cases"]) < 1e-10


def test_stft_impulse_has_expected_unscaled_fft():
    x = np.zeros(1024)
    x[512] = 1
    spectrum = stft(x)[0]
    np.testing.assert_allclose(spectrum, (-1.0) ** np.arange(513), atol=1e-12)


def test_mfcc_matches_direct_cosine_transform():
    rng = np.random.default_rng(10)
    values = rng.normal(size=(8, 64))
    bins = np.arange(64) + 0.5
    basis = np.sqrt(2 / 64) * np.cos(np.pi * np.arange(20)[:, None] * bins / 64)
    basis[0] /= np.sqrt(2)
    np.testing.assert_allclose(mfcc_from_log_mel(values), values @ basis.T, atol=2e-14)


def test_mel_triangles_are_nonnegative_local_and_cover_spectrum():
    bank = mel_filterbank()
    assert bank.shape == (64, 513)
    assert np.all(bank >= 0)
    assert np.all(bank.max(axis=1) <= 1)
    assert np.all(bank.sum(axis=1) > 0)
    assert np.max((bank > 0).sum(axis=0)) <= 2
    # Adjacent triangles form a partition of unity only between filter centres;
    # outside the first/last centre the edge filters taper toward zero.
    interior = (bank > 0).sum(axis=0) == 2
    np.testing.assert_allclose(bank.sum(axis=0)[interior], 1, atol=1e-14)
    assert bank[:, 0].sum() == 0
    assert bank[:, -1].sum() < 1e-12


def test_delta_has_correct_slope_and_boundary_values():
    values = np.arange(10, dtype=float)[:, None]
    result = temporal_delta(values)[:, 0]
    np.testing.assert_allclose(result, [0.5, 0.8, 1, 1, 1, 1, 1, 1, 0.8, 0.5])
    assert np.all(temporal_delta(np.ones((1, 20))) == 0)


def test_tonal_features_have_physical_meaning():
    # Integer-bin tone and complete frames avoid leakage from padding.
    x = np.sin(2 * np.pi * 32 * np.arange(2048) / 1024)
    values = frame_features(x)
    np.testing.assert_allclose(values[:, 0], 1 / np.sqrt(2), atol=1e-14)
    np.testing.assert_allclose(values[:, 2], 32 * 44100 / 1024, atol=1e-9)
    np.testing.assert_allclose(values[:, 3], 44100 / 1024 / np.sqrt(3), atol=1e-9)
    np.testing.assert_allclose(values[:, 5], 0, atol=1e-13)


def test_silence_features_are_finite_and_spectral_descriptors_zero():
    values = frame_features(np.zeros(2048))
    assert np.all(values[:, :6] == 0)
    assert np.all(values[:, 26:] == 0)
    pooled = extract_features(np.zeros(2048))
    assert pooled.shape == (92,)
    assert np.isfinite(pooled).all()
    np.testing.assert_allclose(pooled[46:], 0, atol=1e-13)
    assert len(feature_names()) == len(set(feature_names())) == 92


def test_pooling_matches_frame_statistics():
    x = np.random.default_rng(2).normal(size=4410)
    frames = frame_features(x)
    pooled = extract_features(x)
    np.testing.assert_allclose(pooled[:46], frames.mean(axis=0))
    np.testing.assert_allclose(pooled[46:], frames.std(axis=0, ddof=0))
    assert log_mel(x).shape == (8, 64)


@pytest.mark.parametrize("bad", [[], [[1, 2]], [np.nan], [np.inf]])
def test_bad_waveforms_are_rejected(bad):
    with pytest.raises(ValueError):
        extract_features(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hop_length": 2048},
        {"n_fft": 512},
        {"n_mfcc": 65},
        {"rolloff_fraction": 0},
        {"delta_width": 0},
    ],
)
def test_bad_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FeatureConfig(**kwargs)

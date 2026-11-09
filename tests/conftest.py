import csv

import numpy as np
import pytest
import soundfile as sf

TOY_RATE = 44100
TOY_LENGTH = 22050  # half-second clips keep the interference tests fast


def make_tone(frequency, length=TOY_LENGTH, amplitude=0.3):
    return amplitude * np.sin(2 * np.pi * frequency * np.arange(length) / TOY_RATE)


@pytest.fixture
def toy_dataset(tmp_path):
    """Short synthetic WAVs for pipeline verification, never reported as ESC-50 results."""
    root = tmp_path / "toy"
    (root / "audio").mkdir(parents=True)
    (root / "meta").mkdir()
    rng = np.random.default_rng(4)
    rows = []
    for fold in range(1, 6):
        for target in range(2):
            for take in range(2):
                source = f"{fold}{target}{take}"
                name = f"{fold}-{source}-A-{target}.wav"
                time = np.arange(4096) / 44100
                x = 0.3 * np.sin(2 * np.pi * (800 + 2400 * target) * time)
                x += rng.normal(0, 0.001, size=len(x))
                sf.write(root / "audio" / name, x, 44100, subtype="FLOAT")
                rows.append([name, fold, target, f"class_{target}", source])
    with (root / "meta" / "esc50.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "fold", "target", "category", "src_file"])
        writer.writerows(rows)
    return root


@pytest.fixture
def interference_dataset(tmp_path):
    """Four classes from two coarse categories; some clips carry digital silence."""
    root = tmp_path / "toy_interference"
    (root / "audio").mkdir(parents=True)
    (root / "meta").mkdir()
    rng = np.random.default_rng(11)
    rows = []
    for fold in range(1, 6):
        for target in (0, 1, 10, 11):
            for take in range(2):
                source = f"{fold}{target:02d}{take}"
                name = f"{fold}-{source}-A-{target}.wav"
                x = make_tone(500 + 300 * target) + rng.normal(0, 1e-3, TOY_LENGTH)
                if target == 10:
                    x[: TOY_LENGTH // 2] = 0.0  # active only in the second half
                if target == 11:
                    # Active only in the first half, ending a few frames before the
                    # midpoint so that it has no energy inside class 10's active frames.
                    x[TOY_LENGTH // 2 - 2048 :] = 0.0
                sf.write(root / "audio" / name, x, TOY_RATE, subtype="FLOAT")
                rows.append([name, fold, target, f"class_{target}", source])
    with (root / "meta" / "esc50.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "fold", "target", "category", "src_file"])
        writer.writerows(rows)
    return root

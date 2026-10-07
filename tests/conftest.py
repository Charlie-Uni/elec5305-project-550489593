import csv

import numpy as np
import pytest
import soundfile as sf


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

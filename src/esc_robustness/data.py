"""Validate official ESC-50 folds and audit source overlap without rewriting them."""

import csv
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .dsp import FeatureConfig, validate_waveform

COARSE_CATEGORIES = (
    "animals",
    "natural_soundscapes_and_water",
    "human_non_speech",
    "interior_domestic",
    "exterior_urban",
)


@dataclass(frozen=True)
class Clip:
    filename: str
    fold: int
    target: int
    category: str
    source_id: str

    @property
    def coarse_category(self):
        # Official class IDs are grouped into five consecutive blocks of ten.
        return COARSE_CATEGORIES[self.target // 10]


def source_overlap_report(clips):
    groups = {}
    for clip in clips:
        groups.setdefault(clip.source_id, []).append(clip)
    return [
        {
            "source_id": source,
            "folds": sorted({c.fold for c in group}),
            "filenames": sorted(c.filename for c in group),
            "class_ids": sorted({c.target for c in group}),
        }
        for source, group in sorted(groups.items())
        if len({c.fold for c in group}) > 1
    ]


def load_metadata(root, require_complete=True, strict_sources=False):
    root = Path(root)
    path = root / "meta" / "esc50.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; provide the ESC-50 root with audio/ and meta/")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "fold", "target", "category", "src_file"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Metadata requires columns {sorted(required)}")
        clips = [
            Clip(
                row["filename"],
                int(row["fold"]),
                int(row["target"]),
                row["category"],
                row["src_file"],
            )
            for row in reader
        ]
    if not clips:
        raise ValueError("Metadata is empty")
    filenames, class_names, category_labels = set(), {}, {}
    for clip in clips:
        if Path(clip.filename).name != clip.filename or not clip.filename.endswith(".wav"):
            raise ValueError(f"Unsafe or invalid WAV filename: {clip.filename}")
        if clip.filename in filenames:
            raise ValueError(f"Duplicate filename: {clip.filename}")
        filenames.add(clip.filename)
        if clip.fold not in range(1, 6) or clip.target not in range(50):
            raise ValueError(f"Invalid fold or class: {clip}")
        if not clip.category or not clip.source_id:
            raise ValueError("Category and source ID must be nonempty")
        if class_names.setdefault(clip.target, clip.category) != clip.category:
            raise ValueError("One target ID maps to multiple class names")
        if category_labels.setdefault(clip.category, clip.target) != clip.target:
            raise ValueError("One class name maps to multiple target IDs")
        if not (root / "audio" / clip.filename).is_file():
            raise FileNotFoundError(root / "audio" / clip.filename)
    if require_complete:
        counts = Counter((clip.fold, clip.target) for clip in clips)
        if len(clips) != 2000 or any(counts[(f, t)] != 8 for f in range(1, 6) for t in range(50)):
            raise ValueError("Expected complete ESC-50: 2000 clips, 8 per class in each of 5 folds")
    overlaps = source_overlap_report(clips)
    if overlaps:
        message = (
            f"Source overlap across official folds: {', '.join(r['source_id'] for r in overlaps)}"
        )
        if strict_sources:
            raise ValueError(message)
        warnings.warn(
            message + "; retaining official fold assignments and recording the audit", stacklevel=2
        )
    return sorted(clips, key=lambda clip: (clip.fold, clip.target, clip.filename))


def load_audio(path, config=FeatureConfig(), require_five_seconds=False):
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    if sample_rate != config.sample_rate:
        raise ValueError(f"{path}: expected {config.sample_rate} Hz, got {sample_rate}")
    audio = validate_waveform(audio)
    if require_five_seconds and len(audio) != 5 * sample_rate:
        raise ValueError(f"{path}: expected exactly five seconds of audio")
    return audio


def outer_split(folds, test_fold):
    folds = np.asarray(folds)
    if test_fold not in range(1, 6) or set(folds.tolist()) != set(range(1, 6)):
        raise ValueError("Outer evaluation requires folds 1 through 5")
    validation_fold = test_fold % 5 + 1
    test = np.flatnonzero(folds == test_fold)
    validation = np.flatnonzero(folds == validation_fold)
    train = np.flatnonzero((folds != test_fold) & (folds != validation_fold))
    return train, validation, test

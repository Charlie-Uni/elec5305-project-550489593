"""Unnormalised log-Mel caches; learned scaling belongs inside each training split."""

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .artifacts import sha256_file
from .data import Clip, load_audio, load_metadata, source_overlap_report
from .dsp import FeatureConfig, log_mel


def extract_spectrograms(root, output, config=FeatureConfig(), require_complete=True):
    root, output = Path(root), Path(output)
    if output.suffix != ".npz":
        raise ValueError("Spectrogram cache filename must end in .npz")
    if output.exists():
        raise FileExistsError(f"Cache already exists: {output}")
    clips = load_metadata(root, require_complete=require_complete)
    values, hashes = [], []
    with threadpool_limits(limits=1):
        for i, clip in enumerate(clips, 1):
            path = root / "audio" / clip.filename
            audio = load_audio(path, config, require_five_seconds=require_complete)
            values.append(log_mel(audio, config).T.astype(np.float32))
            hashes.append(sha256_file(path))
            if i % 100 == 0 or i == len(clips):
                print(f"Log-Mel: {i}/{len(clips)} clips", flush=True)
    if len({v.shape for v in values}) != 1:
        raise ValueError("A spectrogram cache requires equal-length recordings")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.array("log_mel_v1"),
        X=np.stack(values),
        y=np.array([c.target for c in clips]),
        folds=np.array([c.fold for c in clips]),
        filenames=np.array([c.filename for c in clips]),
        categories=np.array([c.category for c in clips]),
        source_ids=np.array([c.source_id for c in clips]),
        audio_sha256=np.array(hashes),
        config_json=np.array(json.dumps(config.as_dict(), sort_keys=True)),
        metadata_sha256=np.array(sha256_file(root / "meta" / "esc50.csv")),
        source_hashes_json=np.array(
            json.dumps(
                {
                    name: sha256_file(Path(__file__).with_name(name))
                    for name in ("dsp.py", "data.py", "spectrograms.py")
                },
                sort_keys=True,
            )
        ),
        source_overlap_json=np.array(json.dumps(source_overlap_report(clips), sort_keys=True)),
        complete_dataset=np.array(require_complete),
    )
    return {
        "clips": len(clips),
        "shape": [len(clips), *values[0].shape],
        "spectrogram_cache": str(output),
        "sha256": sha256_file(output),
    }


def load_spectrogram_cache(path, require_complete=True):
    with np.load(path, allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    required = {
        "schema_version",
        "X",
        "y",
        "folds",
        "filenames",
        "categories",
        "source_ids",
        "audio_sha256",
        "config_json",
        "metadata_sha256",
        "source_hashes_json",
        "source_overlap_json",
        "complete_dataset",
    }
    if not required.issubset(cache) or cache["schema_version"].item() != "log_mel_v1":
        raise ValueError("Missing or unsupported log-Mel cache schema")
    config = FeatureConfig(**json.loads(cache["config_json"].item()))
    X, y, folds = cache["X"], cache["y"], cache["folds"]
    n = len(y)
    if (
        X.ndim != 3
        or X.shape[:2] != (n, config.n_mels)
        or X.shape[2] < 1
        or n == 0
        or not np.isfinite(X).all()
    ):
        raise ValueError("Invalid log-Mel cache shape or values")
    for key in ("y", "folds", "filenames", "categories", "source_ids", "audio_sha256"):
        if cache[key].shape != (n,):
            raise ValueError(f"Invalid cache array: {key}")
    if len(set(cache["filenames"].tolist())) != n:
        raise ValueError("Duplicate cached filenames")
    if (
        not np.issubdtype(y.dtype, np.integer)
        or np.any((y < 0) | (y >= 50))
        or not np.issubdtype(folds.dtype, np.integer)
        or set(folds.tolist()) != set(range(1, 6))
    ):
        raise ValueError("Invalid class IDs or folds")
    classes, names = {}, {}
    for label, category in zip(y, cache["categories"]):
        if not category or classes.setdefault(int(label), str(category)) != category:
            raise ValueError("Inconsistent cached class names")
        if names.setdefault(str(category), int(label)) != label:
            raise ValueError("Class name maps to multiple IDs")
    clips = [
        Clip(str(n), int(f), int(y), str(c), str(s))
        for n, f, y, c, s in zip(
            cache["filenames"], folds, y, cache["categories"], cache["source_ids"]
        )
    ]
    if any(not c.source_id for c in clips):
        raise ValueError("Missing source recording ID")
    if source_overlap_report(clips) != json.loads(cache["source_overlap_json"].item()):
        raise ValueError("Cached source-overlap audit does not match its rows")
    expected_frames = 1 + max(
        0,
        (5 * config.sample_rate - config.frame_length + config.hop_length - 1) // config.hop_length,
    )
    if require_complete and (
        not bool(cache["complete_dataset"].item())
        or n != 2000
        or X.shape[2] != expected_frames
        or any(np.sum((folds == f) & (y == c)) != 8 for f in range(1, 6) for c in range(50))
    ):
        raise ValueError("Official evaluation requires a complete ESC-50 spectrogram cache")
    return cache, config

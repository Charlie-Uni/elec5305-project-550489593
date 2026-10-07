"""Feature caches with recorded inputs, conventions and extraction provenance."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .data import Clip, load_audio, load_metadata, source_overlap_report
from .dsp import FeatureConfig
from .features import extract_features, feature_names


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def extract_dataset(root, output, config=FeatureConfig(), require_complete=True):
    root, output = Path(root), Path(output)
    if output.suffix != ".npz":
        raise ValueError("Feature cache filename must end in .npz")
    clips = load_metadata(root, require_complete=require_complete)
    features, hashes, timings = [], [], []
    with threadpool_limits(limits=1):
        for i, clip in enumerate(clips, 1):
            path = root / "audio" / clip.filename
            audio = load_audio(path, config, require_five_seconds=require_complete)
            start = perf_counter()
            features.append(extract_features(audio, config))
            timings.append(perf_counter() - start)
            hashes.append(sha256_file(path))
            if i % 100 == 0 or i == len(clips):
                print(f"Extracted {i}/{len(clips)} clips", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ("dsp.py", "features.py", "data.py", "artifacts.py")
    }
    np.savez_compressed(
        output,
        schema_version=np.array(1),
        X=np.asarray(features),
        y=np.array([c.target for c in clips]),
        folds=np.array([c.fold for c in clips]),
        filenames=np.array([c.filename for c in clips]),
        categories=np.array([c.category for c in clips]),
        source_ids=np.array([c.source_id for c in clips]),
        audio_sha256=np.array(hashes),
        extraction_seconds=np.array(timings),
        feature_names=np.array(feature_names(config)),
        config_json=np.array(json.dumps(config.as_dict(), sort_keys=True)),
        metadata_sha256=np.array(sha256_file(root / "meta" / "esc50.csv")),
        source_hashes_json=np.array(json.dumps(source_hashes, sort_keys=True)),
        source_overlap_json=np.array(json.dumps(source_overlap_report(clips), sort_keys=True)),
        complete_dataset=np.array(require_complete),
    )
    return {
        "clips": len(clips),
        "features": len(feature_names(config)),
        "feature_cache": str(output),
        "sha256": sha256_file(output),
    }


def load_feature_cache(path):
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
        "config_json",
        "feature_names",
        "complete_dataset",
        "source_hashes_json",
        "source_overlap_json",
    }
    if not required.issubset(cache) or cache["schema_version"].item() != 1:
        raise ValueError("Missing or unsupported feature-cache schema")
    config = FeatureConfig(**json.loads(cache["config_json"].item()))
    n = len(cache["y"])
    if cache["X"].shape != (n, len(feature_names(config))) or n == 0:
        raise ValueError("Inconsistent feature-cache shape")
    if not np.isfinite(cache["X"]).all():
        raise ValueError("Feature cache contains nonfinite values")
    if cache["feature_names"].tolist() != feature_names(config):
        raise ValueError("Feature names do not match the feature configuration")
    for key in ("y", "folds", "filenames", "categories", "source_ids"):
        if cache[key].shape != (n,):
            raise ValueError(f"Invalid cache array: {key}")
    if len(set(cache["filenames"].tolist())) != n:
        raise ValueError("Duplicate cached filenames")
    if not np.issubdtype(cache["y"].dtype, np.integer) or np.any(
        (cache["y"] < 0) | (cache["y"] >= 50)
    ):
        raise ValueError("Invalid cached class IDs")
    if not np.issubdtype(cache["folds"].dtype, np.integer) or set(cache["folds"].tolist()) != set(
        range(1, 6)
    ):
        raise ValueError("Feature cache requires official folds 1-5")
    classes = {}
    for source, fold, target, category in zip(
        cache["source_ids"], cache["folds"], cache["y"], cache["categories"]
    ):
        if not str(source):
            raise ValueError("Missing source recording ID in feature cache")
        if classes.setdefault(int(target), str(category)) != category:
            raise ValueError("Inconsistent cached class names")
    clips = [
        Clip(str(name), int(fold), int(target), str(category), str(source))
        for name, fold, target, category, source in zip(
            cache["filenames"], cache["folds"], cache["y"], cache["categories"], cache["source_ids"]
        )
    ]
    if source_overlap_report(clips) != json.loads(cache["source_overlap_json"].item()):
        raise ValueError("Cached source-overlap audit does not match its rows")
    return cache, config

"""Check every training view against official metadata and held-out partitions."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from esc_robustness.artifacts import sha256_file, write_json


def audit(run, dataset, output):
    with (dataset / "meta/esc50.csv").open() as handle:
        metadata = {r["filename"]: r for r in csv.DictReader(handle)}
    if len(metadata) != 2000:
        raise ValueError("Full official metadata required")
    source_folds = defaultdict(set)
    for row in metadata.values():
        source_folds[row["src_file"]].add(int(row["fold"]))
    with np.load(run / "clean_log_mel.npz") as cache:
        names = cache["filenames"].tolist()
        hashes = dict(zip(names, cache["audio_sha256"].tolist()))
    if len(names) != 2000 or len(set(names)) != 2000:
        raise ValueError("Clean cache coverage differs")
    for name in names:
        if sha256_file(dataset / "audio" / name) != hashes[name]:
            raise ValueError("Dataset changed since clean-cache generation")
    cache_hash = sha256_file(run / "clean_log_mel.npz")
    banks = []
    for mode in ("wgn", "semantic"):
        for fold in range(1, 6):
            for phase in ("selection", "refit"):
                directory = run / "banks" / mode / f"fold_{fold}_{phase}"
                meta = json.loads((directory / "bank_meta.json").read_text())
                fingerprint = meta["fingerprint"]
                excluded = {fold, fold % 5 + 1} if phase == "selection" else {fold}
                allowed = {name for name in names if int(metadata[name]["fold"]) not in excluded}
                if {names[i] for i in fingerprint["indices"]} != allowed or fingerprint[
                    "cache_sha256"
                ] != cache_hash:
                    raise ValueError("Bank training partition or source cache differs")
                if (
                    fingerprint["mode"] != mode
                    or fingerprint["fold"] != fold
                    or fingerprint["phase"] != phase
                ):
                    raise ValueError("Bank protocol differs")
                for name, digest in fingerprint["source_sha256"].items():
                    if sha256_file(run / "augmentation_source/esc_robustness" / name) != digest:
                        raise ValueError("Generation source snapshot differs")
                for name, digest in meta["files"].items():
                    if sha256_file(directory / name) != digest:
                        raise ValueError("Bank file checksum differs")
                with (directory / "manifest.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                if len(rows) != 3 * len(allowed) or len(rows) != meta["rows"]:
                    raise ValueError("Missing augmented views")
                pairs, levels, pairing_counts = set(), Counter(), Counter()
                max_ratio_error = 0.0
                for row in rows:
                    index, position, view = [
                        int(row[k]) for k in ("target_index", "position", "view")
                    ]
                    name = row["target_file"]
                    if (position, view) in pairs or view not in (0, 1, 2):
                        raise ValueError("Duplicate or unexpected view")
                    pairs.add((position, view))
                    if (
                        name != names[index]
                        or fingerprint["indices"][position] != index
                        or name not in allowed
                    ):
                        raise ValueError("Training row outside the declared partition")
                    original = metadata[name]
                    if (
                        row["target_class"] != original["target"]
                        or row["target_fold"] != original["fold"]
                        or row["target_sha256"] != hashes[name]
                    ):
                        raise ValueError("Training target identity differs")
                    level, achieved = float(row["level_db"]), float(row["achieved_db"])
                    if level != (5, 10, 15)[view] or abs(level - achieved) > 1e-8:
                        raise ValueError("Training SNR/SIR differs")
                    max_ratio_error = max(max_ratio_error, abs(level - achieved))
                    common, peak = float(row["common_gain"]), float(row["peak_before_gain"])
                    if not 0 < common <= 1 or common * peak > 0.99 + 1e-12:
                        raise ValueError("Clipped or invalid training gain")
                    levels[level] += 1
                    if mode == "semantic":
                        partner = row["interferer"]
                        if partner not in allowed:
                            raise ValueError("Validation/test audio used as a distractor")
                        other = metadata[partner]
                        if (
                            other["src_file"] == original["src_file"]
                            or len(source_folds[other["src_file"]]) != 1
                            or other["target"] == original["target"]
                        ):
                            raise ValueError("Invalid distractor source or class")
                        if (
                            row["interferer_class"] != other["target"]
                            or row["interferer_fold"] != other["fold"]
                            or row["interferer_sha256"] != hashes[partner]
                        ):
                            raise ValueError("Distractor identity differs")
                        same = int(other["target"]) // 10 == int(original["target"]) // 10
                        if same != ((position + view) % 2 == 0) or row["pairing"] != (
                            "same_category" if same else "different_category"
                        ):
                            raise ValueError("Category pairing schedule differs")
                        pairing_counts[row["pairing"]] += 1
                    elif row["interferer"] != "white" or row["interferer_class"]:
                        raise ValueError("Unexpected WGN source")
                if pairs != {(i, v) for i in range(len(allowed)) for v in range(3)}:
                    raise ValueError("Incomplete view coverage")
                for filename, trailing in (("features.npy", (92,)), ("log_mel.npy", (64, 430))):
                    array = np.load(directory / filename, mmap_mode="r")
                    if array.shape != (len(allowed), 3, *trailing):
                        raise ValueError("Bank tensor shape differs")
                    for start in range(0, len(array), 64):
                        if not np.isfinite(array[start : start + 64]).all():
                            raise ValueError("Nonfinite augmentation tensor")
                banks.append(
                    dict(
                        mode=mode,
                        fold=fold,
                        phase=phase,
                        training_clips=len(allowed),
                        views=len(rows),
                        excluded_folds=sorted(excluded),
                        levels=dict(levels),
                        pairings=dict(pairing_counts),
                        max_achieved_ratio_error_db=max_ratio_error,
                        metadata_sha256=sha256_file(directory / "bank_meta.json"),
                        files=meta["files"],
                    )
                )
                print(f"Verified augmentation bank: {mode}, fold {fold}, {phase}", flush=True)
    report = dict(
        verified=True,
        banks=len(banks),
        views=sum(b["views"] for b in banks),
        audio_files_verified=2000,
        selection_excludes_validation_and_test=True,
        refit_excludes_test=True,
        semantic_distractors_exclude_cross_fold_sources=True,
        bank_audits=banks,
    )
    write_json(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.dataset, args.output)

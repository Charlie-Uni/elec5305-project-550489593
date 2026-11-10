"""Grouped resampling must preserve repeated recordings as one sampled unit."""

import runpy
from pathlib import Path

cluster_difference = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/check_source_cluster_uncertainty.py")
)["cluster_difference"]


def test_unequal_source_sizes_keep_whole_cluster_and_clip_weighted_estimand():
    names = ["a1", "a2", "a3", "b1"]
    sources = {name: name[0] for name in names}
    rows = [
        {"target_file": n, "target_class": "0", "prediction": "0" if n[0] == "a" else "1"}
        for n in names
    ]
    reference = {n: {"prediction": "1" if n[0] == "a" else "0"} for n in names}
    result = cluster_difference(rows, reference, sources, replicates=2000)
    # Two sampled sources give exactly three possible differences: -1, 0.5, 1.
    # The three copies of source A are never independently sampled.
    assert result["target_source_clusters"] == 2
    assert result["maximum_clips_per_source"] == 3
    assert result["accuracy_difference"] == 0.5
    assert result["source_ci_low"] == -1
    assert result["source_ci_high"] == 1


def test_identical_predictions_have_zero_difference_for_every_source_sample():
    rows = [
        {"target_file": str(i), "target_class": "0", "prediction": str(i % 2)} for i in range(10)
    ]
    reference = {r["target_file"]: r for r in rows}
    sources = {str(i): str(i // 2) for i in range(10)}
    result = cluster_difference(rows, reference, sources, replicates=2000)
    assert result["accuracy_difference"] == 0
    assert result["source_ci_low"] == 0
    assert result["source_ci_high"] == 0

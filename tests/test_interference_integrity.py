import json

import numpy as np
import pytest
import soundfile as sf

from esc_robustness.analysis import capture_statistics, condition_table
from esc_robustness.artifacts import sha256_file
from esc_robustness.mixing import (
    DemandCache,
    InterferenceConfig,
    build_manifest,
    renderer_from_manifest,
)


def test_same_power_modified_audio_is_rejected(toy_dataset, tmp_path):
    manifest = tmp_path / "manifest"
    build_manifest(
        toy_dataset,
        manifest,
        InterferenceConfig(conditions=("white",), levels_db=(5.0,)),
        require_complete=False,
        limit_per_fold=1,
    )
    renderer, _, rows = renderer_from_manifest(manifest)
    path = toy_dataset / "audio" / rows[0]["target_file"]
    audio, rate = sf.read(path)
    sf.write(path, -audio, rate, subtype="FLOAT")
    with pytest.raises(ValueError, match="checksum"):
        renderer.render(rows[0])


def test_changed_metadata_is_rejected(toy_dataset, tmp_path):
    manifest = tmp_path / "manifest"
    build_manifest(
        toy_dataset,
        manifest,
        InterferenceConfig(conditions=("white",), levels_db=(5.0,)),
        require_complete=False,
        limit_per_fold=1,
    )
    with (toy_dataset / "meta/esc50.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="metadata checksum"):
        renderer_from_manifest(manifest)


def test_modified_demand_cache_is_rejected(tmp_path):
    path = tmp_path / "noise.npy"
    np.save(path, np.ones(20))
    meta = {"sample_rate": 44100, "ENV": {"cache": path.name, "cache_sha256": sha256_file(path)}}
    (tmp_path / "demand_cache.json").write_text(json.dumps(meta))
    np.save(path, -np.ones(20))
    with pytest.raises(ValueError, match="checksum"):
        DemandCache(tmp_path).audio("ENV")


def test_capture_reference_uses_class_count_and_explicit_assumption():
    rows = [{"interferer_class": 1, "correct": 0, "prediction": 1}]
    result = capture_statistics(rows, 4)
    assert result["capture_rate"] == 1
    assert result["uniform_wrong_class_reference"] == pytest.approx(1 / 3)
    assert "not an empirical null" in result["reference_assumption"]


def test_clean_fold_mismatch_is_rejected():
    row = {
        "target_file": "x.wav",
        "condition": "white",
        "level_db": 5,
        "fold": 1,
        "target_class": 0,
    }
    with pytest.raises(ValueError, match="fold/label"):
        condition_table([row], {"x.wav": (2, 0, 0)}, [0, 1])


def test_clean_reference_matches_condition_after_silent_pair_exclusion():
    clean = {"a": (1, 0, 0), "b": (1, 1, 0), "c": (2, 0, 0), "d": (2, 1, 1)}
    rows = []
    for condition, names in (("white", "abcd"), ("semantic_same", "acd")):
        for name in names:
            fold, target, prediction = clean[name]
            semantic = condition.startswith("semantic")
            rows.append(
                {
                    "target_file": name,
                    "fold": fold,
                    "target_class": target,
                    "prediction": prediction,
                    "correct": int(target == prediction),
                    "control_prediction": prediction,
                    "common_gain": 1.0,
                    "condition": condition,
                    "level_db": 5.0,
                    "ratio_type": "SIR" if semantic else "SNR",
                    "interferer_class": 1 - target if semantic else None,
                    "margin": 1.0 if semantic else None,
                }
            )
    result = condition_table(rows, clean, [0, 1])
    semantic = result["semantic_same@+5dB"]
    assert result["clean"]["n"] == 4
    assert semantic["matched_clean"]["n"] == 3
    assert semantic["matched_clean"]["macro_f1"]["mean"] == pytest.approx(0.75)
    assert semantic["relative_macro_f1_loss"] == pytest.approx(0.0)
    assert semantic["control_macro_f1_minus_clean"] == pytest.approx(0.0)


def test_renderer_uses_recorded_gains_after_tolerance_check(toy_dataset, tmp_path):
    manifest = tmp_path / "manifest"
    build_manifest(
        toy_dataset,
        manifest,
        InterferenceConfig(conditions=("white",), levels_db=(5.0,)),
        require_complete=False,
        limit_per_fold=1,
    )
    renderer, _, rows = renderer_from_manifest(manifest)
    row = dict(rows[0])
    row["interferer_gain"] = np.nextafter(row["interferer_gain"], np.inf)
    row["common_gain"] = np.nextafter(row["common_gain"], 0.0)
    target = renderer.clip(row["target_file"])
    noise = np.random.default_rng(row["row_seed"]).standard_normal(target.size)
    expected = (target + row["interferer_gain"] * noise) * row["common_gain"]
    rendered = renderer.render(row)
    assert np.array_equal(rendered["mixture"], expected)
    assert np.array_equal(rendered["clean_matched"], target * row["common_gain"])

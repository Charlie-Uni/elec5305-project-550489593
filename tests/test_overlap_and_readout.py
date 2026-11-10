import csv
import json

import numpy as np
import pytest
from conftest import TOY_LENGTH, make_tone

from esc_robustness.artifacts import extract_dataset
from esc_robustness.cli import main
from esc_robustness.dsp import FeatureConfig
from esc_robustness.interference_eval import evaluate_manifest
from esc_robustness.mixing import InterferenceConfig, build_manifest
from esc_robustness.overlap_analysis import (
    analyse_overlap,
    compute_overlap_features,
    logistic_decomposition,
    matched_semantic_vs_demand,
    overlap_measures,
    read_features,
    stratified_accuracy,
)
from esc_robustness.readout_analysis import OUTCOMES, analyse_readout, outcome, summarise
from esc_robustness.svm import evaluate_svm

pytest.importorskip("matplotlib")


def test_overlap_measures_follow_spectral_and_temporal_structure():
    config = FeatureConfig()
    tone = make_tone(500, amplitude=0.3)
    same_band_louder = make_tone(520, amplitude=0.6)
    other_band = make_tone(6000, amplitude=0.3)
    same = overlap_measures(tone, same_band_louder, config)
    other = overlap_measures(tone, other_band, config)
    assert same["spectral_correlation_db"] > 0.9 and other["spectral_correlation_db"] < 0.5
    assert same["masked_cell_fraction"] > 0.75  # louder interferer in the target's cells
    assert other["masked_cell_fraction"] < 0.1  # disjoint bands leave the target unmasked
    assert same["masked_energy_fraction"] > other["masked_energy_fraction"]
    half = tone.copy()
    half[TOY_LENGTH // 2 :] = 0.0
    late = same_band_louder.copy()
    late[: TOY_LENGTH // 2] = 0.0
    disjoint_time = overlap_measures(half, late, config)
    assert disjoint_time["spectral_correlation_db"] > 0.9  # same bands...
    assert disjoint_time["masked_cell_fraction"] < 0.3  # ...but mostly not at the same time
    assert 0 < same["interferer_energy_share"] < 1
    with pytest.raises(ValueError):
        overlap_measures(np.zeros(TOY_LENGTH), tone, config)


@pytest.fixture
def toy_evaluation(interference_dataset, tmp_path):
    cache = tmp_path / "features.npz"
    extract_dataset(interference_dataset, cache, require_complete=False)
    svm_dir = tmp_path / "svm"
    evaluate_svm(cache, svm_dir, {"svc__C": [1.0], "svc__gamma": ["scale"]}, require_complete=False)
    manifest = tmp_path / "manifest"
    build_manifest(
        interference_dataset,
        manifest,
        InterferenceConfig(conditions=("white", "semantic_same", "semantic_different")),
        require_complete=False,
    )
    evaluation = tmp_path / "eval"
    evaluate_manifest(manifest, evaluation, svm_dir=svm_dir)
    return manifest, evaluation


def test_overlap_features_and_analysis_on_toy_evaluation(toy_evaluation, tmp_path):
    manifest, evaluation = toy_evaluation
    features_dir = tmp_path / "overlap"
    result = compute_overlap_features(manifest, features_dir)
    assert result["rows_written"] == 40 * 3 * 3 and result["demand_rows_skipped"] == 0
    features = read_features(features_dir / "overlap_features.csv")
    assert len(features) == 360
    assert all(0 <= f["masked_cell_fraction"] <= 1 for f in features.values())
    assert all(-1 - 1e-9 <= f["spectral_correlation_db"] <= 1 + 1e-9 for f in features.values())
    meta = json.loads((features_dir / "overlap_features_meta.json").read_text())
    assert meta["rows_written"] == 360 and "manifest_sha256" in meta
    with pytest.raises(FileExistsError):
        compute_overlap_features(manifest, features_dir)
    output = tmp_path / "overlap_tables"
    summary = analyse_overlap(features_dir, {"svm": evaluation}, output, level=5.0)
    assert "svm" in summary["systems"]
    models = summary["systems"]["svm"]["models"]
    assert set(models) == {"condition_only", "acoustic_only", "condition_plus_acoustic"}
    assert set(models["condition_plus_acoustic"]["coefficients"]) >= {
        "masked_cell_fraction",
        "spectral_correlation_db",
        "activity_overlap",
    }
    with (output / "stratified_accuracy.csv").open(newline="") as handle:
        strata = list(csv.DictReader(handle))
    assert strata and all(0 <= float(r["accuracy"]) <= 1 for r in strata)
    # No DEMAND condition in this manifest: no matched table, no matched figure.
    assert not (output / "matched_semantic_vs_demand.csv").exists()
    assert (output / "overlap_accuracy_vs_masking.png").exists()
    assert (output / "measure_summary.csv").exists()


def test_matched_and_logistic_helpers_with_synthetic_rows():
    # Two targets, DEMAND and same-category rows at +5 dB; semantic masks more for one.
    features = {
        1: {"masked_cell_fraction": 0.1, "spectral_correlation_db": 0.5, "activity_overlap": 1.0},
        2: {"masked_cell_fraction": 0.6, "spectral_correlation_db": 0.9, "activity_overlap": 1.0},
        3: {"masked_cell_fraction": 0.3, "spectral_correlation_db": 0.4, "activity_overlap": 1.0},
        4: {"masked_cell_fraction": 0.2, "spectral_correlation_db": 0.3, "activity_overlap": 0.5},
    }
    predictions = [
        {
            "row": 1,
            "condition": "demand",
            "level_db": 5.0,
            "target_file": "a",
            "correct": 1,
            "target_class": 0,
            "prediction": 0,
            "interferer_class": None,
        },
        {
            "row": 2,
            "condition": "semantic_same",
            "level_db": 5.0,
            "target_file": "a",
            "correct": 0,
            "target_class": 0,
            "prediction": 7,
            "interferer_class": 7,
        },
        {
            "row": 3,
            "condition": "demand",
            "level_db": 5.0,
            "target_file": "b",
            "correct": 0,
            "target_class": 1,
            "prediction": 3,
            "interferer_class": None,
        },
        {
            "row": 4,
            "condition": "semantic_same",
            "level_db": 5.0,
            "target_file": "b",
            "correct": 1,
            "target_class": 1,
            "prediction": 1,
            "interferer_class": 9,
        },
    ]
    matched = matched_semantic_vs_demand(features, predictions, 5.0)
    all_pairs = [m for m in matched if m["stratum"] == "all pairs"][0]
    assert all_pairs["n"] == 2 and all_pairs["semantic_minus_demand_pp"] == pytest.approx(0.0)
    strata = stratified_accuracy(features, predictions, 5.0)
    assert {s["condition"] for s in strata} == {"demand", "semantic_same"}
    assert all(s["capture_fraction"] is None for s in strata if s["condition"] == "demand")
    decomposition = logistic_decomposition(features, predictions, 5.0)
    assert (
        decomposition["n"] == 4
        and "semantic_same" in decomposition["models"]["condition_only"]["coefficients"]
    )


def test_topk_outcomes_and_readout_summary(tmp_path):
    labels = outcome([1, 1, 3, 4, np.nan], [2, 3, 1, 5, 1], k=2)
    assert list(labels[:4]) == ["both", "target_only", "distractor_only", "neither"]
    assert labels[4] is None
    trials = []
    for i, (tr, dr, ctr, cdr, pred_cap) in enumerate(
        [(1, 2, 1, 7, 0), (2, 1, 1, 9, 1), (5, 1, 1, 3, 1), (1, 6, 1, 2, 0)]
    ):
        trials.append(
            {
                "region": "own",
                "row": i,
                "fold": 1,
                "condition": "semantic_same",
                "level_db": 5.0,
                "target_class": 0,
                "distractor_class": 4,
                "correct": int(tr == 1),
                "argmax_captured": pred_cap,
                "target_rank": tr,
                "distractor_rank": dr,
                "control_target_rank": ctr,
                "control_distractor_rank": cdr,
            }
        )
    table = summarise(trials, k=2)
    assert len(table) == 1
    entry = table[0]
    assert entry["n"] == 4 and entry["argmax_accuracy"] == 0.5
    assert entry["mixture_both"] == 0.5 and entry["mixture_distractor_only"] == 0.25
    assert entry["mixture_target_only"] == 0.25 and entry["mixture_neither"] == 0.0
    assert entry["argmax_captured_target_rank_le_k"] == 0.5  # one of two captured had rank 2
    assert entry["control_distractor_in_topk"] == 0.25
    assert sum(entry[f"mixture_{name}"] for name in OUTCOMES) == pytest.approx(1.0)
    summary = analyse_readout(trials, tmp_path / "readout", k=2)
    assert summary["control_ranks_exact"] is True
    assert (tmp_path / "readout" / "readout_top2_outcomes.png").exists()
    assert (tmp_path / "readout" / "readout_outcomes.csv").exists()


def test_cli_overlap_commands(toy_evaluation, tmp_path, capsys):
    manifest, evaluation = toy_evaluation
    main(
        [
            "overlap-features",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "ovl"),
            "--limit",
            "30",
        ]
    )
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{") :])["rows_written"] == 30
    main(
        [
            "analyse-overlap",
            "--features",
            str(tmp_path / "ovl"),
            "--output",
            str(tmp_path / "ovl_tables"),
            "--svm-eval",
            str(evaluation),
            "--level",
            "5",
        ]
    )
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{") :])["systems"] == ["svm"]


def test_exact_topk_has_deterministic_tie_breaking_and_rejects_nonfinite():
    from esc_robustness.readout_analysis import ordinal_ranks

    scores = np.array([[0.4, 0.4, 0.4, 0.1], [0.1, 0.9, 0.3, 0.3]])
    ranks = ordinal_ranks(scores)
    assert ranks.tolist() == [[1, 2, 3, 4], [4, 1, 2, 3]]
    assert ((ranks <= 2).sum(axis=1) == 2).all()
    with pytest.raises(ValueError):
        ordinal_ranks([[0.1, np.nan]])
    for invalid in (0, -1, 51, 2.5):
        with pytest.raises(ValueError):
            summarise([], invalid)


def test_target_active_spectrum_distinguishes_nonoverlap_and_gain():
    config = FeatureConfig()
    t = make_tone(500, amplitude=0.3)
    d = make_tone(6000, amplitude=0.4)
    normal = overlap_measures(t, d, config)
    quiet = overlap_measures(t * 0.01, d * 0.01, config)
    for key in (
        "target_activity_spectral_cosine",
        "masked_cell_fraction",
        "density_spectral_correlation_db",
    ):
        assert quiet[key] == pytest.approx(normal[key], abs=1e-10)
    silence = overlap_measures(t, np.zeros_like(t), config)
    assert silence["target_activity_spectral_cosine"] is None
    assert silence["density_spectral_correlation_db"] is None


def test_ranked_reader_joins_exact_rows(toy_evaluation):
    from esc_robustness.readout_analysis import trials_from_scores

    _, evaluation = toy_evaluation
    trials = trials_from_scores("own", evaluation, "svm")
    assert len(trials) == 240
    assert all(t["target_file"] and t["control_target_rank"] >= 1 for t in trials)


def test_source_bootstrap_keeps_multiple_clips_together():
    import importlib.util
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "scripts/complete_closing_analyses.py"
    spec = importlib.util.spec_from_file_location("closing_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    draws = module.bootstrap_means([0, 0, 1], ["a", "a", "b"], replicates=200)[:, 0]
    # Two clusters have sizes 2 and 1. Resampling clusters can yield only these means.
    assert set(np.round(draws, 10)).issubset({0.0, round(1 / 3, 10), 1.0})

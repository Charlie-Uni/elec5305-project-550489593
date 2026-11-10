import csv
import json

import numpy as np
import pytest

from esc_robustness.artifacts import extract_dataset
from esc_robustness.cli import main
from esc_robustness.figures import EvaluationData, displacement
from esc_robustness.interference_eval import PREDICTION_FIELDS, evaluate_manifest
from esc_robustness.mixing import InterferenceConfig, build_manifest
from esc_robustness.svm import evaluate_svm

pytest.importorskip("matplotlib")

FIGURES = (
    "fig1_macro_f1_vs_level",
    "fig2_capture_rate_vs_sir",
    "fig3_margin_vs_sir",
    "fig4_same_vs_different",
    "fig5_representation_displacement",
    "fig6_confusion_flow",
    "fig7_robustness_vs_cost",
)


def test_make_figures_writes_every_figure_with_its_data(interference_dataset, tmp_path):
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
    cost = tmp_path / "cost.json"
    cost.write_text(
        json.dumps(
            {"preprocessing_median_ms": 3.0, "inference_median_ms": 7.0, "model_bytes": 1000}
        )
    )
    output = tmp_path / "figures"
    main(
        [
            "make-figures",
            "--evaluation",
            str(evaluation),
            "--output",
            str(output),
            "--svm-clean",
            str(svm_dir),
            "--svm-cost",
            str(cost),
            "--level",
            "5",
        ]
    )
    meta = json.loads((output / "figures_meta.json").read_text())
    assert meta["systems"] == ["svm"] and meta["focus_level_db"] == 5.0
    for stem in FIGURES:
        assert (output / f"{stem}.png").stat().st_size > 0
        assert (output / f"{stem}.svg").stat().st_size > 0
        with (output / f"{stem}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, stem
    # No DEMAND condition in this manifest, so the overlap figure is skipped.
    assert not (output / "fig8_overlap_strata.png").exists()
    with (output / "fig1_macro_f1_vs_level.csv").open(newline="") as handle:
        first = list(csv.DictReader(handle))
    assert {r["level_db"] for r in first} == {"clean", "10.0", "5.0", "0.0"}
    assert all(0 <= float(r["macro_f1_mean"]) <= 1 for r in first)


def test_displacement_uses_training_fold_scales_and_not_control_population():
    arrays = {
        "row": np.array([1, 2]),
        "row_folds": np.array([1, 2]),
        "representation": np.array([[2.0, 4.0], [2.0, 4.0]]),
        "control_representation": np.zeros((2, 2)),
        "representation_type": "handcrafted_features",
        "training_scales": {1: np.array([2.0, 4.0]), 2: np.array([20.0, 40.0])},
    }
    rows = [{"row": 2}, {"row": 1}]
    values, _ = displacement("svm", arrays, rows)
    assert values == pytest.approx([0.1, 1.0])
    # A held-out control shift cannot refit the feature scaling.
    arrays["representation"] += 1000
    arrays["control_representation"] += 1000
    assert displacement("svm", arrays, rows)[0] == pytest.approx(values)


def _minimal_saved_evaluation(root, system):
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps({"manifest_sha256": "same-manifest", "systems": {system: {}}})
    )
    row = dict.fromkeys(PREDICTION_FIELDS, "")
    row.update(
        row=1,
        condition="semantic_same",
        ratio_type="SIR",
        level_db=5,
        fold=1,
        target_file="a.wav",
        target_class=0,
        target_category="dog",
        interferer="b.wav",
        interferer_class=1,
        common_gain=1,
        activity_overlap=1,
        prediction=1,
        correct=0,
        control_prediction=0,
        margin=-0.2,
    )
    with (root / f"{system}_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    np.savez(
        root / f"{system}_scores.npz",
        class_ids=np.array([0, 1]),
        score_type=np.array("cosine"),
        row=np.array([1]),
        representation=np.array([[1.0, 0.0]]),
        control_representation=np.array([[0.0, 1.0]]),
        representation_type=np.array("embedding"),
    )
    return root


def test_additional_evaluation_requires_identical_manifest_and_trials(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "predictions.csv").write_text("filename,fold,target,prediction\na.wav,1,0,0\n")
    cnn = _minimal_saved_evaluation(tmp_path / "cnn", "cnn")
    clap = _minimal_saved_evaluation(tmp_path / "clap", "clap")
    refs = {"cnn": clean, "clap": clean}
    data = EvaluationData(cnn, refs, additional_evaluations=[clap])
    assert data.systems == ["cnn", "clap"]
    summary = json.loads((clap / "summary.json").read_text())
    summary["manifest_sha256"] = "different-manifest"
    (clap / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="identical manifest"):
        EvaluationData(cnn, refs, additional_evaluations=[clap])
    summary["manifest_sha256"] = "same-manifest"
    (clap / "summary.json").write_text(json.dumps(summary))
    path = clap / "clap_predictions.csv"
    path.write_text(path.read_text().replace("b.wav", "another.wav"))
    with pytest.raises(ValueError, match="same ordered trials"):
        EvaluationData(cnn, refs, additional_evaluations=[clap])


def test_prediction_representation_row_misalignment_is_rejected(tmp_path):
    root = _minimal_saved_evaluation(tmp_path / "cnn", "cnn")
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "predictions.csv").write_text("filename,fold,target,prediction\na.wav,1,0,0\n")
    path = root / "cnn_scores.npz"
    with np.load(path) as original:
        arrays = {key: original[key] for key in original.files}
    arrays["row"] = np.array([99])
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="row IDs"):
        EvaluationData(root, {"cnn": clean})

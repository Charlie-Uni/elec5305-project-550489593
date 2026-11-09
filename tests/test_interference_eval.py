import csv
import json

import numpy as np
import pytest

from esc_robustness.artifacts import extract_dataset
from esc_robustness.cli import main
from esc_robustness.interference_eval import evaluate_manifest
from esc_robustness.mixing import InterferenceConfig, build_manifest
from esc_robustness.svm import evaluate_svm

SMALL_GRID = {"svc__C": [1.0], "svc__gamma": ["scale"]}


@pytest.fixture
def toy_svm(interference_dataset, tmp_path):
    cache = tmp_path / "features.npz"
    extract_dataset(interference_dataset, cache, require_complete=False)
    model_dir = tmp_path / "svm"
    evaluate_svm(cache, model_dir, SMALL_GRID, require_complete=False)
    return model_dir


def test_manifest_rows_are_classified_by_fold_matched_models(
    interference_dataset,
    toy_svm,
    tmp_path,
):
    config = InterferenceConfig(
        conditions=("white", "semantic_different"), levels_db=(10.0, 0.0), seed=3
    )
    manifest = tmp_path / "manifest"
    build_manifest(interference_dataset, manifest, config, require_complete=False)
    output = tmp_path / "eval"
    summary = evaluate_manifest(manifest, output, svm_dir=toy_svm, limit=24)
    assert summary["rows_evaluated"] == 24 and set(summary["systems"]) == {"svm"}
    with (output / "svm_predictions.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 24
    with np.load(output / "svm_scores.npz") as arrays:
        assert arrays["scores"].shape == (24, 4)
        assert arrays["representation"].shape == (24, 92)
        assert arrays["control_representation"].shape == (24, 92)
        assert np.isfinite(arrays["scores"]).all()
        class_ids = arrays["class_ids"].tolist()
        scores = arrays["scores"]
    for i, record in enumerate(records):
        predicted = class_ids[int(np.argmax(scores[i]))]
        assert int(record["prediction"]) == predicted
        assert record["correct"] == str(int(predicted == int(record["target_class"])))
        if record["condition"] == "white":
            assert record["interferer_class"] == "" and record["margin"] == ""
        else:
            target = float(record["target_score"])
            interferer = float(record["interferer_score"])
            assert float(record["margin"]) == pytest.approx(target - interferer)
            assert record["predicted_interferer"] == str(
                int(int(record["prediction"]) == int(record["interferer_class"]))
            )
        # The tones are easy: a matched clean control at full gain is classified correctly.
        if float(record["common_gain"]) == 1.0:
            assert record["control_correct"] == "1"
    table = summary["systems"]["svm"]["conditions"]
    for cell in table.values():
        assert 0 <= cell["accuracy"] <= 1 and cell["n"] > 0
        assert cell["capture_rate"] is None or 0 <= cell["capture_rate"] <= 1
    with pytest.raises(FileExistsError):
        evaluate_manifest(manifest, output, svm_dir=toy_svm, limit=1)


def test_cli_evaluate_interference(interference_dataset, toy_svm, tmp_path, capsys):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps({"interference": {"conditions": ["white"], "levels_db": [5.0]}})
    )
    manifest = tmp_path / "manifest"
    main(
        [
            "build-manifest",
            "--data-root",
            str(interference_dataset),
            "--output",
            str(manifest),
            "--config",
            str(config_path),
            "--allow-incomplete",
            "--limit-per-fold",
            "2",
        ]
    )
    capsys.readouterr()
    main(
        [
            "evaluate-interference",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "eval"),
            "--svm-dir",
            str(toy_svm),
        ]
    )
    out = capsys.readouterr().out
    result = json.loads(out[out.index("{") :])
    assert result["rows_evaluated"] == 10 and "white@+5dB" in result["systems"]["svm"]


def test_cnn_system_reports_embeddings(toy_dataset, tmp_path):
    torch = pytest.importorskip("torch")
    from esc_robustness.cnn import CNNConfig, evaluate_cnn
    from esc_robustness.spectrograms import extract_spectrograms

    # The CNN trainer needs contiguous class IDs, so this uses the two-class toy set
    # with the white-noise condition only.
    cache = tmp_path / "log_mel.npz"
    extract_spectrograms(toy_dataset, cache, require_complete=False)
    cnn_dir = tmp_path / "cnn"
    evaluate_cnn(
        cache,
        cnn_dir,
        CNNConfig(channels=(2, 4, 4), max_epochs=1, patience=1, num_threads=1),
        "cpu",
        require_complete=False,
    )
    manifest = tmp_path / "manifest"
    build_manifest(
        toy_dataset,
        manifest,
        InterferenceConfig(conditions=("white",), levels_db=(0.0,)),
        require_complete=False,
        limit_per_fold=1,
    )
    summary = evaluate_manifest(manifest, tmp_path / "eval", cnn_dir=cnn_dir)
    assert set(summary["systems"]) == {"cnn"}
    with np.load(tmp_path / "eval" / "cnn_scores.npz") as arrays:
        assert arrays["scores"].shape == (5, 2)
        assert arrays["representation"].shape == (5, 16)  # 4 channels x 4 frequency bins
        assert arrays["control_representation"].shape == (5, 16)
        assert str(arrays["representation_type"]) == "cnn_embedding"
        assert str(arrays["score_type"]) == "logits"
    assert torch.get_num_threads() >= 1
    from esc_robustness.extension_eval import evaluate_cached, prepare_frontends

    frontends = tmp_path / "frontends"
    prepare_frontends(manifest, frontends)
    cached = tmp_path / "cached_eval"
    evaluate_cached(manifest, cached, cnn_dir, "cnn", frontend_cache=frontends, batch_size=3)
    with (
        np.load(tmp_path / "eval/cnn_scores.npz") as single,
        np.load(cached / "cnn_scores.npz") as batched,
    ):
        for key in ("scores", "control_scores", "representation", "control_representation"):
            np.testing.assert_allclose(single[key], batched[key], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(single["scores"].argmax(1), batched["scores"].argmax(1))
    meta_path = frontends / "cache_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["fingerprint"]["manifest_sha256"] = "invalid"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Manifest differs"):
        evaluate_cached(manifest, tmp_path / "bad_eval", cnn_dir, "cnn", frontend_cache=frontends)


def test_cached_svm_vectors_match_direct_waveform_evaluation(
    interference_dataset, toy_svm, tmp_path
):
    from esc_robustness.extension_eval import evaluate_cached

    manifest = tmp_path / "manifest"
    build_manifest(
        interference_dataset,
        manifest,
        InterferenceConfig(conditions=("white",), levels_db=(5.0,)),
        require_complete=False,
        limit_per_fold=1,
    )
    direct, cached = tmp_path / "direct", tmp_path / "cached"
    evaluate_manifest(manifest, direct, svm_dir=toy_svm)
    evaluate_cached(manifest, cached, toy_svm, "svm", feature_evaluation=direct, batch_size=3)
    with np.load(direct / "svm_scores.npz") as a, np.load(cached / "svm_scores.npz") as b:
        np.testing.assert_array_equal(a["scores"], b["scores"])
        np.testing.assert_array_equal(a["control_scores"], b["control_scores"])

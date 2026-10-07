import csv
import json

import joblib
import numpy as np
import pytest
import soundfile as sf
from sklearn.preprocessing import StandardScaler

from esc_robustness.artifacts import extract_dataset, load_feature_cache
from esc_robustness.cli import benchmark, main, predict
from esc_robustness.data import load_audio, load_metadata, outer_split
from esc_robustness.svm import evaluate_svm, fit_outer_fold

SMALL_GRID = {"svc__C": [1.0, 10.0], "svc__gamma": ["scale"]}


def test_official_data_is_required_by_default(toy_dataset):
    with pytest.raises(ValueError, match="complete ESC-50"):
        load_metadata(toy_dataset)
    clips = load_metadata(toy_dataset, require_complete=False)
    assert len(clips) == 20
    assert clips[0].coarse_category == "animals"


def test_source_leakage_is_rejected(toy_dataset):
    path = toy_dataset / "meta" / "esc50.csv"
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    rows[5][-1] = rows[1][-1]
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError, match="Source overlap"):
        load_metadata(toy_dataset, require_complete=False, strict_sources=True)
    with pytest.warns(UserWarning, match="retaining official"):
        assert len(load_metadata(toy_dataset, require_complete=False)) == 20


def test_missing_file_and_duplicate_rows_are_rejected(toy_dataset):
    path = toy_dataset / "meta" / "esc50.csv"
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    with path.open("a", newline="") as handle:
        csv.writer(handle).writerow(rows[1])
    with pytest.raises(ValueError, match="Duplicate"):
        load_metadata(toy_dataset, require_complete=False)
    (toy_dataset / "audio" / rows[1][0]).unlink()
    with pytest.raises(FileNotFoundError):
        load_metadata(toy_dataset, require_complete=False)


@pytest.mark.parametrize("fold", range(1, 6))
def test_outer_splits_are_disjoint_and_cover_every_row(fold):
    folds = np.repeat(np.arange(1, 6), 4)
    train, validation, test = outer_split(folds, fold)
    assert len(train) == 12 and len(validation) == len(test) == 4
    assert len(set(train) | set(validation) | set(test)) == len(folds)
    assert not (
        set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test)
    )
    assert np.all(folds[test] == fold)
    assert np.all(folds[validation] == fold % 5 + 1)


def test_sample_rate_and_stereo_are_not_silently_changed(tmp_path):
    path = tmp_path / "bad.wav"
    sf.write(path, np.zeros(100), 16000)
    with pytest.raises(ValueError, match="44100 Hz"):
        load_audio(path)
    sf.write(path, np.zeros((100, 2)), 44100)
    with pytest.raises(ValueError, match="mono"):
        load_audio(path)


def test_scaler_is_fitted_only_on_selection_train_then_final_non_test(monkeypatch):
    rng = np.random.default_rng(10)
    folds = np.repeat(np.arange(1, 6), 8)
    y = np.tile(np.repeat([0, 1], 4), 5)
    X = rng.normal(size=(40, 3)) + y[:, None] * 4
    train, validation, test = outer_split(folds, 1)
    calls = []
    original_fit = StandardScaler.fit

    def record_fit(self, values, *args, **kwargs):
        calls.append(values.copy())
        return original_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    model, result, _, _, _ = fit_outer_fold(X, y, folds, 1, SMALL_GRID)
    np.testing.assert_array_equal(calls[0], X[train])
    np.testing.assert_array_equal(calls[1], X[train])
    np.testing.assert_array_equal(calls[2], X[np.concatenate((train, validation))])
    np.testing.assert_allclose(model.named_steps["scale"].mean_, X[folds != 1].mean(axis=0))
    changed = X.copy()
    changed[test] += 1e6
    altered, other, _, _, _ = fit_outer_fold(changed, y, folds, 1, SMALL_GRID)
    assert result["selected_parameters"] == other["selected_parameters"]
    assert result["candidate_results"] == other["candidate_results"]
    np.testing.assert_array_equal(
        model.named_steps["scale"].mean_, altered.named_steps["scale"].mean_
    )
    np.testing.assert_array_equal(
        model.named_steps["svc"].support_vectors_, altered.named_steps["svc"].support_vectors_
    )


def test_feature_cache_to_saved_predictions_and_demo(toy_dataset, tmp_path):
    path = tmp_path / "features.npz"
    extract_dataset(toy_dataset, path, require_complete=False)
    cache, config = load_feature_cache(path)
    assert cache["X"].shape == (20, 92)
    assert len(cache["audio_sha256"]) == 20
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="complete"):
        evaluate_svm(path, output)
    report = evaluate_svm(path, output, SMALL_GRID, require_complete=False)
    assert report["evaluation"] == "diagnostic_subset"
    assert report["accuracy"]["mean"] >= 0.9
    assert len(report["folds"]) == 5
    assert json.loads((output / "summary.json").read_text())["clips"] == 20
    with (output / "predictions.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 20
    with np.load(output / "decision_scores.npz") as predictions:
        assert predictions["scores"].shape == (20, 2)
        assert np.isfinite(predictions["scores"]).all()
    model_path = output / "svm_fold_1.joblib"
    model = joblib.load(model_path)
    np.testing.assert_allclose(
        model["model"].named_steps["scale"].mean_, cache["X"][cache["folds"] != 1].mean(axis=0)
    )
    audio = toy_dataset / "audio" / cache["filenames"][0]
    assert predict(model_path, audio)["prediction"] == int(cache["y"][0])
    cost = benchmark(model_path, audio, runs=2, warmup=1)
    assert cost["preprocessing_median_ms"] > 0
    assert cost["inference_median_ms"] > 0
    with pytest.raises(FileExistsError):
        evaluate_svm(path, output, SMALL_GRID, require_complete=False)


def test_cli_validation_writes_readable_report(tmp_path, capsys):
    report = tmp_path / "stft.json"
    main(["validate-stft", "--output", str(report)])
    assert json.loads(report.read_text())["passed"]
    assert json.loads(capsys.readouterr().out)["passed"]

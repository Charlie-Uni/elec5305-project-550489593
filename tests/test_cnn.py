import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esc_robustness.cnn import (  # noqa: E402
    CNNConfig,
    CompactCNN,
    benchmark_cnn,
    evaluate_cnn,
    fit_outer_fold_cnn,
    load_cnn,
    predict_arrays,
    predict_cnn,
    seed_training,
)
from esc_robustness.data import load_audio, outer_split  # noqa: E402
from esc_robustness.dsp import log_mel  # noqa: E402
from esc_robustness.spectrograms import (  # noqa: E402
    extract_spectrograms,
    load_spectrogram_cache,
)

SMALL_CNN = CNNConfig(
    channels=(2, 4, 4), max_epochs=3, patience=2, batch_size=8, num_threads=1, dropout=0.0
)


@pytest.mark.parametrize("backend", ["cpu", "mps", "cuda"])
def test_accelerator_determinism_policy(backend):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        # A device descriptor exercises policy without requiring that hardware.
        seed_training(5305, torch.device(backend))
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled() == (backend != "cpu")
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_pooling_backward_can_train():
    device = torch.device("cuda")
    seed_training(5305, device)
    model = CompactCNN(64, 2, SMALL_CNN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.randn(4, 1, 64, 16, device=device)
    targets = torch.tensor([0, 1, 0, 1], device=device)
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()


def test_cache_matches_existing_frontend_and_rejects_corruption(toy_dataset, tmp_path):
    path = tmp_path / "log_mel.npz"
    extract_spectrograms(toy_dataset, path, require_complete=False)
    cache, config = load_spectrogram_cache(path, require_complete=False)
    audio = load_audio(toy_dataset / "audio" / cache["filenames"][0])
    np.testing.assert_allclose(cache["X"][0], log_mel(audio, config).T, rtol=1e-6)
    with pytest.raises(ValueError, match="complete ESC-50"):
        load_spectrogram_cache(path)
    with pytest.raises(FileExistsError):
        extract_spectrograms(toy_dataset, path, require_complete=False)
    cache["folds"][0] = 0
    np.savez_compressed(tmp_path / "bad.npz", **cache)
    with pytest.raises(ValueError, match="folds"):
        load_spectrogram_cache(tmp_path / "bad.npz", require_complete=False)


def test_test_fold_cannot_change_selection_scaler_or_refitted_weights():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 64, 16)).astype(np.float32)
    y = np.tile([0, 0, 1, 1], 5)
    folds = np.repeat(np.arange(1, 6), 4)
    train, validation, test = outer_split(folds, 1)
    model, report, _, _, _, _ = fit_outer_fold_cnn(X, y, folds, 1, SMALL_CNN)
    np.testing.assert_allclose(
        report["selection_normalisation"]["mean"],
        X[train].mean(axis=(0, 2), dtype=np.float64),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        model.input_mean.flatten().numpy(),
        X[np.r_[train, validation]].mean(axis=(0, 2), dtype=np.float64),
        atol=1e-7,
    )
    changed_X, changed_y = X.copy(), y.copy()
    changed_X[test] += 100
    changed_y[test] = 1 - changed_y[test]
    other, changed, _, _, _, _ = fit_outer_fold_cnn(changed_X, changed_y, folds, 1, SMALL_CNN)
    assert changed["selected_epochs"] == report["selected_epochs"]
    assert len(report["refit_history"]) == report["selected_epochs"]
    assert [r["validation"] for r in report["selection_history"]] == [
        r["validation"] for r in changed["selection_history"]
    ]
    for key, values in model.state_dict().items():
        torch.testing.assert_close(values, other.state_dict()[key], rtol=0, atol=0)


def test_early_stop_uses_validation_and_keeps_earliest_tie(monkeypatch):
    import esc_robustness.cnn as cnn

    X = np.zeros((20, 64, 16), dtype=np.float32)
    y, folds = np.tile([0, 0, 1, 1], 5), np.repeat(np.arange(1, 6), 4)
    validations = iter([0.2, 0.5, 0.5, 0.4, 0.1])
    monkeypatch.setattr(cnn, "metrics", lambda *args: {"macro_f1": next(validations)})
    config = CNNConfig(channels=(2, 4, 4), max_epochs=8, patience=2, num_threads=1)
    _, result, _, _, _, _ = fit_outer_fold_cnn(X, y, folds, 1, config)
    assert result["selected_epochs"] == 2
    assert len(result["selection_history"]) == 4
    assert len(result["refit_history"]) == 2


def test_network_can_learn_small_frequency_separation():
    from esc_robustness.cnn import new_model, train_epochs

    rng = np.random.default_rng(99)
    X = rng.normal(0, 0.1, (16, 64, 16)).astype(np.float32)
    y = np.repeat([0, 1], 8)
    X[:8, 6:12] += 3
    X[8:, 48:54] += 3
    config = CNNConfig(
        channels=(4, 8, 8), batch_size=8, learning_rate=0.01, dropout=0.0, num_threads=1
    )
    indices, device = np.arange(16), torch.device("cpu")
    model = new_model(X, indices, 2, config, device)
    history, _ = train_epochs(model, X, y, indices, config, device, 15)
    scores, _ = predict_arrays(model, X, indices, 8, device)
    assert history[-1]["train_loss"] < history[0]["train_loss"] * 0.25
    assert np.mean(scores.argmax(axis=1) == y) == 1.0


def test_five_fold_save_reload_demo_and_cpu_cost(toy_dataset, tmp_path):
    path, output = tmp_path / "log_mel.npz", tmp_path / "run"
    extract_spectrograms(toy_dataset, path, require_complete=False)
    config = CNNConfig(channels=(2, 4, 4), max_epochs=1, patience=1, num_threads=1)
    report = evaluate_cnn(path, output, config, "cpu", require_complete=False)
    assert report["evaluation"] == "diagnostic_subset"
    assert len(report["folds"]) == 5
    with (output / "predictions.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 20
    model, _, bundle = load_cnn(output / "cnn_fold_1.pt")
    assert bundle["test_fold"] == 1
    cache, _ = load_spectrogram_cache(path, require_complete=False)
    test = np.flatnonzero(cache["folds"] == 1)
    logits, embeddings = predict_arrays(model, cache["X"], test, 4, torch.device("cpu"))
    with np.load(output / "decision_scores.npz") as saved:
        np.testing.assert_allclose(logits, saved["scores"][test], atol=1e-6)
        np.testing.assert_allclose(embeddings, saved["embeddings"][test], atol=1e-6)
    audio = toy_dataset / "audio" / cache["filenames"][0]
    demo = predict_cnn(output / "cnn_fold_1.pt", audio)
    assert demo["prediction"] == int(logits[0].argmax())
    assert demo["score_type"] == "logits"
    cost = benchmark_cnn(output / "cnn_fold_1.pt", audio, runs=2, warmup=1)
    assert cost["inference_median_ms"] > 0 and cost["numerical_threads"] == 1
    assert json.loads((output / "progress.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        evaluate_cnn(path, output, config, "cpu", require_complete=False)


def test_invalid_configuration_and_short_input():
    with pytest.raises(ValueError):
        CNNConfig(patience=0)
    with pytest.raises(ValueError):
        CNNConfig(channels=(16, 32))
    with pytest.raises(ValueError):
        CNNConfig(learning_rate=float("nan"))
    model = CompactCNN(64, 50).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 1, 64, 1)).shape == (1, 50)

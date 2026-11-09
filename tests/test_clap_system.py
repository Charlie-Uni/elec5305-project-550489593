import csv
import json
import zlib

import numpy as np
import pytest

from esc_robustness.clap_system import (
    CLAP_SAMPLE_RATE,
    CLAPSystem,
    FrozenLAIONBackend,
    benchmark_clap,
    build_prompts,
    evaluate_clean,
    int16_round_trip,
    load_laion_clap,
    resample_for_clap,
)
from esc_robustness.data import load_metadata
from esc_robustness.dsp import FeatureConfig
from esc_robustness.interference_eval import evaluate_manifest
from esc_robustness.mixing import InterferenceConfig, build_manifest


class FakeCLAP:
    """Stands in for laion_clap.CLAP_Module: deterministic, spectrum-based embeddings."""

    dimension = 32

    def __init__(self):
        self.audio_calls = 0

    def get_text_embedding(self, prompts, use_tensor=False):
        assert not use_tensor
        rows = []
        for prompt in prompts:
            rng = np.random.default_rng(zlib.crc32(prompt.encode()))
            rows.append(rng.standard_normal(self.dimension))
        return np.asarray(rows)

    def get_audio_embedding_from_data(self, x, use_tensor=False):
        assert not use_tensor and x.ndim == 2
        assert x.dtype == np.float32 and np.abs(x).max() <= 1.0
        self.audio_calls += len(x)
        vectors = []
        for audio in x:
            spectrum = np.abs(np.fft.rfft(audio))
            vectors.append([segment.mean() for segment in np.array_split(spectrum, self.dimension)])
        return (np.asarray(vectors) + 1e-3) * 3.0  # un-normalised on purpose


def test_checkpoint_checksum_checked_before_model_loading(tmp_path):
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not the fixed reference checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        load_laion_clap(wrong)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_laion_clap(tmp_path / "missing.pt")


def test_prompts_and_preprocessing():
    names = {0: "dog", 1: "sea_waves", 2: "door_wood_knock"}
    assert build_prompts(names) == [
        "the sound of dog",
        "the sound of sea waves",
        "the sound of door wood knock",
    ]
    with pytest.raises(ValueError):
        build_prompts(names, "no placeholder")
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
    resampled = resample_for_clap(audio, 44100)
    assert resampled.shape == (CLAP_SAMPLE_RATE,)
    assert np.array_equal(resample_for_clap(audio, CLAP_SAMPLE_RATE), audio)
    quantised = int16_round_trip(np.array([1.5, -2.0, 0.25, 1 / 32767]))
    assert quantised.dtype == np.float32
    assert quantised[0] == pytest.approx(1.0) and quantised[1] == pytest.approx(-1.0)
    assert quantised[2] == pytest.approx(0.25, abs=1 / 32767)
    assert np.array_equal(int16_round_trip(np.array([0.75, -0.75]) / 32767), [0.0, 0.0])
    with pytest.raises(ValueError, match="Sample rates"):
        resample_for_clap(audio, 0)


def test_clap_system_scores_are_cosines_and_fold_independent(interference_dataset):
    clips = load_metadata(interference_dataset, require_complete=False)
    names = {c.target: c.category for c in clips}
    with pytest.raises(ValueError):
        CLAPSystem(FakeCLAP(), names)  # IDs 0, 1, 10, 11 are not contiguous
    contiguous = {i: f"class_{i}" for i in range(4)}
    fake = FakeCLAP()
    system = CLAPSystem(fake, contiguous)
    assert np.allclose(np.linalg.norm(system.text_embeddings, axis=1), 1.0)
    audio = 0.3 * np.sin(2 * np.pi * 800 * np.arange(22050) / 44100)
    scores_a, embedding_a = system.evaluate(1, audio)
    scores_b, embedding_b = system.evaluate(4, audio)
    assert np.array_equal(scores_a, scores_b) and np.array_equal(embedding_a, embedding_b)
    assert scores_a.shape == (4,) and embedding_a.shape == (FakeCLAP.dimension,)
    assert np.all(np.abs(scores_a) <= 1 + 1e-9)
    assert np.linalg.norm(embedding_a) == pytest.approx(1.0)
    assert fake.audio_calls == 2
    info = system.provenance()
    assert info["fold_independent"] and info["prompts"][2] == "the sound of class 2"


def test_clean_reference_and_manifest_evaluation_with_fake_clap(toy_dataset, tmp_path):
    clips = load_metadata(toy_dataset, require_complete=False)
    names = {c.target: c.category for c in clips}
    system = CLAPSystem(FakeCLAP(), names)
    clean_dir = tmp_path / "clap_clean"
    summary = evaluate_clean(system, toy_dataset, clean_dir, require_complete=False)
    assert summary["clips"] == 20 and set(summary["folds"]) == {1, 2, 3, 4, 5}
    with (clean_dir / "predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20 and set(rows[0]) >= {"filename", "fold", "target", "prediction"}
    with np.load(clean_dir / "decision_scores.npz") as arrays:
        assert arrays["scores"].shape == (20, 2)
        assert arrays["embeddings"].shape == (20, FakeCLAP.dimension)
        assert arrays["text_embeddings"].shape == (2, FakeCLAP.dimension)
        assert list(arrays["prompts"]) == ["the sound of class 0", "the sound of class 1"]
    manifest = tmp_path / "manifest"
    build_manifest(
        toy_dataset,
        manifest,
        InterferenceConfig(conditions=("white",), levels_db=(0.0,)),
        require_complete=False,
        limit_per_fold=1,
    )
    result = evaluate_manifest(manifest, tmp_path / "eval", extra_systems=[system])
    assert set(result["systems"]) == {"clap"}
    assert result["systems"]["clap"]["provenance"]["frozen"] is True
    with np.load(tmp_path / "eval" / "clap_scores.npz") as arrays:
        assert arrays["scores"].shape == (5, 2)
        assert str(arrays["score_type"]) == "cosine_similarity"
        assert str(arrays["representation_type"]) == "clap_audio_embedding"
        assert arrays["representation"].shape == (5, FakeCLAP.dimension)
    cost = benchmark_clap(system, toy_dataset / "audio" / clips[0].filename, runs=2, warmup=1)
    assert cost["inference_median_ms"] > 0 and cost["text_embeddings_precomputed"]
    mismatched = CLAPSystem(FakeCLAP(), names, FeatureConfig(sample_rate=16000))
    assert mismatched.input_rate == 16000
    with pytest.raises(ValueError, match="sample rate"):
        evaluate_manifest(manifest, tmp_path / "eval2", extra_systems=[mismatched])
    assert json.loads((tmp_path / "eval" / "summary.json").read_text())["rows_evaluated"] == 5


def test_tensor_backend_avoids_second_quantisation():
    torch = pytest.importorskip("torch")

    class OfficialAPIStub:
        def get_audio_embedding_from_data(self, x, use_tensor=False):
            assert use_tensor is True
            assert torch.is_inference_mode_enabled()
            self.input = x.clone()
            return torch.ones((len(x), 4))

    official = OfficialAPIStub()
    backend = FrozenLAIONBackend(official, "cpu", "test", "HTSAT-tiny", "roberta")
    prepared = int16_round_trip(np.linspace(-0.9, 0.9, 100))[None, :]
    output = backend.get_audio_embedding_from_data(prepared)
    assert np.array_equal(official.input.numpy(), prepared)
    assert output.shape == (1, 4)


def test_batched_evaluation_preserves_rows_scores_controls_and_text(toy_dataset, tmp_path):
    clips = load_metadata(toy_dataset, require_complete=False)
    names = {c.target: c.category for c in clips}
    manifest = tmp_path / "manifest"
    build_manifest(
        toy_dataset,
        manifest,
        InterferenceConfig(conditions=("white",)),
        require_complete=False,
        limit_per_fold=1,
    )
    for size in (1, 4):
        evaluate_manifest(
            manifest,
            tmp_path / f"eval_{size}",
            extra_systems=[CLAPSystem(FakeCLAP(), names)],
            batch_size=size,
        )
    with (
        np.load(tmp_path / "eval_1/clap_scores.npz") as serial,
        np.load(tmp_path / "eval_4/clap_scores.npz") as batch,
    ):
        for field in (
            "scores",
            "control_scores",
            "representation",
            "control_representation",
            "text_embeddings",
        ):
            assert np.allclose(serial[field], batch[field], rtol=0, atol=1e-12)
        assert np.array_equal(serial["row"], batch["row"])
        assert np.array_equal(serial["prompts"], batch["prompts"])


def test_cli_forwards_clap_device_and_batch_size(toy_dataset, tmp_path, monkeypatch, capsys):
    from esc_robustness import cli

    captured = {}

    def loader(checkpoint, data_root, prompt, device="cpu", feature_config=None):
        captured.update(device=device, prompt=prompt)
        clips = load_metadata(data_root, require_complete=False)
        return CLAPSystem(FakeCLAP(), {c.target: c.category for c in clips})

    monkeypatch.setattr(cli, "load_clap_system", loader)
    manifest = tmp_path / "manifest"
    build_manifest(
        toy_dataset,
        manifest,
        InterferenceConfig(conditions=("white",)),
        require_complete=False,
        limit_per_fold=1,
    )
    cli.main(
        [
            "evaluate-interference",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "run"),
            "--clap-checkpoint",
            "unused.pt",
            "--clap-device",
            "mps",
            "--batch-size",
            "4",
        ]
    )
    assert captured["device"] == "mps"
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["batch_size_mixtures"] == 4
    capsys.readouterr()

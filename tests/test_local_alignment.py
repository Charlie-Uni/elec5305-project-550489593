import csv
import json

import numpy as np
import pytest
import soundfile as sf
from conftest import TOY_LENGTH, TOY_RATE, make_tone
from test_clap_system import FakeCLAP

from esc_robustness.clap_system import CLAPSystem, evaluate_clean, unit_rows
from esc_robustness.data import load_metadata
from esc_robustness.interference_eval import evaluate_manifest
from esc_robustness.local_alignment import (
    FIXED_RULE,
    READOUTS,
    TEMPERATURES,
    LocalAlignmentSystem,
    _rank_of_target,
    alignment_components,
    analyse,
    apply_selection,
    benchmark_alignment,
    calibration_mask,
    compare_clean_run,
    fixed_choice,
    fused_scores,
    fusion_grid,
    grid_correct,
    load_cached_ensemble,
    pack_components,
    paired_change,
    select_cross_fold,
    unpack_components,
    verify_reference,
)
from esc_robustness.mixing import InterferenceConfig, build_manifest


@pytest.fixture
def twelve_class_dataset(tmp_path):
    """Contiguous class IDs 0-11, so two coarse categories exist for semantic pairing."""
    root = tmp_path / "toy_twelve"
    (root / "audio").mkdir(parents=True)
    (root / "meta").mkdir()
    rng = np.random.default_rng(12)
    rows = []
    for fold in range(1, 6):
        for target in range(12):
            source = f"{fold}{target:02d}"
            name = f"{fold}-{source}-A-{target}.wav"
            x = make_tone(400 + 250 * target) + rng.normal(0, 1e-3, TOY_LENGTH)
            sf.write(root / "audio" / name, x, TOY_RATE, subtype="FLOAT")
            rows.append([name, fold, target, f"class_{target}", source])
    with (root / "meta" / "esc50.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "fold", "target", "category", "src_file"])
        writer.writerows(rows)
    return root


def released_reference(audio_global, audio_local, feat_t):
    """Independent transcription of the released per-clip loop (inference_zs.py @ 8910ebb)."""
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    feat_t = torch.tensor(feat_t).T  # (d, K), as in the release
    logits1, logits2 = [], []
    for g, local in zip(torch.tensor(audio_global), torch.tensor(audio_local)):
        feat_a = local.permute(1, 0)  # (d, c): the release permutes (c, d) frames
        a_weight = torch.matmul(feat_a.permute(1, 0), feat_t) * 2
        weight1 = functional.softmax(a_weight, dim=0)
        weight2 = functional.softmax(a_weight, dim=1)
        feat_t_a = torch.matmul(feat_a, weight1)
        feat_a_a = torch.matmul(weight2, feat_t.permute(1, 0))
        feat_a_a = feat_a_a.mean(0) + feat_a_a.max(0)[0]
        logits1.append(g @ feat_t_a)
        logits2.append(feat_a_a @ feat_t)
    return torch.stack(logits1).numpy(), torch.stack(logits2).numpy()


def test_components_reproduce_the_released_arithmetic():
    rng = np.random.default_rng(0)
    pooled, frames = rng.normal(size=(3, 6)), rng.normal(size=(3, 7, 6))
    text = unit_rows(rng.normal(size=(4, 6)))
    expected_text, expected_audio = released_reference(pooled, frames, text)
    text_guided, audio_guided = alignment_components(pooled, frames, text, temperature=2)
    np.testing.assert_allclose(text_guided, expected_text, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(audio_guided, expected_audio, rtol=1e-10, atol=1e-12)
    # Exactly repeated frames change nothing, so collapsing them is not a modelling choice.
    doubled = alignment_components(pooled, np.repeat(frames, 3, axis=1), text, 2)
    np.testing.assert_allclose(doubled[0], text_guided)
    np.testing.assert_allclose(doubled[1], audio_guided)
    with pytest.raises(ValueError):
        alignment_components(pooled, frames[:, :, :5], text, 2)


def test_grid_starts_at_the_ensemble_and_packing_round_trips():
    grid = fusion_grid()
    assert grid[0] == (0, 0.0, 0.0) and len(set(grid)) == len(grid)
    assert len(grid) == 1 + (9 * 9 - 1) * len(TEMPERATURES)
    assert fixed_choice(TEMPERATURES) == (2, FIXED_RULE["beta_text"], FIXED_RULE["beta_audio"])
    with pytest.raises(ValueError):
        fusion_grid(betas=(0.1, 0.2))
    with pytest.raises(ValueError):
        fixed_choice((1.0, 2.0))
    rng = np.random.default_rng(1)
    pooled = rng.normal(size=(5, 8))
    text_guided = [rng.normal(size=(5, 3)) for _ in range(2)]
    audio_guided = [rng.normal(size=(5, 3)) for _ in range(2)]
    packed = pack_components(pooled, text_guided, audio_guided)
    assert packed.dtype == np.float32 and packed.shape == (5, 8 + 2 * 2 * 3)
    back = unpack_components(packed, 3, (16.0, 64.0))
    np.testing.assert_allclose(back[0], pooled, atol=1e-6)
    np.testing.assert_allclose(back[1][:, 1], text_guided[1], atol=1e-6)
    np.testing.assert_allclose(back[2][:, 0], audio_guided[0], atol=1e-6)
    base = rng.normal(size=(5, 3))
    np.testing.assert_array_equal(fused_scores(base, back[1], back[2], grid[0]), base)


def test_target_rank_breaks_ties_consistently_with_argmax():
    scores = np.array([[0.5, 0.5, 0.5], [0.8, 0.2, 0.8], [0.2, 0.9, 0.1]])
    classes = np.arange(3)
    np.testing.assert_array_equal(_rank_of_target(scores, classes, np.array([2, 2, 1])), [3, 2, 1])
    predictions = classes[scores.argmax(1)]
    np.testing.assert_array_equal(_rank_of_target(scores, classes, predictions), [1, 1, 1])


def test_calibration_ignores_the_test_fold_and_shared_sources():
    rng = np.random.default_rng(2)
    classes, per_fold = np.arange(3), 12
    folds = np.repeat(np.arange(1, 6), per_fold)
    targets = rng.integers(0, 3, size=len(folds))
    sources = np.array([f"s{i}" for i in range(len(folds))])
    sources[per_fold] = sources[0]  # one fold-2 clip shares its source with fold 1
    base = rng.normal(scale=0.1, size=(len(folds), 3))
    helpful = np.eye(3)[targets]  # the text-guided branch knows the answer
    text_guided = np.stack([helpful, helpful], axis=1)
    audio_guided = np.zeros_like(text_guided)
    grid = fusion_grid((16.0, 64.0), (0.0, 0.5, 2.0))
    assert calibration_mask(folds, sources, 1).sum() == 4 * per_fold - 1
    assert calibration_mask(folds, sources, 3).sum() == 4 * per_fold
    selection = select_cross_fold(
        base, text_guided, audio_guided, targets, folds, sources, classes, grid
    )
    assert selection[1]["excluded_shared_source_clips"] == 1
    assert all(s["choice"][1] > 0 and s["choice"][2] == 0 for s in selection.values())
    # Changing only the test fold's labels and scores cannot change that fold's rule.
    changed_targets, changed_base = targets.copy(), base.copy()
    changed_targets[folds == 1] = (targets[folds == 1] + 1) % 3
    changed_base[folds == 1] += 5.0
    again = select_cross_fold(
        changed_base, text_guided, audio_guided, changed_targets, folds, sources, classes, grid
    )
    assert again[1] == selection[1]
    # Uninformative branches leave the ensemble selected through the tie rule.
    flat = np.zeros_like(text_guided)
    none = select_cross_fold(base, flat, flat, targets, folds, sources, classes, grid)
    assert all(s["choice"] == (0, 0.0, 0.0) for s in none.values())
    scores = apply_selection(base, text_guided, audio_guided, folds, selection)
    assert np.all(classes[scores.argmax(1)] == targets)
    assert grid_correct(base, text_guided, audio_guided, targets, classes, grid)[0] < len(folds)
    with pytest.raises(ValueError):
        apply_selection(base, text_guided, audio_guided, folds, {1: selection[1]})
    change = paired_change(classes[base.argmax(1)], classes[scores.argmax(1)], targets)
    assert change["lost"] == 0 and change["gained"] > 0 and change["exact_mcnemar_p"] < 0.05
    assert change["accuracy_difference_pp"] == pytest.approx(100 * change["gained"] / len(folds))


def test_frame_hook_uses_one_official_pass_and_checks_identities():
    torch = pytest.importorskip("torch")
    from esc_robustness.local_alignment import audio_frames, collapse_repeated_frames

    frames = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    repeated = frames[:, :, None, :].repeat(1, 1, 5, 1).reshape(2, 15, 4)
    assert torch.equal(collapse_repeated_frames(repeated), frames)
    assert torch.equal(collapse_repeated_frames(frames), frames)

    class Tower(torch.nn.Module):
        def __init__(self, consistent=True):
            super().__init__()
            self.proj = torch.nn.Linear(16, 6)
            self.consistent = consistent

        def forward(self, x):
            fine = torch.tanh(self.proj(x.reshape(len(x), 4, 16)))  # (B, 4 frames, 6)
            pooled = fine.mean(dim=1) if self.consistent else fine[:, 0]
            repeated = fine[:, :, None, :].repeat(1, 1, 8, 1).reshape(len(x), 32, 6)
            return {"embedding": pooled, "fine_grained_embedding": repeated}

    class Model(torch.nn.Module):
        def __init__(self, consistent=True):
            super().__init__()
            self.audio_branch = Tower(consistent)
            self.audio_projection = torch.nn.Sequential(
                torch.nn.Linear(6, 5), torch.nn.ReLU(), torch.nn.Linear(5, 5)
            )

    class Official:
        def __init__(self, consistent=True):
            torch.manual_seed(3)
            self.model = Model(consistent).eval()

        def get_audio_embedding_from_data(self, x, use_tensor=False):
            assert use_tensor and x.dtype == torch.float32
            latent = self.model.audio_branch(x)["embedding"]
            return torch.nn.functional.normalize(self.model.audio_projection(latent), dim=-1)

    class Backend:
        def __init__(self, consistent=True):
            self.module = Official(consistent)

    x = np.random.default_rng(4).normal(size=(3, 64)).astype(np.float32)
    backend = Backend()
    pooled, local = audio_frames(backend, x)
    assert pooled.shape == (3, 5) and local.shape == (3, 4, 5)
    np.testing.assert_allclose(np.linalg.norm(local, axis=-1), 1, atol=1e-6)
    expected = backend.module.get_audio_embedding_from_data(torch.from_numpy(x), use_tensor=True)
    np.testing.assert_allclose(pooled, expected.detach().numpy(), atol=1e-6)
    assert not backend.module.model.audio_branch._forward_hooks  # hook always removed
    with pytest.raises(RuntimeError, match="average"):
        audio_frames(Backend(consistent=False), x)


def fake_frames(module, prepared):
    """Four frame embeddings per clip whose mean direction is the fake pooled embedding."""
    pooled = unit_rows(module.get_audio_embedding_from_data(x=prepared, use_tensor=False))
    chunks = np.array_split(np.asarray(prepared), 4, axis=1)
    frames = np.stack(
        [module.get_audio_embedding_from_data(x=np.ascontiguousarray(c)) for c in chunks], axis=1
    )
    return pooled, unit_rows(frames)


def test_identical_input_cache_preserves_scores_and_skips_reencoding():
    names = {0: "first", 1: "second"}
    text = np.random.default_rng(19).normal(size=(2, FakeCLAP.dimension))
    calls = []

    def counted(module, prepared):
        calls.append(len(prepared))
        return fake_frames(module, prepared)

    cached = LocalAlignmentSystem(
        FakeCLAP(), names, text, frame_source=counted, cache_encodings=True
    )
    plain = LocalAlignmentSystem(FakeCLAP(), names, text, frame_source=fake_frames)
    first = np.linspace(-0.05, 0.05, 128)
    second = np.linspace(0.02, 0.03, 128)
    cached.evaluate_batch([1, 1], [first, first])
    actual = cached.evaluate_batch([1, 1, 1], [second, first, second])
    expected = plain.evaluate_batch([1, 1, 1], [second, first, second])
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-7)
    assert calls == [1, 1]
    assert cached.provenance()["encoding_cache"]["unique_inputs"] == 2
    assert cached.provenance()["encoding_cache"]["cache_hits"] == 3


def test_text_cache_rejects_changed_model_templates_or_class_order(tmp_path):
    from esc_robustness.artifacts import sha256_file

    templates = tmp_path / "templates.txt"
    templates.write_text("the sound of <sound>\na recording of <sound>\n")
    names = {0: "first", 1: "second"}
    per_prompt = unit_rows(np.random.default_rng(20).normal(size=(2, 2, 5)))
    mean = unit_rows(per_prompt.mean(1))
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "text_embeddings.npz"
    np.savez_compressed(
        path,
        ensemble=mean,
        per_prompt=per_prompt,
        templates=np.array(templates.read_text().splitlines()),
        class_names=np.array(list(names.values())),
    )
    (cache / "provenance.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": "fixed-model",
                "templates_sha256": sha256_file(templates),
                "text_embeddings_sha256": sha256_file(path),
            }
        )
    )
    np.testing.assert_array_equal(load_cached_ensemble(cache, "fixed-model", templates, names), mean)
    with pytest.raises(ValueError, match="checkpoint"):
        load_cached_ensemble(cache, "different-model", templates, names)
    with pytest.raises(ValueError, match="class order"):
        load_cached_ensemble(cache, "fixed-model", templates, {0: "second", 1: "first"})
    templates.write_text("changed <sound>\n")
    with pytest.raises(ValueError, match="templates"):
        load_cached_ensemble(cache, "fixed-model", templates, names)


def test_system_run_and_analysis_on_toy_data(twelve_class_dataset, tmp_path):
    toy_dataset = twelve_class_dataset
    clips = load_metadata(toy_dataset, require_complete=False)
    names = {c.target: c.category for c in clips}
    backend = FakeCLAP()
    ensemble = np.random.default_rng(5).normal(size=(12, FakeCLAP.dimension))
    system = LocalAlignmentSystem(backend, names, ensemble, frame_source=fake_frames)
    assert system.name == "clap_local" and "calibrated" in system.provenance()["not_zero_shot"]
    with pytest.raises(ValueError):
        LocalAlignmentSystem(backend, names, ensemble[:1], frame_source=fake_frames)
    run = tmp_path / "run"
    run.mkdir()
    evaluate_clean(system, toy_dataset, run / "clean", require_complete=False, batch_size=3)
    manifest = tmp_path / "manifest"
    build_manifest(
        toy_dataset,
        manifest,
        InterferenceConfig(conditions=("white", "semantic_different"), levels_db=(5.0, 0.0)),
        require_complete=False,
        limit_per_fold=2,
    )
    evaluate_manifest(manifest, run / "eval_own", extra_systems=[system], batch_size=2)
    np.savez_compressed(
        run / "text_embeddings.npz",
        ensemble=system.text_embeddings,
        single=system.single_text_embeddings,
        temperatures=np.array(system.temperatures),
    )
    np.savez_compressed(
        run / "clean_sources.npz",
        filenames=np.array([c.filename for c in clips]),
        source_ids=np.array([c.source_id for c in clips]),
    )
    summary = analyse(run, regions=("own", "target"))  # the absent region is skipped
    with (run / "conditions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    cells = {(r["region"], r["condition"], r["level_db"]) for r in rows}
    assert len(rows) == len(cells) * len(READOUTS) == summary["cells"]
    assert ("clean", "clean", "") in cells and ("own", "semantic_different", "5") in cells
    assert not any("label" in r["readout"] for r in rows)
    # The ensemble readout must reproduce the standard prediction file of the same run.
    with (run / "eval_own" / "clap_local_predictions.csv").open(newline="") as handle:
        standard = list(csv.DictReader(handle))
    for condition, level in {(r["condition"], r["level_db"]) for r in standard}:
        subset = [r for r in standard if (r["condition"], r["level_db"]) == (condition, level)]
        cell = next(
            r
            for r in rows
            if r["readout"] == "ensemble"
            and (r["region"], r["condition"]) == ("own", condition)
            and float(r["level_db"]) == float(level)
        )
        assert int(cell["correct"]) == sum(int(r["correct"]) for r in subset)
        assert int(cell["control_correct"]) == sum(int(r["control_correct"]) for r in subset)
        if condition.startswith("semantic"):
            captured = sum(r["predicted_interferer"] == "1" for r in subset)
            assert int(cell["captures"]) == captured
        else:
            assert cell["captures"] == ""
    selection = json.loads((run / "selection.json").read_text())
    assert selection["fixed_rule"] == FIXED_RULE and set(selection["folds"]) == set("12345")
    assert all(f["calibration_clips"] == 48 for f in selection["folds"].values())
    with (run / "label_selected_reference.csv").open(newline="") as handle:
        reference = list(csv.DictReader(handle))
    assert len(reference) == len(cells)
    assert all(int(r["correct"]) >= int(r["ensemble_correct"]) for r in reference)
    with np.load(run / "eval_own" / "fused_scores.npz") as fused:
        assert "not a cosine" in str(fused["score_type"])
        assert np.isfinite(fused["mixture_aligned_calibrated"]).all()
        assert fused["control_aligned_fixed"].shape == (len(standard), 12)
    assert set(summary["paired_against_ensemble"]["own/white@+5"]) == {
        "aligned_fixed",
        "aligned_calibrated",
    }
    assert summary["primary_test_family"]["tests"] == 4
    for result in summary["paired_against_ensemble"]["own/semantic_different@+5"].values():
        assert result["primary_holm_p"] >= result["exact_mcnemar_p"]
    assert (
        "capture_difference"
        in (summary["paired_against_ensemble"]["own/semantic_different@+0"]["aligned_fixed"])
    )
    cost = benchmark_alignment(system, toy_dataset / "audio" / clips[0].filename, runs=2, warmup=1)
    assert cost["aligned_readout_median_ms"] > 0 and cost["pooled_ensemble_readout_median_ms"] > 0


def test_reference_check_detects_a_changed_extraction_path(toy_dataset, tmp_path, monkeypatch):
    clips = load_metadata(toy_dataset, require_complete=False)
    names = {c.target: c.category for c in clips}
    ensemble = np.random.default_rng(6).normal(size=(2, FakeCLAP.dimension))
    from esc_robustness import local_alignment

    monkeypatch.setattr(
        local_alignment,
        "load_metadata",
        lambda root: load_metadata(root, require_complete=False),
    )
    monkeypatch.setattr(
        local_alignment,
        "load_audio",
        lambda path, config, complete: sf.read(path, dtype="float64")[0],
    )
    single = CLAPSystem(FakeCLAP(), names)
    evaluate_clean(single, toy_dataset, tmp_path / "single", require_complete=False)
    system = LocalAlignmentSystem(FakeCLAP(), names, ensemble, frame_source=fake_frames)
    good = verify_reference(system, toy_dataset, tmp_path / "single", clips=7)
    assert good["passed"] and good["clips_checked"] == 7
    assert good["pooled_embedding_max_abs_difference"] < 1e-9

    run = tmp_path / "run"
    run.mkdir()
    evaluate_clean(system, toy_dataset, run / "clean", require_complete=False)
    np.savez_compressed(
        run / "text_embeddings.npz",
        single=system.single_text_embeddings,
        temperatures=np.array(system.temperatures),
    )
    full = compare_clean_run(run, tmp_path / "single")
    assert full["passed"] and full["clips_checked"] == len(clips)
    with np.load(tmp_path / "single" / "decision_scores.npz") as f:
        altered = dict(f)
    altered["embeddings"] = unit_rows(altered["embeddings"] + 0.01)
    (tmp_path / "altered").mkdir()
    np.savez_compressed(tmp_path / "altered" / "decision_scores.npz", **altered)
    assert not compare_clean_run(run, tmp_path / "altered")["passed"]

    def shifted(module, prepared):
        pooled, frames = fake_frames(module, prepared)
        return unit_rows(pooled + 0.01), frames

    broken = LocalAlignmentSystem(FakeCLAP(), names, ensemble, frame_source=shifted)
    assert not verify_reference(broken, toy_dataset, tmp_path / "single")["passed"]

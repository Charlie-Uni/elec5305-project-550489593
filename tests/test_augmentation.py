import csv

import numpy as np
import pytest

from esc_robustness.augmentation import AugmentationBanks, BatchAugmenter
from esc_robustness.data import outer_split
from esc_robustness.spectrograms import extract_spectrograms
from esc_robustness.svm import fit_outer_fold


def test_batch_augmentation_cannot_touch_held_out_rows_and_is_deterministic():
    bank = np.full((2, 3, 4, 5), 7, dtype=np.float32)
    a = BatchAugmenter(bank, [2, 5], 10, {}, probability=1)
    clean = np.zeros((2, 4, 5), dtype=np.float32)
    assert np.all(a([2, 5], clean) == 7)
    assert np.all(clean == 0)
    with pytest.raises(ValueError, match="outside"):
        a([2, 6], clean)
    b = BatchAugmenter(bank, [2, 5], 10, {}, probability=0)
    assert np.all(b([2, 5], clean) == 0)


def test_semantic_bank_uses_only_current_training_partition(interference_dataset, tmp_path):
    cache = tmp_path / "mel.npz"
    extract_spectrograms(interference_dataset, cache, require_complete=False)
    builder = AugmentationBanks(
        cache, interference_dataset, tmp_path / "banks", "semantic", require_complete=False
    )
    train, validation, test = outer_split(builder.cache["folds"], 1)
    augmented = builder(1, "selection", train)
    manifest = tmp_path / "banks/semantic/fold_1_selection/manifest.csv"
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(train) * 3
    assert {int(r["target_fold"]) for r in rows} == {3, 4, 5}
    assert {int(r["interferer_fold"]) for r in rows} <= {3, 4, 5}
    assert all(r["target_class"] != r["interferer_class"] for r in rows)
    assert all(abs(float(r["achieved_db"]) - float(r["level_db"])) < 1e-8 for r in rows)
    assert augmented.bank.shape[:2] == (len(train), 3)
    with pytest.raises(ValueError, match="partition"):
        builder(1, "selection", np.concatenate((train, validation)))
    with pytest.raises(ValueError, match="outside"):
        augmented(test[:1], np.zeros((1, *augmented.bank.shape[2:])))
    clean = np.zeros((len(builder.cache["y"]), 92))
    combined, labels, _ = builder.training_features(
        clean, builder.cache["y"], train, 1, "selection"
    )
    assert combined.shape == (2 * len(train), 92)
    np.testing.assert_array_equal(labels, np.tile(builder.cache["y"][train], 2))


def test_augmented_svm_does_not_fit_or_select_on_test_examples():
    rng = np.random.default_rng(5305)
    folds = np.repeat(np.arange(1, 6), 8)
    y = np.tile(np.repeat([0, 1], 4), 5)
    X = rng.normal(size=(40, 3)) + y[:, None] * 3
    calls = []

    class FakeBanks:
        def training_features(self, clean, labels, indices, fold, phase):
            train, validation, _ = outer_split(folds, fold)
            expected = train if phase == "selection" else np.concatenate((train, validation))
            np.testing.assert_array_equal(indices, expected)
            calls.append(phase)
            return (
                np.concatenate((clean[indices], clean[indices] + 0.25)),
                np.tile(labels[indices], 2),
                {},
            )

    grid = {"svc__C": [1.0], "svc__gamma": ["scale"]}
    model, result, _, _, _ = fit_outer_fold(X, y, folds, 1, grid, FakeBanks())
    X[folds == 1] += 1e6
    other, result2, _, _, _ = fit_outer_fold(X, y, folds, 1, grid, FakeBanks())
    np.testing.assert_array_equal(
        model.named_steps["scale"].mean_, other.named_steps["scale"].mean_
    )
    np.testing.assert_array_equal(
        model.named_steps["svc"].support_vectors_, other.named_steps["svc"].support_vectors_
    )
    assert result["candidate_results"] == result2["candidate_results"]
    assert calls == ["selection", "refit", "selection", "refit"]

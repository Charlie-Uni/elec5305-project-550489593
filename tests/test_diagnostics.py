import numpy as np
import pytest

from esc_robustness.diagnostics import (
    f1_from_counts,
    feature_group_indices,
    holm_adjust,
    paired_bootstrap,
    score_ranks,
)
from esc_robustness.features import feature_names


def test_paired_bootstrap_identical_predictions_are_exactly_zero():
    y = np.tile([0, 1, 2, 0], 5)
    pred = np.tile([0, 2, 2, 0], 5)
    result = paired_bootstrap(
        y, pred, pred, np.repeat(np.arange(1, 6), 4), replicates=200, n_classes=3
    )
    for key in (
        "macro_f1_difference",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
        "accuracy_difference",
        "accuracy_ci_low",
        "accuracy_ci_high",
    ):
        assert result[key] == 0
    assert result["mcnemar_exact_p"] == 1


def test_bootstrap_difference_sign_and_holm_correction():
    y = np.tile([0, 1], 10)
    result = paired_bootstrap(y, y, 1 - y, np.repeat([1, 2], 10), replicates=200, n_classes=2)
    assert result["macro_f1_difference"] == 1
    assert result["accuracy_ci_low"] == result["accuracy_ci_high"] == 1
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])


def test_groups_partition_all_92_coordinates_and_ranks_handle_ties():
    groups = feature_group_indices(feature_names())
    assert sorted(np.concatenate(list(groups.values())).tolist()) == list(range(92))
    assert len(groups["mfcc"]) == len(groups["delta_mfcc"]) == 40
    scores = np.array([[0.5, 0.5, 0.1], [0.1, 0.8, 0.2]])
    assert score_ranks(scores, np.array([1, 0])).tolist() == [1, 3]
    assert f1_from_counts(np.array([0, 1]), np.array([0, 1]), n_classes=3) == pytest.approx(2 / 3)

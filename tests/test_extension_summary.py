from pathlib import Path

import pytest

from esc_robustness.extension_summary import comparison_specs, matched_vectors, variant_paths


def test_intervention_comparison_keeps_targets_and_mixtures_paired():
    a = [
        dict(
            target_file="a",
            fold=1,
            target_class=0,
            prediction=0,
            condition="white",
            level_db=5.0,
            activity_overlap=1.0,
            interferer="white",
            common_gain=0.5,
        ),
        dict(
            target_file="b",
            fold=2,
            target_class=1,
            prediction=0,
            condition="white",
            level_db=5.0,
            activity_overlap=1.0,
            interferer="white",
            common_gain=0.8,
        ),
    ]
    b = [dict(a[1], prediction=1), dict(a[0], prediction=1)]
    y, aa, bb, folds = matched_vectors(a, b, "white")
    assert y.tolist() == [0, 1] and aa.tolist() == [0, 0] and bb.tolist() == [1, 1]
    assert folds.tolist() == [1, 2]
    with pytest.raises(ValueError, match="targets differ"):
        matched_vectors(a, b[:1], "white")
    with pytest.raises(ValueError, match="mixture identity"):
        matched_vectors(a, [dict(b[0], common_gain=0.9), b[1]], "white")


def test_semantic_wgn_contrasts_keep_optimisation_seed_matched():
    contrasts = list(comparison_specs(variant_paths(Path("repo"), Path("runtime"), Path("run"))))
    direct = [r for r in contrasts if r[-1] == "semantic_vs_wgn"]
    assert len(contrasts) == 13 and len(direct) == 4
    assert {(r[0], r[1]) for r in direct} == {
        ("svm_semantic", "svm_wgn"),
        *((f"cnn_semantic_seed_{seed}", f"cnn_wgn_seed_{seed}") for seed in (5305, 5306, 5307)),
    }
    assert len(contrasts) * 2 * 6 == 156

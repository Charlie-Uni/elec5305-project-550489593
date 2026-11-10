"""One audited numerical source for slides, narration and the public project page."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def local_alignment_claims():
    """Return the additional alignment arm only after independent full-run verification."""
    base = ROOT / "results/extensions_20261111/clap_local_alignment"
    audit_file = base / "independent_audit.json"
    if not audit_file.exists():
        return None
    audit = json.loads(audit_file.read_text())
    if not audit["verified"] or audit["condition_rows_checked"] != 100:
        raise ValueError("Local alignment requires a verified full run")
    for name, field in (
        ("conditions.csv", "conditions_sha256"),
        ("primary_comparisons.csv", "primary_comparisons_sha256"),
        ("trial_evidence.csv.gz", "trial_evidence_sha256"),
        ("setup_cost.json", "setup_cost_sha256"),
    ):
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != audit[field]:
            raise ValueError(f"Audited local alignment input changed: {name}")
    selection = json.loads((base / "selection.json").read_text())
    result = {
        "name": "PAT-style local alignment on LAION-CLAP",
        "readout_names": {
            "single": "Single prompt",
            "ensemble": "Fixed prompt ensemble",
            "aligned_fixed": "Local alignment / preset",
            "aligned_calibrated": "Local alignment / calibrated",
        },
        "labels_used": {
            "single": "none", "ensemble": "none", "aligned_fixed": "none",
            "aligned_calibrated": "clean labels from non-test folds, shared sources excluded",
        },
        "conditions": read_csv(base / "conditions.csv"),
        "primary": read_csv(base / "primary_comparisons.csv"),
        "selection": selection,
        "cost": json.loads((base / "cost.json").read_text()),
        "encoding_cache": json.loads((base / "encoding_cache.json").read_text()),
        "setup_cost": json.loads((base / "setup_cost.json").read_text()),
        "author_listening": json.loads((base / "author_listening_confirmation.json").read_text()),
        "source_sha256": {
            name: hashlib.sha256((base / name).read_bytes()).hexdigest()
            for name in ("conditions.csv", "primary_comparisons.csv", "selection.json", "cost.json")
        },
        "interpretation": (
            "A project adaptation of released feedback arithmetic, not official or full PAT. "
            "Preset weights use no labels; calibrated weights use non-test clean labels. "
            "Four primary paired tests share Holm correction; cluster intervals are pointwise "
            "and condition on the fitted calibration rules and observed source pairs."
        ),
    }
    balanced = base / "cost_balanced"
    if (balanced / "audit.json").exists():
        cost_audit = json.loads((balanced / "audit.json").read_text())
        assert cost_audit["verified"] and cost_audit["queries"] == 4500
        for name, digest in cost_audit["output_sha256"].items():
            if hashlib.sha256((balanced / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Balanced alignment cost changed: {name}")
        trials = read_csv(balanced / "trials.csv")
        summaries = read_csv(balanced / "summary.csv")
        assert len(trials) == 4500 and len(summaries) == 6
        for row in summaries:
            group = [r for r in trials if r["readout"] == row["readout"] and r["order"] == row["order"]]
            assert len(group) == 750 and len({r["filename"] for r in group}) == 250
            for stage in ("preprocessing", "inference_scoring", "total"):
                value = statistics.median(float(r[stage + "_ms"]) for r in group)
                assert abs(value - float(row[stage + "_median_ms"])) < 1e-9
            p95 = statistics.quantiles([float(r["total_ms"]) for r in group], n=100, method="inclusive")[94]
            assert abs(p95 - float(row["total_p95_ms"])) < 1e-9
        assert sum(r["prediction"] != r["reference_prediction"] for r in trials) == cost_audit["prediction_disagreements"]
        assert max(float(r["max_abs_score_difference"]) for r in trials) <= 1e-4
        result["balanced_costs"] = summaries
        result["balanced_cost_audit"] = cost_audit
        result["balanced_costs_verified_from_trials"] = True
    return result


def editorial_claims():
    """Core diagnostics added to the concise report, with explicit denominators."""
    base = ROOT / "results/figures_three_systems_own_20261109"
    capture = [
        r for r in read_csv(base / "fig2_capture_rate_vs_sir.csv") if float(r["level_db"]) == 5
    ]
    assert len(capture) == 6
    for row in capture:
        assert int(row["n"]) == 2000
        assert (
            abs(
                float(row["capture_rate"])
                - int(row["errors_equal_distractor"]) / int(row["errors"])
            )
            < 1e-12
        )
    overlap = [
        r
        for r in read_csv(
            ROOT / "results/extensions_20261109/clean_replication/cnn_overlap_replication.csv"
        )
        if r["region"] == "own"
        and r["seed"] == "5305"
        and float(r["level_db"]) == 5
        and r["stratum"] == "overlap_ge_0.8"
    ]
    margin = [
        r
        for r in read_csv(base / "fig3_margin_vs_sir.csv")
        if r["system"] == "clap" and r["condition"] == "semantic_same" and float(r["level_db"]) == 5
    ]
    displacement = [
        r
        for r in read_csv(base / "fig5_representation_displacement.csv")
        if r["system"] == "clap" and r["condition"] == "semantic_same" and float(r["level_db"]) == 5
    ]
    assert len(overlap) == len(displacement) == 2 and len(margin) == 1
    return {
        "region": "own",
        "level_db": 5,
        "capture": capture,
        "cnn_high_overlap": overlap,
        "clap_same_margin": margin[0],
        "clap_same_displacement": displacement,
        "capture_denominator": "All incorrect predictions, pooled across the five folds",
        "interpretation": "High-overlap original-seed contrasts do not pass Holm correction; margin and displacement are descriptive associations.",
    }


def closing_claims():
    base = ROOT / "results/closing_20261110"
    audit = json.loads((base / "independent_audit.json").read_text())
    if not audit["verified"]:
        raise ValueError("Closing claims require the independent audit")
    for name, digest in audit["output_hashes"].items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Audited input changed: {name}")
    readout = read_csv(base / "readout_outcomes.csv")
    paired = read_csv(base / "acoustic_matched_comparisons.csv")
    medians = []
    for region in ("own", "target"):
        rows = read_csv(base / f"overlap_{region}/overlap_features.csv")
        for condition in ("white", "demand", "semantic_same", "semantic_different"):
            selected = [
                r for r in rows if r["condition"] == condition and float(r["level_db"]) == 5
            ]
            record = dict(region=region, condition=condition, n=len(selected))
            for metric in (
                "spectral_correlation_db",
                "density_spectral_correlation_db",
                "masked_cell_fraction",
            ):
                record[metric] = statistics.median(float(r[metric]) for r in selected)
            medians.append(record)
    return {
        "readout": readout,
        "acoustic_primary": [r for r in paired if r["stratum"] == "close_primary"],
        "acoustic_medians": medians,
        "acoustic_rows": {"own": 24000, "target": 23991},
        "demand_rows_per_protocol": 6000,
        "semantic_rows_per_variant": 23991,
        "readout_variants": 4,
        "bootstrap_replicates": 2000,
        "bootstrap_unit": "original target-source recording",
        "calipers": {
            "target_activity_spectral_cosine": 0.10,
            "masked_cell_fraction": 0.05,
            "activity_overlap": 0.10,
        },
        "interpretation": "Exploratory selected-population associations; ranked inclusion is not calibrated detection or proof of hearing; no semantic causal claim.",
        "sources_sha256": audit["output_hashes"],
    }

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

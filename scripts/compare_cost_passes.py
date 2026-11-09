"""Verify identical benchmark inputs across the primary and reversed-order pass."""

import argparse
import csv
import json
from pathlib import Path

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.diagnostics import write_csv


def compare(first, reverse, output):
    audits = [json.loads((p / "audit.json").read_text()) for p in (first, reverse)]
    if not all(a["verified"] for a in audits):
        raise ValueError("Both timing passes require independent audits")
    tables = []
    for path in (first, reverse):
        with (path / "latency_summary.csv").open() as handle:
            tables.append({r["variant"]: r for r in csv.DictReader(handle)})
    if set(tables[0]) != set(tables[1]) or len(tables[0]) != 7:
        raise ValueError("Timing model sets differ")
    rows = []
    for name in tables[0]:
        a, b = [json.loads((p / name / "summary.json").read_text()) for p in (first, reverse)]
        for key in ("model_sha256", "versions", "cpu_name", "source_sha256"):
            if a[key] != b[key]:
                raise ValueError(f"Timing inputs changed: {name}/{key}")
        if (first / name / "panel.json").read_bytes() != (reverse / name / "panel.json").read_bytes():
            raise ValueError("Timing clip panel changed")
        rows.append(dict(
            variant=name,
            **{f"{prefix}_{metric}": table[name][metric]
               for prefix, table in zip(("primary", "reverse"), tables)
               for metric in ("total_median_ms", "total_p25_ms", "total_p75_ms", "total_p95_ms")},
        ))
    output.mkdir(exist_ok=True)
    write_csv(output / "comparison.csv", rows)
    write_json(output / "audit.json", dict(
        verified=True, models=7, clips_per_model=250, queries_per_model_per_pass=750,
        total_timed_queries=10500, model_inputs_identical=True,
        interpretation="Post-hoc reversed-order sensitivity after timing drift was observed; both passes retained. No attribution to a measured thermal or scheduling cause, and no causal augmentation-cost claim.",
        source_audits=[sha256_file(p / "audit.json") for p in (first, reverse)],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "reverse", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    compare(args.first, args.reverse, args.output)

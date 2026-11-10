"""Verify the balanced CPU benchmark and export comparable total-query costs."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.diagnostics import write_csv
from esc_robustness.extension_summary import variant_paths

VARIANTS = (
    "svm_clean",
    "cnn_clean",
    "svm_wgn",
    "cnn_wgn",
    "svm_semantic",
    "cnn_semantic",
    "clap_baseline",
)


def audit(source, output, repo, runtime, run):
    output.mkdir(parents=True, exist_ok=True)
    rows, references, panels = [], {}, []
    clean_paths = {
        name: clean / "predictions.csv"
        for name, _, _, _, clean, _ in variant_paths(repo, runtime, run)
    }
    for variant in VARIANTS:
        directory = source / variant
        summary = json.loads((directory / "summary.json").read_text())
        panel = json.loads((directory / "panel.json").read_text())
        panels.append(panel)
        with (directory / "trials.csv").open() as handle:
            trials = list(csv.DictReader(handle))
        if len(panel) != 250 or len(trials) != 750 or summary["trials"] != 750:
            raise ValueError("Benchmark must cover 250 clips with three repetitions")
        if {(r["fold"], r["target"]) for r in panel} != {
            (f, c) for f in range(1, 6) for c in range(50)
        }:
            raise ValueError("Benchmark panel is not balanced")
        indexed = {r["filename"]: r for r in panel}
        prediction_variant = variant + "_seed_5305" if variant.startswith("cnn") else variant
        clean_path = clean_paths[prediction_variant]
        with clean_path.open() as handle:
            clean = {r["filename"]: r for r in csv.DictReader(handle)}
        if len(clean) != 2000:
            raise ValueError("Incomplete clean prediction reference for benchmark")
        seen, predictions = set(), {}
        for trial in trials:
            name, repeat = trial["filename"], int(trial["repetition"])
            if name not in indexed or repeat not in (0, 1, 2) or (name, repeat) in seen:
                raise ValueError("Invalid repeated benchmark trial")
            seen.add((name, repeat))
            original = indexed[name]
            if (int(trial["fold"]), int(trial["target"])) != (original["fold"], original["target"]):
                raise ValueError("Benchmark trial identity differs")
            if predictions.setdefault(name, trial["prediction"]) != trial["prediction"]:
                raise ValueError("Repeated deterministic predictions differ")
            reference = clean[name]
            if any(trial[key] != reference[key] for key in ("fold", "target", "prediction")):
                raise ValueError(f"Timed path disagrees with clean evaluation: {variant}/{name}")
            pp, inference, total = [
                float(trial[k]) for k in ("preprocessing_ms", "inference_scoring_ms", "total_ms")
            ]
            if not np.isfinite([pp, inference, total]).all() or min(pp, inference, total) <= 0:
                raise ValueError("Invalid latency")
            np.testing.assert_allclose(total, pp + inference, atol=1e-9, rtol=0)
        if summary["trials_sha256"] != sha256_file(directory / "trials.csv"):
            raise ValueError("Timed data changed")
        if (
            summary["numerical_threads"] != 1
            or summary["device"] != "cpu"
            or summary["batch_size"] != 1
        ):
            raise ValueError("CPU timing protocol differs")
        if any(pool["num_threads"] != 1 for pool in summary["observed_thread_pools"]):
            raise ValueError("Observed thread count differs")
        row = dict(
            variant=variant,
            clips=250,
            timed_trials=750,
            cpu=summary["cpu_name"],
            parameter_count=summary["parameter_count"],
        )
        for stage in ("preprocessing_ms", "inference_scoring_ms", "total_ms"):
            values = np.array([float(r[stage]) for r in trials])
            actual = np.quantile(values, [0.5, 0.25, 0.75, 0.95])
            for label, value in zip(("median", "p25", "p75", "p95"), actual):
                np.testing.assert_allclose(summary[stage][label], value, atol=1e-9, rtol=0)
                row[stage.removesuffix("_ms") + "_" + label + "_ms"] = float(value)
        sizes = summary["serialized_model_bytes"]
        row["serialized_bytes_per_fold_median"] = (
            float(np.median(list(sizes.values()))) if isinstance(sizes, dict) else sizes
        )
        rows.append(row)
        destination = output / variant
        destination.mkdir(exist_ok=True)
        for name in ("summary.json", "trials.csv", "panel.json"):
            shutil.copyfile(directory / name, destination / name)
        references[variant] = {
            "trials_sha256": sha256_file(directory / "trials.csv"),
            "summary_sha256": sha256_file(directory / "summary.json"),
            "model_sha256": summary["model_sha256"],
            "clean_predictions_sha256": sha256_file(clean_path),
            "timed_prediction_changes_from_clean": 0,
        }
    if any(panel != panels[0] for panel in panels[1:]) or len({row["cpu"] for row in rows}) != 1:
        raise ValueError("Model benchmarks use different panels or CPUs")
    write_csv(output / "latency_summary.csv", rows)
    result = dict(
        verified=True,
        models=7,
        clips_per_model=250,
        total_timed_queries=5250,
        predictions="All timed predictions match the corresponding official clean evaluation",
        total_cost="Direct median of total query time, not sum of stage medians",
        cnn_scope="Seed 5305 only for latency; all seeds retained for accuracy",
        clap_scope="Single-prompt query benchmark; ensemble uses cached 50-vector readout but is not independently timed here",
        caveats=[
            "Same Apple M3 CPU, model-specific pinned software environments",
            "One fixed model order, no claim of hardware-general latency",
            "Disk I/O, model loading and one-off prompt encoding excluded",
        ],
        provenance=references,
    )
    write_json(output / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("repo", "runtime", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.input, args.output, args.repo, args.runtime, args.run)

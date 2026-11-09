"""Recalculate intervention tables directly from per-trial CSVs with sklearn.

This verifier does not use the summary module's metric or pairing functions.
Bootstrap samples are not rerun: their bounds are checked for validity; point
estimates, discordance counts and exact tests are independently recomputed.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.extension_summary import variant_paths


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    np.testing.assert_allclose(float(a), float(b), atol=1e-12, rtol=0)


def metrics(rows, prediction="prediction", clean=False):
    target = "target" if clean else "target_class"
    values = {"accuracy": [], "macro_f1": []}
    for fold in range(1, 6):
        selected = [r for r in rows if int(r["fold"]) == fold]
        if not selected:
            raise ValueError("An official fold is missing")
        y, p = [[int(r[key]) for r in selected] for key in (target, prediction)]
        values["accuracy"].append(accuracy_score(y, p))
        values["macro_f1"].append(
            f1_score(y, p, labels=range(50), average="macro", zero_division=0)
        )
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in values.items()}


def select(rows, condition, level, overlap=False):
    selected = [
        r
        for r in rows
        if r["condition"] == condition
        and float(r["level_db"]) == level
        and (not overlap or float(r["activity_overlap"]) >= 0.8)
    ]
    indexed = {r["target_file"]: r for r in selected}
    if len(indexed) != len(selected) or not selected:
        raise ValueError("Empty or duplicate target pairing")
    return indexed


def check_pair(row, a, b):
    if set(a) != set(b) or len(a) != int(row["n"]):
        raise ValueError("Unmatched paired contrast")
    aa, bb, yy = [], [], []
    for name in sorted(a):
        if (a[name]["target_class"], a[name]["fold"]) != (b[name]["target_class"], b[name]["fold"]):
            raise ValueError("Paired trial identity differs")
        aa.append(int(a[name]["prediction"]))
        bb.append(int(b[name]["prediction"]))
        yy.append(int(a[name]["target_class"]))
    aa, bb, yy = map(np.asarray, (aa, bb, yy))
    ca, cb = aa == yy, bb == yy
    only_a, only_b = int(np.sum(ca & ~cb)), int(np.sum(cb & ~ca))
    if (only_a, only_b) != (int(row["a_only_correct"]), int(row["b_only_correct"])):
        raise ValueError("Discordance counts differ")
    close(row["accuracy_difference"], np.mean(ca) - np.mean(cb))
    close(
        row["macro_f1_difference"],
        metrics(list(a.values()))["macro_f1"][0] - metrics(list(b.values()))["macro_f1"][0],
    )
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    close(row["mcnemar_exact_p"], p)
    for metric in ("accuracy", "macro_f1"):
        lo, hi = (float(row[metric + "_ci_" + key]) for key in ("low", "high"))
        if not -1 <= lo <= hi <= 1 or int(row["replicates"]) != 2000:
            raise ValueError("Invalid bootstrap specification or bounds")
    return p


def check_holm(rows, pvalues, key):
    maximum = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        maximum = max(maximum, (len(pvalues) - rank) * pvalues[index])
        close(rows[index][key], min(1.0, maximum))


def audit(repo, runtime, root, directory):
    declared = json.loads((directory / "summary_audit.json").read_text())
    if not declared["complete"] or declared["variants"] != 14:
        raise ValueError("The complete declared study is required")
    clean_table = read(directory / "clean_variants.csv")
    conditions = read(directory / "condition_variants.csv")
    if len(clean_table) != 14 or len(conditions) != 336:
        raise ValueError("Unexpected summary dimensions")
    loaded, hashes = {}, {}
    for name, system, _, _, clean_dir, directories in variant_paths(repo, runtime, root):
        clean = read(clean_dir / "predictions.csv")
        clean_index = {r["filename"]: r for r in clean}
        if len(clean_index) != 2000 or any(
            sum(int(r["fold"]) == f for r in clean) != 400 for f in range(1, 6)
        ):
            raise ValueError("Incomplete or duplicated clean trials")
        calculated = metrics(clean, clean=True)
        entry = next(r for r in clean_table if r["variant"] == name)
        for metric, (mean, sd) in calculated.items():
            close(entry[metric], mean)
            close(entry[metric + "_fold_sd"], sd)
        for region, path in directories.items():
            origin = declared["provenance"][f"{region}/{name}"]
            if (
                sha256_file(path / f"{system}_predictions.csv") != origin["predictions_sha256"]
                or sha256_file(clean_dir / "predictions.csv") != origin["clean_sha256"]
            ):
                raise ValueError("Audited source hashes changed")
            hashes.setdefault(region, set()).add(origin["manifest_sha256"])
            rows = read(path / f"{system}_predictions.csv")
            if len(rows) != (24000 if region == "own" else 23991) or len(
                {r["row"] for r in rows}
            ) != len(rows):
                raise ValueError("Incomplete or duplicated interference rows")
            loaded[(region, name)] = rows
            for r in rows:
                c = clean_index[r["target_file"]]
                if (r["target_class"], r["fold"]) != (c["target"], c["fold"]):
                    raise ValueError("Clean/interference identity differs")
            for cell in (r for r in conditions if r["variant"] == name and r["region"] == region):
                indexed = select(rows, cell["condition"], float(cell["level_db"]))
                selected = list(indexed.values())
                if len(selected) != int(cell["n"]):
                    raise ValueError("Condition denominator differs")
                for metric, (mean, sd) in metrics(selected).items():
                    close(cell[metric], mean)
                    close(cell[metric + "_fold_sd"], sd)
                close(
                    cell["control_macro_f1"], metrics(selected, "control_prediction")["macro_f1"][0]
                )
                close(
                    cell["matched_clean_macro_f1"],
                    metrics([clean_index[n] for n in indexed], clean=True)["macro_f1"][0],
                )
                if cell["condition"].startswith("semantic"):
                    captures = sum(r["prediction"] == r["interferer_class"] for r in selected)
                    errors = sum(r["prediction"] != r["target_class"] for r in selected)
                    close(cell["capture_fraction"], captures / len(selected))
                    close(cell["capture_rate"], captures / errors)
    if any(len(v) != 1 for v in hashes.values()):
        raise ValueError("Different variants used different manifests")
    comparisons = read(directory / "paired_intervention_effects.csv")
    if len(comparisons) != 108:
        raise ValueError("Changed intervention test family")
    probabilities = []
    for row in comparisons:
        overlap = row["stratum"] == "overlap_ge_0.8"
        a, b = [
            select(loaded[(row["region"], row[key])], row["condition"], 5.0, overlap)
            for key in ("variant", "reference")
        ]
        for name in a:
            for key in (
                "interferer",
                "interferer_class",
                "level_db",
                "common_gain",
                "activity_overlap",
            ):
                if a[name][key] != b[name][key]:
                    raise ValueError("Intervention mixture changed")
        probabilities.append(check_pair(row, a, b))
    check_holm(comparisons, probabilities, "holm_p_across_extension_comparisons")
    replication = read(directory / "cnn_overlap_replication.csv")
    probabilities = []
    if len(replication) != 12:
        raise ValueError("Changed replication test family")
    for row in replication:
        rows = loaded[(row["region"], "cnn_clean_seed_" + row["seed"])]
        a = select(rows, row["condition"], 5.0, True)
        demand = select(rows, "demand", 5.0)
        probabilities.append(check_pair(row, a, {n: demand[n] for n in a}))
    check_holm(replication, probabilities, "holm_p_across_cnn_replication")
    aggregates = read(directory / "cnn_three_seed_summary.csv")
    if len(aggregates) != 72:
        raise ValueError("Missing CNN seed aggregates")
    for row in aggregates:
        matches = [
            r
            for r in conditions
            if r["system"] == "cnn"
            and all(r[k] == row[k] for k in ("region", "regime", "condition", "level_db"))
        ]
        if {r["seed"] for r in matches} != {"5305", "5306", "5307"}:
            raise ValueError("Seed selection or omission")
        for metric in ("accuracy", "macro_f1"):
            values = [float(r[metric]) for r in matches]
            close(row[metric + "_mean"], np.mean(values))
            close(row[metric + "_seed_sd"], np.std(values, ddof=1))
    report = dict(
        verified=True,
        clean_variants=14,
        condition_cells=336,
        paired_effects=108,
        replication_contrasts=12,
        cnn_seed_aggregates=72,
        method="Direct CSV/sklearn recomputation; independently calculated exact tests and Holm adjustment",
        bootstrap_scope="Bounds and point estimates checked; stochastic replicates not independently regenerated",
        source_summary_sha256=sha256_file(directory / "summary_audit.json"),
    )
    write_json(directory / "independent_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "runtime", "run", "input"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    audit(args.repo, args.runtime, args.run, args.input)

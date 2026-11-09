"""Summarise fixed intervention variants without selecting the best test seed."""

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import condition_table, fold_metrics, read_clean_predictions, read_predictions
from .artifacts import sha256_file, write_json
from .diagnostics import holm_adjust, paired_bootstrap, write_csv


def variant_paths(repo, runtime, root, trial_evidence=None):
    core = runtime / "runs/clap_v4_20261109"
    variants = []
    for system in ("svm", "cnn"):
        name = "cnn_clean_seed_5305" if system == "cnn" else "svm_clean"
        evaluation = {
            region: root / "evaluations" / region / name
            if system == "cnn"
            else repo / f"artifacts/interference/v3_20261109/eval_{region}"
            for region in ("own", "target")
        }
        variants.append(
            (
                name,
                system,
                "clean",
                5305 if system == "cnn" else None,
                repo / f"results/clean_{system}",
                evaluation,
            )
        )
    for seed in (5306, 5307):
        name = f"cnn_clean_seed_{seed}"
        variants.append(
            (
                name,
                "cnn",
                "clean",
                seed,
                root / f"clean_seed_{seed}",
                {region: root / "evaluations" / region / name for region in ("own", "target")},
            )
        )
    for mode in ("wgn", "semantic"):
        for system, seeds in (("svm", (None,)), ("cnn", (5305, 5306, 5307))):
            for seed in seeds:
                name = f"{system}_{mode}" + (f"_seed_{seed}" if seed else "")
                variants.append(
                    (
                        name,
                        system,
                        mode,
                        seed,
                        root / name,
                        {
                            region: root / "evaluations" / region / name
                            for region in ("own", "target")
                        },
                    )
                )
    for regime, base in (("baseline", core), ("prompt_ensemble", root / "clap_prompt_ensemble")):
        variants.append(
            (
                f"clap_{regime}",
                "clap",
                regime,
                None,
                base / "clean",
                {region: base / f"eval_{region}" for region in ("own", "target")},
            )
        )
    if trial_evidence is not None:
        evidence = Path(trial_evidence)
        return [
            (
                name, system, regime, seed, evidence / "clean" / name,
                {region: evidence / region / name for region in ("own", "target")},
            )
            for name, system, regime, seed, _, _ in variants
        ]
    return variants


def comparison_specs(variants):
    """Retain clean-reference contrasts and direct semantic-versus-WGN pairs."""
    for name, system, regime, seed, _, _ in variants:
        if regime in ("clean", "baseline"):
            continue
        reference = (
            f"cnn_clean_seed_{seed}"
            if system == "cnn"
            else ("svm_clean" if system == "svm" else "clap_baseline")
        )
        yield name, reference, system, regime, seed, "against_clean_reference"
        if regime == "semantic":
            reference = f"cnn_wgn_seed_{seed}" if system == "cnn" else "svm_wgn"
            yield name, reference, system, regime, seed, "semantic_vs_wgn"


def matched_vectors(a, b, condition, level=5.0, high_overlap=False):
    def index(rows):
        selected = [
            r
            for r in rows
            if r["condition"] == condition
            and r["level_db"] == level
            and (not high_overlap or float(r["activity_overlap"]) >= 0.8)
        ]
        result = {r["target_file"]: r for r in selected}
        if len(result) != len(selected):
            raise ValueError("Duplicate paired target")
        return result

    aa, bb = index(a), index(b)
    if set(aa) != set(bb) or not aa:
        raise ValueError("Intervention and baseline targets differ")
    names = sorted(aa)
    for name in names:
        if any(
            aa[name][key] != bb[name][key]
            for key in ("fold", "target_class", "interferer", "common_gain")
        ):
            raise ValueError("Intervention and baseline mixture identity differs")
    return tuple(
        np.array([aa[name][key] for name in names], dtype=int)
        for key in ("target_class", "prediction")
    ) + (
        np.array([bb[name]["prediction"] for name in names], dtype=int),
        np.array([aa[name]["fold"] for name in names], dtype=int),
    )


def replicate_overlap(loaded, replicates=2000):
    # Replicate the original overlap finding without treating three predictions
    # of one test recording as three independent observations.
    replication = []
    for region in ("own", "target"):
        for seed in (5305, 5306, 5307):
            rows = loaded[(region, f"cnn_clean_seed_{seed}")]
            demand = {
                r["target_file"]: r
                for r in rows
                if r["condition"] == "demand" and r["level_db"] == 5.0
            }
            for condition in ("semantic_same", "semantic_different"):
                selected = sorted(
                    [
                        r
                        for r in rows
                        if r["condition"] == condition
                        and r["level_db"] == 5.0
                        and float(r["activity_overlap"]) >= 0.8
                    ],
                    key=lambda r: r["target_file"],
                )
                controls = [demand[r["target_file"]] for r in selected]
                if any(
                    a["target_class"] != b["target_class"] or a["fold"] != b["fold"]
                    for a, b in zip(selected, controls)
                ):
                    raise ValueError("Semantic/DEMAND target matching failed")
                result = paired_bootstrap(
                    [r["target_class"] for r in selected],
                    [r["prediction"] for r in selected],
                    [r["prediction"] for r in controls],
                    [r["fold"] for r in selected],
                    replicates=replicates,
                    seed=530509,
                )
                replication.append(
                    dict(
                        region=region,
                        seed=seed,
                        condition=condition,
                        level_db=5.0,
                        stratum="overlap_ge_0.8",
                        **result,
                    )
                )
    for row, adjusted in zip(replication, holm_adjust([r["mcnemar_exact_p"] for r in replication])):
        row["holm_p_across_cnn_replication"] = float(adjusted)
    return replication


def summarise(repo, runtime, root, output, replicates=2000, trial_evidence=None):
    output.mkdir(parents=True, exist_ok=True)
    tables, clean_rows, loaded, provenance = [], [], {}, {}
    variants = variant_paths(repo, runtime, root, trial_evidence)
    for name, system, regime, seed, clean_dir, dirs in variants:
        clean = read_clean_predictions(clean_dir / "predictions.csv")
        if len(clean) != 2000:
            raise ValueError("Incomplete clean reference")
        reference = list(clean.values())
        clean_metrics = fold_metrics(
            [r[1] for r in reference],
            [r[2] for r in reference],
            [r[0] for r in reference],
            np.arange(50),
        )
        common = {"variant": name, "system": system, "regime": regime, "seed": seed}
        clean_rows.append(
            {
                **common,
                "accuracy": clean_metrics["accuracy"]["mean"],
                "accuracy_fold_sd": clean_metrics["accuracy"]["sample_std"],
                "macro_f1": clean_metrics["macro_f1"]["mean"],
                "macro_f1_fold_sd": clean_metrics["macro_f1"]["sample_std"],
            }
        )
        for region, directory in dirs.items():
            rows = read_predictions(directory / f"{system}_predictions.csv")
            summary = json.loads((directory / "summary.json").read_text())
            expected = 24000 if region == "own" else 23991
            if len(rows) != expected or summary["rows_evaluated"] != expected:
                raise ValueError("Incomplete intervention evaluation")
            loaded[(region, name)] = rows
            provenance[f"{region}/{name}"] = {
                "directory": str(directory),
                "manifest_sha256": summary["manifest_sha256"],
                "predictions_sha256": sha256_file(directory / f"{system}_predictions.csv"),
                "clean_sha256": sha256_file(clean_dir / "predictions.csv"),
            }
            detailed = condition_table(rows, clean, np.arange(50))
            for condition in ("white", "demand", "semantic_same", "semantic_different"):
                for level in (10.0, 5.0, 0.0):
                    entry = detailed[f"{condition}@{level:+.0f}dB"]
                    capture = entry["capture"] or {}
                    tables.append(
                        {
                            **common,
                            "region": region,
                            "condition": condition,
                            "level_db": level,
                            "n": entry["n"],
                            "accuracy": entry["accuracy"]["mean"],
                            "accuracy_fold_sd": entry["accuracy"]["sample_std"],
                            "macro_f1": entry["macro_f1"]["mean"],
                            "macro_f1_fold_sd": entry["macro_f1"]["sample_std"],
                            "capture_rate": capture.get("capture_rate"),
                            "capture_fraction": capture.get("capture_fraction"),
                            "control_macro_f1": entry["control"]["macro_f1"]["mean"],
                            "matched_clean_macro_f1": entry["matched_clean"]["macro_f1"]["mean"],
                        }
                    )
            write_json(output / f"{region}_{name}_conditions.json", detailed)
    write_csv(output / "clean_variants.csv", clean_rows)
    write_csv(output / "condition_variants.csv", tables)

    # Seed SD describes optimisation variation, separately from within-seed fold SD.
    aggregates = []
    for region in ("own", "target"):
        for regime in ("clean", "wgn", "semantic"):
            for condition in ("white", "demand", "semantic_same", "semantic_different"):
                for level in (10.0, 5.0, 0.0):
                    selected = [
                        r
                        for r in tables
                        if r["system"] == "cnn"
                        and r["region"] == region
                        and r["regime"] == regime
                        and r["condition"] == condition
                        and r["level_db"] == level
                    ]
                    if {r["seed"] for r in selected} != {5305, 5306, 5307} or len(selected) != 3:
                        raise ValueError("All three declared CNN seeds are required")
                    item = dict(
                        region=region, regime=regime, condition=condition, level_db=level, seeds=3
                    )
                    for metric in ("accuracy", "macro_f1"):
                        values = [r[metric] for r in selected]
                        item[metric + "_mean"] = float(np.mean(values))
                        item[metric + "_seed_sd"] = float(np.std(values, ddof=1))
                    aggregates.append(item)
    write_csv(output / "cnn_three_seed_summary.csv", aggregates)

    comparisons = []
    for region in ("own", "target"):
        for name, reference, system, regime, seed, comparison_type in comparison_specs(
            variants
        ):
            comparisons_to_run = [
                (c, False) for c in ("white", "demand", "semantic_same", "semantic_different")
            ]
            comparisons_to_run += [(c, True) for c in ("semantic_same", "semantic_different")]
            for condition, high_overlap in comparisons_to_run:
                y, a, b, folds = matched_vectors(
                    loaded[(region, name)],
                    loaded[(region, reference)],
                    condition,
                    high_overlap=high_overlap,
                )
                result = paired_bootstrap(y, a, b, folds, replicates=replicates, seed=530509)
                comparisons.append(
                    dict(
                        region=region,
                        variant=name,
                        reference=reference,
                        comparison_type=comparison_type,
                        system=system,
                        regime=regime,
                        seed=seed,
                        condition=condition,
                        level_db=5.0,
                        stratum="overlap_ge_0.8" if high_overlap else "all",
                        **result,
                    )
                )
    adjusted = holm_adjust([r["mcnemar_exact_p"] for r in comparisons])
    for row, pvalue in zip(comparisons, adjusted):
        row["holm_p_across_extension_comparisons"] = float(pvalue)
    write_csv(output / "paired_intervention_effects.csv", comparisons)
    replication = replicate_overlap(loaded, replicates)
    write_csv(output / "cnn_overlap_replication.csv", replication)
    report = {
        "complete": True,
        "variants": len(clean_rows),
        "condition_rows": len(tables),
        "paired_comparisons": len(comparisons),
        "comparison_family": "108 clean-reference contrasts plus 48 direct semantic-versus-WGN contrasts; one Holm family of 156",
        "analysis_amendment": "2026-11-09 22:45 AEDT: direct contrasts added to answer the stated second question, after some WGN results but before any completed semantic-intervention result; training settings unchanged",
        "cnn_overlap_replications": len(replication),
        "bootstrap_replicates": replicates,
        "provenance": provenance,
        "caveats": [
            "No best-seed selection; all three CNN seeds are retained",
            "Seed SD and five-fold SD describe different sources of variation",
            "Paired CIs are per-seed and conditional on fitted models; do not pool repeated target predictions as independent samples",
            "All extension +5 dB exact McNemar tests share a single Holm correction family",
            "Prompt ensemble is not full PAT; no local-feature alignment or label-based beta tuning",
        ],
    }
    write_json(output / "summary_audit.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "runtime", "run", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--replication-only", action="store_true")
    p.add_argument("--trial-evidence", type=Path, help="Portable evidence ZIP's trial_evidence directory")
    a = p.parse_args()
    if a.replication_only and a.trial_evidence:
        p.error("Use --trial-evidence for the full portable summary, without --replication-only")
    if a.replication_only:
        loaded = {}
        provenance = {}
        for region in ("own", "target"):
            for seed in (5305, 5306, 5307):
                name = f"cnn_clean_seed_{seed}"
                directory = a.run / "evaluations" / region / name
                audit = json.loads((directory / "audit.json").read_text())
                path = directory / "cnn_predictions.csv"
                if sha256_file(path) != audit["predictions_sha256"]:
                    raise ValueError("Clean replication prediction hash differs")
                loaded[(region, name)] = read_predictions(path)
                provenance[f"{region}/{name}"] = audit
        a.output.mkdir(parents=True, exist_ok=True)
        rows = replicate_overlap(loaded, a.replicates)
        write_csv(a.output / "cnn_overlap_replication.csv", rows)
        write_json(
            a.output / "audit.json",
            {
                "complete": True,
                "scope": "clean CNN overlap replication only",
                "provenance": provenance,
            },
        )
    else:
        summarise(a.repo, a.runtime, a.run, a.output, a.replicates, a.trial_evidence)

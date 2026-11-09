"""Recheck the core figure data and expose controls and third-class competition."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.diagnostics import write_csv
from esc_robustness.extension_summary import variant_paths


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def close(actual, expected):
    if not np.isclose(float(actual), float(expected), atol=1e-10, rtol=0):
        raise ValueError(f"Figure value differs: {actual} versus {expected}")


def metrics(y, p, folds):
    scores = [
        f1_score(
            y[folds == k], p[folds == k], labels=np.arange(50), average="macro", zero_division=0
        )
        for k in np.unique(folds)
    ]
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def run(repo, runtime, root, output):
    output.mkdir(parents=True, exist_ok=True)
    variants = variant_paths(repo, runtime, root)
    names = {"svm_clean", "cnn_clean_seed_5305", "clap_baseline"}
    checks, provenance, competitors = {}, {}, []
    for name, system, _, _, clean_dir, directories in variants:
        if name not in names:
            continue
        # Core figures use the original CPU evaluation. The extension reuses
        # identical CNN predictions from its MPS cache, with small score drift.
        if system == "cnn":
            directories = {
                region: repo / f"artifacts/interference/v3_20261109/eval_{region}"
                for region in ("own", "target")
            }
        clean = {r["filename"]: int(r["prediction"]) for r in read(clean_dir / "predictions.csv")}
        for region, directory in directories.items():
            path = directory / f"{system}_predictions.csv"
            rows = read(path)
            y = np.array([int(r["target_class"]) for r in rows])
            p = np.array([int(r["prediction"]) for r in rows])
            cp = np.array([int(r["control_prediction"]) for r in rows])
            folds = np.array([int(r["fold"]) for r in rows])
            labels = np.array(
                [int(r["interferer_class"]) if r["interferer_class"] else -1 for r in rows]
            )
            conditions = np.array([r["condition"] for r in rows])
            levels = np.array([float(r["level_db"]) for r in rows])
            targets = np.array([r["target_file"] for r in rows])
            overlaps = np.array([float(r["activity_overlap"]) for r in rows])
            clean_p = np.array([clean[t] for t in targets])
            score_path = directory / f"{system}_scores.npz"
            with np.load(score_path) as arrays:
                assert np.array_equal(arrays["row"], [int(r["row"]) for r in rows])
                assert np.array_equal(arrays["class_ids"], np.arange(50))
                scores, control_scores = arrays["scores"], arrays["control_scores"]
                a, b = (
                    arrays["representation"].astype(float),
                    arrays["control_representation"].astype(float),
                )
            if system == "svm":
                d = a - b
                for k in range(1, 6):
                    model_path = (
                        runtime
                        / f"artifacts/interference/v3_20261109/clean_svm_reconstructed/svm_fold_{k}.joblib"
                    )
                    d[folds == k] /= joblib.load(model_path)["model"].named_steps["scale"].scale_
                distances = np.sqrt(np.mean(d * d, axis=1))
            else:
                distances = 1 - np.clip(
                    np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)),
                    -1,
                    1,
                )
            del a, b
            figures = repo / f"results/figures_three_systems_{region}_20261109"
            count = 0
            for number, filename in (
                (1, "macro_f1_vs_level"),
                (2, "capture_rate_vs_sir"),
                (3, "margin_vs_sir"),
                (4, "same_vs_different"),
                (5, "representation_displacement"),
                (6, "confusion_flow"),
                (8, "overlap_strata"),
            ):
                file = figures / f"fig{number}_{filename}.csv"
                for row in read(file):
                    if row["system"] != system:
                        continue
                    level = 10 if row["level_db"] == "clean" else float(row["level_db"])
                    condition = row.get("condition", row.get("semantic_condition"))
                    mask = (levels == level) & ((conditions == condition) if condition else True)
                    if number == 1:
                        pred = clean_p if row["level_db"] == "clean" else p
                        mean, sd = metrics(y[mask], pred[mask], folds[mask])
                        close(row["macro_f1_mean"], mean)
                        close(row["macro_f1_sd"], sd)
                    elif number == 2:
                        errors = int(np.sum(mask & (p != y)))
                        captured = int(np.sum(mask & (p == labels)))
                        close(row["errors"], errors)
                        close(row["errors_equal_distractor"], captured)
                        close(row["capture_rate"], captured / errors)
                        close(row["capture_fraction"], captured / mask.sum())
                    elif number == 3:
                        indices = np.flatnonzero(mask)
                        values = scores[indices, y[mask]] - scores[indices, labels[mask]]
                        for key, quantile in (("q1", 25), ("median", 50), ("q3", 75)):
                            close(row[key], np.percentile(values, quantile))
                        close(row["fraction_negative"], np.mean(values < 0))
                    elif number == 4:
                        common = set(
                            targets[(levels == level) & (conditions == "semantic_same")]
                        ) & set(targets[(levels == level) & (conditions == "semantic_different")])
                        mask &= np.isin(targets, list(common))
                        clean_f1 = metrics(y[mask], clean_p[mask], folds[mask])[0]
                        f1 = metrics(y[mask], p[mask], folds[mask])[0]
                        close(row["clean_macro_f1"], clean_f1)
                        close(row["macro_f1"], f1)
                        close(row["relative_macro_f1_loss"], 1 - f1 / clean_f1)
                    elif number == 5:
                        mask &= (p == y) if row["outcome"] == "correct" else (p != y)
                        for key, quantile in (("q1", 25), ("median", 50), ("q3", 75)):
                            close(row[key], np.percentile(distances[mask], quantile))
                    elif number == 6:
                        mask &= (y == int(row["target_class"])) & (
                            labels == int(row["distractor_class"])
                        )
                        close(row["pairs"], mask.sum())
                        close(row["captured"], np.sum(mask & (p == labels)))
                        close(row["control_correct_pairs"], np.sum(mask & (cp == y)))
                        close(
                            row["new_captures_from_correct_control"],
                            np.sum(mask & (cp == y) & (p == labels)),
                        )
                    elif number == 8:
                        lo, hi = {
                            "[0.0, 0.5)": (0, 0.5),
                            "[0.5, 0.8)": (0.5, 0.8),
                            "[0.8, 1.0]": (0.8, 1.01),
                        }[row["overlap_stratum"]]
                        mask &= (overlaps >= lo) & (overlaps < hi)
                        demand = {
                            targets[i]: p[i]
                            for i in np.flatnonzero((levels == level) & (conditions == "demand"))
                        }
                        dp = np.array([demand[t] for t in targets[mask]])
                        close(row["semantic_accuracy"], np.mean(p[mask] == y[mask]))
                        close(row["demand_accuracy_same_targets"], np.mean(dp == y[mask]))
                        close(row["semantic_macro_f1"], metrics(y[mask], p[mask], folds[mask])[0])
                        close(
                            row["demand_macro_f1_same_targets"],
                            metrics(y[mask], dp, folds[mask])[0],
                        )
                        close(row["classes_present"], len(np.unique(y[mask])))
                    if "n" in row:
                        close(row["n"], mask.sum())
                    count += 1
            if system == "clap":
                for condition in ("semantic_same", "semantic_different"):
                    mask = (levels == 5) & (conditions == condition)
                    indices = np.flatnonzero(mask)
                    others = scores[mask].copy()
                    others[np.arange(len(indices)), y[mask]] = -np.inf
                    others[np.arange(len(indices)), labels[mask]] = -np.inf
                    winner = others.argmax(axis=1)
                    competitors.append(
                        dict(
                            region=region,
                            condition=condition,
                            n=len(indices),
                            target_predictions=int(np.sum(p[mask] == y[mask])),
                            distractor_predictions=int(np.sum(p[mask] == labels[mask])),
                            other_predictions=int(
                                np.sum((p[mask] != y[mask]) & (p[mask] != labels[mask]))
                            ),
                            median_target_minus_best_other=float(
                                np.median(scores[indices, y[mask]] - scores[indices, winner])
                            ),
                            median_selected_other_score_change=float(
                                np.median(scores[indices, winner] - control_scores[indices, winner])
                            ),
                        )
                    )
            checks[f"{region}/{system}"] = count
            provenance[f"{region}/{system}"] = {
                "predictions": sha256_file(path),
                "scores": sha256_file(score_path),
            }
    controls = []
    for region in ("own", "target"):
        for name in ("svm_clean", "cnn_clean_seed_5305", "clap_baseline", "clap_prompt_ensemble"):
            table = json.loads(
                (
                    repo
                    / f"results/extensions_20261109/interventions/{region}_{name}_conditions.json"
                ).read_text()
            )
            for condition in ("white", "demand", "semantic_same", "semantic_different"):
                for level in (10, 5, 0):
                    row = table[f"{condition}@+{level}dB"]
                    controls.append(
                        dict(
                            region=region,
                            variant=name,
                            condition=condition,
                            level_db=level,
                            n=row["n"],
                            clean_macro_f1=row["matched_clean"]["macro_f1"]["mean"],
                            control_macro_f1=row["control"]["macro_f1"]["mean"],
                            mixture_macro_f1=row["macro_f1"]["mean"],
                            control_minus_clean_pp=100 * row["control_macro_f1_minus_clean"],
                        )
                    )
    write_csv(output / "gain_control_effects.csv", controls)
    write_csv(output / "clap_other_competitors.csv", competitors)
    write_json(
        output / "audit.json",
        dict(
            verified=True,
            figure_records_checked=checks,
            tolerance=1e-10,
            checks="Independent per-trial/sklearn recomputation of figures 1–6 and 8 for both protocols; cost figure 7 retains its audited historical microbenchmark definition",
            source_hashes=provenance,
            selected_other_caveat="Best other class is selected on each mixture; its score change is a descriptive selected-competitor diagnostic, not a prespecified class effect",
        ),
    )
    print(json.dumps({"verified": True, "figure_records_checked": checks}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "runtime", "run", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.runtime, args.run, args.output)

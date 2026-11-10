"""Tables from interference evaluations: degradation, capture and margins per fold.

Metrics follow the baseline convention: compute accuracy and macro-F1 inside each
official test fold, then report the mean and sample standard deviation across the five
folds. Clean reference scores come from the baseline run's out-of-fold predictions, so
every comparison uses the same fold-matched model.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .artifacts import write_json

LEVEL_ORDER = (10.0, 5.0, 0.0)


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No predictions in {path}")
    for row in rows:
        row["fold"] = int(row["fold"])
        row["target_class"] = int(row["target_class"])
        row["prediction"] = int(row["prediction"])
        row["level_db"] = float(row["level_db"])
        row["correct"] = int(row["correct"])
        row["control_prediction"] = int(row["control_prediction"])
        row["common_gain"] = float(row["common_gain"])
        row["interferer_class"] = (
            int(row["interferer_class"]) if row["interferer_class"] != "" else None
        )
        row["margin"] = float(row["margin"]) if row["margin"] != "" else None
    return rows


def read_clean_predictions(path):
    """Baseline predictions.csv → {filename: (fold, target, prediction)}."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {
            r["filename"]: (int(r["fold"]), int(r["target"]), int(r["prediction"]))
            for r in csv.DictReader(handle)
        }


def fold_metrics(y_true, y_pred, folds, labels):
    """Per-fold accuracy and macro-F1 with mean and sample SD across folds present."""
    y_true, y_pred, folds = (np.asarray(a) for a in (y_true, y_pred, folds))
    per_fold = {}
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        per_fold[int(fold)] = {
            "n": int(mask.sum()),
            "accuracy": float(np.mean(y_true[mask] == y_pred[mask])),
            "macro_f1": float(
                f1_score(
                    y_true[mask], y_pred[mask], labels=labels, average="macro", zero_division=0
                )
            ),
        }
    result = {"folds": per_fold}
    for name in ("accuracy", "macro_f1"):
        values = [v[name] for v in per_fold.values()]
        result[name] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    result["pooled"] = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }
    return result


def capture_statistics(rows, n_classes=50):
    """Distractor capture: errors that name the distractor, with explicit counts."""
    semantic = [r for r in rows if r["interferer_class"] is not None]
    if not semantic:
        return None
    errors = [r for r in semantic if not r["correct"]]
    captured = [r for r in errors if r["prediction"] == r["interferer_class"]]
    return {
        "n": len(semantic),
        "errors": len(errors),
        "errors_equal_distractor": len(captured),
        "capture_rate": len(captured) / len(errors) if errors else None,
        "capture_fraction": len(captured) / len(semantic),
        "uniform_wrong_class_reference": 1 / (n_classes - 1) if n_classes > 1 else None,
        "reference_assumption": "uniform choice among wrong labels; not an empirical null",
    }


def margin_statistics(rows):
    margins = np.array([r["margin"] for r in rows if r["margin"] is not None])
    if margins.size == 0:
        return None
    quartiles = np.percentile(margins, [25, 50, 75])
    return {
        "n": int(margins.size),
        "median": float(quartiles[1]),
        "q1": float(quartiles[0]),
        "q3": float(quartiles[2]),
        "fraction_negative": float(np.mean(margins < 0)),
    }


def condition_table(rows, clean, labels):
    """One entry per (condition, level) plus the clean reference for the same targets."""
    seen = set()
    for row in rows:
        key = (row["target_file"], row["condition"], row["level_db"])
        if key in seen:
            raise ValueError("Duplicate target within a condition and level")
        seen.add(key)
        if clean[row["target_file"]][:2] != (row["fold"], row["target_class"]):
            raise ValueError("Clean baseline fold/label differs from the interference trial")
    targets = sorted({r["target_file"] for r in rows})
    clean_rows = [clean[t] for t in targets]
    clean_metrics = fold_metrics(
        [c[1] for c in clean_rows], [c[2] for c in clean_rows], [c[0] for c in clean_rows], labels
    )
    table = {"clean": {"n": len(targets), **clean_metrics}}
    conditions = sorted({r["condition"] for r in rows})
    for condition in conditions:
        levels = sorted({r["level_db"] for r in rows if r["condition"] == condition}, reverse=True)
        for level in levels:
            subset = [r for r in rows if r["condition"] == condition and r["level_db"] == level]
            metrics = fold_metrics(
                [r["target_class"] for r in subset],
                [r["prediction"] for r in subset],
                [r["fold"] for r in subset],
                labels,
            )
            control = fold_metrics(
                [r["target_class"] for r in subset],
                [r["control_prediction"] for r in subset],
                [r["fold"] for r in subset],
                labels,
            )
            # Some power definitions cannot scale an interferer that is silent on the
            # target mask. Match the clean reference to this condition's retained rows.
            matched_clean_rows = [clean[r["target_file"]] for r in subset]
            matched_clean = fold_metrics(
                [c[1] for c in matched_clean_rows],
                [c[2] for c in matched_clean_rows],
                [c[0] for c in matched_clean_rows],
                labels,
            )
            clean_f1 = matched_clean["macro_f1"]["mean"]
            control_f1 = control["macro_f1"]["mean"]
            control_correct = [r for r in subset if r["control_prediction"] == r["target_class"]]
            lost = [r for r in control_correct if r["prediction"] != r["target_class"]]
            captured = [r for r in lost if r["prediction"] == r["interferer_class"]]
            entry = {
                "n": len(subset),
                "ratio_type": subset[0]["ratio_type"],
                **metrics,
                "matched_clean": {"n": len(subset), **matched_clean},
                "relative_macro_f1_loss": (clean_f1 - metrics["macro_f1"]["mean"]) / clean_f1
                if clean_f1 > 0
                else None,
                "relative_control_macro_f1_loss": (control_f1 - metrics["macro_f1"]["mean"])
                / control_f1
                if control_f1 > 0
                else None,
                "control_macro_f1_minus_clean": control_f1 - clean_f1,
                "control_correct_transitions": {
                    "n": len(control_correct),
                    "remain_correct": len(control_correct) - len(lost),
                    "become_incorrect": len(lost),
                    "become_distractor": len(captured),
                    "distractor_fraction_of_new_errors": len(captured) / len(lost)
                    if lost and subset[0]["interferer_class"] is not None
                    else None,
                },
                "control": {
                    "accuracy": control["accuracy"],
                    "macro_f1": control["macro_f1"],
                    "rows_with_common_gain": int(sum(r["common_gain"] < 1 for r in subset)),
                },
                "capture": capture_statistics(subset, len(labels)),
                "margin": margin_statistics(subset),
            }
            table[f"{condition}@{level:+.0f}dB"] = entry
    return table


def summarise_evaluation(eval_dir, clean_dirs, output_dir=None):
    """Build tables for every system in an evaluation directory.

    ``clean_dirs`` maps system name → directory holding the baseline predictions.csv.
    """
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir) if output_dir else eval_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((eval_dir / "summary.json").read_text())
    report = {
        "evaluation": str(eval_dir),
        "manifest_sha256": summary["manifest_sha256"],
        "power_definition": summary.get("power_definition"),
        "systems": {},
    }
    for system in summary["systems"]:
        if system not in clean_dirs:
            raise ValueError(f"No clean baseline directory given for {system}")
        rows = read_predictions(eval_dir / f"{system}_predictions.csv")
        clean = read_clean_predictions(Path(clean_dirs[system]) / "predictions.csv")
        missing = {r["target_file"] for r in rows} - set(clean)
        if missing:
            raise ValueError(f"{system}: {len(missing)} targets lack clean predictions")
        with np.load(eval_dir / f"{system}_scores.npz") as arrays:
            labels = arrays["class_ids"].tolist()
        table = condition_table(rows, clean, labels)
        report["systems"][system] = table
        _write_condition_csv(output_dir / f"{system}_conditions.csv", table)
    write_json(output_dir / "tables.json", report)
    return report


def _write_condition_csv(path, table):
    fields = [
        "condition",
        "n",
        "accuracy_mean",
        "accuracy_sd",
        "macro_f1_mean",
        "macro_f1_sd",
        "matched_clean_n",
        "matched_clean_macro_f1_mean",
        "relative_macro_f1_loss",
        "relative_control_macro_f1_loss",
        "control_macro_f1_minus_clean",
        "control_macro_f1_mean",
        "rows_with_common_gain",
        "errors",
        "errors_equal_distractor",
        "capture_rate",
        "capture_fraction",
        "margin_median",
        "margin_fraction_negative",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, entry in table.items():
            capture, margin = entry.get("capture"), entry.get("margin")
            control = entry.get("control")
            writer.writerow(
                {
                    "condition": name,
                    "n": entry["n"],
                    "accuracy_mean": entry["accuracy"]["mean"],
                    "accuracy_sd": entry["accuracy"]["sample_std"],
                    "macro_f1_mean": entry["macro_f1"]["mean"],
                    "macro_f1_sd": entry["macro_f1"]["sample_std"],
                    "matched_clean_n": entry["matched_clean"]["n"]
                    if "matched_clean" in entry
                    else entry["n"],
                    "matched_clean_macro_f1_mean": entry["matched_clean"]["macro_f1"]["mean"]
                    if "matched_clean" in entry
                    else entry["macro_f1"]["mean"],
                    "relative_macro_f1_loss": entry.get("relative_macro_f1_loss"),
                    "relative_control_macro_f1_loss": entry.get("relative_control_macro_f1_loss"),
                    "control_macro_f1_minus_clean": entry.get("control_macro_f1_minus_clean"),
                    "control_macro_f1_mean": control["macro_f1"]["mean"] if control else "",
                    "rows_with_common_gain": control["rows_with_common_gain"] if control else "",
                    "errors": capture["errors"] if capture else "",
                    "errors_equal_distractor": capture["errors_equal_distractor"]
                    if capture
                    else "",
                    "capture_rate": capture["capture_rate"] if capture else "",
                    "capture_fraction": capture["capture_fraction"] if capture else "",
                    "margin_median": margin["median"] if margin else "",
                    "margin_fraction_negative": margin["fraction_negative"] if margin else "",
                }
            )

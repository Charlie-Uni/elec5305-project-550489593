"""Paired uncertainty and representation/ranking diagnostics from saved predictions.

The bootstrap resamples target clips within official folds and keeps paired
conditions together. It conditions on fitted models; it is not a training-seed CI.
"""

import csv
import json
import zlib
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from .analysis import read_predictions
from .artifacts import sha256_file, write_json
from .features import feature_names


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f1_from_counts(y, predictions, weights=None, n_classes=50):
    counts = np.bincount(
        np.asarray(y) * n_classes + np.asarray(predictions),
        weights=weights,
        minlength=n_classes**2,
    ).reshape(n_classes, n_classes)
    denominator = counts.sum(0) + counts.sum(1)
    return np.divide(
        2 * counts.diagonal(), denominator, out=np.zeros(n_classes), where=denominator > 0
    ).mean()


def paired_bootstrap(y, a, b, folds, replicates=2000, seed=530509, n_classes=50):
    """Difference A-B; one multinomial weight vector shared by each target pair."""
    y, a, b, folds = (np.asarray(v, dtype=int) for v in (y, a, b, folds))
    if not (y.shape == a.shape == b.shape == folds.shape) or y.ndim != 1 or not y.size:
        raise ValueError("Paired vectors must be nonempty and have identical shapes")
    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    if any(np.any((v < 0) | (v >= n_classes)) for v in (y, a, b)):
        raise ValueError("Class IDs outside the declared class set")
    rng = np.random.default_rng(seed)
    fold_ids = np.unique(folds)
    f1_differences = np.zeros(replicates)
    accuracy_differences = np.zeros(replicates)
    point = 0.0
    for fold in fold_ids:
        mask = folds == fold
        yy, aa, bb = y[mask], a[mask], b[mask]
        n = len(yy)
        point += (
            f1_from_counts(yy, aa, n_classes=n_classes)
            - f1_from_counts(yy, bb, n_classes=n_classes)
        ) / len(fold_ids)
        correct_difference = (yy == aa).astype(int) - (yy == bb).astype(int)
        # Batching limits RAM; repeated mixtures of a target are never resampled separately.
        for start in range(0, replicates, 100):
            stop = min(start + 100, replicates)
            weights = rng.multinomial(n, np.full(n, 1 / n), size=stop - start)
            accuracy_differences[start:stop] += weights @ correct_difference / len(y)
            for offset, w in enumerate(weights):
                f1_differences[start + offset] += (
                    f1_from_counts(yy, aa, w, n_classes) - f1_from_counts(yy, bb, w, n_classes)
                ) / len(fold_ids)
    aa, bb = a == y, b == y
    only_a, only_b = int(np.sum(aa & ~bb)), int(np.sum(bb & ~aa))
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {
        "n": len(y),
        "folds": len(fold_ids),
        "replicates": replicates,
        "macro_f1_difference": float(point),
        "macro_f1_ci_low": float(np.quantile(f1_differences, 0.025)),
        "macro_f1_ci_high": float(np.quantile(f1_differences, 0.975)),
        "accuracy_difference": float(np.mean(aa) - np.mean(bb)),
        "accuracy_ci_low": float(np.quantile(accuracy_differences, 0.025)),
        "accuracy_ci_high": float(np.quantile(accuracy_differences, 0.975)),
        "a_only_correct": only_a,
        "b_only_correct": only_b,
        "mcnemar_exact_p": float(p),
    }


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    corrected = np.empty(len(pvalues))
    corrected[order] = np.minimum(
        1, np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1))
    )
    return corrected


def feature_group_indices(names):
    groups = {}
    for group in (
        "rms",
        "zcr",
        "spectral_centroid",
        "spectral_spread",
        "spectral_rolloff",
        "spectral_flux",
        "mfcc",
        "delta_mfcc",
    ):
        prefix = group + "_"
        groups[group] = np.array([i for i, name in enumerate(names) if name.startswith(prefix)])
    covered = np.concatenate(list(groups.values()))
    if len(covered) != len(names) or len(np.unique(covered)) != len(names):
        raise ValueError("Feature groups must partition the representation exactly")
    return groups


def score_ranks(scores, labels):
    """Competition rank: 1 + number of strictly larger scores (ties share rank)."""
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=int)
    if scores.ndim != 2 or labels.shape != (len(scores),) or not np.isfinite(scores).all():
        raise ValueError("Finite score matrix and one label per row are required")
    return 1 + np.sum(scores > scores[np.arange(len(scores)), labels, None], axis=1)


def run_diagnostics(evaluations, output_dir, svm_models, replicates=2000, seed=530509):
    """evaluations maps own/target to a list of directories containing disjoint systems."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparisons, groups_table, rankings, rank_summary, aucs = [], [], [], [], []
    provenance = {}
    for region, directories in evaluations.items():
        hashes, seen = set(), set()
        for directory in map(Path, directories):
            summary = json.loads((directory / "summary.json").read_text())
            hashes.add(summary["manifest_sha256"])
            if len(hashes) != 1:
                raise ValueError("Protocols cannot combine different manifests")
            for system in summary["systems"]:
                if system in seen:
                    raise ValueError("Duplicate system within one power protocol")
                seen.add(system)
                rows = read_predictions(directory / f"{system}_predictions.csv")
                with np.load(directory / f"{system}_scores.npz") as archive:
                    data = dict(archive)
                if not np.array_equal(data["row"], [int(r["row"]) for r in rows]):
                    raise ValueError("Prediction and score row IDs differ")
                if not np.array_equal(data["class_ids"], np.arange(50)):
                    raise ValueError("Expected all 50 ordered ESC-50 classes")
                for key in ("scores", "control_scores", "representation", "control_representation"):
                    if not np.isfinite(data[key]).all():
                        raise ValueError("Nonfinite stored outputs")
                provenance[f"{region}/{system}"] = {
                    "directory": str(directory),
                    "manifest_sha256": summary["manifest_sha256"],
                    "predictions_sha256": sha256_file(directory / f"{system}_predictions.csv"),
                    "scores_sha256": sha256_file(directory / f"{system}_scores.npz"),
                }
                y = np.array([r["target_class"] for r in rows])
                prediction = np.array([r["prediction"] for r in rows])
                folds = np.array([r["fold"] for r in rows])
                conditions = np.array([r["condition"] for r in rows])
                levels = np.array([r["level_db"] for r in rows])
                error = prediction != y
                feature_distances = {}
                if system == "svm":
                    difference = data["representation"] - data["control_representation"]
                    for fold in np.unique(folds):
                        path = Path(svm_models) / f"svm_fold_{fold}.joblib"
                        expected = (
                            summary["systems"]["svm"].get("checkpoint_sha256", {}).get(str(fold))
                        )
                        if expected and sha256_file(path) != expected:
                            raise ValueError("SVM scaler hash differs from evaluated model")
                        scale = joblib.load(path)["model"].named_steps["scale"].scale_
                        difference[folds == fold] /= scale
                    distance = np.sqrt(np.mean(difference**2, axis=1))
                    names = data.get("representation_names", np.array(feature_names())).tolist()
                    for name, indices in feature_group_indices(names).items():
                        feature_distances[name] = np.sqrt(
                            np.mean(difference[:, indices] ** 2, axis=1)
                        )
                else:
                    a, b = data["representation"], data["control_representation"]
                    distance = 1 - np.clip(
                        np.einsum("ij,ij->i", a, b)
                        / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)),
                        -1,
                        1,
                    )
                for condition in sorted(set(conditions)):
                    for level in sorted(set(levels), reverse=True):
                        mask = (conditions == condition) & (levels == level)
                        for fold in (0, 1, 2, 3, 4, 5):
                            selected = mask if fold == 0 else mask & (folds == fold)
                            yy = error[selected]
                            value = (
                                float(roc_auc_score(yy, distance[selected]))
                                if len(set(yy)) == 2
                                else None
                            )
                            aucs.append(
                                dict(
                                    region=region,
                                    system=system,
                                    condition=condition,
                                    level_db=level,
                                    fold="pooled" if fold == 0 else fold,
                                    n=int(selected.sum()),
                                    errors=int(yy.sum()),
                                    error_discrimination_auc=value,
                                )
                            )
                        for name, values in feature_distances.items():
                            for outcome in ("all", "correct", "incorrect"):
                                selected = (
                                    mask
                                    if outcome == "all"
                                    else mask & (error == (outcome == "incorrect"))
                                )
                                vv = values[selected]
                                if len(vv):
                                    groups_table.append(
                                        dict(
                                            region=region,
                                            condition=condition,
                                            level_db=level,
                                            feature_group=name,
                                            outcome=outcome,
                                            n=len(vv),
                                            mean=float(vv.mean()),
                                            q1=float(np.quantile(vv, 0.25)),
                                            median=float(np.median(vv)),
                                            q3=float(np.quantile(vv, 0.75)),
                                        )
                                    )
                for level in sorted(set(levels), reverse=True):
                    demand = {
                        r["target_file"]: i
                        for i, r in enumerate(rows)
                        if r["condition"] == "demand" and r["level_db"] == level
                    }
                    for condition in ("semantic_same", "semantic_different"):
                        for stratum in ("all", "overlap_ge_0.8"):
                            indices = [
                                i
                                for i, r in enumerate(rows)
                                if r["condition"] == condition
                                and r["level_db"] == level
                                and r["target_file"] in demand
                                and (stratum == "all" or float(r["activity_overlap"]) >= 0.8)
                            ]
                            control_indices = [demand[rows[i]["target_file"]] for i in indices]
                            if not indices:
                                continue
                            salt = zlib.crc32(f"{region}:{condition}:{level}:{stratum}".encode())
                            stats = paired_bootstrap(
                                y[indices],
                                prediction[indices],
                                prediction[control_indices],
                                folds[indices],
                                replicates,
                                (seed + salt) % (2**32),
                            )
                            comparisons.append(
                                dict(
                                    region=region,
                                    system=system,
                                    semantic_condition=condition,
                                    level_db=level,
                                    stratum=stratum,
                                    comparison="semantic_minus_matched_DEMAND",
                                    **stats,
                                )
                            )
                if system == "clap":
                    semantic = np.flatnonzero(np.char.startswith(conditions, "semantic"))
                    labels = np.array([rows[i]["interferer_class"] for i in semantic])
                    target_ranks = score_ranks(data["scores"][semantic], y[semantic])
                    distractor_ranks = score_ranks(data["scores"][semantic], labels)
                    control_ranks = score_ranks(data["control_scores"][semantic], labels)
                    names = {r["target_class"]: r["target_category"] for r in rows}
                    for j, i in enumerate(semantic):
                        r = rows[i]
                        other = data["scores"][i].copy()
                        other[[y[i], labels[j]]] = -np.inf
                        ordered = np.argsort(-other, kind="stable")[:3]
                        entry = dict(
                            region=region,
                            row=r["row"],
                            target_file=r["target_file"],
                            fold=r["fold"],
                            condition=r["condition"],
                            level_db=r["level_db"],
                            target_class=r["target_class"],
                            distractor_class=labels[j],
                            target_rank=int(target_ranks[j]),
                            distractor_rank=int(distractor_ranks[j]),
                            control_distractor_rank=int(control_ranks[j]),
                            distractor_score_change=float(
                                data["scores"][i, labels[j]] - data["control_scores"][i, labels[j]]
                            ),
                        )
                        for k, c in enumerate(ordered, 1):
                            entry.update(
                                {
                                    f"other_{k}_class": int(c),
                                    f"other_{k}_name": names[int(c)],
                                    f"other_{k}_score": float(data["scores"][i, c]),
                                    f"other_{k}_score_change": float(
                                        data["scores"][i, c] - data["control_scores"][i, c]
                                    ),
                                }
                            )
                        rankings.append(entry)
                    for condition in ("semantic_same", "semantic_different"):
                        for level in sorted(set(levels), reverse=True):
                            selected = (conditions[semantic] == condition) & (
                                levels[semantic] == level
                            )
                            dr, tr, cr = (
                                distractor_ranks[selected],
                                target_ranks[selected],
                                control_ranks[selected],
                            )
                            rank_summary.append(
                                dict(
                                    region=region,
                                    condition=condition,
                                    level_db=level,
                                    n=len(dr),
                                    target_rank_median=float(np.median(tr)),
                                    distractor_rank_median=float(np.median(dr)),
                                    control_distractor_rank_median=float(np.median(cr)),
                                    distractor_top1_fraction=float(np.mean(dr == 1)),
                                    distractor_top5_fraction=float(np.mean(dr <= 5)),
                                    distractor_reciprocal_rank_mean=float(np.mean(1 / dr)),
                                )
                            )
                print(f"Diagnostics: {region}/{system} complete", flush=True)
    primary = [r for r in comparisons if r["level_db"] == 5 and r["stratum"] == "overlap_ge_0.8"]
    adjusted = holm_adjust([r["mcnemar_exact_p"] for r in primary])
    for r, p in zip(primary, adjusted):
        r["primary_holm_p"] = float(p)
    for r in comparisons:
        r.setdefault("primary_holm_p", "")
    for name, values in [
        ("paired_uncertainty", comparisons),
        ("feature_group_displacement", groups_table),
        ("clap_competition_trials", rankings),
        ("clap_rank_summary", rank_summary),
        ("displacement_error_auc", aucs),
    ]:
        write_csv(out / f"{name}.csv", values)
    meta = dict(
        input_provenance=provenance,
        bootstrap_replicates=replicates,
        seed=seed,
        bootstrap="paired target clips, stratified by test fold; percentile 95% CI; fixed fitted models",
        multiplicity="Holm adjustment across +5 dB high-overlap exact McNemar tests only",
        primary_tests=len(primary),
        ranking_ties="competition rank; strictly larger scores",
        caveats=[
            "CLAP ranks use all 50 fixed class prompts; no prompt tuning",
            "AUC describes association of displacement with errors, not causation",
            "Clip bootstrap does not capture model refitting or source-recording dependence",
            "All new analyses are extensions motivated by previously observed baseline results",
        ],
    )
    write_json(out / "diagnostics_meta.json", meta)
    return meta

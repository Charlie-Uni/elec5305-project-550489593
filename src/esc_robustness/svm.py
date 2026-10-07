"""Validation-selected RBF SVM with five official outer folds and saved OOF scores."""

import csv
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .artifacts import load_feature_cache, sha256_file, write_json
from .data import outer_split

DEFAULT_GRID = {"svc__C": [1.0, 10.0, 100.0], "svc__gamma": ["scale", 0.001, 0.1]}


def make_pipeline():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    decision_function_shape="ovr",
                    break_ties=True,
                    cache_size=256,
                ),
            ),
        ]
    )


def class_scores(model, features):
    scores = model.decision_function(features)
    # Binary SVC supplies one signed score. Keep an explicit column per class.
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    return scores


def metrics(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }


def fit_outer_fold(X, y, folds, test_fold, grid=None):
    """The test indices are never passed to a candidate fit or selection score."""
    train, validation, test = outer_split(folds, test_fold)
    labels = np.unique(y)
    for indices in (train, validation, test):
        if not np.array_equal(np.unique(y[indices]), labels):
            raise ValueError("Every train/validation/test partition must contain all classes")
    template = make_pipeline()
    selection = []
    for params in ParameterGrid(DEFAULT_GRID if grid is None else grid):
        candidate = clone(template).set_params(**params)
        candidate.fit(X[train], y[train])
        scores = metrics(y[validation], candidate.predict(X[validation]), labels)
        selection.append({"parameters": params, **scores})
    if not selection:
        raise ValueError("SVM grid must contain at least one candidate")
    # Stable tie handling: first candidate in the declared ParameterGrid order.
    best = max(selection, key=lambda row: row["macro_f1"])
    final_train = np.concatenate((train, validation))
    model = clone(template).set_params(**best["parameters"])
    model.fit(X[final_train], y[final_train])
    predictions = model.predict(X[test])
    result = {
        "test_fold": test_fold,
        "validation_fold": test_fold % 5 + 1,
        "selection_training_folds": sorted(set(folds[train].tolist())),
        "final_training_folds": sorted(set(folds[final_train].tolist())),
        "n_selection_train": len(train),
        "n_validation": len(validation),
        "n_final_train": len(final_train),
        "n_test": len(test),
        "selected_parameters": best["parameters"],
        "validation_macro_f1": best["macro_f1"],
        "candidate_results": selection,
        "test_metrics": metrics(y[test], predictions, labels),
        "support_vectors": int(model.named_steps["svc"].support_vectors_.shape[0]),
    }
    return model, result, test, predictions, class_scores(model, X[test])


def evaluate_svm(cache_path, output_dir, grid=None, require_complete=True):
    cache_path, output_dir = Path(cache_path), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose a new run directory: {output_dir}"
        )
    cache, config = load_feature_cache(cache_path)
    X, y, folds = cache["X"], cache["y"], cache["folds"]
    labels = np.unique(y)
    if require_complete and (
        not bool(cache["complete_dataset"])
        or len(y) != 2000
        or len(labels) != 50
        or any(np.sum((folds == f) & (y == c)) != 8 for f in range(1, 6) for c in range(50))
    ):
        raise ValueError("Official evaluation requires a complete, validated ESC-50 feature cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = {
        int(label): str(cache["categories"][np.flatnonzero(y == label)[0]]) for label in labels
    }
    predictions = np.full(len(y), -1, dtype=int)
    scores = np.full((len(y), len(labels)), np.nan)
    results = []
    with threadpool_limits(limits=1):
        for fold in range(1, 6):
            print(f"Training outer fold {fold}/5", flush=True)
            model, result, test, pred, decision = fit_outer_fold(X, y, folds, fold, grid)
            if not np.array_equal(model.classes_, labels):
                raise ValueError("Inconsistent class ordering")
            predictions[test], scores[test] = pred, decision
            model_path = output_dir / f"svm_fold_{fold}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "feature_config": config.as_dict(),
                    "class_names": categories,
                    "test_fold": fold,
                    "feature_cache_sha256": sha256_file(cache_path),
                    "feature_source_hashes": json.loads(cache["source_hashes_json"].item()),
                },
                model_path,
            )
            result["model_bytes"] = model_path.stat().st_size
            results.append(result)
            write_json(output_dir / f"fold_{fold}.json", result)
            print(
                f"Fold {fold}: accuracy={result['test_metrics']['accuracy']:.4f}, "
                f"macro-F1={result['test_metrics']['macro_f1']:.4f}",
                flush=True,
            )
    if np.any(predictions < 0) or not np.isfinite(scores).all():
        raise RuntimeError("Incomplete out-of-fold predictions")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["filename", "fold", "target", "category", "prediction", "predicted_category"]
        )
        for i, pred in enumerate(predictions):
            writer.writerow(
                [
                    cache["filenames"][i],
                    folds[i],
                    y[i],
                    categories[int(y[i])],
                    pred,
                    categories[int(pred)],
                ]
            )
    np.savez_compressed(
        output_dir / "decision_scores.npz",
        scores=scores,
        class_ids=labels,
        filenames=cache["filenames"],
        folds=folds,
        targets=y,
        predictions=predictions,
    )
    cm = confusion_matrix(y, predictions, labels=labels)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class/predicted_class", *[categories[int(c)] for c in labels]])
        for label, row in zip(labels, cm):
            writer.writerow([categories[int(label)], *row])
    summary = {}
    for name in ("accuracy", "macro_f1"):
        values = [r["test_metrics"][name] for r in results]
        summary[name] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)),
        }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    summary.update(
        {
            "evaluation": "clean_ESC-50_official_five_fold"
            if require_complete
            else "diagnostic_subset",
            "clips": len(y),
            "n_features": X.shape[1],
            "labels": labels.tolist(),
            "feature_config": config.as_dict(),
            "selection_grid": DEFAULT_GRID if grid is None else grid,
            "feature_cache_sha256": sha256_file(cache_path),
            "source_sha256": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in ("dsp.py", "features.py", "data.py", "artifacts.py", "svm.py")
            },
            "feature_source_sha256": json.loads(cache["source_hashes_json"].item()),
            "metadata_source_overlaps": json.loads(cache["source_overlap_json"].item()),
            "git_head": revision,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu": platform.processor(),
                "numerical_threads": 1,
                "packages": {
                    package: importlib.metadata.version(package)
                    for package in ("numpy", "scipy", "scikit-learn", "soundfile", "joblib")
                },
            },
            "folds": results,
            "pooled_out_of_fold": metrics(y, predictions, labels),
            "classification_report": classification_report(
                y,
                predictions,
                labels=labels,
                target_names=[categories[int(c)] for c in labels],
                output_dict=True,
                zero_division=0,
            ),
        }
    )
    write_json(output_dir / "summary.json", summary)
    return summary

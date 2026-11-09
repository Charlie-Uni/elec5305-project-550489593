"""Independently recompute the prompt ablation from saved vectors and predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from esc_robustness.artifacts import sha256_file, write_json


def archive(path):
    with np.load(path) as data:
        return dict(data)


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def audit(core, ensemble, output):
    if not (ensemble / "complete.json").exists():
        raise ValueError("Prompt ablation is not complete")
    texts = archive(ensemble / "text_embeddings.npz")
    individual = texts["per_prompt"]
    np.testing.assert_allclose(np.linalg.norm(individual, axis=2), 1, atol=1e-6)
    averaged = individual.mean(axis=1)
    averaged /= np.linalg.norm(averaged, axis=1, keepdims=True)
    np.testing.assert_allclose(averaged, texts["ensemble"], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(texts["ensemble"], axis=1), 1, atol=1e-6)
    report = dict(
        verified=True,
        templates_per_class=individual.shape[1],
        unique_templates=len(set(texts["templates"].tolist())),
        conditions=[],
    )
    for name, expected in (("clean", 2000), ("eval_own", 24000), ("eval_target", 23991)):
        clean = name == "clean"
        filename = "decision_scores.npz" if clean else "clap_scores.npz"
        source = archive(core / name / filename)
        current = archive(ensemble / name / filename)
        key = "embeddings" if clean else "representation"
        if len(current[key]) != expected:
            raise ValueError("Incomplete embedding set")
        np.testing.assert_array_equal(current["class_ids"], np.arange(50))
        for field in (
            ("filenames", "folds", "targets", "embeddings")
            if clean
            else ("row", "representation", "control_representation")
        ):
            np.testing.assert_array_equal(source[field], current[field])
        expected_scores = current[key] @ texts["ensemble"].T
        np.testing.assert_allclose(current["scores"], expected_scores, atol=1e-6)
        rows = csv_rows(ensemble / name / ("predictions.csv" if clean else "clap_predictions.csv"))
        prediction = np.array([int(r["prediction"]) for r in rows])
        np.testing.assert_array_equal(expected_scores.argmax(1), prediction)
        y = np.array([int(r["target"] if clean else r["target_class"]) for r in rows])
        folds = np.array([int(r["fold"]) for r in rows])
        if clean:
            saved = json.loads((ensemble / name / "summary.json").read_text())
            for metric, function in (("accuracy", accuracy_score), ("macro_f1", f1_score)):
                values = [
                    function(
                        y[folds == f],
                        prediction[folds == f],
                        **(
                            {"average": "macro", "labels": np.arange(50), "zero_division": 0}
                            if metric == "macro_f1"
                            else {}
                        ),
                    )
                    for f in range(1, 6)
                ]
                np.testing.assert_allclose(saved[metric]["mean"], np.mean(values), atol=1e-14)
                report["clean_" + metric] = {
                    "folds": values,
                    "mean": float(np.mean(values)),
                    "fold_sd": float(np.std(values, ddof=1)),
                }
        else:
            source_meta = json.loads((core / name / "summary.json").read_text())
            summary = json.loads((ensemble / name / "summary.json").read_text())
            if source_meta["manifest_sha256"] != summary["manifest_sha256"]:
                raise ValueError("Manifest differs from core")
            controls = current["control_representation"] @ texts["ensemble"].T
            np.testing.assert_allclose(current["control_scores"], controls, atol=1e-6)
            np.testing.assert_array_equal(
                controls.argmax(1), [int(r["control_prediction"]) for r in rows]
            )
            np.testing.assert_allclose(
                current["scores"][np.arange(expected), y],
                [float(r["target_score"]) for r in rows],
                atol=0,
                rtol=0,
            )
            for condition in ("white", "demand", "semantic_same", "semantic_different"):
                for level in (10.0, 5.0, 0.0):
                    ids = [
                        i
                        for i, r in enumerate(rows)
                        if r["condition"] == condition and float(r["level_db"]) == level
                    ]
                    yy, pp = y[ids], prediction[ids]
                    accuracy = float(np.mean(yy == pp))
                    cell = summary["systems"]["clap"]["conditions"][f"{condition}@{level:+.0f}dB"]
                    np.testing.assert_allclose(accuracy, cell["accuracy"], atol=0, rtol=0)
                    capture = None
                    if condition.startswith("semantic"):
                        distractors = np.array([int(rows[i]["interferer_class"]) for i in ids])
                        capture = float(np.sum(pp == distractors) / np.sum(pp != yy))
                        np.testing.assert_allclose(capture, cell["capture_rate"], atol=0, rtol=0)
                    report["conditions"].append(
                        dict(
                            region=name[5:],
                            condition=condition,
                            level_db=level,
                            n=len(ids),
                            accuracy=accuracy,
                            capture_rate=capture,
                        )
                    )
        report[name] = {
            "rows": expected,
            "audio_embeddings_unchanged": True,
            "score_max_abs_difference": float(np.max(np.abs(current["scores"] - expected_scores))),
            "scores_sha256": sha256_file(ensemble / name / filename),
        }
    write_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "conditions"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "ensemble", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        audit(args.core, args.ensemble, args.output)

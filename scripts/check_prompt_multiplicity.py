"""Descriptive post-hoc sensitivity to duplicate released prompt templates.

The declared 300-entry ensemble remains the intervention. This equal-weight
275-template check is not used to choose the reported classifier or prompts.
Audio inference is not repeated or changed.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.diagnostics import write_csv


def archive(path):
    with np.load(path) as data:
        return dict(data)


def check(source, output):
    output.mkdir(parents=True, exist_ok=False)
    texts = archive(source / "text_embeddings.npz")
    first = {}
    for i, template in enumerate(texts["templates"].tolist()):
        first.setdefault(template, i)
    unique = texts["per_prompt"][:, list(first.values())].mean(1)
    unique /= np.linalg.norm(unique, axis=1, keepdims=True)
    np.testing.assert_allclose(np.linalg.norm(unique, axis=1), 1, atol=1e-12)
    records, saved = [], {}
    for region, filename in (
        ("clean", "decision_scores.npz"),
        ("own", "clap_scores.npz"),
        ("target", "clap_scores.npz"),
    ):
        directory = source / ("clean" if region == "clean" else "eval_" + region)
        arrays = archive(directory / filename)
        clean = region == "clean"
        csvfile = directory / ("predictions.csv" if clean else "clap_predictions.csv")
        with csvfile.open() as handle:
            rows = list(csv.DictReader(handle))
        audio = arrays["embeddings" if clean else "representation"]
        original = (audio @ texts["ensemble"].T).argmax(1)
        np.testing.assert_array_equal(original, [int(r["prediction"]) for r in rows])
        prediction = (audio @ unique.T).argmax(1)
        y = np.array([int(r["target" if clean else "target_class"]) for r in rows])
        folds = np.array([int(r["fold"]) for r in rows])
        groups = (
            [("clean", None)]
            if clean
            else [
                (c, level)
                for c in ("white", "demand", "semantic_same", "semantic_different")
                for level in (10.0, 5.0, 0.0)
            ]
        )
        for condition, level in groups:
            mask = (
                np.ones(len(rows), dtype=bool)
                if clean
                else np.array(
                    [r["condition"] == condition and float(r["level_db"]) == level for r in rows]
                )
            )
            cell = dict(
                region=region,
                condition=condition,
                level_db=level,
                n=int(mask.sum()),
                changed_predictions=int(np.sum(original[mask] != prediction[mask])),
            )
            for name, p in (("released_300", original), ("unique_275", prediction)):
                f1 = [
                    f1_score(
                        y[mask & (folds == f)],
                        p[mask & (folds == f)],
                        labels=range(50),
                        average="macro",
                        zero_division=0,
                    )
                    for f in range(1, 6)
                ]
                cell[name + "_accuracy"] = float(np.mean(y[mask] == p[mask]))
                cell[name + "_macro_f1"] = float(np.mean(f1))
                if condition.startswith("semantic"):
                    distractors = np.array(
                        [int(r["interferer_class"]) for i, r in enumerate(rows) if mask[i]]
                    )
                    count = int(np.sum(p[mask] == distractors))
                    errors = int(np.sum(p[mask] != y[mask]))
                    cell[name + "_capture_fraction"] = count / int(mask.sum())
                    cell[name + "_capture_rate"] = count / errors
                else:
                    cell[name + "_capture_fraction"] = cell[name + "_capture_rate"] = None
            records.append(cell)
        saved[region + "_predictions"] = prediction
        saved[region + "_row_ids"] = arrays["filenames" if clean else "row"]
    np.savez_compressed(output / "predictions.npz", **saved)
    write_csv(output / "sensitivity.csv", records)
    delta = [100 * (r["unique_275_macro_f1"] - r["released_300_macro_f1"]) for r in records]
    audit = dict(
        complete=True,
        scope="Exploratory descriptive duplicate-weighting sensitivity; original 300-entry intervention retained",
        templates=300,
        unique_templates=len(first),
        condition_cells=len(records),
        largest_absolute_macro_f1_change_pp=float(max(map(abs, delta))),
        text_centroid_cosine_min=float(np.min(np.einsum("ij,ij->i", unique, texts["ensemble"]))),
        source_text_sha256=sha256_file(source / "text_embeddings.npz"),
        predictions_sha256=sha256_file(output / "predictions.npz"),
        table_sha256=sha256_file(output / "sensitivity.csv"),
    )
    write_json(output / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        check(args.ensemble, args.output)

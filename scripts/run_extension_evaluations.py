"""Consume completed extension models and caches, then verify saved predictions.

Runs are serial to bound memory. Each result is separate from the core experiment.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from esc_robustness.analysis import read_predictions
from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.extension_eval import evaluate_cached
from esc_robustness.mixing import manifest_digest


def verify(directory, manifest, model, system):
    meta = json.loads((directory / "summary.json").read_text())
    if meta["manifest_sha256"] != manifest_digest(manifest):
        raise ValueError("Evaluation manifest hash differs")
    suffix = "pt" if system == "cnn" else "joblib"
    for fold in range(1, 6):
        actual = sha256_file(model / f"{system}_fold_{fold}.{suffix}")
        if meta["systems"][system]["checkpoint_sha256"][str(fold)] != actual:
            raise ValueError("Evaluation model hash differs")
    rows = read_predictions(directory / f"{system}_predictions.csv")
    with np.load(directory / f"{system}_scores.npz") as f:
        a = dict(f)
    if len(rows) != meta["rows_evaluated"] or not np.array_equal(
        a["row"], [int(r["row"]) for r in rows]
    ):
        raise ValueError("Prediction row identity differs")
    for key, pred_key in (("scores", "prediction"), ("control_scores", "control_prediction")):
        if not np.isfinite(a[key]).all():
            raise ValueError("Nonfinite scores")
        np.testing.assert_array_equal(a["class_ids"][a[key].argmax(1)], [r[pred_key] for r in rows])
    targets = [np.flatnonzero(a["class_ids"] == r["target_class"])[0] for r in rows]
    np.testing.assert_allclose(
        a["scores"][np.arange(len(rows)), targets],
        [float(r["target_score"]) for r in rows],
        atol=0,
        rtol=0,
    )
    return {
        "rows": len(rows),
        "prediction_scores_verified": True,
        "predictions_sha256": sha256_file(directory / f"{system}_predictions.csv"),
        "scores_sha256": sha256_file(directory / f"{system}_scores.npz"),
    }


def run(repo, runtime, root, device="cpu"):
    core = runtime / "runs/clap_v4_20261109"
    models = [("cnn_clean_seed_5305", "cnn", runtime / "artifacts/baselines/clean_cnn")]
    models += [(f"cnn_clean_seed_{s}", "cnn", root / f"clean_seed_{s}") for s in (5306, 5307)]
    models += [
        (f"cnn_{mode}_seed_{s}", "cnn", root / f"cnn_{mode}_seed_{s}")
        for mode in ("wgn", "semantic")
        for s in (5305, 5306, 5307)
    ]
    models += [(f"svm_{mode}", "svm", root / f"svm_{mode}") for mode in ("wgn", "semantic")]
    completed = {}

    def model_ready(name, system, path):
        if name == "cnn_clean_seed_5305":
            return all((path / f"cnn_fold_{fold}.pt").is_file() for fold in range(1, 6))
        return (path / "summary.json").is_file()
    (root / "evaluations").mkdir(exist_ok=True)
    while len(completed) < len(models) * 2:
        progressed = False
        for region in ("own", "target"):
            manifest = core / f"manifest_{region}"
            cache = root / f"frontend_{region}"
            for name, system, model in models:
                key = f"{region}/{name}"
                if key in completed or not model_ready(name, system, model):
                    continue
                if system == "cnn" and not (cache / "cache_meta.json").exists():
                    continue
                destination = root / "evaluations" / region / name
                write_json(
                    root / "evaluation_queue.json",
                    {
                        "status": "running",
                        "active": key,
                        "completed": completed,
                        "total": len(models) * 2,
                    },
                )
                print(f"Starting {key}", flush=True)
                if not (destination / "summary.json").exists():
                    evaluate_cached(
                        manifest,
                        destination,
                        model,
                        system,
                        data_root=runtime / "data/ESC-50",
                        frontend_cache=cache,
                        feature_evaluation=repo
                        / f"artifacts/interference/v3_20261109/eval_{region}",
                        device=device,
                        batch_size=64,
                    )
                audit = verify(destination, manifest, model, system)
                if name == "cnn_clean_seed_5305":
                    source = (
                        repo / f"artifacts/interference/v3_20261109/eval_{region}/cnn_scores.npz"
                    )
                    with (
                        np.load(source) as original,
                        np.load(destination / "cnn_scores.npz") as current,
                    ):
                        for item in ("scores", "control_scores"):
                            error = float(np.max(np.abs(original[item] - current[item])))
                            changed = int(
                                np.sum(original[item].argmax(1) != current[item].argmax(1))
                            )
                            audit[item + "_original_max_abs_difference"] = error
                            audit[item + "_original_prediction_changes"] = changed
                            if error > 1e-3:
                                raise ValueError("Cached baseline diverges from original CNN")
                write_json(destination / "audit.json", audit)
                completed[key] = audit
                progressed = True
                print(f"Verified {key}", flush=True)
        write_json(
            root / "evaluation_queue.json",
            {
                "status": "complete" if len(completed) == len(models) * 2 else "waiting",
                "active": None,
                "completed": completed,
                "total": len(models) * 2,
            },
        )
        if not progressed and len(completed) < len(models) * 2:
            # Surface a failed producer rather than waiting forever on missing models.
            missing = [p for name, system, p in models if not model_ready(name, system, p)]
            launch = json.loads((root / "augmentation_launch.json").read_text())
            if missing:
                try:
                    os.kill(launch["pid"], 0)
                except ProcessLookupError as error:
                    raise RuntimeError(
                        f"Training producer stopped; missing models: {missing}"
                    ) from error
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "runtime", "run"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.repo, args.runtime, args.run, args.device)

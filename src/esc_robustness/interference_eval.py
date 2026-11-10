"""Run the fold-matched SVM and CNN, and optionally frozen CLAP, on every mixture.

Each mixture is rendered deterministically from the manifest and classified by the
model whose outer test fold contains the target clip, so no interfering recording or
target recording was ever seen in training. Frozen CLAP is fold independent and uses
one checkpoint and one prompt set for every row. The gain-matched clean control
(target scaled by the row's common gain) is classified alongside each mixture.

Outputs per system: ``<system>_predictions.csv`` (one row per mixture) and
``<system>_scores.npz`` (class scores, representations and the clean-control scores).
"""

import csv
import importlib.metadata
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .artifacts import sha256_file, write_json
from .dsp import FeatureConfig, log_mel
from .features import extract_features, feature_names
from .mixing import manifest_digest, renderer_from_manifest
from .svm import class_scores

SYSTEMS = ("svm", "cnn", "clap")

PREDICTION_FIELDS = (
    "row",
    "condition",
    "ratio_type",
    "level_db",
    "fold",
    "target_file",
    "target_class",
    "target_category",
    "target_coarse",
    "interferer",
    "interferer_class",
    "interferer_coarse",
    "common_gain",
    "activity_overlap",
    "prediction",
    "correct",
    "predicted_interferer",
    "target_score",
    "interferer_score",
    "margin",
    "control_prediction",
    "control_correct",
    "control_target_score",
)


class SVMSystem:
    name = "svm"

    def __init__(self, model_dir, feature_config):
        self.models = {}
        for fold in range(1, 6):
            bundle = joblib.load(Path(model_dir) / f"svm_fold_{fold}.joblib")
            if bundle["test_fold"] != fold:
                raise ValueError(f"svm_fold_{fold}.joblib reports test fold {bundle['test_fold']}")
            if FeatureConfig(**bundle["feature_config"]) != feature_config:
                raise ValueError("SVM feature configuration differs from the manifest")
            self.models[fold] = bundle
        self.class_ids = np.asarray(self.models[1]["model"].classes_)
        if any(
            not np.array_equal(np.asarray(b["model"].classes_), self.class_ids)
            for b in self.models.values()
        ):
            raise ValueError("SVM folds disagree on class ordering")
        self.config = feature_config
        self.representation_names = feature_names(feature_config)
        self.representation_type = "handcrafted_features"
        self.score_type = "ovr_decision_function"

    def evaluate(self, fold, audio):
        """Return (class scores, 92-dimensional handcrafted feature vector)."""
        features = extract_features(audio, self.config)
        return class_scores(self.models[fold]["model"], features[None, :])[0], features


class CNNSystem:
    name = "cnn"

    def __init__(self, model_dir, feature_config):
        import torch

        from .cnn import load_cnn

        torch.set_num_threads(1)
        self.torch = torch
        self.models = {}
        class_ids = None
        for fold in range(1, 6):
            model, config, bundle = load_cnn(Path(model_dir) / f"cnn_fold_{fold}.pt")
            if bundle["test_fold"] != fold:
                raise ValueError(f"cnn_fold_{fold}.pt reports test fold {bundle['test_fold']}")
            if config != feature_config:
                raise ValueError("CNN feature configuration differs from the manifest")
            if class_ids is None:
                class_ids = list(bundle["class_ids"])
            elif list(bundle["class_ids"]) != class_ids:
                raise ValueError("CNN folds disagree on class ordering")
            self.models[fold] = model
        self.class_ids = np.asarray(class_ids)
        self.config = feature_config
        self.representation_names = None  # penultimate embedding, no per-dimension names
        self.representation_type = "cnn_embedding"
        self.score_type = "logits"

    def evaluate(self, fold, audio):
        """Return (logits, penultimate embedding) for one clip on the CPU."""
        spectrogram = log_mel(audio, self.config).T.astype(np.float32)
        with self.torch.inference_mode():
            x = self.torch.from_numpy(spectrogram[None, None])
            embedding = self.models[fold].embedding(x)
            logits = self.models[fold].classifier(embedding)
        return logits.numpy()[0].astype(np.float64), embedding.numpy()[0].astype(np.float64)


def _score_for(class_ids, scores, class_id):
    if class_id is None:
        return None
    index = np.flatnonzero(class_ids == class_id)
    if index.size != 1:
        raise ValueError(f"Class {class_id} is not a model output")
    return float(scores[index[0]])


def evaluate_manifest(
    manifest_dir,
    output_dir,
    data_root=None,
    svm_dir=None,
    cnn_dir=None,
    limit=None,
    extra_systems=(),
    batch_size=1,
):
    """``extra_systems`` holds ready-made systems such as a loaded ``CLAPSystem``."""
    manifest_dir, output_dir = Path(manifest_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if svm_dir is None and cnn_dir is None and not extra_systems:
        raise ValueError("Provide at least one system: svm_dir, cnn_dir or extra_systems")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    renderer, meta, rows = renderer_from_manifest(manifest_dir, data_root)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    systems = []
    if svm_dir is not None:
        systems.append(SVMSystem(svm_dir, renderer.feature_config))
    if cnn_dir is not None:
        systems.append(CNNSystem(cnn_dir, renderer.feature_config))
    systems.extend(extra_systems)
    if len({s.name for s in systems}) != len(systems):
        raise ValueError("System names must be unique")
    if any(s.input_rate != renderer.feature_config.sample_rate for s in extra_systems):
        raise ValueError("Extra systems must expect the manifest's sample rate")
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    store = {
        s.name: {
            "scores": np.full((n, len(s.class_ids)), np.nan),
            "control_scores": np.full((n, len(s.class_ids)), np.nan),
            "representation": [],
            "control_representation": [],
            "rows": [],
        }
        for s in systems
    }
    start = perf_counter()
    with threadpool_limits(limits=1):
        for start_index in range(0, n, batch_size):
            batch = rows[start_index : start_index + batch_size]
            rendered = [renderer.render(row) for row in batch]
            for system in systems:
                evaluated = None
                if batch_size > 1 and hasattr(system, "evaluate_batch"):
                    audio = [
                        signal
                        for item in rendered
                        for signal in (item["mixture"], item["clean_matched"])
                    ]
                    folds = [row["fold"] for row in batch for _ in range(2)]
                    evaluated = system.evaluate_batch(folds, audio)
                for j, (row, item) in enumerate(zip(batch, rendered)):
                    values = (
                        None
                        if evaluated is None
                        else (
                            evaluated[0][2 * j],
                            evaluated[1][2 * j],
                            evaluated[0][2 * j + 1],
                            evaluated[1][2 * j + 1],
                        )
                    )
                    record = _classify(
                        system, row, item, store[system.name], start_index + j, values
                    )
                    store[system.name]["rows"].append(record)
            completed = start_index + len(batch)
            if completed // 500 > start_index // 500 or completed == n:
                print(f"Evaluated {completed}/{n} mixtures", flush=True)
                write_json(
                    output_dir / "progress.json",
                    {
                        "status": "running",
                        "completed": completed,
                        "total": n,
                        "elapsed_seconds": perf_counter() - start,
                    },
                )
    seconds = perf_counter() - start
    summary = {
        "manifest": str(manifest_dir),
        "manifest_sha256": manifest_digest(manifest_dir),
        "rows_evaluated": n,
        "subset_limit": limit,
        "batch_size_mixtures": batch_size,
        "batch_note": "Batch-capable systems embed interleaved mixtures and controls; up to 2*batch_size audio inputs.",
        "data_root": str(renderer.data_root),
        "interference_config": meta["interference_config"],
        "power_definition": meta["power_definition"],
        "feature_config": renderer.feature_config.as_dict(),
        "control": "target scaled by the row's common gain, classified by the same model",
        "seconds": seconds,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numerical_threads": 1,
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "scikit-learn", "soundfile")
            },
        },
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("interference_eval.py", "mixing.py", "dsp.py", "features.py")
        },
        "systems": {},
    }
    for system in systems:
        data = store[system.name]
        csv_path = output_dir / f"{system.name}_predictions.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            writer.writerows(data["rows"])
        arrays = {
            "scores": data["scores"],
            "control_scores": data["control_scores"],
            "representation": np.asarray(data["representation"]),
            "control_representation": np.asarray(data["control_representation"]),
            "class_ids": system.class_ids,
            "row": np.array([r["row"] for r in data["rows"]]),
            "score_type": np.array(system.score_type),
            "representation_type": np.array(system.representation_type),
        }
        if system.representation_names:
            arrays["representation_names"] = np.array(system.representation_names)
        if hasattr(system, "text_embeddings"):
            arrays["text_embeddings"] = system.text_embeddings
            arrays["prompts"] = np.array(system.prompts)
        np.savez_compressed(output_dir / f"{system.name}_scores.npz", **arrays)
        summary["systems"][system.name] = _summarise(data["rows"], system)
        if system.name in ("svm", "cnn"):
            directory = Path(svm_dir if system.name == "svm" else cnn_dir)
            suffix = "joblib" if system.name == "svm" else "pt"
            summary["systems"][system.name]["model_dir"] = str(directory)
            summary["systems"][system.name]["checkpoint_sha256"] = {
                str(fold): sha256_file(directory / f"{system.name}_fold_{fold}.{suffix}")
                for fold in range(1, 6)
            }
        elif hasattr(system, "provenance"):
            summary["systems"][system.name]["provenance"] = system.provenance()
    if cnn_dir is not None or any(s.name == "clap" for s in systems):
        summary["environment"]["packages"]["torch"] = importlib.metadata.version("torch")
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "progress.json",
        {
            "status": "complete",
            "completed": n,
            "total": n,
            "elapsed_seconds": seconds,
        },
    )
    return summary


def _classify(system, row, rendered, data, index, evaluated=None):
    if evaluated is None:
        scores, vector = system.evaluate(row["fold"], rendered["mixture"])
        control_scores, control_vector = system.evaluate(row["fold"], rendered["clean_matched"])
    else:
        scores, vector, control_scores, control_vector = evaluated
    if not all(np.isfinite(x).all() for x in (scores, control_scores, vector, control_vector)):
        raise ValueError(f"Row {row['row']}: nonfinite {system.name} scores")
    data["scores"][index] = scores
    data["control_scores"][index] = control_scores
    data["representation"].append(vector)
    data["control_representation"].append(control_vector)
    prediction = int(system.class_ids[int(np.argmax(scores))])
    control_prediction = int(system.class_ids[int(np.argmax(control_scores))])
    interferer_class = row["interferer_class"]
    target_score = _score_for(system.class_ids, scores, row["target_class"])
    interferer_score = _score_for(system.class_ids, scores, interferer_class)
    return {
        "row": row["row"],
        "condition": row["condition"],
        "ratio_type": row["ratio_type"],
        "level_db": row["level_db"],
        "fold": row["fold"],
        "target_file": row["target_file"],
        "target_class": row["target_class"],
        "target_category": row["target_category"],
        "target_coarse": row["target_coarse"],
        "interferer": row["interferer"],
        "interferer_class": "" if interferer_class is None else interferer_class,
        "interferer_coarse": row["interferer_coarse"],
        "common_gain": row["common_gain"],
        "activity_overlap": row["activity_overlap"],
        "prediction": prediction,
        "correct": int(prediction == row["target_class"]),
        "predicted_interferer": ""
        if interferer_class is None
        else int(prediction == interferer_class),
        "target_score": target_score,
        "interferer_score": "" if interferer_score is None else interferer_score,
        "margin": "" if interferer_score is None else target_score - interferer_score,
        "control_prediction": control_prediction,
        "control_correct": int(control_prediction == row["target_class"]),
        "control_target_score": _score_for(system.class_ids, control_scores, row["target_class"]),
    }


def _summarise(records, system):
    """Accuracy per condition and level plus capture statistics (counts, not claims)."""
    table = {}
    for record in records:
        key = f"{record['condition']}@{record['level_db']:+.0f}dB"
        cell = table.setdefault(
            key,
            {
                "n": 0,
                "correct": 0,
                "control_correct": 0,
                "errors": 0,
                "errors_equal_interferer": 0,
                "predicted_interferer": 0,
                "semantic": record["interferer_class"] != "",
            },
        )
        cell["n"] += 1
        cell["correct"] += record["correct"]
        cell["control_correct"] += record["control_correct"]
        if not record["correct"]:
            cell["errors"] += 1
            if record["predicted_interferer"] == 1:
                cell["errors_equal_interferer"] += 1
        if record["predicted_interferer"] == 1:
            cell["predicted_interferer"] += 1
    for cell in table.values():
        cell["accuracy"] = cell["correct"] / cell["n"]
        cell["control_accuracy"] = cell["control_correct"] / cell["n"]
        cell["capture_rate"] = (
            cell["errors_equal_interferer"] / cell["errors"]
            if cell["semantic"] and cell["errors"]
            else None
        )
        cell["capture_fraction"] = (
            cell["predicted_interferer"] / cell["n"] if cell["semantic"] else None
        )
    return {"score_type": system.score_type, "conditions": table}

"""Compact log-Mel CNN, validation-selected duration and fresh four-fold refits."""

import csv
import importlib.metadata
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from torch import nn

from .artifacts import sha256_file, write_json
from .data import load_audio, outer_split
from .dsp import FeatureConfig, log_mel
from .spectrograms import load_spectrogram_cache
from .svm import metrics


@dataclass(frozen=True)
class CNNConfig:
    seed: int = 5305
    batch_size: int = 32
    max_epochs: int = 40
    patience: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout: float = 0.3
    channels: tuple = (16, 32, 64)
    num_threads: int = 4

    def __post_init__(self):
        for name in ("batch_size", "max_epochs", "patience", "num_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if (
            not np.isfinite([self.learning_rate, self.weight_decay, self.dropout]).all()
            or self.learning_rate <= 0
            or self.weight_decay < 0
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("Invalid learning rate, weight decay or dropout")
        if len(self.channels) != 3 or any(type(c) is not int or c < 1 for c in self.channels):
            raise ValueError("channels must contain three positive integers")


def resolve_device(name="auto"):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if name not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps or cuda")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    return torch.device(name)


def seed_training(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    # MPS has operations without deterministic implementations; log this limitation.
    torch.use_deterministic_algorithms(True, warn_only=device.type == "mps")


def fit_normalisation(X, indices):
    """One mean/std per Mel band, estimated only on the explicit training rows."""
    selected = X[indices]
    mean = selected.mean(axis=(0, 2), dtype=np.float64)
    scale = selected.std(axis=(0, 2), dtype=np.float64)
    return mean.astype(np.float32), np.maximum(scale, 1e-6).astype(np.float32)


class CompactCNN(nn.Module):
    """Three conv/BN/ReLU/pool blocks; 4 frequency bins and mean temporal pooling."""

    def __init__(self, n_mels, n_classes, config=CNNConfig()):
        super().__init__()
        if n_mels < 32 or n_classes < 2:
            raise ValueError("CNN requires at least 32 Mel bands and two classes")
        self.register_buffer("input_mean", torch.zeros(1, 1, n_mels, 1))
        self.register_buffer("input_scale", torch.ones(1, 1, n_mels, 1))
        layers, incoming = [], 1
        for i, outgoing in enumerate(config.channels):
            layers.extend(
                [
                    nn.Conv2d(
                        incoming, outgoing, 3, stride=(1, 2) if i == 0 else 1, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(outgoing),
                    nn.ReLU(),
                    nn.MaxPool2d(2, ceil_mode=True),
                ]
            )
            incoming = outgoing
        self.convolution = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 1))
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.channels[-1] * 4, n_classes)

    def embedding(self, x):
        x = (x - self.input_mean) / self.input_scale
        return self.pool(self.convolution(x)).flatten(1)

    def forward(self, x):
        return self.classifier(self.dropout(self.embedding(x)))


def new_model(X, training_indices, n_classes, config, device):
    seed_training(config.seed, device)
    mean, scale = fit_normalisation(X, training_indices)
    model = CompactCNN(X.shape[1], n_classes, config)
    model.input_mean.copy_(torch.from_numpy(mean)[None, None, :, None])
    model.input_scale.copy_(torch.from_numpy(scale)[None, None, :, None])
    return model.to(device)


def predict_arrays(model, X, indices, batch_size, device):
    model.eval()
    logits, embeddings = [], []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = np.asarray(X[indices[start : start + batch_size]], dtype=np.float32)
            x = torch.from_numpy(batch[:, None]).to(device)
            embedding = model.embedding(x)
            logits.append(model.classifier(embedding).cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    logits, embeddings = np.concatenate(logits), np.concatenate(embeddings)
    if not np.isfinite(logits).all() or not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite CNN predictions")
    return logits, embeddings


def train_epochs(model, X, y, indices, config, device, epochs, validation=None, progress=None):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(config.seed)
    history, best_score, best_epoch = [], -1.0, 0
    labels = np.arange(model.classifier.out_features)
    for epoch in range(1, epochs + 1):
        start = perf_counter()
        model.train()
        order = indices[torch.randperm(len(indices), generator=generator).numpy()]
        total_loss = 0.0
        for offset in range(0, len(order), config.batch_size):
            rows = order[offset : offset + config.batch_size]
            inputs = torch.from_numpy(np.asarray(X[rows], dtype=np.float32)[:, None]).to(device)
            targets = torch.from_numpy(y[rows].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite CNN training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rows)
        row = {"epoch": epoch, "train_loss": total_loss / len(indices)}
        if validation is not None:
            scores, _ = predict_arrays(model, X, validation, config.batch_size, device)
            row["validation"] = metrics(y[validation], scores.argmax(axis=1), labels)
            if row["validation"]["macro_f1"] > best_score:
                best_score, best_epoch = row["validation"]["macro_f1"], epoch
        row["seconds"] = perf_counter() - start
        history.append(row)
        if progress:
            progress(row)
        if validation is not None and epoch - best_epoch >= config.patience:
            break
    return history, best_epoch


def fit_outer_fold_cnn(X, y, folds, test_fold, config=CNNConfig(), device="cpu", progress=None):
    device = resolve_device(str(device))
    torch.set_num_threads(config.num_threads)
    labels = np.unique(y)
    if not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError("CNN class IDs must be contiguous starting at zero")
    train, validation, test = outer_split(folds, test_fold)
    for indices in (train, validation, test):
        if not np.array_equal(np.unique(y[indices]), labels):
            raise ValueError("Every partition must contain all classes")
    model = new_model(X, train, len(labels), config, device)
    selection_mean = model.input_mean.cpu().flatten().tolist()
    selection_scale = model.input_scale.cpu().flatten().tolist()

    def callback(phase):
        return (lambda row: progress(phase, row)) if progress else None

    selection_history, best_epoch = train_epochs(
        model, X, y, train, config, device, config.max_epochs, validation, callback("selection")
    )
    # Discard selection weights and optimizer. Fresh seed, scaler, model and optimizer.
    del model
    final_train = np.concatenate((train, validation))
    model = new_model(X, final_train, len(labels), config, device)
    refit_history, _ = train_epochs(
        model, X, y, final_train, config, device, best_epoch, progress=callback("refit")
    )
    logits, embeddings = predict_arrays(model, X, test, config.batch_size, device)
    predictions = logits.argmax(axis=1)
    result = {
        "test_fold": test_fold,
        "validation_fold": test_fold % 5 + 1,
        "selection_training_folds": sorted(set(folds[train].tolist())),
        "final_training_folds": sorted(set(folds[final_train].tolist())),
        "n_selection_train": len(train),
        "n_validation": len(validation),
        "n_final_train": len(final_train),
        "n_test": len(test),
        "seed": config.seed,
        "selected_epochs": best_epoch,
        "validation_macro_f1": selection_history[best_epoch - 1]["validation"]["macro_f1"],
        "selection_normalisation": {"mean": selection_mean, "scale": selection_scale},
        "final_normalisation": {
            "mean": model.input_mean.cpu().flatten().tolist(),
            "scale": model.input_scale.cpu().flatten().tolist(),
        },
        "selection_history": selection_history,
        "refit_history": refit_history,
        "test_metrics": metrics(y[test], predictions, labels),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return model.cpu(), result, test, predictions, logits, embeddings


def evaluate_cnn(cache_path, output_dir, config=CNNConfig(), device="auto", require_complete=True):
    cache_path, output_dir = Path(cache_path), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new run: {output_dir}")
    cache, feature_config = load_spectrogram_cache(cache_path, require_complete)
    device = resolve_device(device)
    X, y, folds = cache["X"], cache["y"], cache["folds"]
    labels = np.unique(y)
    categories = {int(c): str(cache["categories"][np.flatnonzero(y == c)[0]]) for c in labels}
    source_hashes = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ("cnn.py", "dsp.py", "spectrograms.py", "data.py", "svm.py")
    }
    run = {
        "evaluation": "clean_ESC-50_official_five_fold"
        if require_complete
        else "diagnostic_subset",
        "clips": len(y),
        "labels": labels.tolist(),
        "cnn_config": asdict(config),
        "feature_config": feature_config.as_dict(),
        "spectrogram_cache_sha256": sha256_file(cache_path),
        "metadata_sha256": cache["metadata_sha256"].item(),
        "source_sha256": source_hashes,
        "feature_source_sha256": json.loads(cache["source_hashes_json"].item()),
        "metadata_source_overlaps": json.loads(cache["source_overlap_json"].item()),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "device": str(device),
            "numerical_threads": config.num_threads,
            "deterministic_algorithms": True,
            "deterministic_warn_only": device.type == "mps",
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("torch", "numpy", "scipy", "scikit-learn")
            },
        },
        "selection_rule": "highest validation macro-F1; ties keep earliest epoch; fresh refit",
        "augmentation": "none",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", run)
    logits = np.full((len(y), len(labels)), np.nan, dtype=np.float32)
    embeddings = np.full((len(y), config.channels[-1] * 4), np.nan, dtype=np.float32)
    predictions = np.full(len(y), -1, dtype=int)
    results = []
    start = perf_counter()
    with threadpool_limits(limits=config.num_threads):
        for fold in range(1, 6):

            def progress(phase, row):
                write_json(output_dir / "progress.json", {"fold": fold, "phase": phase, **row})
                score = row.get("validation", {}).get("macro_f1")
                suffix = f" val_F1={score:.4f}" if score is not None else ""
                print(
                    f"Fold {fold} {phase} epoch {row['epoch']}: "
                    f"loss={row['train_loss']:.4f}{suffix} ({row['seconds']:.1f}s)",
                    flush=True,
                )

            model, result, test, pred, scores, vectors = fit_outer_fold_cnn(
                X, y, folds, fold, config, str(device), progress
            )
            predictions[test], logits[test], embeddings[test] = pred, scores, vectors
            model_path = output_dir / f"cnn_fold_{fold}.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "architecture": "compact_cnn_v1",
                    "state_dict": model.state_dict(),
                    "cnn_config": asdict(config),
                    "feature_config": feature_config.as_dict(),
                    "class_names": categories,
                    "class_ids": labels.tolist(),
                    "test_fold": fold,
                    "selected_epochs": result["selected_epochs"],
                    "spectrogram_cache_sha256": run["spectrogram_cache_sha256"],
                    "source_sha256": source_hashes,
                },
                model_path,
            )
            result["model_bytes"] = model_path.stat().st_size
            results.append(result)
            write_json(output_dir / f"fold_{fold}.json", result)
            print(f"Fold {fold} complete: {result['test_metrics']}", flush=True)
    if (
        np.any(predictions < 0)
        or not np.isfinite(logits).all()
        or not np.isfinite(embeddings).all()
    ):
        raise RuntimeError("Incomplete out-of-fold CNN predictions")
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
        scores=logits,
        embeddings=embeddings,
        score_type=np.array("logits"),
        class_ids=labels,
        filenames=cache["filenames"],
        folds=folds,
        targets=y,
        predictions=predictions,
    )
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class/predicted_class", *[categories[int(c)] for c in labels]])
        for label, row in zip(labels, confusion_matrix(y, predictions, labels=labels)):
            writer.writerow([categories[int(label)], *row])
    for name in ("accuracy", "macro_f1"):
        values = [r["test_metrics"][name] for r in results]
        run[name] = {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}
    run.update(
        {
            "folds": results,
            "training_and_evaluation_seconds": perf_counter() - start,
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
    write_json(output_dir / "summary.json", run)
    write_json(output_dir / "progress.json", {"status": "complete", "folds": 5})
    return run


def load_cnn(model_path):
    bundle = torch.load(model_path, map_location="cpu", weights_only=True)
    if bundle.get("schema_version") != 1 or bundle.get("architecture") != "compact_cnn_v1":
        raise ValueError("Unsupported CNN checkpoint")
    config = FeatureConfig(**bundle["feature_config"])
    model = CompactCNN(config.n_mels, len(bundle["class_ids"]), CNNConfig(**bundle["cnn_config"]))
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    return model, config, bundle


def predict_cnn(model_path, audio_path):
    model, config, bundle = load_cnn(model_path)
    audio = load_audio(audio_path, config)
    X = log_mel(audio, config).T.astype(np.float32)[None]
    scores, _ = predict_arrays(model, X, np.array([0]), 1, torch.device("cpu"))
    order = np.argsort(-scores[0])[:5]
    label = int(order[0])
    return {
        "prediction": label,
        "category": bundle["class_names"][label],
        "model_test_fold": bundle["test_fold"],
        "score_type": "logits",
        "top_scores": [
            {
                "class_id": int(i),
                "category": bundle["class_names"][int(i)],
                "logit": float(scores[0, i]),
            }
            for i in order
        ],
    }


def benchmark_cnn(model_path, audio_path, runs=100, warmup=10):
    if runs < 1 or warmup < 1:
        raise ValueError("runs and warmup must be positive")
    model, config, _ = load_cnn(model_path)
    audio = load_audio(audio_path, config)
    processing, inference = [], []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1), torch.inference_mode():
            for i in range(warmup + runs):
                start = perf_counter()
                values = log_mel(audio, config).T.astype(np.float32)
                inputs = torch.from_numpy(values[None, None])
                processed = perf_counter()
                model(inputs)
                finished = perf_counter()
                if i >= warmup:
                    processing.append(processed - start)
                    inference.append(finished - processed)
    finally:
        torch.set_num_threads(old_threads)
    return {
        "scope": "single_clip_microbenchmark",
        "audio_file": str(audio_path),
        "audio_seconds": len(audio) / config.sample_rate,
        "batch_size": 1,
        "runs": runs,
        "warmup": warmup,
        "numerical_threads": 1,
        "device": "cpu",
        "preprocessing_median_ms": float(1000 * np.median(processing)),
        "inference_median_ms": float(1000 * np.median(inference)),
        "model_bytes": Path(model_path).stat().st_size,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "excludes": ["disk I/O", "model loading"],
    }

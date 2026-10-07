"""Command-line entry points for clean SVM and compact CNN baselines."""

import argparse
import json
import platform
from collections import Counter
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .artifacts import extract_dataset, load_feature_cache, write_json
from .data import load_audio, load_metadata, source_overlap_report
from .dsp import FeatureConfig
from .features import extract_features
from .spectrograms import extract_spectrograms, load_spectrogram_cache
from .svm import class_scores, evaluate_svm
from .validation import validate_stft


def read_config(path):
    values = json.loads(path.read_text()) if path else {}
    return FeatureConfig(**values.get("features", {})), values.get("svm_grid")


def predict(model_path, audio_path):
    bundle = joblib.load(model_path)
    config = FeatureConfig(**bundle["feature_config"])
    audio = load_audio(audio_path, config)
    model = bundle["model"]
    with threadpool_limits(limits=1):
        features = extract_features(audio, config)[None, :]
        label = int(model.predict(features)[0])
        scores = class_scores(model, features)[0]
    order = np.argsort(-scores)[:5]
    return {
        "prediction": label,
        "category": bundle["class_names"][label],
        "model_test_fold": bundle["test_fold"],
        "top_scores": [
            {
                "class_id": int(model.classes_[i]),
                "category": bundle["class_names"][int(model.classes_[i])],
                "decision_score": float(scores[i]),
            }
            for i in order
        ],
    }


def benchmark(model_path, audio_path, runs=100, warmup=10):
    if runs < 1 or warmup < 1:
        raise ValueError("runs and warmup must be positive")
    bundle = joblib.load(model_path)
    config = FeatureConfig(**bundle["feature_config"])
    audio = load_audio(audio_path, config)
    model = bundle["model"]
    processing, inference = [], []
    with threadpool_limits(limits=1):
        for _ in range(warmup):
            model.predict(extract_features(audio, config)[None, :])
        for _ in range(runs):
            start = perf_counter()
            features = extract_features(audio, config)[None, :]
            processing.append(perf_counter() - start)
            start = perf_counter()
            model.predict(features)
            inference.append(perf_counter() - start)
    return {
        "scope": "single_clip_microbenchmark",
        "audio_file": str(audio_path),
        "audio_seconds": len(audio) / config.sample_rate,
        "batch_size": 1,
        "runs": runs,
        "warmup": warmup,
        "numerical_threads": 1,
        "preprocessing_median_ms": float(1000 * np.median(processing)),
        "inference_median_ms": float(1000 * np.median(inference)),
        "model_bytes": Path(model_path).stat().st_size,
        "support_vectors": int(model.named_steps["svc"].support_vectors_.shape[0]),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "excludes": ["disk I/O", "model loading", "external pretraining"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-stft", help="Compare custom STFT with SciPy")
    validate.add_argument("--config", type=Path)
    validate.add_argument("--output", type=Path)
    inspect = commands.add_parser("inspect-data", help="Validate official metadata and audio paths")
    inspect.add_argument("--data-root", type=Path, required=True)
    inspect.add_argument(
        "--strict-sources",
        action="store_true",
        help="Fail if any source ID appears in more than one fold",
    )
    extract = commands.add_parser("extract-features", help="Build a recorded 92-feature cache")
    extract.add_argument("--data-root", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--config", type=Path)
    train = commands.add_parser("train-svm", help="Run all five official outer folds")
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--config", type=Path)
    infer = commands.add_parser("predict", help="Classify mono audio at the model's sample rate")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--audio", type=Path, required=True)
    cost = commands.add_parser("benchmark", help="Measure CPU preprocessing and SVM inference")
    cost.add_argument("--model", type=Path, required=True)
    cost.add_argument("--audio", type=Path, required=True)
    cost.add_argument("--runs", type=int, default=100)
    cost.add_argument("--warmup", type=int, default=10)
    cost.add_argument("--output", type=Path)
    mel = commands.add_parser("extract-log-mel", help="Cache unnormalised log-Mel inputs")
    mel.add_argument("--data-root", type=Path, required=True)
    mel.add_argument("--output", type=Path, required=True)
    mel.add_argument("--config", type=Path)
    cnn = commands.add_parser("train-cnn", help="Select duration and refit CNN on five folds")
    cnn.add_argument("--spectrograms", type=Path, required=True)
    cnn.add_argument("--output", type=Path, required=True)
    cnn.add_argument("--config", type=Path)
    cnn.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    for name in ("predict-cnn", "benchmark-cnn"):
        command = commands.add_parser(name)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--audio", type=Path, required=True)
        if name == "benchmark-cnn":
            command.add_argument("--runs", type=int, default=100)
            command.add_argument("--warmup", type=int, default=10)
            command.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-stft":
            config, _ = read_config(args.config)
            result = validate_stft(config)
        elif args.command == "inspect-data":
            clips = load_metadata(args.data_root, strict_sources=args.strict_sources)
            result = {
                "clips": len(clips),
                "classes": len({c.target for c in clips}),
                "fold_counts": dict(sorted(Counter(c.fold for c in clips).items())),
                "source_recordings": len({c.source_id for c in clips}),
                "source_overlaps": source_overlap_report(clips),
            }
        elif args.command == "extract-features":
            config, _ = read_config(args.config)
            result = extract_dataset(args.data_root, args.output, config)
        elif args.command == "train-svm":
            config, grid = read_config(args.config)
            if args.config and load_feature_cache(args.features)[1] != config:
                raise ValueError("Training config does not match the cached feature configuration")
            summary = evaluate_svm(args.features, args.output, grid)
            result = {
                "accuracy": summary["accuracy"],
                "macro_f1": summary["macro_f1"],
                "summary": str(args.output / "summary.json"),
            }
        elif args.command == "predict":
            result = predict(args.model, args.audio)
        elif args.command == "extract-log-mel":
            config, _ = read_config(args.config)
            result = extract_spectrograms(args.data_root, args.output, config)
        elif args.command == "train-cnn":
            from .cnn import CNNConfig, evaluate_cnn

            values = json.loads(args.config.read_text()) if args.config else {}
            if args.config and load_spectrogram_cache(args.spectrograms)[1] != FeatureConfig(
                **values.get("features", {})
            ):
                raise ValueError("Training config does not match the cached feature configuration")
            summary = evaluate_cnn(
                args.spectrograms, args.output, CNNConfig(**values.get("cnn", {})), args.device
            )
            result = {
                "accuracy": summary["accuracy"],
                "macro_f1": summary["macro_f1"],
                "summary": str(args.output / "summary.json"),
            }
        elif args.command == "predict-cnn":
            from .cnn import predict_cnn

            result = predict_cnn(args.model, args.audio)
        elif args.command == "benchmark-cnn":
            from .cnn import benchmark_cnn

            result = benchmark_cnn(args.model, args.audio, args.runs, args.warmup)
        else:
            result = benchmark(args.model, args.audio, args.runs, args.warmup)
        if args.command in {"validate-stft", "benchmark", "benchmark-cnn"} and args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        parser.exit(2, "error: CNN commands require PyTorch; install the 'cnn' extra.\n")


if __name__ == "__main__":
    main()

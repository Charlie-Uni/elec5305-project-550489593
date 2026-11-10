"""Command-line entry points for SVM, compact CNN and frozen CLAP experiments."""

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


def load_clap_system(checkpoint, data_root, prompt_template, device="cpu", feature_config=None):
    """Class names come from the dataset metadata so prompts match the official labels."""
    from .clap_system import clap_system_from_checkpoint

    if data_root is None:
        raise ValueError("CLAP needs --data-root to read the class names")
    clips = load_metadata(data_root, require_complete=False)
    class_names = {c.target: c.category for c in clips}
    return clap_system_from_checkpoint(
        checkpoint,
        class_names,
        feature_config=feature_config or FeatureConfig(),
        prompt_template=prompt_template,
        device=device,
    )


def read_interference_config(path):
    from .mixing import interference_config_from_json

    values = json.loads(path.read_text()) if path else {}
    return interference_config_from_json(values.get("interference", {}))


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
    demand = commands.add_parser("prepare-demand", help="Resample DEMAND channels to 44.1 kHz")
    demand.add_argument("--demand-root", type=Path, required=True)
    demand.add_argument("--output", type=Path, required=True)
    demand.add_argument("--config", type=Path)
    manifest = commands.add_parser(
        "build-manifest", help="Plan every canonical noise and distractor mixture"
    )
    manifest.add_argument("--data-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--config", type=Path)
    manifest.add_argument("--demand-cache", type=Path)
    manifest.add_argument("--limit-per-fold", type=int, help="Diagnostic subset: targets per fold")
    manifest.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Diagnostic only: accept a dataset that is not the complete ESC-50",
    )
    render = commands.add_parser("render-mixtures", help="Write WAV examples from a manifest")
    render.add_argument("--manifest", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--data-root", type=Path)
    render.add_argument("--rows", type=int, nargs="*")
    render.add_argument("--per-condition", type=int)
    evaluate = commands.add_parser(
        "evaluate-interference", help="Classify every manifest mixture with the fold models"
    )
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--data-root", type=Path)
    evaluate.add_argument("--svm-dir", type=Path, help="Directory holding svm_fold_1..5.joblib")
    evaluate.add_argument("--cnn-dir", type=Path, help="Directory holding cnn_fold_1..5.pt")
    evaluate.add_argument("--limit", type=int, help="Diagnostic: first N manifest rows only")
    evaluate.add_argument("--clap-checkpoint", type=Path, help="Frozen LAION-CLAP checkpoint")
    evaluate.add_argument("--clap-prompt", default="the sound of {label}")
    evaluate.add_argument("--clap-device", choices=("cpu", "mps", "cuda"), default="cpu")
    evaluate.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Mixtures per batch; CLAP also embeds their controls",
    )
    clap_clean = commands.add_parser(
        "evaluate-clap-clean", help="Zero-shot CLAP on every clean clip (clean reference)"
    )
    clap_clean.add_argument("--data-root", type=Path, required=True)
    clap_clean.add_argument("--clap-checkpoint", type=Path, required=True)
    clap_clean.add_argument("--output", type=Path, required=True)
    clap_clean.add_argument("--clap-prompt", default="the sound of {label}")
    clap_clean.add_argument("--clap-device", choices=("cpu", "mps", "cuda"), default="cpu")
    clap_clean.add_argument("--batch-size", type=int, default=1)
    clap_cost = commands.add_parser("benchmark-clap", help="Single-clip cost of frozen CLAP")
    clap_cost.add_argument("--data-root", type=Path, required=True)
    clap_cost.add_argument("--clap-checkpoint", type=Path, required=True)
    clap_cost.add_argument("--audio", type=Path, required=True)
    clap_cost.add_argument("--runs", type=int, default=100)
    clap_cost.add_argument("--warmup", type=int, default=10)
    clap_cost.add_argument("--output", type=Path)
    clap_cost.add_argument("--clap-prompt", default="the sound of {label}")
    clap_cost.add_argument("--clap-device", choices=("cpu", "mps", "cuda"), default="cpu")
    tables = commands.add_parser(
        "summarise-interference", help="Per-fold degradation, capture and margin tables"
    )
    tables.add_argument("--evaluation", type=Path, required=True)
    tables.add_argument("--svm-clean", type=Path, help="Baseline run with predictions.csv")
    tables.add_argument("--cnn-clean", type=Path, help="Baseline run with predictions.csv")
    tables.add_argument("--clap-clean", type=Path, help="evaluate-clap-clean output directory")
    tables.add_argument("--output", type=Path)
    figures = commands.add_parser("make-figures", help="Report figures with their data tables")
    figures.add_argument("--evaluation", type=Path, required=True)
    figures.add_argument("--output", type=Path, required=True)
    figures.add_argument("--svm-clean", type=Path)
    figures.add_argument("--cnn-clean", type=Path)
    figures.add_argument("--clap-clean", type=Path)
    figures.add_argument("--svm-cost", type=Path, help="cost.json from benchmark")
    figures.add_argument("--cnn-cost", type=Path, help="cost.json from benchmark-cnn")
    figures.add_argument("--clap-cost", type=Path, help="cost.json from benchmark-clap")
    figures.add_argument("--level", type=float, default=5.0, help="Focus level for Figs 5-8")
    figures.add_argument(
        "--additional-evaluation",
        type=Path,
        action="append",
        default=[],
        help="Additional systems from the identical manifest (for example, CLAP)",
    )
    figures.add_argument(
        "--svm-models",
        type=Path,
        help="Override the recorded SVM model directory for training-fold scales",
    )
    diagnostics = commands.add_parser(
        "analyse-diagnostics", help="Paired bootstrap, feature groups and CLAP competition"
    )
    diagnostics.add_argument("--own-evaluation", type=Path, action="append", default=[])
    diagnostics.add_argument("--target-evaluation", type=Path, action="append", default=[])
    diagnostics.add_argument("--svm-models", type=Path, required=True)
    diagnostics.add_argument("--output", type=Path, required=True)
    diagnostics.add_argument("--replicates", type=int, default=2000)
    diagnostics.add_argument("--seed", type=int, default=530509)
    augmented = commands.add_parser(
        "train-augmented-cnn", help="Train CNN with isolated WGN or semantic augmentation"
    )
    augmented.add_argument("--spectrograms", type=Path, required=True)
    augmented.add_argument("--data-root", type=Path, required=True)
    augmented.add_argument("--bank-root", type=Path, required=True)
    augmented.add_argument("--mode", choices=["wgn", "semantic"], required=True)
    augmented.add_argument("--config", type=Path)
    augmented.add_argument("--output", type=Path, required=True)
    augmented.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    augmented.add_argument("--prepare-only", action="store_true")
    augmented_svm = commands.add_parser("train-augmented-svm")
    augmented_svm.add_argument("--features", type=Path, required=True)
    augmented_svm.add_argument("--spectrograms", type=Path, required=True)
    augmented_svm.add_argument("--data-root", type=Path, required=True)
    augmented_svm.add_argument("--bank-root", type=Path, required=True)
    augmented_svm.add_argument("--mode", choices=["wgn", "semantic"], required=True)
    augmented_svm.add_argument("--config", type=Path)
    augmented_svm.add_argument("--output", type=Path, required=True)
    overlap = commands.add_parser(
        "overlap-features", help="Spectral/temporal overlap measures for every manifest row"
    )
    overlap.add_argument("--manifest", type=Path, required=True)
    overlap.add_argument("--output", type=Path, required=True)
    overlap.add_argument("--data-root", type=Path)
    overlap.add_argument("--limit", type=int)
    overlap.add_argument("--require-demand", action="store_true")
    overlap_tables = commands.add_parser(
        "analyse-overlap", help="Accuracy by acoustic overlap; matched semantic vs DEMAND"
    )
    overlap_tables.add_argument("--features", type=Path, required=True)
    overlap_tables.add_argument("--output", type=Path, required=True)
    overlap_tables.add_argument("--svm-eval", type=Path)
    overlap_tables.add_argument("--cnn-eval", type=Path)
    overlap_tables.add_argument("--clap-eval", type=Path)
    overlap_tables.add_argument("--level", type=float, default=5.0)
    readout = commands.add_parser(
        "analyse-readout", help="Exact top-k readout outcomes for semantic mixtures"
    )
    readout.add_argument("--output", type=Path, required=True)
    readout.add_argument("--k", type=int, default=2)
    readout.add_argument("--system", choices=["svm", "cnn", "clap"], default="clap")
    readout.add_argument(
        "--scores",
        type=Path,
        nargs="*",
        default=[],
        help="region=path/to/eval_dir pairs holding SYSTEM_scores.npz and SYSTEM_predictions.csv",
    )
    readout.add_argument("--competition-csv", type=Path, help="clap_competition_trials.csv")
    readout.add_argument(
        "--predictions",
        type=Path,
        nargs="*",
        default=[],
        help="region=path/to/clap_predictions.csv pairs (with --competition-csv)",
    )
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
        elif args.command == "train-augmented-cnn":
            from .augmentation import AugmentationBanks
            from .cnn import CNNConfig, evaluate_cnn
            from .data import outer_split

            values = json.loads(args.config.read_text()) if args.config else {}
            config = CNNConfig(**values.get("cnn", {}))
            banks = AugmentationBanks(
                args.spectrograms, args.data_root, args.bank_root, args.mode, config.seed
            )
            if args.config and banks.config != FeatureConfig(**values.get("features", {})):
                raise ValueError("Training config differs from the augmentation bank")
            if args.prepare_only:
                prepared = []
                for fold in range(1, 6):
                    train, validation, _ = outer_split(banks.cache["folds"], fold)
                    for phase, indices in (
                        ("selection", train),
                        ("refit", np.concatenate((train, validation))),
                    ):
                        prepared.append(banks(fold, phase, indices).metadata)
                write_json(args.output, {"settings": banks.settings, "banks": prepared})
                result = {"prepared_banks": len(prepared), "output": str(args.output)}
            else:
                summary = evaluate_cnn(
                    args.spectrograms, args.output, config, args.device, augmentation=banks
                )
                result = {"accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"]}
        elif args.command == "train-augmented-svm":
            from .augmentation import AugmentationBanks

            config, grid = read_config(args.config)
            banks = AugmentationBanks(args.spectrograms, args.data_root, args.bank_root, args.mode)
            if args.config and config != banks.config:
                raise ValueError("Training config differs from the augmentation bank")
            summary = evaluate_svm(args.features, args.output, grid, augmentation=banks)
            result = {"accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"]}
        elif args.command == "predict-cnn":
            from .cnn import predict_cnn

            result = predict_cnn(args.model, args.audio)
        elif args.command == "benchmark-cnn":
            from .cnn import benchmark_cnn

            result = benchmark_cnn(args.model, args.audio, args.runs, args.warmup)
        elif args.command == "prepare-demand":
            from .mixing import prepare_demand

            interference = read_interference_config(args.config)
            result = prepare_demand(
                args.demand_root,
                args.output,
                interference.demand_environments,
                interference.demand_channel,
            )
        elif args.command == "build-manifest":
            from .mixing import build_manifest

            config, _ = read_config(args.config)
            result = build_manifest(
                args.data_root,
                args.output,
                read_interference_config(args.config),
                config,
                args.demand_cache,
                args.limit_per_fold,
                require_complete=not args.allow_incomplete,
            )
            result = {"manifest": str(args.output / "manifest.csv"), **result["summary"]}
        elif args.command == "render-mixtures":
            from .mixing import render_examples

            written = render_examples(
                args.manifest, args.output, args.data_root, args.rows, args.per_condition
            )
            result = {"output": str(args.output), "files": len(written)}
        elif args.command == "evaluate-interference":
            from .interference_eval import evaluate_manifest

            extra = []
            if args.clap_checkpoint is not None:
                meta = json.loads((args.manifest / "manifest_meta.json").read_text())
                root = args.data_root or Path(meta["data_root"])
                extra.append(
                    load_clap_system(
                        args.clap_checkpoint,
                        root,
                        args.clap_prompt,
                        args.clap_device,
                        FeatureConfig(**meta["feature_config"]),
                    )
                )
            summary = evaluate_manifest(
                args.manifest,
                args.output,
                args.data_root,
                args.svm_dir,
                args.cnn_dir,
                args.limit,
                extra_systems=extra,
                batch_size=args.batch_size,
            )
            result = {
                "summary": str(args.output / "summary.json"),
                "rows_evaluated": summary["rows_evaluated"],
                "systems": {
                    name: {k: round(v["accuracy"], 4) for k, v in s["conditions"].items()}
                    for name, s in summary["systems"].items()
                },
            }
        elif args.command == "evaluate-clap-clean":
            from .clap_system import evaluate_clean

            system = load_clap_system(
                args.clap_checkpoint, args.data_root, args.clap_prompt, args.clap_device
            )
            summary = evaluate_clean(
                system, args.data_root, args.output, batch_size=args.batch_size
            )
            result = {
                "accuracy": summary["accuracy"],
                "macro_f1": summary["macro_f1"],
                "summary": str(args.output / "summary.json"),
            }
        elif args.command == "benchmark-clap":
            from .clap_system import benchmark_clap

            system = load_clap_system(
                args.clap_checkpoint, args.data_root, args.clap_prompt, args.clap_device
            )
            result = benchmark_clap(system, args.audio, runs=args.runs, warmup=args.warmup)
        elif args.command == "summarise-interference":
            from .analysis import summarise_evaluation

            clean_dirs = {
                name: path
                for name, path in (
                    ("svm", args.svm_clean),
                    ("cnn", args.cnn_clean),
                    ("clap", args.clap_clean),
                )
                if path is not None
            }
            report = summarise_evaluation(args.evaluation, clean_dirs, args.output)
            result = {
                system: {
                    name: {
                        "macro_f1": round(entry["macro_f1"]["mean"], 4),
                        "capture_rate": None
                        if not entry.get("capture") or entry["capture"]["capture_rate"] is None
                        else round(entry["capture"]["capture_rate"], 4),
                    }
                    for name, entry in table.items()
                }
                for system, table in report["systems"].items()
            }
        elif args.command == "analyse-diagnostics":
            from .diagnostics import run_diagnostics

            evaluations = {
                name: paths
                for name, paths in (
                    ("own", args.own_evaluation),
                    ("target", args.target_evaluation),
                )
                if paths
            }
            if not evaluations:
                raise ValueError("At least one evaluation directory is required")
            result = run_diagnostics(
                evaluations, args.output, args.svm_models, args.replicates, args.seed
            )
        elif args.command == "make-figures":
            from .figures import make_figures

            clean_dirs = {
                name: path
                for name, path in (
                    ("svm", args.svm_clean),
                    ("cnn", args.cnn_clean),
                    ("clap", args.clap_clean),
                )
                if path is not None
            }
            costs = {
                name: path
                for name, path in (
                    ("svm", args.svm_cost),
                    ("cnn", args.cnn_cost),
                    ("clap", args.clap_cost),
                )
                if path is not None
            }
            result = make_figures(
                args.evaluation,
                clean_dirs,
                args.output,
                costs,
                args.level,
                args.additional_evaluation,
                args.svm_models,
            )
        elif args.command == "overlap-features":
            from .overlap_analysis import compute_overlap_features

            result = compute_overlap_features(
                args.manifest, args.output, args.data_root, args.limit, args.require_demand
            )
        elif args.command == "analyse-overlap":
            from .overlap_analysis import analyse_overlap

            prediction_dirs = {
                name: path
                for name, path in (
                    ("svm", args.svm_eval),
                    ("cnn", args.cnn_eval),
                    ("clap", args.clap_eval),
                )
                if path is not None
            }
            summary = analyse_overlap(args.features, prediction_dirs, args.output, args.level)
            result = {"output": str(args.output), "systems": list(summary["systems"])}
        elif args.command == "analyse-readout":
            from .readout_analysis import analyse_readout, load_trials

            def pairs(values):
                parsed = {}
                for value in values:
                    region, _, path = str(value).partition("=")
                    if not path:
                        raise ValueError("Expected region=path")
                    parsed[region] = Path(path)
                return parsed

            trials = load_trials(
                scores=pairs(args.scores),
                competition_csv=args.competition_csv,
                predictions=pairs(args.predictions),
                system=args.system,
            )
            summary = analyse_readout(trials, args.output, args.k)
            result = {"output": str(args.output), "trials": summary["trials"]}
        else:
            result = benchmark(args.model, args.audio, args.runs, args.warmup)
        if (
            args.command in {"validate-stft", "benchmark", "benchmark-cnn", "benchmark-clap"}
            and args.output
        ):
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

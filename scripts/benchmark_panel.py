"""Balanced 250-clip CPU latency panel, run after training/evaluation is idle.

One deterministic clip per official fold/class, three timed repetitions per clip.
Preprocessing and inference plus all-class scoring are timed separately. Loading
audio/models and cached text preparation are excluded, and explicitly recorded.
"""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.data import load_audio, load_metadata
from esc_robustness.diagnostics import write_csv
from esc_robustness.dsp import FeatureConfig, log_mel
from esc_robustness.features import extract_features
from esc_robustness.svm import class_scores


def run(system_name, model_path, data_root, output, repeats=3):
    if repeats < 3:
        raise ValueError("Use at least three repetitions per clip")
    output.mkdir(parents=True, exist_ok=False)
    config = FeatureConfig()
    metadata = load_metadata(data_root)
    panel = {}
    for clip in sorted(metadata, key=lambda c: c.filename):
        panel.setdefault((clip.fold, clip.target), clip)
    if set(panel) != {(fold, cls) for fold in range(1, 6) for cls in range(50)}:
        raise ValueError("Expected the complete five-fold, 50-class panel")
    write_json(
        output / "panel.json",
        [
            dict(
                filename=c.filename,
                fold=c.fold,
                target=c.target,
                audio_sha256=sha256_file(data_root / "audio" / c.filename),
            )
            for _, c in sorted(panel.items())
        ],
    )
    if system_name == "clap":
        from esc_robustness.clap_system import clap_system_from_checkpoint

        system = clap_system_from_checkpoint(
            model_path, {c.target: c.category for c in metadata}, device="cpu"
        )

        def preprocess(audio):
            return system.preprocess(audio)

        def infer(fold, prepared):
            return system.text_embeddings @ system.embed(prepared)

        model_hashes = {"checkpoint": sha256_file(model_path)}
        model_bytes = model_path.stat().st_size
        parameter_count = sum(p.numel() for p in system.module.module.parameters())
    elif system_name == "cnn":
        import torch

        from esc_robustness.interference_eval import CNNSystem

        system = CNNSystem(model_path, config)

        def preprocess(audio):
            spec = log_mel(audio, config).T.astype(np.float32)
            return torch.from_numpy(spec[None, None])

        def infer(fold, prepared):
            with torch.inference_mode():
                model = system.models[fold]
                return model.classifier(model.embedding(prepared)).numpy()[0]

        paths = [model_path / f"cnn_fold_{f}.pt" for f in range(1, 6)]
        model_hashes = {p.name: sha256_file(p) for p in paths}
        model_bytes = {p.name: p.stat().st_size for p in paths}
        parameter_count = sum(p.numel() for p in system.models[1].parameters())
    else:
        from esc_robustness.interference_eval import SVMSystem

        system = SVMSystem(model_path, config)

        def preprocess(audio):
            return extract_features(audio, config)[None, :]

        def infer(fold, prepared):
            return class_scores(system.models[fold]["model"], prepared)[0]

        paths = [model_path / f"svm_fold_{f}.joblib" for f in range(1, 6)]
        model_hashes = {p.name: sha256_file(p) for p in paths}
        model_bytes = {p.name: p.stat().st_size for p in paths}
        parameter_count = None
    records = []
    started = perf_counter()
    with threadpool_limits(limits=1):
        observed_pools = threadpool_info()
        if any(pool["num_threads"] != 1 for pool in observed_pools):
            raise ValueError("Benchmark numerical thread count differs")
        for fold in range(1, 6):
            first = load_audio(data_root / "audio" / panel[(fold, 0)].filename, config)
            for _ in range(10):
                infer(fold, preprocess(first))
            # Rotate class order by fold; all systems use identical ordering.
            for cls in np.roll(np.arange(50), fold * 7):
                clip = panel[(fold, int(cls))]
                audio_path = data_root / "audio" / clip.filename
                audio = load_audio(audio_path, config)
                for repetition in range(repeats):
                    start = perf_counter()
                    prepared = preprocess(audio)
                    processed = perf_counter()
                    scores = infer(fold, prepared)
                    prediction = int(np.argmax(scores))
                    finished = perf_counter()
                    if scores.shape != (50,) or not np.isfinite(scores).all():
                        raise ValueError("Invalid benchmark scores")
                    records.append(
                        dict(
                            system=system_name,
                            filename=clip.filename,
                            fold=fold,
                            target=clip.target,
                            repetition=repetition,
                            prediction=prediction,
                            preprocessing_ms=1000 * (processed - start),
                            inference_scoring_ms=1000 * (finished - processed),
                            total_ms=1000 * (finished - start),
                        )
                    )
            print(f"Benchmark {system_name}: fold {fold}/5", flush=True)
    write_csv(output / "trials.csv", records)
    summary = {
        "scope": "one clip per class and official fold, 250 unique five-second clips",
        "system": system_name,
        "batch_size": 1,
        "device": "cpu",
        "numerical_threads": 1,
        "observed_thread_pools": observed_pools,
        "repetitions_per_clip": repeats,
        "warmup_per_model": 10,
        "trials": len(records),
        "elapsed_seconds_including_loading_between_trials": perf_counter() - started,
        "model_sha256": model_hashes,
        "serialized_model_bytes": model_bytes,
        "parameter_count": parameter_count,
        "excludes": [
            "disk I/O",
            "model loading",
            "text embedding preparation",
            "external pretraining",
        ],
        "includes": ["frontend", "audio representation", "50-class scoring", "argmax"],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_name": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if platform.system() == "Darwin"
        else platform.processor(),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")},
        "interpretation": "Latency distribution over clips and repetitions; not independent accuracy observations or a general hardware guarantee",
        "trials_sha256": sha256_file(output / "trials.csv"),
        "source_sha256": sha256_file(Path(__file__)),
    }
    if system_name == "svm":
        summary["support_vectors_per_fold"] = {
            str(fold): int(bundle["model"].named_steps["svc"].support_vectors_.shape[0])
            for fold, bundle in system.models.items()
        }
    for package in (
        ["torch"]
        if system_name == "cnn"
        else ["torch", "laion-clap", "transformers"]
        if system_name == "clap"
        else []
    ):
        summary["versions"][package] = importlib.metadata.version(package)
    for key in ("preprocessing_ms", "inference_scoring_ms", "total_ms"):
        values = np.array([r[key] for r in records])
        summary[key] = dict(
            zip(
                ("median", "p25", "p75", "p95"),
                map(float, np.quantile(values, [0.5, 0.25, 0.75, 0.95])),
            )
        )
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("svm", "cnn", "clap"), required=True)
    for argument in ("model", "data-root", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.system, args.model, args.data_root, args.output, args.repeats), indent=2
        )
    )

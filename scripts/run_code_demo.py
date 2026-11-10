"""Recompute one published mixture and its predictions, then check saved evidence."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def progress(message):
    print(message, flush=True)


def run_system(args):
    import numpy as np
    import soundfile as sf

    from esc_robustness.artifacts import sha256_file
    from esc_robustness.dsp import FeatureConfig
    from esc_robustness.mixing import MixtureRenderer, interference_config_from_json, parse_row

    manifest = ROOT / "results/mixture_manifests_20261109/manifest_own"
    meta = json.loads((manifest / "manifest_meta.json").read_text())
    with gzip.open(manifest / "manifest.csv.gz", "rb") as f:
        manifest_bytes = f.read()
    row = parse_row(
        next(r for r in csv.DictReader(io.StringIO(manifest_bytes.decode())) if r["row"] == "91")
    )
    if row["target_file"] != "1-101404-A-34.wav" or row["interferer"] != "1-13613-A-37.wav":
        raise ValueError("The documented demonstration pair has changed")
    data_root = args.runtime_root / "data/ESC-50"
    if sha256_file(data_root / "meta/esc50.csv") != meta["metadata_sha256"]:
        raise ValueError("ESC-50 metadata hash differs")
    config = FeatureConfig(**meta["feature_config"])
    renderer = MixtureRenderer(
        data_root, config, interference_config_from_json(meta["interference_config"])
    )
    progress(
        f"[{args.system.upper()}] Reading the two source recordings; rebuilding own-power +5 dB mixture."
    )
    rendered = renderer.render(row)
    peak = float(np.max(np.abs(rendered["mixture"])))
    assert peak <= 0.990000001
    progress(
        f"[DSP] Achieved SIR {rendered['measurements']['achieved_db']:.6f} dB; peak {peak:.6f}; no hard clipping."
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["clean_matched", "mixture"]:
        sf.write(args.output / f"{name}.wav", rendered[name], config.sample_rate, subtype="PCM_16")
    with (data_root / "meta/esc50.csv").open() as f:
        class_names = {int(r["target"]): r["category"] for r in csv.DictReader(f)}
    start = time.perf_counter()
    progress(
        f"[{args.system.upper()}] Loading the original model weights. Predictions will be computed from the rebuilt waveforms."
    )
    if args.system == "svm":
        from esc_robustness.interference_eval import SVMSystem

        model_path = (
            args.runtime_root / "artifacts/interference/v3_20261109/clean_svm_reconstructed"
        )
        system = SVMSystem(model_path, config)
        weight = model_path / "svm_fold_1.joblib"
    elif args.system == "cnn":
        from esc_robustness.interference_eval import CNNSystem

        model_path = args.runtime_root / "artifacts/baselines/clean_cnn"
        system = CNNSystem(model_path, config)
        weight = model_path / "cnn_fold_1.pt"
    else:
        from esc_robustness.clap_system import CLAPSystem, load_laion_clap

        weight = args.runtime_root / "models/clap/630k-audioset-best.pt"
        system = CLAPSystem(load_laion_clap(weight), class_names, config, checkpoint=weight)
    results = {}
    for name in ["clean_matched", "mixture"]:
        scores, _ = system.evaluate(row["fold"], rendered[name])
        if len(scores) != 50 or not np.isfinite(scores).all():
            raise ValueError("Invalid class scores")
        order = np.argsort(-scores, kind="stable")
        pred = int(system.class_ids[order[0]])
        results[name] = dict(
            prediction=pred,
            label=class_names[pred],
            top_two=[class_names[int(system.class_ids[i])] for i in order[:2]],
            target_rank=int(np.where(system.class_ids[order] == row["target_class"])[0][0] + 1),
            target_score=float(scores[np.where(system.class_ids == row["target_class"])[0][0]]),
            distractor_score=float(
                scores[np.where(system.class_ids == row["interferer_class"])[0][0]]
            ),
            scores=scores.tolist(),
        )
        progress(
            f"[{args.system.upper()}] {name}: {class_names[pred]}; target rank {results[name]['target_rank']}; top two {', '.join(results[name]['top_two'])}."
        )
    # Only after actual inference, read the historical predictions for verification.
    example = json.loads((ROOT / "examples/listening_own_20261109/examples.json").read_text())
    expected = next(r for r in example["files"] if r["row"] == 91)["predictions"][args.system]
    for name, key in [("clean_matched", "control_prediction"), ("mixture", "prediction")]:
        if results[name]["prediction"] != int(expected[key]):
            raise ValueError(f"{args.system} prediction differs from the original evaluated case")
    max_error = max(
        abs(results["mixture"]["target_score"] - float(expected["target_score"])),
        abs(results["mixture"]["distractor_score"] - float(expected["interferer_score"])),
    )
    if max_error > 1e-4:
        raise ValueError(f"Score discrepancy exceeds tolerance: {max_error}")
    result = dict(
        system=args.system,
        row=91,
        computed_from_source_audio=True,
        score_reference_read_after_inference=True,
        score_max_abs_error=max_error,
        verified=True,
        source_hashes={k: row[k] for k in ["target_sha256", "interferer_sha256"]},
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        model_sha256=sha256_file(weight),
        measurements=rendered["measurements"],
        peak=peak,
        results=results,
        elapsed_including_model_load_seconds=time.perf_counter() - start,
    )
    (args.output / f"{args.system}.json").write_text(json.dumps(result, indent=2) + "\n")
    progress(
        f"[{args.system.upper()}] Fresh outputs match the saved evaluation. Maximum checked score error {max_error:.3g}."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", type=Path, default=ROOT.parent)
    p.add_argument(
        "--output", type=Path, default=ROOT / "results/final_submission_20261110/demo_run"
    )
    p.add_argument("--system", choices=["all", "svm", "cnn", "clap"], default="all")
    args = p.parse_args()
    args.runtime_root = args.runtime_root.resolve()
    args.output = args.output.resolve()
    if args.system != "all":
        run_system(args)
        return
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        HF_HOME=str(args.runtime_root / "hf-cache"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    progress(
        "REAL CODE DEMONSTRATION: can opening + clock alarm. One illustrative saved pair; no model training."
    )
    for system in ["svm", "cnn", "clap"]:
        python = (
            args.runtime_root
            / ("clap-venv" if system == "clap" else "baseline-venv")
            / "bin/python"
        )
        subprocess.run(
            [
                str(python),
                str(Path(__file__).resolve()),
                "--runtime-root",
                str(args.runtime_root),
                "--output",
                str(args.output),
                "--system",
                system,
            ],
            env=env,
            check=True,
        )
    progress(
        "COMPLETE: all three models ran on source-derived waveforms and passed the reference check."
    )


if __name__ == "__main__":
    main()

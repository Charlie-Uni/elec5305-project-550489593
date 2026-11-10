"""Measure the added readouts on the original balanced CPU panel in both orders."""

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from esc_robustness.artifacts import sha256_file
from esc_robustness.data import load_audio
from esc_robustness.local_alignment import (
    _all_readouts,
    build_system,
    fixed_choice,
    fused_scores,
    unpack_components,
)


def benchmark(args):
    run, output = args.run_dir, args.output
    assert json.loads((run / "complete.json").read_text()) == {"complete": True, "limit": None}
    output.mkdir(parents=True, exist_ok=False)
    panel = json.loads(args.panel.read_text())
    assert len(panel) == 250 and len({(r["fold"], r["target"]) for r in panel}) == 250
    chosen = json.loads((run / "selection.json").read_text())
    with np.load(run / "clean/decision_scores.npz") as f:
        clean = {name: f[name] for name in f.files}
    with np.load(run / "text_embeddings.npz") as f:
        single, temperatures = f["single"], tuple(f["temperatures"])
    rules = {
        "fixed": fixed_choice(temperatures),
        "calibrated": {
            int(fold): {"choice": (temperatures.index(row["choice"]["temperature"]),
                                    row["choice"]["beta_text"], row["choice"]["beta_audio"])}
            for fold, row in chosen["folds"].items()
        },
    }
    expected = _all_readouts(clean["scores"], *unpack_components(clean["embeddings"], 50, temperatures),
                             clean["folds"], single, rules)
    indices = {name: i for i, name in enumerate(clean["filenames"])}
    with threadpool_limits(limits=1):
        system, preparation = build_system(
            args.checkpoint, args.templates, args.data_root, "cpu", args.text_cache
        )

        def score(prepared, readout, fold):
            if readout == "ensemble":
                return system.text_embeddings @ system.embed(prepared)
            rule = chosen["fixed_rule"] if readout == "aligned_fixed" else chosen["folds"][str(fold)]["choice"]
            if rule["beta_text"] == 0 and rule["beta_audio"] == 0:
                return system.text_embeddings @ system.embed(prepared)
            pooled, text, audio = system.branch_scores(prepared[None], [rule["temperature"]])
            return fused_scores(
                pooled @ system.text_embeddings.T,
                np.stack(text, 1).astype(np.float32).astype(float),
                np.stack(audio, 1).astype(np.float32).astype(float),
                (0, rule["beta_text"], rule["beta_audio"]),
            )[0]

        first = panel[0]
        warm_audio = load_audio(args.data_root / "audio" / first["filename"])
        variants = ["ensemble", "aligned_fixed", "aligned_calibrated"]
        for readout in variants:
            for _ in range(5):
                score(system.preprocess(warm_audio), readout, first["fold"])
        trials = []
        for order, sequence in (("forward", variants), ("reverse", list(reversed(variants)))):
            for row in np.random.default_rng(5311).permutation(panel):
                path = args.data_root / "audio" / row["filename"]
                assert sha256_file(path) == row["audio_sha256"]
                audio = load_audio(path)
                for repetition in range(3):
                    for readout in sequence:
                        system.synchronize()
                        start = perf_counter()
                        prepared = system.preprocess(audio)
                        ready = perf_counter()
                        scores = score(prepared, readout, row["fold"])
                        system.synchronize()
                        end = perf_counter()
                        reference = expected[readout][indices[row["filename"]]]
                        difference = float(np.max(np.abs(scores - reference)))
                        assert difference <= 1e-4
                        trials.append({
                            "order": order, "readout": readout, "filename": row["filename"],
                            "fold": row["fold"], "target": row["target"], "repetition": repetition,
                            "preprocessing_ms": 1000 * (ready - start),
                            "inference_scoring_ms": 1000 * (end - ready),
                            "total_ms": 1000 * (end - start), "prediction": int(scores.argmax()),
                            "reference_prediction": int(reference.argmax()),
                            "max_abs_score_difference": difference,
                        })
            print(f"Balanced local-alignment cost: {order} complete", flush=True)
    summary = []
    for order in ("forward", "reverse"):
        for readout in variants:
            selected = [r for r in trials if r["order"] == order and r["readout"] == readout]
            assert len(selected) == 750
            summary.append({
                "order": order, "readout": readout, "queries": len(selected),
                **{name + "_median_ms": float(np.median([r[name + "_ms"] for r in selected]))
                   for name in ("preprocessing", "inference_scoring", "total")},
                "total_p95_ms": float(np.quantile([r["total_ms"] for r in selected], .95)),
            })
    for name, values in (("trials.csv", trials), ("summary.csv", summary)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    audit = {
        "verified": True, "clips": 250, "repetitions": 3, "orders": 2,
        "queries": len(trials), "numerical_threads": 1, "batch_size": 1,
        "input_cache_bypassed": True, "panel_sha256": sha256_file(args.panel),
        "prediction_disagreements": sum(r["prediction"] != r["reference_prediction"] for r in trials),
        "max_abs_score_difference": max(r["max_abs_score_difference"] for r in trials),
        "source_sha256": sha256_file(Path(__file__)),
        "platform": platform.platform(), "checkpoint_sha256": system.module.checkpoint_sha256,
        "text_preparation": preparation,
        "scope": "same original 250-clip class/fold-balanced panel; paired readout orders; disk/model/text setup excluded",
        "output_sha256": {name: sha256_file(output / name) for name in ("trials.csv", "summary.csv")},
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "checkpoint", "templates", "data-root", "text-cache", "panel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    benchmark(parser.parse_args())

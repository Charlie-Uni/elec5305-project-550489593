"""Post-hoc latency sensitivity: repeat the identical panel in reverse model order.

Retain the first pass. This checks timing stability after observing different
frontend latency for CNNs with identical architecture and input processing.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(repo, runtime, root):
    phase = json.loads((root / "finalization_progress.json").read_text())["phase"]
    if phase != "ready_for_result_and_visual_review":
        raise ValueError("Training, evaluation and primary timing must have finished")
    variants = [
        ("svm_clean", "svm", runtime / "artifacts/interference/v3_20261109/clean_svm_reconstructed"),
        ("cnn_clean", "cnn", runtime / "artifacts/baselines/clean_cnn"),
        ("svm_wgn", "svm", root / "svm_wgn"),
        ("cnn_wgn", "cnn", root / "cnn_wgn_seed_5305"),
        ("svm_semantic", "svm", root / "svm_semantic"),
        ("cnn_semantic", "cnn", root / "cnn_semantic_seed_5305"),
        ("clap_baseline", "clap", runtime / "models/clap/630k-audioset-best.pt"),
    ]
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Post-hoc check of latency drift after unequal CNN frontend timings; original pass retained",
        "primary_order": [v[0] for v in variants],
        "repeat_order": [v[0] for v in reversed(variants)],
        "completed": [],
    }
    destination = root / "cost_panel_reverse"
    destination.mkdir(exist_ok=True)
    env = dict(
        os.environ, PYTHONPATH=str(repo / "src"),
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        HF_HOME=str(runtime / "hf-cache"), HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
    )
    progress = root / "repeat_cost_progress.json"
    for name, system, model in reversed(variants):
        record["active"] = name
        progress.write_text(json.dumps(record, indent=2))
        python = runtime / ("clap-venv/bin/python" if system == "clap" else "baseline-venv/bin/python")
        with (root / f"repeat_cost_{name}.log").open("w") as log:
            subprocess.run(
                [str(python), str(repo / "scripts/benchmark_panel.py"),
                 "--system", system, "--model", str(model),
                 "--data-root", str(runtime / "data/ESC-50"),
                 "--output", str(destination / name)],
                cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        record["completed"].append(name)
    record.update(active=None, complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    progress.write_text(json.dumps(record, indent=2))
    print("Reverse-order timing complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "runtime", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.runtime, args.run)

"""Wait for declared runs, summarise, render previews and measure idle CPU cost.

This is a one-run experiment coordinator, not a recurring automation. Final
figures and report conclusions still require inspection of the saved results.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main(repo, runtime, root):
    status_path = root / "finalization_progress.json"
    python = runtime / "baseline-venv/bin/python"
    env = dict(
        os.environ,
        PYTHONPATH=str(repo / "src"),
        PYTHONPYCACHEPREFIX=str(runtime / "pycache"),
        MPLCONFIGDIR=str(runtime / "mpl-cache"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )

    def status(phase, **extra):
        status_path.write_text(json.dumps(dict(phase=phase, **extra), indent=2))

    def execute(args, label, executable=python, environment=env):
        status(label)
        with (root / f"finalize_{label}.log").open("w") as log:
            subprocess.run(
                [str(executable), *map(str, args)],
                env=environment,
                cwd=repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        status("waiting_for_verified_evaluations")
        while True:
            queue = json.loads((root / "evaluation_queue.json").read_text())
            if (
                queue["status"] == "complete"
                and (root / "clap_prompt_ensemble/complete.json").exists()
            ):
                break
            for filename in ("evaluation_launch.json",):
                launch = json.loads((root / filename).read_text())
                if queue["status"] != "complete":
                    os.kill(launch["pid"], 0)
            time.sleep(30)
        destination = repo / "results/extensions_20261109/interventions"
        execute(
            [
                "-m",
                "esc_robustness.extension_summary",
                "--repo",
                repo,
                "--runtime",
                runtime,
                "--run",
                root,
                "--output",
                destination,
            ],
            "summary",
        )
        execute(
            [
                repo / "scripts/plot_extensions.py",
                "--input",
                destination,
                "--output",
                root / "intervention_preview",
            ],
            "figure_preview",
        )
        # Wait until the final producer exits, so concurrent project computation
        # is not counted as model deployment cost.
        producer = json.loads((root / "augmentation_launch.json").read_text())["pid"]
        while True:
            process = subprocess.run(
                ["ps", "-p", str(producer), "-o", "stat="], capture_output=True, text=True
            )
            if not process.stdout.strip() or process.stdout.strip().startswith("Z"):
                break
            time.sleep(5)
        models = [
            (
                "svm_clean",
                "svm",
                runtime / "artifacts/interference/v3_20261109/clean_svm_reconstructed",
            ),
            ("cnn_clean", "cnn", runtime / "artifacts/baselines/clean_cnn"),
        ]
        models += [
            (
                f"{system}_{regime}",
                system,
                root / (f"cnn_{regime}_seed_5305" if system == "cnn" else f"svm_{regime}"),
            )
            for regime in ("wgn", "semantic")
            for system in ("svm", "cnn")
        ]
        models += [("clap_baseline", "clap", runtime / "models/clap/630k-audioset-best.pt")]
        for name, system, model in models:
            output = root / "cost_panel" / name
            if (output / "summary.json").exists():
                continue
            selected_python = runtime / "clap-venv/bin/python" if system == "clap" else python
            environment = dict(
                env,
                HF_HOME=str(runtime / "hf-cache"),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
            )
            execute(
                [
                    repo / "scripts/benchmark_panel.py",
                    "--system",
                    system,
                    "--model",
                    model,
                    "--data-root",
                    runtime / "data/ESC-50",
                    "--output",
                    output,
                ],
                "cost_" + name,
                selected_python,
                environment,
            )
        status(
            "ready_for_result_and_visual_review",
            summary=str(destination),
            preview=str(root / "intervention_preview"),
            costs=str(root / "cost_panel"),
        )
    except Exception as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "runtime", "run"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    main(args.repo, args.runtime, args.run)

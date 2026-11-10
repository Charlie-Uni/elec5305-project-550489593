"""Package verified tables, per-trial predictions and author-review deliverables."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.extension_summary import variant_paths


def package(repo, runtime, run, destination):
    extension = repo / "results/extensions_20261109"
    for relative in (
        "interventions/independent_audit.json",
        "cost_panel/audit.json",
        "cost_panel_reverse/audit.json",
        "cost_repeatability/audit.json",
        "augmentation_banks_audit.json",
        "prompt_ensemble_audit.json",
    ):
        if not json.loads((extension / relative).read_text())["verified"]:
            raise ValueError("Evidence bundle requires completed independent audits")
    queue = json.loads((run / "evaluation_queue.json").read_text())
    if queue["status"] != "complete" or len(queue["completed"]) != 22:
        raise ValueError("Declared supervised evaluation queue is incomplete")
    files = {}
    for folder in ("src", "scripts", "configs", "tests", "docs", "results", "examples", "demo"):
        for path in sorted((repo / folder).rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.name.startswith(".")
                or path.suffix.lower()
                in (".pyc", ".png", ".log", ".zip", ".npz", ".npy", ".pt", ".joblib")
            ):
                continue
            files[str(path.relative_to(repo))] = path
    for name in (
        "README.md",
        ".nojekyll",
        "pyproject.toml",
        "uv.lock",
        "requirements-clap-macos.txt",
        "index.html",
        "site-data.json",
        "ELEC5305_Project_proposal_550489593.pdf",
    ):
        files[name] = repo / name
    for variant, system, _, _, clean, dirs in variant_paths(repo, runtime, run):
        for region, directory in dirs.items():
            for filename in (f"{system}_predictions.csv", "summary.json", "audit.json"):
                if (directory / filename).exists():
                    files[f"trial_evidence/{region}/{variant}/{filename}"] = directory / filename
        for filename in ("predictions.csv", "summary.json", "run_config.json"):
            if (clean / filename).exists():
                files[f"trial_evidence/clean/{variant}/{filename}"] = clean / filename
        for path in sorted(clean.glob("fold_*.json")):
            files[f"trial_evidence/training/{variant}/{path.name}"] = path
    for region in ("own", "target"):
        for filename in ("manifest.csv", "manifest_meta.json"):
            files[f"test_manifests/manifest_{region}/{filename}"] = (
                runtime / "runs/clap_v4_20261109" / f"manifest_{region}" / filename
            )
    for path in sorted((run / "configs").glob("*.json")):
        files[f"execution_configs/{path.name}"] = path
    for path in sorted((run / "banks").rglob("bank_meta.json")):
        files[f"training_banks/{path.parent.parent.name}/{path.parent.name}/bank_meta.json"] = path
        files[f"training_banks/{path.parent.parent.name}/{path.parent.name}/manifest.csv"] = (
            path.with_name("manifest.csv")
        )
    snapshots = {
        name: run / name for name in ("augmentation_source", "evaluation_source", "ensemble_source")
    }
    snapshots["core_inference_source"] = runtime / "runs/clap_v4_20261109/execution_v4_snapshot"
    for name, source in snapshots.items():
        for path in sorted(source.rglob("*.py")):
            files[f"executed_source/{name}/{path.relative_to(source)}"] = path
    # PNGs are included for offline site previews; vector versions remain the
    # primary scientific exports. Avoid copying unrelated intermediate renders.
    for folder in (
        "figures_three_systems_own_20261109",
        "figures_three_systems_target_20261109",
        "extensions_20261109/figures",
        "extensions_20261109/intervention_figures",
        "extensions_20261109/replication_figures",
        "extensions_20261109/cost_figures",
        "closing_20261110/figures",
        "final_submission_20261110/figures",
        "final_submission_20261110",
    ):
        for path in sorted((repo / "results" / folder).glob("*.png")):
            if not path.name.endswith("_gray.png"):
                files[str(path.relative_to(repo))] = path
    metadata = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        repository_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        files={name: sha256_file(path) for name, path in files.items()},
        scope="Verified core, extensions and closing analyses; 10-page main report, unchanged 26-page supplement, 12-slide presentation, exact narration, actual code execution video and Pages",
        excluded=[
            "Model checkpoints",
            "Full ESC-50 and DEMAND data",
            "Large embedding/feature arrays",
            "Private local caches",
        ],
        recording_status="Actual code execution video with source-derived audio is included; the author-narrated full presentation is not recorded",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pending.zip")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(metadata, indent=2))
        archive.writestr(
            "READ_THIS_FIRST.txt",
            "Verified final research evidence bundle\n\nStart with results/final_submission_20261110/final_report.pdf (10 pages) and results/final_submission_20261110/presentation.pptx (12 slides). The original 26-page report remains unchanged at results/research_report_20261109_draft/research_report.pdf.\n\nThe new deck contains editable charts, exact notes and two embedded audio clips. The 728-word script is docs/video_narration_final.txt, approximately 5.4–6.1 minutes with pauses. code_demo.mp4 records actual source reconstruction and all three model inferences; sound examples are included, but there is no author narration. No final Canvas submission is implied. See docs/rubric_review_20261110.md and docs/code_demo.md.\n\ntrial_evidence/ retains per-trial clean and interference predictions for 14 variants and both power protocols. Model weights, full datasets and large embedding arrays are external; hashes and commands identify them. Source attribution and licences remain in examples/listening_own_20261109/README.md. For offline website preview, serve this folder over localhost. The live demo additionally requires its documented local inference environments and assets.\n\nBUNDLE_MANIFEST.json provides SHA-256 for every source file.\n",
        )
        for name, path in files.items():
            archive.write(path, name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Archive integrity failure")
        for name, digest in metadata["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Packaged evidence differs from source")
    temporary.replace(destination)
    result = dict(
        verified=True,
        file=str(destination),
        bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        source_files=len(files),
        repository_commit=metadata["repository_commit"],
    )
    write_json(destination.with_suffix(".json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "runtime", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    package(args.repo, args.runtime, args.run, args.output)

"""Build the editable research-report draft using verified result tables."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def build(compiler, extension_run=None, only_cached=False):
    core = ROOT / "results/clap_full_20261109"
    clean_dirs = {
        "svm": ROOT / "results/clean_svm",
        "cnn": ROOT / "results/clean_cnn",
        "clap": ROOT / "results/clap_integration_20261109/clean",
    }
    rows = []
    for system, path in clean_dirs.items():
        d = json.loads((path / "summary.json").read_text())
        values = [
            f"{100 * d[key]['mean']:.2f} $\\pm$ {100 * d[key]['sample_std']:.2f}"
            for key in ("accuracy", "macro_f1")
        ]
        rows.append(system.upper() + " & " + " & ".join(values) + r" \\")
    text = (
        r"""\begin{table}[htbp]\centering
\caption{Clean five-fold results (percent; mean $\pm$ sample SD). Core CNN seed: 5305.}
\label{tab:clean}\begin{tabular}{lrr}\toprule System & Accuracy & Macro-F1\\\midrule
"""
        + "\n".join(rows)
        + r"\bottomrule\end{tabular}\end{table}"
        + "\n"
    )
    with (core / "all_conditions_verified.csv").open() as f:
        data = list(csv.DictReader(f))
    text += r"""\begin{table}[htbp]\centering
\caption{Macro-F1 at +5 dB (percent; five-fold mean $\pm$ sample SD). Each cell uses its retained targets.}
\label{tab:five}\begin{tabular}{llrrr}\toprule Power & Condition & SVM & CNN & CLAP\\\midrule
"""
    for region in ("own", "target"):
        for condition, label in (
            ("white", "White noise"),
            ("demand", "DEMAND"),
            ("semantic_same", "Same category"),
            ("semantic_different", "Different category"),
        ):
            values = []
            for system in ("svm", "cnn", "clap"):
                matched = [
                    r
                    for r in data
                    if r["region"] == region
                    and r["system"] == system
                    and r["condition"] == f"{condition}@+5dB"
                ]
                if len(matched) != 1:
                    raise ValueError("Missing or duplicated verified condition")
                row = matched[0]
                values.append(
                    f"{100 * float(row['macro_f1_mean']):.2f} $\\pm$ {100 * float(row['macro_f1_sd']):.2f}"
                )
            text += region.capitalize() + " & " + label + " & " + " & ".join(values) + r" \\" + "\n"
        if region == "own":
            text += r"\midrule" + "\n"
    text += r"\bottomrule\end{tabular}\end{table}" + "\n"
    (ROOT / "docs/report_tables.tex").write_text(text)
    status = "The core and diagnostic results above are verified. The extended augmentation, multi-seed interference and prompt-ensemble results are still being generated and are not used to support the conclusions of this draft.\n"
    if extension_run:
        complete = []
        for seed in (5306, 5307):
            path = extension_run / f"clean_seed_{seed}/summary.json"
            if path.exists():
                d = json.loads(path.read_text())
                complete.append(
                    f"seed {seed}: clean accuracy {100 * d['accuracy']['mean']:.2f}\\%, macro-F1 {100 * d['macro_f1']['mean']:.2f}\\%"
                )
        if complete:
            status += (
                "Completed additional clean training runs at the build time are "
                + "; ".join(complete)
                + ". Their interference comparisons are reported only after evaluation and audit.\n"
            )
    (ROOT / "docs/report_extension_status.tex").write_text(status)
    destination = ROOT / "results/research_report_20261109_draft"
    destination.mkdir(exist_ok=True)
    command = [
        str(compiler),
        "--keep-logs",
        "--outdir",
        str(destination),
        str(ROOT / "docs/research_report.tex"),
    ]
    if only_cached:
        command.insert(1, "--only-cached")
    result = subprocess.run(
        command, cwd=ROOT / "docs", text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    (destination / "build_output.txt").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout)
    reader = PdfReader(destination / "research_report.pdf")
    metadata = {
        "pages": len(reader.pages),
        "draft": True,
        "source": "docs/research_report.tex",
        "table_source": "results/clap_full_20261109/all_conditions_verified.csv",
        "extension_status": status,
        "warnings": [
            line
            for line in result.stdout.splitlines()
            if "warning" in line.lower() or "overfull" in line.lower()
        ],
    }
    (destination / "build_audit.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compiler", type=Path, default=ROOT / ".venv/bin/tectonic")
    p.add_argument("--extension-run", type=Path)
    p.add_argument("--only-cached", action="store_true")
    a = p.parse_args()
    build(a.compiler, a.extension_run, a.only_cached)

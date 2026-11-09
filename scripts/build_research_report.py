"""Build the editable research-report draft using verified result tables."""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def extension_status(extension_run):
    root = ROOT / "results/extensions_20261109"
    complete = (root / "interventions/summary_audit.json").exists()
    text = (
        "All declared intervention variants have completed evaluation.\n"
        if complete
        else "Supervised augmentation comparisons are still running. The completed replication and prompt-ablation results below are separately verified; pending interventions do not support the conclusions of this draft.\n"
    )
    if extension_run:
        values = []
        for seed in (5306, 5307):
            path = extension_run / f"clean_seed_{seed}/summary.json"
            if path.exists():
                d = json.loads(path.read_text())
                values.append(
                    f"seed {seed}: clean accuracy {100 * d['accuracy']['mean']:.2f}\\%, macro-F1 {100 * d['macro_f1']['mean']:.2f}\\%"
                )
        if values:
            text += "Additional clean runs are " + "; ".join(values) + ".\n"
    replication = root / "clean_replication/cnn_overlap_replication.csv"
    if replication.exists():
        with replication.open() as handle:
            rows = list(csv.DictReader(handle))
        text += r"""
\subsubsection*{CNN replication across optimisation seeds}
All three clean-trained seeds show lower accuracy for semantic interference than
matched DEMAND at +5 dB and overlap at least 0.8. The two new seeds have larger
differences than the core seed. Their eight comparisons pass the declared
12-comparison Holm family, while seed 5305 remains above 0.05. This supports
directional replication; it does not turn repeated predictions of the same clips
into independent samples or isolate a causal semantic effect.
\begin{table}[htbp]\centering\small
\caption{Semantic minus matched DEMAND accuracy (percentage points), with paired 95\% target-bootstrap intervals and Holm-adjusted exact McNemar $p$ values. Each seed is analysed separately.}
\begin{tabular}{lrcrcr}\toprule Power & Seed & Same category & $p$ & Different category & $p$\\\midrule
"""
        for region in ("own", "target"):
            for seed in (5305, 5306, 5307):
                cells = []
                for condition in ("semantic_same", "semantic_different"):
                    row = next(
                        r
                        for r in rows
                        if r["region"] == region
                        and int(r["seed"]) == seed
                        and r["condition"] == condition
                    )
                    a, lo, hi = [
                        100 * float(row[k])
                        for k in ("accuracy_difference", "accuracy_ci_low", "accuracy_ci_high")
                    ]
                    p = float(row["holm_p_across_cnn_replication"])
                    cells += [
                        f"${a:.2f}\\;[{lo:.2f},{hi:.2f}]$",
                        "$<0.001$" if p < 0.001 else f"{p:.3f}",
                    ]
                text += (
                    region.capitalize()
                    + " & "
                    + str(seed)
                    + " & "
                    + " & ".join(cells)
                    + r" \\"
                    + "\n"
                )
            if region == "own":
                text += r"\midrule" + "\n"
        text += r"\bottomrule\end{tabular}\end{table}" + "\n"
        if (root / "replication_figures/17_cnn_overlap_replication.pdf").exists():
            text += r"""\begin{figure}[htbp]\centering
\includegraphics{../results/extensions_20261109/replication_figures/17_cnn_overlap_replication.pdf}
\caption{The high-overlap CNN contrast is negative for all three seeds in both power protocols. Intervals condition on each fitted model.}
\end{figure}
"""
    path = root / "prompt_ensemble_audit.json"
    if path.exists():
        audit = json.loads(path.read_text())
        if not audit["verified"]:
            raise ValueError("Unverified prompt ensemble")
        acc, f1 = audit["clean_accuracy"], audit["clean_macro_f1"]
        text += r"\subsubsection*{Fixed prompt-ensemble ablation}" + "\n"
        text += f"The released list contains {audit['templates_per_class']} entries and {audit['unique_templates']} unique templates. All entries, including their released multiplicities, are retained. Clean accuracy is {100 * acc['mean']:.2f} $\\pm$ {100 * acc['fold_sd']:.2f}\\%, and macro-F1 is {100 * f1['mean']:.2f} $\\pm$ {100 * f1['fold_sd']:.2f}\\% across five folds. No prompts were selected using the test labels.\n"
        text += r"""The saved audio embeddings are bit-identical to the core run, and
scores have been independently recomputed from the new unit text vectors. Thus
the changed classifier output comes from the text references, not a new audio
encoder or new training. The gain in clean accuracy does not remove the
semantic-interference gap. Conditional capture rates must still be interpreted
together with the unconditional capture fraction and total error count.
\begin{table}[htbp]\centering
\caption{Fixed prompt ensemble at +5 dB: five-fold macro-F1 (percent). The single-prompt reference is retained in Table~\ref{tab:five}.}
\begin{tabular}{lrrrr}\toprule Power & White & DEMAND & Same category & Different category\\\midrule
"""
        for region in ("own", "target"):
            table = json.loads(
                (root / f"clap_prompt_ensemble/{region}_conditions.json").read_text()
            )
            vals = [
                f"{100 * table[c + '@+5dB']['macro_f1']['mean']:.2f}"
                for c in ("white", "demand", "semantic_same", "semantic_different")
            ]
            text += region.capitalize() + " & " + " & ".join(vals) + r" \\" + "\n"
        text += r"\bottomrule\end{tabular}\end{table}" + "\n"
    return text


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
    status = extension_status(extension_run)
    (ROOT / "docs/report_extension_status.tex").write_text(status)
    destination = ROOT / "results/research_report_20261109_draft"
    destination.mkdir(exist_ok=True)
    build_directory = ROOT / "tmp/research_report_build"
    build_directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(compiler),
        "--keep-logs",
        "--outdir",
        str(build_directory),
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
    generated = build_directory / "research_report.pdf"
    if not PdfReader(generated).pages:
        raise ValueError("Generated report contains no pages")
    temporary = destination / "research_report.pending.pdf"
    shutil.copyfile(generated, temporary)
    temporary.replace(destination / "research_report.pdf")
    shutil.copyfile(build_directory / "research_report.log", destination / "research_report.log")
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

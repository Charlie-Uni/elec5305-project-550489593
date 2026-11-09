"""Build the editable research-report draft using verified result tables."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def cost_table():
    directory = ROOT / "results/extensions_20261109/cost_panel"
    if not (directory / "audit.json").exists():
        return r"""The core CPU microbenchmark uses one numerical thread, batch size one,
100 repetitions and ten warm-ups on one representative five-second clip. The
sums of median preprocessing and median inference times are approximately
11.10 ms for SVM, 2.61 ms for CNN and 98.13 ms for CLAP. These sums approximate
query cost; they are not the median of per-trial total times or a dataset-wide
tail-latency estimate. The balanced 250-clip extension is still pending.
"""
    if not json.loads((directory / "audit.json").read_text())["verified"]:
        raise ValueError("CPU panel requires its independent audit")
    with (directory / "latency_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    repeat_directory = directory.with_name("cost_panel_reverse")
    repeat_audit = json.loads((directory.with_name("cost_repeatability") / "audit.json").read_text())
    if not repeat_audit["verified"]:
        raise ValueError("Repeated timing must use verified identical model inputs")
    with (repeat_directory / "latency_summary.csv").open() as handle:
        repeats = {r["variant"]: r for r in csv.DictReader(handle)}
    labels = {
        "svm_clean": "SVM / clean",
        "svm_wgn": "SVM / WGN",
        "svm_semantic": "SVM / semantic",
        "cnn_clean": "CNN / clean",
        "cnn_wgn": "CNN / WGN",
        "cnn_semantic": "CNN / semantic",
        "clap_baseline": "CLAP / single prompt",
    }
    text = r"""The expanded benchmark uses the same Apple M3 CPU, one numerical
thread and batch size one for seven model variants, after project training and
evaluation have stopped. One deterministic recording from each class and official
fold gives 250 unique five-second clips; three repetitions give 750 timed queries
per model in each pass. After observing different frontend times for CNNs with
identical processing, a post-hoc second pass reverses the model order. Both passes
are retained, giving 10,500 timed queries in total. Model, audio, software and
benchmark-source hashes match across passes, and all timed predictions match the
formal clean evaluations. Ten warm-ups precede each fold's panel. Total latency directly times
preprocessing, representation extraction, all-class scoring and argmax. Disk I/O,
model loading and one-off text preparation are excluded. Model-specific pinned
software environments are recorded. CNN latency uses seed 5305; its three-seed
accuracy analysis is separate. This is a two-order descriptive sensitivity on
one machine, not a randomised hardware study. The timing shift cannot be assigned
to a measured thermal or scheduling cause. In particular, differences among
identically shaped CNNs are not evidence that augmentation changes operation count.
\begin{table}[htbp]\centering\small
\caption{Balanced CPU panel: both timing passes. Medians and p95 values are in milliseconds. Serialized size is one fold's model (median across folds), or the CLAP checkpoint; it is not resident memory.}
\begin{tabular}{lrrrrr}\toprule Model / training & Median 1 & Median 2 & p95 1 & p95 2 & File (MB)\\\midrule
"""
    for name in labels:
        row = next(r for r in rows if r["variant"] == name)
        text += (
            labels[name]
            + " & "
            + " & ".join(
                [
                    f"{float(row['total_median_ms']):.2f}",
                    f"{float(repeats[name]['total_median_ms']):.2f}",
                    f"{float(row['total_p95_ms']):.2f}",
                    f"{float(repeats[name]['total_p95_ms']):.2f}",
                    f"{float(row['serialized_bytes_per_fold_median']) / 1e6:.2f}",
                ]
            )
            + r" \\"
            + "\n"
        )
    text += r"\bottomrule\end{tabular}\end{table}" + "\n"
    if (ROOT / "results/extensions_20261109/cost_figures/18_balanced_deployment_cost.pdf").exists():
        text += r"""
\begin{figure}[htbp]\centering
\includegraphics{../results/extensions_20261109/cost_figures/18_balanced_deployment_cost.pdf}
\caption{Balanced CPU latency alongside clean and +5-dB different-category macro-F1 under own-activity power. Both timing passes are retained; whiskers show each pass's interquartile range of 750 queries, not confidence intervals. CNN performance here uses seed 5305; the intervention analysis retains all three seeds.}
\end{figure}
"""
    return text


def intervention_tables(directory):
    """Only promote completed comparisons after independent CSV recalculation."""
    audit = json.loads((directory / "independent_audit.json").read_text())
    if not audit["verified"] or audit["condition_cells"] != 336:
        raise ValueError("Intervention tables require the complete independent audit")
    with (directory / "clean_variants.csv").open() as handle:
        clean = list(csv.DictReader(handle))
    with (directory / "condition_variants.csv").open() as handle:
        conditions = list(csv.DictReader(handle))

    def value(rows, system):
        if len(rows) != (3 if system == "cnn" else 1):
            raise ValueError("An intervention table is missing a declared replicate")
        means = [float(r["macro_f1"]) for r in rows]
        sd = np.std(means, ddof=1) if system == "cnn" else float(rows[0]["macro_f1_fold_sd"])
        return f"{100 * np.mean(means):.2f} $\\pm$ {100 * sd:.2f}"

    text = r"""
\subsubsection*{Supervised augmentation transfer}
All nine CNN variants (three regimes by three seeds) and all three SVM regimes
are retained. The following tables include clean performance as well as transfer
to each held-out interference condition. CNN entries summarise three separate
five-fold means; their SD measures optimisation variation, not uncertainty from
15 independent test sets. SVM SD describes the five official folds. Paired
accuracy effects are tested separately for each trained variant, with all 156
declared +5-dB intervention contrasts in one Holm family. Full paired intervals,
discordance counts and gain-matched control scores accompany the exported data.
The analysis was amended after initial WGN results to add direct semantic-versus-
WGN contrasts answering the stated second question, before any semantic-trained
result was available. The amendment changes the comparison family, not training
settings or test mixtures; it is disclosed in the saved protocol.
"""
    for region in ("own", "target"):
        text += (
            r"\begin{table}[htbp]\centering\footnotesize"
            + "\n"
            + r"\caption{Supervised intervention macro-F1 (percent): "
            + region
            + r"-activity protocol at +5 dB. CNN: mean $\pm$ seed SD; SVM: mean $\pm$ fold SD.}"
            + "\n"
            + r"\begin{tabular}{llrrrrr}\toprule Model & Training & Clean & White & DEMAND & Same & Different\\\midrule"
            + "\n"
        )
        for system in ("svm", "cnn"):
            for regime, label in (("clean", "Clean"), ("wgn", "WGN"), ("semantic", "Semantic")):
                clean_rows = [r for r in clean if r["system"] == system and r["regime"] == regime]
                cells = [value(clean_rows, system)]
                for condition in ("white", "demand", "semantic_same", "semantic_different"):
                    selected = [
                        r
                        for r in conditions
                        if r["system"] == system
                        and r["regime"] == regime
                        and r["region"] == region
                        and r["condition"] == condition
                        and float(r["level_db"]) == 5
                    ]
                    cells.append(value(selected, system))
                text += system.upper() + " & " + label + " & " + " & ".join(cells) + r" \\" + "\n"
            if system == "svm":
                text += r"\midrule" + "\n"
        text += r"\bottomrule\end{tabular}\end{table}" + "\n"
    def cnn_f1(regime, condition):
        selected = [r for r in conditions if r["system"] == "cnn" and r["regime"] == regime
                    and r["region"] == "own" and r["condition"] == condition
                    and float(r["level_db"]) == 5]
        return 100 * np.mean([float(r["macro_f1"]) for r in selected])

    clean_means = {regime: 100 * np.mean([float(r["macro_f1"]) for r in clean
                   if r["system"] == "cnn" and r["regime"] == regime])
                   for regime in ("clean", "wgn", "semantic")}
    text += (
        f"Under own-activity power, WGN augmentation raises three-seed CNN white-noise macro-F1 "
        f"from {cnn_f1('clean', 'white'):.2f}\\% to {cnn_f1('wgn', 'white'):.2f}\\%, "
        f"whereas same-category semantic performance changes only from {cnn_f1('clean', 'semantic_same'):.2f}\\% "
        f"to {cnn_f1('wgn', 'semantic_same'):.2f}\\%. Semantic training reaches "
        f"{cnn_f1('semantic', 'semantic_same'):.2f}\\% and {cnn_f1('semantic', 'semantic_different'):.2f}\\% "
        f"on same- and different-category mixtures, but its clean macro-F1 falls to "
        f"{clean_means['semantic']:.2f}\\%, compared with {clean_means['clean']:.2f}\\% "
        f"for clean training and {clean_means['wgn']:.2f}\\% for WGN training. "
        "Thus transfer is interference-specific, and the semantic intervention has a clean-performance cost.\n"
    )
    with (directory / "paired_intervention_effects.csv").open() as handle:
        direct = [r for r in csv.DictReader(handle) if r["comparison_type"] == "semantic_vs_wgn"
                  and r["stratum"] == "all" and r["condition"].startswith("semantic")]
    for system, total in (("cnn", 12), ("svm", 4)):
        selected = [r for r in direct if r["system"] == system]
        if len(selected) != total:
            raise ValueError("Missing direct semantic-training accuracy contrasts")
        positive = sum(float(r["accuracy_difference"]) > 0 for r in selected)
        significant = sum(float(r["holm_p_across_extension_comparisons"]) < 0.05 for r in selected)
        text += (
            f"For {system.upper()}, {positive} of {total} full-condition semantic-versus-WGN "
            f"paired accuracy point estimates are positive, but {significant} of {total} "
            "pass the common 156-test Holm correction. "
        )
    text += "These tests concern per-variant paired accuracy, not significance of the three-seed mean macro-F1. The modest CNN advantage is therefore not established by the corrected tests.\n"
    figure_root = ROOT / "results/extensions_20261109/intervention_figures"
    for filename, caption in (
        (
            "13_augmentation_transfer",
            "Augmentation transfer at +5 dB. Small CNN points retain each optimisation seed; large points and whiskers show the three-seed mean and sample SD. SVM points are five-fold means.",
        ),
        (
            "16_clap_prompt_ensemble",
            "Frozen CLAP class readout with the original single prompt and the complete fixed prompt ensemble. The audio vectors are identical. Both conditional and unconditional capture metrics are retained in the numerical tables.",
        ),
        (
            "19_semantic_vs_wgn_training",
            "Direct semantic-augmentation minus WGN-augmentation accuracy effects at +5 dB. Each CNN comparison retains the same optimisation seed. Intervals are paired target-bootstrap intervals conditional on fitted models; exact tests join the common 156-comparison Holm family.",
        ),
    ):
        if (figure_root / (filename + ".pdf")).exists():
            text += (
                r"\begin{figure}[p]\centering\includegraphics{../results/extensions_20261109/intervention_figures/"
                + filename
                + r".pdf}\caption{"
                + caption
                + r"}\end{figure}"
                + "\n"
            )
    return text


def extension_status(extension_run):
    root = ROOT / "results/extensions_20261109"
    complete = (root / "interventions/independent_audit.json").exists()
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
        if (root / "source_cluster_sensitivity/audit.json").exists():
            text += r"""The exploratory target-source bootstrap preserves this
direction: all 12 clean-CNN pointwise source-cluster intervals remain negative,
whereas the four SVM intervals include zero. This sensitivity check changes the
resampling unit, not the primary test families or their multiplicity conclusions.
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
The ensemble requires 15,000 text encodings to initialise its cache, compared
with 50 for the single template. After averaging, both query paths score against
50 class vectors; one-off text preparation is excluded from query latency.
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
        transitions = root / "capture_transitions_core/capture_transitions.csv"
        if transitions.exists():
            with transitions.open() as handle:
                captures = list(csv.DictReader(handle))
            text += r"""
\begin{table}[htbp]\centering
\caption{CLAP error destinations at +5 dB under own-activity power. Each condition contains 2,000 trials. Captures are counts among all trials, not error-conditioned percentages.}
\begin{tabular}{lrrrr}\toprule & \multicolumn{2}{c}{Single prompt} & \multicolumn{2}{c}{Fixed ensemble}\\
Condition & Errors & Captures & Errors & Captures\\\midrule
"""
            for condition, label in (
                ("semantic_same", "Same category"),
                ("semantic_different", "Different category"),
            ):
                cells = []
                for variant in ("clap_baseline", "clap_prompt_ensemble"):
                    row = next(
                        r
                        for r in captures
                        if r["variant"] == variant
                        and r["region"] == "own"
                        and r["condition"] == condition
                        and float(r["level_db"]) == 5
                        and r["stratum"] == "all"
                    )
                    cells.extend((row["errors"], row["distractor_captures"]))
                text += label + " & " + " & ".join(cells) + r" \\" + "\n"
            text += r"""\bottomrule\end{tabular}\end{table}
The ensemble reduces other wrong-label predictions, while distractor-capture
counts in these conditions are slightly higher. A larger fraction of the
remaining errors therefore indicates a change in error composition as well as
competition. Saved transition tables additionally distinguish new captures from
predictions that already named the distractor in the gain-matched clean control.
Correct-control-only rates use different subsets for different classifiers and
are descriptive diagnostics rather than matched cross-model effects.
"""
        multiplicity = root / "prompt_multiplicity/audit.json"
        if multiplicity.exists():
            sensitivity = json.loads(multiplicity.read_text())
            text += f"An exploratory post-hoc sensitivity check gives each of the 275 unique templates equal weight. Across clean audio and all 24 interference cells, its largest absolute macro-F1 change is {sensitivity['largest_absolute_macro_f1_change_pp']:.2f} percentage points. The declared 300-entry ensemble remains the reported intervention; this check is not used to select a better-performing prompt set.\n"
        if (root / "source_cluster_sensitivity/audit.json").exists():
            text += r"""For high-overlap targets, the ensemble's own-protocol
semantic-minus-matched-DEMAND accuracy differences are $-31.56$ points for same
category and $-34.13$ points for different category. Their exploratory 95\%
target-source bootstrap intervals are $[-34.36,-28.84]$ and $[-36.77,-31.42]$.
The target-activity protocol gives similar contrasts ($-31.28$ and $-33.92$
points). Thus improved clean readout does not eliminate the conditional
semantic-interference gap; the gap against the ordinary-background control is
larger than in the single-prompt reference.
"""
    if complete:
        text += intervention_tables(root / "interventions")
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
    (ROOT / "docs/report_cost_panel.tex").write_text(cost_table())
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
    inputs = [
        Path(__file__).resolve(),
        ROOT / "docs/research_report.tex",
        ROOT / "docs/report_tables.tex",
        ROOT / "docs/report_extension_status.tex",
        ROOT / "docs/report_cost_panel.tex",
        ROOT / "docs/extension_protocol_20261109.md",
        ROOT / "results/clap_full_20261109/all_conditions_verified.csv",
    ]
    for folder in (
        "report_figure_assets_20261109",
        "extensions_20261109/figures",
        "extensions_20261109/replication_figures",
        "extensions_20261109/intervention_figures",
        "extensions_20261109/cost_figures",
    ):
        inputs.extend(sorted((ROOT / "results" / folder).glob("*.pdf")))
    metadata = {
        "pages": len(reader.pages),
        "draft": True,
        "source": "docs/research_report.tex",
        "table_source": "results/clap_full_20261109/all_conditions_verified.csv",
        "extension_status": status,
        "pdf_sha256": hashlib.sha256((destination / "research_report.pdf").read_bytes()).hexdigest(),
        "source_files_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
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

"""Balanced CPU latency and semantic robustness with explicit interval meanings."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from esc_robustness.diagnostics import write_csv


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def plot(costs, interventions, output, repeat_costs, export=False, audit=None):
    if (
        not json.loads((costs / "audit.json").read_text())["verified"]
        or not json.loads((repeat_costs / "audit.json").read_text())["verified"]
        or not json.loads((interventions / "independent_audit.json").read_text())["verified"]
    ):
        raise ValueError("Both datasets require their independent audits")
    latency = {r["variant"]: r for r in read(costs / "latency_summary.csv")}
    repeated = {r["variant"]: r for r in read(repeat_costs / "latency_summary.csv")}
    clean = {r["variant"]: r for r in read(interventions / "clean_variants.csv")}
    conditions = read(interventions / "condition_variants.csv")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    names = [f"{s}_{r}" for s in ("svm", "cnn") for r in ("clean", "wgn", "semantic")] + [
        "clap_baseline"
    ]
    labels = [
        "SVM · clean",
        "SVM · WGN",
        "SVM · semantic",
        "CNN · clean",
        "CNN · WGN",
        "CNN · semantic",
        "CLAP · single prompt",
    ]
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(7.16, 4.4),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.05, 1]},
    )
    records = []
    for index, name in enumerate(names):
        colour = (
            "#0072B2"
            if name.startswith("svm")
            else "#D55E00"
            if name.startswith("cnn")
            else "#009E73"
        )
        timing = latency[name]
        median, q1, q3 = [
            float(timing[k]) for k in ("total_median_ms", "total_p25_ms", "total_p75_ms")
        ]
        variant = name + "_seed_5305" if name.startswith("cnn") else name
        condition = next(
            r
            for r in conditions
            if r["variant"] == variant
            and r["region"] == "own"
            and r["condition"] == "semantic_different"
            and float(r["level_db"]) == 5
        )
        cf, mf = 100 * float(clean[variant]["macro_f1"]), 100 * float(condition["macro_f1"])
        left.errorbar(
            median,
            index - 0.10,
            xerr=[[median - q1], [q3 - median]],
            fmt="o",
            ms=5,
            color=colour,
            capsize=3,
        )
        repeated_timing = repeated[name]
        second, second_q1, second_q3 = [
            float(repeated_timing[k]) for k in ("total_median_ms", "total_p25_ms", "total_p75_ms")
        ]
        left.errorbar(
            second, index + 0.10,
            xerr=[[second - second_q1], [second_q3 - second]],
            fmt="^", ms=5, color=colour, mfc="white", capsize=3,
        )
        right.plot([mf, cf], [index, index], color=colour, lw=1, alpha=0.6)
        right.plot(mf, index, "o", ms=5, color=colour)
        right.plot(cf, index, "o", ms=5, mfc="white", mec=colour)
        records.append(
            dict(
                **timing,
                **{f"reverse_{k}": v for k, v in repeated_timing.items() if k.startswith("total_")},
                performance_variant=variant,
                clean_macro_f1=cf / 100,
                semantic_macro_f1=mf / 100,
                region="own",
                condition="semantic_different",
                level_db=5,
            )
        )
    left.set_xscale("log")
    left.set_xlim(2, 100)
    left.set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    left.xaxis.set_minor_formatter(NullFormatter())
    left.set_yticks(range(7), labels)
    left.invert_yaxis()
    left.set_xlabel("CPU query time (ms, log scale)")
    left.set_title("CPU latency · median and IQR", fontsize=10)
    left.plot([], [], "o", color="0.25", ms=5, label="First pass")
    left.plot([], [], "^", mfc="white", mec="0.25", ms=5, label="Reversed order")
    left.legend(loc="upper left", bbox_to_anchor=(0, -0.17), fontsize=7, frameon=False)
    right.set_xlim(0, 100)
    right.set_xlabel("Macro-F1 (%)")
    right.set_title("Clean vs semantic (+5 dB)", fontsize=10)
    right.plot([], [], "o", color="0.25", ms=5, label="Mixture")
    right.plot([], [], "o", mfc="white", mec="0.25", ms=5, label="Clean")
    right.legend(loc="upper left", bbox_to_anchor=(0, -0.17), ncol=2, fontsize=8, frameon=False)
    for ax in (left, right):
        ax.grid(axis="x", alpha=0.2)
        ax.set_ylim(6.5, -0.5)
    fig.suptitle("Deployment cost and semantic robustness", fontsize=11)
    report = dict(
        layout=audit(fig) if audit else None,
        latency="250 clips × three timed repetitions per model per pass; both first and post-hoc reversed-order passes retained. Whiskers are IQR, not CIs. Do not infer a causal augmentation-cost difference from drift between identically shaped CNNs.",
        performance="Five-fold macro-F1; CNN seed 5305 here, full three-seed analysis elsewhere",
        clap="Single-prompt inference; prompt-ensemble initialisation is not included",
    )
    name = "18_balanced_deployment_cost"
    for suffix in ["png", "svg", "pdf"] if export else ["png"]:
        fig.savefig(output / f"{name}.{suffix}", dpi=600 if export else 160)
    write_csv(output / f"{name}.csv", records)
    (output / "layout_audit.json").write_text(json.dumps(report, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("costs", "interventions", "output", "repeat-costs"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--qa-path", type=Path)
    args = parser.parse_args()
    audit = None
    if args.qa_path:
        import sys

        sys.path.insert(0, str(args.qa_path))
        from visual_qa import audit_layout

        audit = audit_layout
    plot(args.costs, args.interventions, args.output, args.repeat_costs, args.export, audit)

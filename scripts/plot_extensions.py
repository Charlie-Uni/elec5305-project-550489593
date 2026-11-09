"""Intervention figures from complete, audited runs; no best-seed selection."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"svm": "#0072B2", "cnn": "#D55E00", "clap": "#009E73"}
CONDITIONS = ("white", "demand", "semantic_same", "semantic_different")
LABELS = ("White", "DEMAND", "Same", "Different")
REGIMES = ("clean", "wgn", "semantic")
REGIME_LABELS = ("Clean training", "WGN augmentation", "Semantic augmentation")
MARKERS = ("o", "s", "^")


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def replication_figure(replication):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), sharex=True, layout="constrained")
    for ax, region in zip(axes, ("own", "target")):
        for c, condition in enumerate(CONDITIONS[2:]):
            for k, seed in enumerate((5305, 5306, 5307)):
                row = next(
                    r
                    for r in replication
                    if r["region"] == region
                    and r["condition"] == condition
                    and int(r["seed"]) == seed
                )
                value, low, high = [
                    100 * float(row[key])
                    for key in ("accuracy_difference", "accuracy_ci_low", "accuracy_ci_high")
                ]
                ax.errorbar(
                    value,
                    c + (k - 1) * 0.20,
                    xerr=[[value - low], [high - value]],
                    fmt=MARKERS[k],
                    capsize=3,
                    ms=5,
                    color=COLORS["cnn"],
                    mfc="white" if k == 0 else COLORS["cnn"],
                    label=f"Seed {seed}" if c == 0 else None,
                )
        ax.set_yticks([0, 1], ["Same", "Different"])
        ax.invert_yaxis()
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlabel("Semantic − DEMAND accuracy (pp)")
        ax.set_title(f"{region.capitalize()} activity power")
        ax.grid(axis="x", alpha=0.18)
    axes[1].legend(fontsize=7, loc="center", bbox_to_anchor=(0.72, 0.5))
    fig.suptitle("Clean-trained CNN replication: +5 dB, overlap ≥ 0.8", fontsize=10)
    return fig


def plot(directory, output, export=False, audit=None):
    status = json.loads((directory / "summary_audit.json").read_text())
    if not status["complete"] or status["variants"] != 14:
        raise ValueError("All declared variants must be verified before final plots")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    reports = {}

    def save(fig, name, rows, description):
        reports[name] = {"description": description, "layout": audit(fig) if audit else None}
        fig.savefig(output / f"{name}.png", dpi=600 if export else 160)
        if export:
            fig.savefig(output / f"{name}.svg")
            fig.savefig(output / f"{name}.pdf")
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plt.close(fig)

    all_conditions = read(directory / "condition_variants.csv")
    supervised = [r for r in all_conditions if r["system"] != "clap" and float(r["level_db"]) == 5]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6), sharey=True, layout="constrained")
    for i, region in enumerate(("own", "target")):
        for j, system in enumerate(("svm", "cnn")):
            ax = axes[i, j]
            for k, regime in enumerate(REGIMES):
                for c, condition in enumerate(CONDITIONS):
                    rows = [
                        r
                        for r in supervised
                        if r["region"] == region
                        and r["system"] == system
                        and r["regime"] == regime
                        and r["condition"] == condition
                    ]
                    values = np.array([100 * float(r["macro_f1"]) for r in rows])
                    if len(values) != (3 if system == "cnn" else 1):
                        raise ValueError("Missing training replicate")
                    x = c + (k - 1) * 0.24
                    ax.errorbar(
                        x,
                        values.mean(),
                        yerr=values.std(ddof=1) if len(values) > 1 else None,
                        fmt=MARKERS[k],
                        color=COLORS[system],
                        capsize=3,
                        ms=6,
                        mfc="white" if k == 0 else COLORS[system],
                        label=REGIME_LABELS[k] if c == 0 else None,
                    )
                    if len(values) > 1:
                        ax.scatter(
                            x + np.linspace(-0.045, 0.045, 3),
                            values,
                            s=9,
                            color=COLORS[system],
                            alpha=0.45,
                        )
            ax.set_title(f"{system.upper()} · {region} activity power")
            ax.set_xticks(range(4), LABELS)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.18)
            if j == 0:
                ax.set_ylabel("Macro-F1 (%)")
            if i == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Training intervention transfer at +5 dB", fontsize=11)
    save(
        fig,
        "13_augmentation_transfer",
        supervised,
        "SVM points are five-fold means. CNN large markers are means of three five-fold means; whiskers are sample SD across seeds, small dots show each seed. Conditions are categorical and unconnected.",
    )

    effects = read(directory / "paired_intervention_effects.csv")
    for system, number in (("svm", "14"), ("cnn", "15")):
        rows = [r for r in effects if r["system"] == system and r["stratum"] == "all"]
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.6), sharex=True, layout="constrained")
        for i, regime in enumerate(("wgn", "semantic")):
            for j, region in enumerate(("own", "target")):
                ax = axes[i, j]
                for c, condition in enumerate(CONDITIONS):
                    selected = sorted(
                        [
                            r
                            for r in rows
                            if r["region"] == region
                            and r["regime"] == regime
                            and r["condition"] == condition
                        ],
                        key=lambda r: r["seed"],
                    )
                    for k, row in enumerate(selected):
                        value, low, high = [
                            100 * float(row[key])
                            for key in (
                                "accuracy_difference",
                                "accuracy_ci_low",
                                "accuracy_ci_high",
                            )
                        ]
                        y = c + ((k - 1) * 0.20 if system == "cnn" else 0)
                        ax.errorbar(
                            value,
                            y,
                            xerr=[[value - low], [high - value]],
                            fmt=MARKERS[k],
                            color=COLORS[system],
                            capsize=2,
                            ms=4,
                            mfc="white" if k == 0 else COLORS[system],
                            label=f"Seed {row['seed']}" if system == "cnn" and c == 0 else None,
                        )
                ax.axvline(0, color="0.5", lw=0.8)
                ax.set_yticks(range(4), LABELS)
                ax.invert_yaxis()
                ax.grid(axis="x", alpha=0.18)
                ax.set_title(
                    f"{regime.upper() if regime == 'wgn' else 'Semantic'} training · {region}"
                )
                if i == 1:
                    ax.set_xlabel("Augmented − clean accuracy (pp)")
        if system == "cnn":
            axes[0, 1].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"{system.upper()} paired intervention effects at +5 dB", fontsize=11)
        save(
            fig,
            f"{number}_{system}_paired_training_effects",
            rows,
            "Each marker compares augmentation with its matching clean seed on identical target mixtures. 95% paired target bootstrap intervals condition on fitted models; they are not seed-level confidence intervals. See CSV for the common Holm correction family.",
        )

    clap_rows = [r for r in all_conditions if r["system"] == "clap" and float(r["level_db"]) == 5]
    clap_effects = [r for r in effects if r["system"] == "clap" and r["stratum"] == "all"]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.6), layout="constrained")
    for j, region in enumerate(("own", "target")):
        for k, regime in enumerate(("baseline", "prompt_ensemble")):
            for c, condition in enumerate(CONDITIONS):
                row = next(
                    r
                    for r in clap_rows
                    if r["region"] == region
                    and r["regime"] == regime
                    and r["condition"] == condition
                )
                axes[0, j].errorbar(
                    c + (k - 0.5) * 0.23,
                    100 * float(row["macro_f1"]),
                    yerr=100 * float(row["macro_f1_fold_sd"]),
                    fmt=MARKERS[k],
                    capsize=3,
                    color=COLORS["clap"],
                    mfc="white" if k == 0 else COLORS["clap"],
                    label=("Fixed single prompt", "Fixed prompt ensemble")[k] if c == 0 else None,
                )
        axes[0, j].set_xticks(range(4), LABELS)
        axes[0, j].set_ylim(0, 100)
        axes[0, j].set_ylabel("Macro-F1 (%)")
        axes[0, j].set_title(f"{region.capitalize()} activity power")
        axes[0, j].legend(fontsize=7, loc="lower left")
        axes[0, j].grid(axis="y", alpha=0.18)
        for c, condition in enumerate(CONDITIONS):
            row = next(
                r for r in clap_effects if r["region"] == region and r["condition"] == condition
            )
            value, low, high = [
                100 * float(row[key])
                for key in ("accuracy_difference", "accuracy_ci_low", "accuracy_ci_high")
            ]
            axes[1, j].errorbar(
                value,
                c,
                xerr=[[value - low], [high - value]],
                fmt="s",
                capsize=3,
                color=COLORS["clap"],
                ms=5,
            )
        axes[1, j].set_yticks(range(4), LABELS)
        axes[1, j].invert_yaxis()
        axes[1, j].axvline(0, color="0.5", lw=0.8)
        axes[1, j].grid(axis="x", alpha=0.18)
        axes[1, j].set_xlabel("Ensemble − single-prompt accuracy (pp)")
    fig.suptitle("Frozen CLAP: controlled prompt-ensemble ablation at +5 dB", fontsize=10)
    save(
        fig,
        "16_clap_prompt_ensemble",
        clap_rows,
        "Top panels show five-fold macro-F1 means ± fold SD. Bottom panels show paired accuracy differences with 95% target bootstrap intervals; their source is 16_clap_prompt_ensemble_effects.csv. Same audio embeddings, all released fixed templates, no test-label prompt selection; this is not full PAT.",
    )
    with (output / "16_clap_prompt_ensemble_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clap_effects[0]))
        writer.writeheader()
        writer.writerows(clap_effects)

    replication = read(directory / "cnn_overlap_replication.csv")
    fig = replication_figure(replication)
    save(
        fig,
        "17_cnn_overlap_replication",
        replication,
        "Each seed uses the same matched high-overlap targets and a separate paired 95% target bootstrap CI. The 12 exact McNemar comparisons share a Holm correction family. Repeated target predictions are not independent samples.",
    )
    (output / "figure_metadata.json").write_text(
        json.dumps(
            {
                "stage": "export" if export else "preview",
                "figures": reports,
                "source": str(directory),
                "all_seeds_retained": [5305, 5306, 5307],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--qa-path", type=Path)
    parser.add_argument("--replication-only", action="store_true")
    args = parser.parse_args()
    audit = None
    if args.qa_path:
        import sys

        sys.path.insert(0, str(args.qa_path))
        from visual_qa import audit_layout

        audit = audit_layout
    if args.replication_only:
        rows = read(args.input / "cnn_overlap_replication.csv")
        expected = {
            (r, str(s), c)
            for r in ("own", "target")
            for s in (5305, 5306, 5307)
            for c in CONDITIONS[2:]
        }
        if len(rows) != 12 or {(r["region"], r["seed"], r["condition"]) for r in rows} != expected:
            raise ValueError("Incomplete clean-seed replication")
        plt.rcParams.update(
            {
                "font.size": 9,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
        figure = replication_figure(rows)
        args.output.mkdir(parents=True, exist_ok=True)
        report = audit(figure) if audit else None
        for suffix in ["png", "pdf", "svg"] if args.export else ["png"]:
            figure.savefig(
                args.output / f"17_cnn_overlap_replication.{suffix}",
                dpi=600 if args.export else 160,
            )
        with (args.output / "17_cnn_overlap_replication.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output / "layout_audit.json").write_text(json.dumps(report, indent=2))
        plt.close(figure)
    else:
        plot(args.input, args.output, args.export, audit)

"""Four supplementary research figures from verified diagnostic tables.

Use --preview first and inspect the PNGs before --export.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

COLORS = {"svm": "#0072B2", "cnn": "#D55E00", "clap": "#009E73"}
CONDITIONS = ["white", "demand", "semantic_same", "semantic_different"]
LABELS = ["White noise", "DEMAND", "Same category", "Different category"]


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def plot(directory, output, export=False, audit=None):
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
    saved = []
    layout_reports = {}

    def save(fig, name, rows):
        if audit is not None:
            layout_reports[name] = audit(fig)
        fig.savefig(output / f"{name}.png", dpi=600 if export else 160)
        if export:
            fig.savefig(output / f"{name}.svg")
            fig.savefig(output / f"{name}.pdf")
        with (output / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        saved.append(name)
        plt.close(fig)

    uncertainty = [
        r
        for r in read(directory / "paired_uncertainty.csv")
        if float(r["level_db"]) == 5 and r["stratum"] == "overlap_ge_0.8"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.4), sharex=True, layout="constrained")
    for ax, region in zip(axes, ("own", "target")):
        selected = [r for r in uncertainty if r["region"] == region]
        for i, row in enumerate(selected):
            value, low, high = [
                100 * float(row[k])
                for k in ("accuracy_difference", "accuracy_ci_low", "accuracy_ci_high")
            ]
            color = COLORS[row["system"]]
            ax.errorbar(
                value,
                i,
                xerr=[[value - low], [high - value]],
                fmt="o",
                color=color,
                mfc=color if row["semantic_condition"] == "semantic_same" else "white",
                capsize=3,
                ms=5,
            )
        ax.set_yticks(
            range(len(selected)),
            [
                r["system"].upper()
                + (" / same" if r["semantic_condition"] == "semantic_same" else " / different")
                for r in selected
            ],
        )
        ax.invert_yaxis()
        ax.axvline(0, color="0.5", lw=0.8)
        ax.grid(axis="x", alpha=0.2)
        ax.set_title(f"{region.capitalize()} activity power")
        ax.set_xlabel("Semantic − DEMAND accuracy (pp)")
    fig.suptitle("+5 dB, overlap ≥ 0.8: paired 95% bootstrap intervals", fontsize=10)
    save(fig, "09_paired_accuracy_intervals", uncertainty)

    groups = [
        "rms",
        "zcr",
        "spectral_centroid",
        "spectral_spread",
        "spectral_rolloff",
        "spectral_flux",
        "mfcc",
        "delta_mfcc",
    ]
    group_labels = ["RMS", "ZCR", "Centroid", "Spread", "Roll-off", "Flux", "MFCC", "ΔMFCC"]
    features = [
        r
        for r in read(directory / "feature_group_displacement.csv")
        if float(r["level_db"]) == 5 and r["outcome"] == "all"
    ]
    # Resolve names from the extractor, avoiding inferred array positions.
    available = {r["feature_group"] for r in features}
    if set(groups) != available:
        raise ValueError(f"Unexpected feature groups: {available}")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.8), sharey=True, layout="constrained")
    values = [float(r["median"]) for r in features]
    norm = LogNorm(vmin=min(values), vmax=max(values))
    for ax, region in zip(axes, ("own", "target")):
        lookup = {
            (r["feature_group"], r["condition"]): float(r["median"])
            for r in features
            if r["region"] == region
        }
        matrix = np.array(
            [[lookup[(group, condition)] for condition in CONDITIONS] for group in groups]
        )
        im = ax.imshow(matrix, norm=norm, cmap="viridis", aspect="auto")
        for i in range(8):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if norm(matrix[i, j]) > 0.65 else "white",
                )
        ax.set_yticks(range(8), group_labels)
        ax.set_xticks(range(4), ["White", "DEMAND", "Same", "Different"], rotation=25, ha="right")
        ax.set_title(f"{region.capitalize()} activity power")
    fig.colorbar(im, ax=axes, label="Median standardised displacement (log scale)", shrink=0.86)
    fig.suptitle("Handcrafted feature groups at +5 dB", fontsize=10)
    save(fig, "10_feature_group_displacement", features)

    ranks = read(directory / "clap_rank_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0), layout="constrained")
    for i, region in enumerate(("own", "target")):
        for condition, style, marker, label in (
            ("semantic_same", "--", "o", "Same category"),
            ("semantic_different", "-", "s", "Different category"),
        ):
            selected = sorted(
                [r for r in ranks if r["region"] == region and r["condition"] == condition],
                key=lambda r: -float(r["level_db"]),
            )
            x = [float(r["level_db"]) for r in selected]
            for j, key in enumerate(("distractor_rank_median", "distractor_top5_fraction")):
                y = [float(r[key]) * (100 if j else 1) for r in selected]
                axes[i, j].plot(
                    x,
                    y,
                    style,
                    marker=marker,
                    color=COLORS["clap"],
                    mfc=COLORS["clap"] if condition == "semantic_same" else "white",
                    label=label,
                    ms=5,
                )
            axes[i, 0].plot(
                x,
                [float(r["control_distractor_rank_median"]) for r in selected],
                style,
                color="0.5",
                marker=marker,
                ms=3,
                label=f"Control: {'same' if condition == 'semantic_same' else 'different'}",
            )
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([10, 5, 0], ["+10", "+5", "0"])
            ax.invert_xaxis()
            ax.set_xlabel("SIR (dB)")
            ax.set_title(f"{region.capitalize()} activity power")
            ax.grid(alpha=0.2)
        axes[i, 0].set_ylabel("Median distractor rank")
        axes[i, 0].set_ylim(30, 1)
        axes[i, 0].set_yticks([1, 5, 10, 15, 20, 25, 30])
        axes[i, 1].set_ylabel("Distractor in top 5 (%)")
        axes[i, 1].set_ylim(0, 100)
    axes[0, 0].legend(
        fontsize=6.8,
        loc="center",
        bbox_to_anchor=(0.5, 0.64),
        ncol=2,
        columnspacing=0.7,
        handlelength=1.5,
    )
    axes[0, 1].legend(fontsize=8, loc="lower left")
    save(fig, "11_clap_distractor_ranking", ranks)

    auc = [
        r
        for r in read(directory / "displacement_error_auc.csv")
        if float(r["level_db"]) == 5 and r["fold"] != "pooled"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.4), sharey=True, layout="constrained")
    for ax, region in zip(axes, ("own", "target")):
        for j, system in enumerate(("svm", "cnn", "clap")):
            for k, condition in enumerate(CONDITIONS):
                selected = [
                    float(r["error_discrimination_auc"])
                    for r in auc
                    if r["region"] == region
                    and r["system"] == system
                    and r["condition"] == condition
                ]
                x = k + (j - 1) * 0.23
                ax.scatter(
                    x + np.linspace(-0.045, 0.045, len(selected)),
                    selected,
                    s=14,
                    color=COLORS[system],
                    marker=("o", "s", "^")[j],
                    label=system.upper() if k == 0 else None,
                    alpha=0.8,
                )
                ax.plot([x - 0.075, x + 0.075], [np.mean(selected)] * 2, color=COLORS[system], lw=2)
        ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
        ax.set_xticks(range(4), ["White", "DEMAND", "Same", "Different"], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"{region.capitalize()} activity power")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Error discrimination AUC")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Displacement and errors at +5 dB: five folds, bars show means", fontsize=10)
    save(fig, "12_displacement_error_association", auc)
    (output / "figure_metadata.json").write_text(
        json.dumps(
            {
                "figures": saved,
                "stage": "export" if export else "preview",
                "layout_audit": layout_reports,
                "uncertainty": "2000 paired target bootstrap replicates within folds; fixed trained models",
                "feature_displacement": "relative to gain-matched clean control; training-fold SVM scaling; /sqrt(group dimension)",
                "ranking": "competition rank among 50 classes; curves use all targets, not only errors",
                "auc": "Descriptive association only; dots are official folds, bars are unweighted means",
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
    args = parser.parse_args()
    audit = None
    if args.qa_path:
        import sys

        sys.path.insert(0, str(args.qa_path))
        from visual_qa import audit_layout

        audit = audit_layout
    plot(args.input, args.output, args.export, audit)

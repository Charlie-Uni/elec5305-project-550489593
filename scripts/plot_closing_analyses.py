"""Render final readout/overlap diagnostics without changing earlier figures."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from esc_robustness.artifacts import sha256_file, write_json

COLORS = {"svm": "#0072B2", "cnn": "#D55E00", "clap": "#009E73"}
VARIANTS = {
    "svm_baseline": ("SVM", "#0072B2", "o", "-"),
    "cnn_baseline": ("CNN", "#D55E00", "s", "-"),
    "clap_baseline": ("CLAP single", "#009E73", "^", "-"),
    "clap_prompt_ensemble": ("CLAP ensemble", "#009E73", "D", "--"),
}
CONDS = ("white", "demand", "semantic_same", "semantic_different")


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def plot(args):
    data, out = args.data, args.output
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
        }
    )
    qa = None
    if args.qa_tools:
        spec = importlib.util.spec_from_file_location(
            "closing_visual_qa", args.qa_tools / "visual_qa.py"
        )
        qa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qa)
    audits = {}

    def save(fig, name, records):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        w, h = fig.get_size_inches() * fig.dpi
        outside = []
        for text in fig.findobj(matplotlib.text.Text):
            if not text.get_visible() or not text.get_text():
                continue
            b = text.get_window_extent(renderer)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > w + 1 or b.y1 > h + 1:
                outside.append(text.get_text())
        audits[name] = {
            "text_outside_canvas": outside,
            "size_inches": fig.get_size_inches().tolist(),
            "qa": qa.audit_layout(fig) if qa else None,
        }
        if outside:
            raise ValueError(f"{name}: text outside canvas {outside}")
        fig.savefig(out / f"{name}.png", dpi=180 if not args.export else 300)
        if args.export:
            fig.savefig(out / f"{name}.pdf")
            fig.savefig(out / f"{name}.svg")
        with Image.open(out / f"{name}.png") as im:
            im.convert("L").save(out / f"{name}_gray.png")
        if records:
            fields = list(records[0])
            with (out / f"{name}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
        plt.close(fig)

    outcomes = read(data / "readout_outcomes.csv")
    for variant, suffix in [("clap_baseline", "single"), ("clap_prompt_ensemble", "ensemble")]:
        records = [r for r in outcomes if r["variant"] == variant and r["k"] == "2"]
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9), sharey=True, layout="constrained")
        states = ("target_only", "both", "distractor_only", "neither")
        shades = ("#dbefe4", "#009E73", "#006748", "#dddddd")
        hatches = ("", "//", "xx", "..")
        for i, region in enumerate(("own", "target")):
            for j, condition in enumerate(("semantic_same", "semantic_different")):
                ax = axes[i, j]
                rr = sorted(
                    [r for r in records if r["region"] == region and r["condition"] == condition],
                    key=lambda r: -float(r["level_db"]),
                )
                for prefix, shift in [("control", -0.18), ("mixture", 0.18)]:
                    bottom = np.zeros(3)
                    for state, color, hatch in zip(states, shades, hatches):
                        values = np.array([float(r[f"{prefix}_{state}"]) for r in rr])
                        ax.bar(
                            np.arange(3) + shift,
                            values,
                            0.31,
                            bottom=bottom,
                            color=color,
                            edgecolor="#555555",
                            linewidth=0.35,
                            hatch=hatch,
                            label=state.replace("_", " ")
                            if prefix == "mixture" and i == j == 0
                            else None,
                        )
                        bottom += values
                ax.set(
                    xticks=range(3),
                    xticklabels=["+10", "+5", "0"],
                    ylim=(0, 1),
                    xlabel="SIR (dB; 0 dB stress test)",
                )
                ax.set_title(
                    f"{region.capitalize()} / {'same' if j == 0 else 'different'} category · n={rr[0]['n']}"
                )
                if j == 0:
                    ax.set_ylabel("Fraction of designated pairs")
                ax.grid(axis="y", alpha=0.15)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        fig.suptitle(
            f"CLAP {suffix}: exact top-2 inclusion\nEach level: gain-matched control (left), mixture (right)",
            fontsize=10,
        )
        save(fig, f"20_readout_top2_{suffix}", records)
    ci = read(data / "readout_uncertainty.csv")
    rr = [
        r
        for r in ci
        if r["metric"] == "target_inclusion_given_capture" and float(r["level_db"]) == 5
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9), sharey=True, layout="constrained")
    for i, region in enumerate(("own", "target")):
        for j, condition in enumerate(("semantic_same", "semantic_different")):
            ax = axes[i, j]
            for variant, (label, color, marker, style) in VARIANTS.items():
                subset = sorted(
                    [
                        r
                        for r in rr
                        if r["variant"] == variant
                        and r["region"] == region
                        and r["condition"] == condition
                    ],
                    key=lambda r: int(r["k"]),
                )
                y = np.array([float(r["value"]) for r in subset])
                lo = np.array([float(r["ci_low"]) for r in subset])
                hi = np.array([float(r["ci_high"]) for r in subset])
                ax.errorbar(
                    [int(r["k"]) for r in subset],
                    y,
                    yerr=[y - lo, hi - y],
                    label=label,
                    color=color,
                    marker=marker,
                    ls=style,
                    capsize=2,
                    ms=4,
                    lw=1,
                )
            ax.set(xticks=[1, 2, 3, 5], ylim=(0, 1), xlabel="Number of ranked labels, k")
            ax.grid(alpha=0.15)
            ax.set_title(f"{region.capitalize()} / {'same' if j == 0 else 'different'} category")
            if j == 0:
                ax.set_ylabel("Target included | original capture")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(
        "+5 dB: target-rank retention among original captures\nPointwise 95% target-source bootstrap intervals",
        fontsize=10,
    )
    save(fig, "21_capture_rank_retention", rr)
    for region in ("own", "target"):
        features = read(args.run / f"overlap_{region}/overlap_features.csv")
        records = [r for r in features if float(r["level_db"]) == 5]
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0), layout="constrained")
        for ax, key, title in zip(
            axes,
            ("spectral_correlation_db", "density_spectral_correlation_db", "masked_cell_fraction"),
            ("Raw Mel-band correlation", "Bandwidth-normalised correlation", "TF masking proxy"),
        ):
            arrays = [
                np.array([float(r[key]) for r in records if r["condition"] == c and r[key]])
                for c in CONDS
            ]
            ax.boxplot(
                arrays,
                tick_labels=["White", "DEMAND", "Same", "Different"],
                showfliers=False,
                widths=0.5,
                medianprops={"color": "black"},
                boxprops={"color": "#555555"},
            )
            ax.tick_params(axis="x", labelrotation=35)
            ax.set_title(title, fontsize=8)
            ax.set_ylim((-1.05, 1.05) if "correlation" in key else (0, 1.05))
            ax.axhline(0, color="0.7", lw=0.5)
        axes[0].set_ylabel("Pearson r across 64 Mel bands")
        axes[2].set_ylabel("Fraction of active target TF cells")
        fig.suptitle(
            f"{region.capitalize()} power, +5 dB: acoustic descriptor distributions\n"
            f"Boxes: quartiles; whiskers: 1.5 IQR; {'2,000' if region == 'own' else '1,998–2,000'} targets per condition",
            fontsize=9,
        )
        save(fig, f"22_acoustic_distributions_{region}", records)
    matched = read(data / "acoustic_matched_comparisons.csv")
    records = [
        r
        for r in matched
        if r["b_condition"] == "demand"
        and r["stratum"] in ("all", "both_overlap_ge_0.8", "close_primary")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.8), sharex=True, layout="constrained")
    labels = [f"{s.upper()} / {c}" for s in COLORS for c in ("same", "different")]
    for ax, region in zip(axes, ("own", "target")):
        for i, (system, condition) in enumerate(
            (s, c) for s in COLORS for c in ("semantic_same", "semantic_different")
        ):
            for stratum, off, marker, label in [
                ("all", -0.18, "o", "All paired targets"),
                ("both_overlap_ge_0.8", 0, "s", "Both overlaps ≥ 0.8"),
                ("close_primary", 0.18, "D", "Three-metric closeness"),
            ]:
                r = next(
                    r
                    for r in records
                    if r["region"] == region
                    and r["system"] == system
                    and r["a_condition"] == condition
                    and r["stratum"] == stratum
                )
                value, lo, hi = [
                    100 * float(r[k]) for k in ("accuracy_difference", "ci_low", "ci_high")
                ]
                color = "0.6" if stratum == "all" else COLORS[system]
                ax.errorbar(
                    value,
                    i + off,
                    xerr=[[value - lo], [hi - value]],
                    fmt=marker,
                    color=color,
                    mfc="white" if stratum == "both_overlap_ge_0.8" else color,
                    ms=4,
                    capsize=2,
                    lw=1,
                    label=label if i == 0 else None,
                )
                if stratum == "close_primary":
                    ax.annotate(
                        "n=" + r["n"],
                        (hi, i + off),
                        xytext=(4, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=6,
                    )
        ax.set(
            yticks=range(6),
            yticklabels=labels,
            xlabel="Semantic − DEMAND accuracy (pp)",
            title=region.capitalize() + " activity power",
            xlim=(-42, 25),
            xticks=[-40, -20, 0, 20],
        )
        ax.invert_yaxis()
        ax.axvline(0, c="0.5", lw=0.8)
        ax.grid(axis="x", alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        "+5 dB: same-target contrasts under acoustic selection\nPointwise 95% target-source bootstrap intervals; subsets differ",
        fontsize=10,
    )
    save(fig, "23_acoustic_matched_contrasts", records)
    strata = read(data / "acoustic_stratified.csv")
    records = [r for r in strata if r["measure"] == "masked_cell_fraction"]
    fig, axes = plt.subplots(
        2, 3, figsize=(7.16, 5.0), sharex=True, sharey=True, layout="constrained"
    )
    for i, region in enumerate(("own", "target")):
        for j, system in enumerate(COLORS):
            ax = axes[i, j]
            for condition, color, marker, style, label in [
                ("white", "0.6", "x", ":", "White"),
                ("demand", "0.2", "s", "--", "DEMAND"),
                ("semantic_same", COLORS[system], "o", "-", "Same"),
                ("semantic_different", COLORS[system], "^", "-", "Different"),
            ]:
                rr = sorted(
                    [
                        r
                        for r in records
                        if r["region"] == region
                        and r["system"] == system
                        and r["condition"] == condition
                        and r["ci_low"]
                    ],
                    key=lambda r: float(r["bin_low"]),
                )
                x = np.array([(float(r["bin_low"]) + float(r["bin_high"])) / 2 for r in rr])
                y = np.array([float(r["accuracy"]) for r in rr])
                lo = np.array([float(r["ci_low"]) for r in rr])
                hi = np.array([float(r["ci_high"]) for r in rr])
                ax.errorbar(
                    x,
                    y,
                    yerr=[y - lo, hi - y],
                    fmt=marker,
                    color=color,
                    mfc="white" if condition == "semantic_different" else color,
                    ms=4,
                    capsize=2,
                    label=label,
                )
            ax.set(xlim=(0, 1), ylim=(0, 1), title=f"{region.capitalize()} / {system.upper()}")
            ax.grid(alpha=0.15)
            if i == 1:
                ax.set_xlabel("Masked fraction of target TF cells")
            if j == 0:
                ax.set_ylabel("Accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(
        "+5 dB: descriptive masking bins, different target compositions\nPointwise 95% source-bootstrap intervals; ≥30 targets and source clusters",
        fontsize=9,
    )
    save(fig, "24_accuracy_by_masking", records)
    write_json(out / "layout_audit.json", audits)
    write_json(
        out / "figure_provenance.json",
        {
            "data_hashes": {p.name: sha256_file(p) for p in data.glob("*.csv")},
            "source_sha256": sha256_file(Path(__file__)),
            "exported": args.export,
        },
    )
    print(json.dumps({"figures": list(audits), "exported": args.export}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "run", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--qa-tools", type=Path)
    p.add_argument("--export", action="store_true")
    plot(p.parse_args())

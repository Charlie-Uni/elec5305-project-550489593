"""Draw the two diagnostic figures added to the concise report from committed figure tables.

Inputs are the audited CSVs behind the earlier figure set (own-activity power). No model is run.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/figures_three_systems_own_20261109"
OUT = ROOT / "results/final_submission_20261110/figures"
COLORS = {"svm": "#0072B2", "cnn": "#D55E00", "clap": "#009E73"}
NAMES = {"svm": "SVM", "cnn": "CNN", "clap": "CLAP"}
MARKERS = {"svm": "o", "cnn": "s", "clap": "^"}
LEVELS = (10.0, 5.0, 0.0)
WIDTH = 170 / 25.4


def read(name):
    with (SOURCE / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name, rows):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    renderer = fig.canvas.get_renderer()
    displayed = list(fig.texts)
    for ax in fig.axes:
        displayed.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if lo <= tick.get_loc() <= hi:
                    displayed.extend([tick.label1, tick.label2])
        if ax.get_legend() is not None:
            displayed.extend(ax.get_legend().get_texts())
    for item in displayed:
        if not item.get_visible() or not item.get_text():
            continue
        bounds = item.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > width + 1 or bounds.y1 > height + 1:
            raise ValueError(f"Text outside figure bounds: {name}: {item.get_text()}")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=240)
    Image.open(OUT / f"{name}.png").convert("L").save(OUT / f"{name}_gray.png")
    with (OUT / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.close(fig)


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def capture_and_margin():
    capture, margin = read("fig2_capture_rate_vs_sir.csv"), read("fig3_margin_vs_sir.csv")
    fig, axes = plt.subplots(
        1, 4, figsize=(WIDTH, 2.25), gridspec_kw={"width_ratios": [1.5, 1, 1, 1]}
    )
    ax = axes[0]
    for system in NAMES:
        for condition, line, fill in (
            ("semantic_same", "-", True),
            ("semantic_different", "--", False),
        ):
            rows = {
                float(r["level_db"]): r
                for r in capture
                if r["system"] == system and r["condition"] == condition
            }
            ax.plot(
                LEVELS,
                [100 * float(rows[level]["capture_rate"]) for level in LEVELS],
                line,
                color=COLORS[system],
                marker=MARKERS[system],
                markersize=4,
                markerfacecolor=COLORS[system] if fill else "white",
                linewidth=1.2,
                label=NAMES[system] if fill else None,
            )
    ax.axhline(100 / 49, color="0.45", linestyle=":", linewidth=1)
    ax.text(0.2, 100 / 49 + 2, "uniform 1/49", fontsize=7, color="0.35", ha="right")
    ax.set_title("(a) Errors naming the distractor")
    ax.set_ylabel("Share of errors (%)")
    ax.set_ylim(0, 80)
    for system, height in (("clap", 61.0), ("cnn", 39.5), ("svm", 5.5)):
        ax.text(
            10.9,
            height,
            NAMES[system],
            color=COLORS[system],
            fontsize=8,
            fontweight="bold",
            va="center",
        )
    for ax, system, label in zip(
        axes[1:], NAMES, ("(b) SVM decision score", "(c) CNN logit", "(d) CLAP cosine")
    ):
        for condition, offset, fill in (
            ("semantic_same", 0.6, True),
            ("semantic_different", -0.6, False),
        ):
            rows = {
                float(r["level_db"]): r
                for r in margin
                if r["system"] == system and r["condition"] == condition
            }
            median = [float(rows[level]["median"]) for level in LEVELS]
            low = [float(rows[level]["median"]) - float(rows[level]["q1"]) for level in LEVELS]
            high = [float(rows[level]["q3"]) - float(rows[level]["median"]) for level in LEVELS]
            ax.errorbar(
                [level + offset for level in LEVELS],
                median,
                yerr=[low, high],
                fmt=MARKERS[system],
                color=COLORS[system],
                markersize=4,
                markerfacecolor=COLORS[system] if fill else "white",
                elinewidth=1,
                capsize=2,
            )
        ax.axhline(0, color="0.45", linewidth=0.8)
        ax.set_title(label)
    axes[1].set_ylabel("Target − distractor score")
    for ax in axes:
        ax.set_xlim(11.5, -1.5)
        ax.set_xticks(LEVELS, ["+10", "+5", "0"])
        ax.set_xlabel("SIR (dB)")
        ax.grid(axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout(w_pad=0.8)
    rows = [
        dict(
            panel="capture",
            system=r["system"],
            condition=r["condition"],
            level_db=r["level_db"],
            value=r["capture_rate"],
            low="",
            high="",
        )
        for r in capture
    ] + [
        dict(
            panel="margin",
            system=r["system"],
            condition=r["condition"],
            level_db=r["level_db"],
            value=r["median"],
            low=r["q1"],
            high=r["q3"],
        )
        for r in margin
    ]
    save(fig, "04_capture_margin", rows)


def displacement():
    data = read("fig5_representation_displacement.csv")
    conditions = (
        ("white", "White"),
        ("demand", "DEMAND"),
        ("semantic_same", "Same"),
        ("semantic_different", "Different"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.05))
    labels = (
        "(a) SVM: standardised feature shift",
        "(b) CNN: cosine distance",
        "(c) CLAP: cosine distance",
    )
    for ax, system, label in zip(axes, NAMES, labels):
        for outcome, offset, fill in (("correct", -0.16, True), ("incorrect", 0.16, False)):
            rows = {
                r["condition"]: r for r in data if r["system"] == system and r["outcome"] == outcome
            }
            median = [float(rows[c]["median"]) for c, _ in conditions]
            low = [float(rows[c]["median"]) - float(rows[c]["q1"]) for c, _ in conditions]
            high = [float(rows[c]["q3"]) - float(rows[c]["median"]) for c, _ in conditions]
            ax.errorbar(
                [i + offset for i in range(4)],
                median,
                yerr=[low, high],
                fmt=MARKERS[system],
                color=COLORS[system],
                markersize=4,
                markerfacecolor=COLORS[system] if fill else "white",
                elinewidth=1,
                capsize=2,
                label="classified correctly" if fill else "misclassified",
            )
        ax.set_title(label)
        ax.set_xticks(range(4), [name for _, name in conditions])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0].set_ylabel("Displacement from control")
    axes[0].legend(frameon=False, loc="upper right", handletextpad=0.2)
    fig.tight_layout(w_pad=0.8)
    rows = [
        dict(
            system=r["system"],
            condition=r["condition"],
            outcome=r["outcome"],
            n=r["n"],
            median=r["median"],
            q1=r["q1"],
            q3=r["q3"],
        )
        for r in data
    ]
    save(fig, "05_displacement", rows)


if __name__ == "__main__":
    style()
    capture_and_margin()
    displacement()
    print("wrote 04_capture_margin and 05_displacement to", OUT)

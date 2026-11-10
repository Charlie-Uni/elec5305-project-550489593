"""Build the concise submission report from unchanged, audited experiment tables."""

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from delivery_claims import closing_claims, editorial_claims, local_alignment_claims
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/final_submission_20261110"
COLORS = {"svm": "#0072B2", "cnn": "#D55E00", "clap": "#009E73"}
NAMES = {"svm": "SVM", "cnn": "CNN", "clap": "CLAP"}
CONDS = ["white", "demand", "semantic_same", "semantic_different"]
LABELS = ["White noise", "DEMAND", "Same category", "Different category"]


def read(name):
    with (ROOT / name).open() as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_assets():
    OUT.mkdir(exist_ok=True)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    closing = closing_claims()
    core = read("results/clap_full_20261109/all_conditions_verified.csv")
    data = json.loads(
        (ROOT / "results/presentation_20261109_draft/presentation_data.json").read_text()
    )
    data["editorial"] = editorial_claims()
    alignment = local_alignment_claims()
    if alignment is not None:
        data["local_alignment"] = alignment
    clean = {
        s: json.loads((ROOT / p).read_text())
        for s, p in {
            "svm": "results/clean_svm/summary.json",
            "cnn": "results/clean_cnn/summary.json",
            "clap": "results/clap_integration_20261109/clean/summary.json",
        }.items()
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    geometry = []

    def save(fig, name):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        issues = []
        for t in fig.findobj(matplotlib.text.Text):
            if not t.get_visible() or not t.get_text():
                continue
            b = t.get_window_extent(fig.canvas.get_renderer())
            if b.x0 < -1 or b.y0 < -1 or b.x1 > w + 1 or b.y1 > h + 1:
                issues.append(t.get_text())
        if issues:
            raise ValueError((name, issues))
        for ext in ("pdf", "svg", "png"):
            fig.savefig(figures / f"{name}.{ext}", dpi=240)
        Image.open(figures / f"{name}.png").convert("L").save(figures / f"{name}_gray.png")
        geometry.append(
            dict(figure=name, size_inches=list(fig.get_size_inches()), out_of_bounds=issues)
        )
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(170 / 25.4, 3.35), sharex=True, sharey=True)
    for ax, c, title in zip(axes.flat, CONDS, LABELS):
        for s, marker in zip(NAMES, ["o", "s", "^"]):
            rows = [
                next(
                    r
                    for r in core
                    if r["region"] == "own"
                    and r["system"] == s
                    and r["condition"] == f"{c}@+{level}dB"
                )
                for level in (10, 5)
            ]
            rows.append(
                next(
                    r
                    for r in core
                    if r["region"] == "own" and r["system"] == s and r["condition"] == f"{c}@+0dB"
                )
            )
            ax.errorbar(
                [10, 5, 0],
                [100 * float(r["macro_f1_mean"]) for r in rows],
                yerr=[100 * float(r["macro_f1_sd"]) for r in rows],
                label=NAMES[s],
                color=COLORS[s],
                marker=marker,
                ms=4,
                lw=1.2,
                capsize=2,
            )
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(0, 100)
        ax.set_xlim(11, -1)
        ax.set_xticks([10, 5, 0])
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1 (%)")
    for ax in axes[1]:
        ax.set_xlabel("SNR / SIR (dB)")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.54, 1.015),
    )
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.10, right=0.985, hspace=0.45, wspace=0.16)
    save(fig, "01_robustness")
    rr = [
        next(
            r
            for r in closing["readout"]
            if r["variant"] == "clap_baseline"
            and r["region"] == "own"
            and r["condition"] == c
            and r["k"] == "2"
            and float(r["level_db"]) == 5
        )
        for c in CONDS[2:]
    ]
    fig, ax = plt.subplots(figsize=(170 / 25.4, 1.8))
    left = np.zeros(2)
    for state, label, color, hatch in zip(
        ["target_only", "both", "distractor_only", "neither"],
        ["Target only", "Both", "Distractor only", "Neither"],
        ["#deeee7", "#009e73", "#175944", "#d4d8d7"],
        ["", "//", "xx", ".."],
    ):
        vals = np.array([100 * float(r["mixture_" + state]) for r in rr])
        ax.barh(
            [1, 0],
            vals,
            left=left,
            label=label,
            color=color,
            edgecolor="#536366",
            hatch=hatch,
            height=0.55,
            lw=0.4,
        )
        for i, (v, offset) in enumerate(zip(vals, left)):
            ax.text(
                offset + v / 2,
                1 - i,
                f"{v:.2f}",
                va="center",
                ha="center",
                fontsize=8,
                color="white" if state == "distractor_only" else "#172a2d",
                bbox=dict(facecolor=color, edgecolor="none", pad=0.7),
            )
        left += vals
    ax.set_yticks([1, 0], ["Same category", "Different category"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("All semantic trials (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.48, 1.02), ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.19, right=0.965, bottom=0.30, top=0.72)
    save(fig, "02_top_two")
    fig, axes = plt.subplots(1, 2, figsize=(170 / 25.4, 2.35), sharey=True)
    for ax, region in zip(axes, ["own", "target"]):
        for i, s in enumerate(NAMES):
            for j, c in enumerate(CONDS[2:]):
                r = next(
                    r
                    for r in closing["acoustic_primary"]
                    if r["region"] == region
                    and r["system"] == s
                    and r["a_condition"] == c
                    and r["b_condition"] == "demand"
                )
                v, lo, hi = [
                    100 * float(r[k]) for k in ["accuracy_difference", "ci_low", "ci_high"]
                ]
                y = 2 - i + (0.12 if j == 0 else -0.12)
                ax.errorbar(
                    v,
                    y,
                    xerr=[[v - lo], [hi - v]],
                    fmt="o" if j == 0 else "s",
                    color=COLORS[s],
                    mfc=COLORS[s] if j == 0 else "white",
                    ms=4,
                    capsize=2,
                    label=("Same category" if j == 0 else "Different category") if i == 0 else None,
                )
        ax.axvline(0, color="#888888", lw=0.8, ls=":")
        ax.set_xlim(-40, 20)
        ax.set_xticks([-40, -20, 0, 20])
        ax.set_yticks([2, 1, 0], ["SVM", "CNN", "CLAP"])
        ax.set_title(region.capitalize() + " activity", loc="left", fontsize=10)
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel("Accuracy difference (pp)")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.55, 1.03),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.24, top=0.78, wspace=0.20)
    save(fig, "03_acoustic_matching")
    rows = []
    for s in NAMES:
        vals = [
            f"{100 * clean[s][m]['mean']:.2f} $\\pm$ {100 * clean[s][m]['sample_std']:.2f}"
            for m in ["accuracy", "macro_f1"]
        ]
        for condition in ("semantic_same", "semantic_different"):
            row = next(
                r
                for r in data["editorial"]["capture"]
                if r["system"] == s and r["condition"] == condition
            )
            vals.append(
                f"{100 * float(row['capture_rate']):.2f} ({row['errors_equal_distractor']}/{row['errors']})"
            )
        rows.append(NAMES[s] + " & " + " & ".join(vals) + r" \\")
    text = "\\newcommand{\\CleanRows}{\n" + "\n".join(rows) + "\n}\n"
    rows = []
    for regime in ["clean", "wgn", "semantic"]:
        vals = []
        for c in ["clean", *CONDS]:
            source = data["clean_variants"] if c == "clean" else data["interventions"]
            rr0 = [
                r
                for r in source
                if r["system"] == "cnn"
                and r["regime"] == regime
                and (
                    c == "clean"
                    or (r["region"] == "own" and r["condition"] == c and float(r["level_db"]) == 5)
                )
            ]
            assert len(rr0) == 3
            v = np.array([100 * float(r["macro_f1"]) for r in rr0])
            vals.append(f"{v.mean():.2f} $\\pm$ {v.std(ddof=1):.2f}")
        rows.append(
            {"clean": "Clean", "wgn": "White noise", "semantic": "Semantic"}[regime]
            + " & "
            + " & ".join(vals)
            + r" \\"
        )
    text += "\\newcommand{\\AugRows}{\n" + "\n".join(rows) + "\n}\n"
    rows = []
    for s in NAMES:
        vals = [
            next(
                r
                for r in data[key]
                if r["variant"] == s + ("_baseline" if s == "clap" else "_clean")
            )["total_median_ms"]
            for key in ["cost_panel", "cost_panel_reverse"]
        ]
        rows.append(NAMES[s] + " & " + " & ".join(f"{float(v):.2f}" for v in vals) + r" \\")
    text += "\\newcommand{\\CostRows}{\n" + "\n".join(rows) + "\n}\n"
    (ROOT / "docs/final_report_tables.tex").write_text(text)

    # Export the exact source rows alongside each new report figure.
    def save_rows(name, rows):
        with (figures / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    save_rows("01_robustness", [r for r in core if r["region"] == "own"])
    save_rows("02_top_two", rr)
    save_rows(
        "03_acoustic_matching",
        [r for r in closing["acoustic_primary"] if r["b_condition"] == "demand"],
    )
    (OUT / "figure_audit.json").write_text(
        json.dumps(
            dict(
                geometry=geometry,
                source="Unchanged audited core and closing tables; all figures use the final 170-mm report width",
                core_rows=len(core),
                readout_rows=len(closing["readout"]),
                acoustic_rows=len(closing["acoustic_primary"]),
            ),
            indent=2,
        )
    )
    (OUT / "presentation_data.json").write_text(json.dumps(data, indent=2))


def build():
    build_assets()
    from build_report_extra_figures import capture_and_margin, displacement, style

    style()
    capture_and_margin()
    displacement()
    temp = ROOT.parent / "runs/final_submission_20261110/report_build"
    temp.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            str(ROOT / ".venv/bin/tectonic"),
            "--only-cached",
            "--keep-logs",
            "--outdir",
            str(temp),
            str(ROOT / "docs/final_report.tex"),
        ],
        cwd=ROOT / "docs",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    (temp / "build_output.txt").write_text(r.stdout)
    if r.returncode:
        raise RuntimeError(r.stdout[-5000:])
    shutil.copyfile(temp / "final_report.pdf", OUT / "final_report.pdf")
    reader = PdfReader(OUT / "final_report.pdf")
    print(json.dumps(dict(pages=len(reader.pages), sha256=sha(OUT / "final_report.pdf"))))


if __name__ == "__main__":
    build()

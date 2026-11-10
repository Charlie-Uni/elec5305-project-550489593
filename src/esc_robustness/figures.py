"""Report figures from an interference evaluation, with the numbers behind each one.

Every figure is built from the per-row prediction CSVs and score archives written by
``evaluate-interference`` plus the clean baseline predictions, so the plotted values
can be re-derived from files in the repository. Each figure writes a PNG, an SVG and
a CSV with its data. Systems are discovered from ``<system>_predictions.csv``; two or
three systems work alike. Colours follow a fixed per-system assignment, condition
contrasts use fill versus hatch, and axes are shared within a figure.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from .analysis import fold_metrics, read_clean_predictions, read_predictions
from .artifacts import sha256_file, write_json

SYSTEM_ORDER = ("svm", "cnn", "clap")
SYSTEM_LABEL = {"svm": "Handcrafted + SVM", "cnn": "Log-Mel CNN", "clap": "Zero-shot CLAP"}
SYSTEM_COLOUR = {"svm": "#2a78d6", "cnn": "#eb6834", "clap": "#1baf7a"}
SYSTEM_MARKER = {"svm": "o", "cnn": "s", "clap": "^"}
SYSTEM_SHORT = {"svm": "SVM", "cnn": "CNN", "clap": "CLAP"}
CONDITION_ORDER = ("white", "demand", "semantic_same", "semantic_different")
CONDITION_LABEL = {
    "white": "White noise (SNR)",
    "demand": "DEMAND background (SNR)",
    "semantic_same": "Same-category event (SIR)",
    "semantic_different": "Different-category event (SIR)",
}
LEVELS = (10.0, 5.0, 0.0)
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
HATCH = "///"


def _matplotlib():
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Figures need matplotlib; install the 'figures' extra (uv sync --extra figures)."
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 7,
            "hatch.linewidth": 0.8,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )
    return plt


class EvaluationData:
    """Predictions, scores and clean references for every system in one evaluation."""

    def __init__(self, eval_dir, clean_dirs, additional_evaluations=(), svm_models=None):
        self.eval_dir = Path(eval_dir)
        self.summary = json.loads((self.eval_dir / "summary.json").read_text())
        self.input_dirs = {s: self.eval_dir for s in self.summary["systems"]}
        for directory in additional_evaluations:
            directory = Path(directory)
            extra = json.loads((directory / "summary.json").read_text())
            if not extra.get("manifest_sha256") or extra["manifest_sha256"] != self.summary.get(
                "manifest_sha256"
            ):
                raise ValueError("Additional evaluations must use the identical manifest")
            if set(extra["systems"]) & set(self.input_dirs):
                raise ValueError("A system occurs in more than one evaluation")
            self.input_dirs.update({s: directory for s in extra["systems"]})
            self.summary["systems"].update(extra["systems"])
        self.systems = [s for s in SYSTEM_ORDER if s in self.summary["systems"]]
        if not self.systems:
            raise ValueError("The evaluation summary lists no known systems")
        self.rows, self.clean, self.labels, self.score_type, self.arrays = {}, {}, {}, {}, {}
        for system in self.systems:
            if system not in clean_dirs:
                raise ValueError(f"No clean baseline directory given for {system}")
            self.rows[system] = read_predictions(
                self.input_dirs[system] / f"{system}_predictions.csv"
            )
            self.clean[system] = read_clean_predictions(
                Path(clean_dirs[system]) / "predictions.csv"
            )
            with np.load(self.input_dirs[system] / f"{system}_scores.npz") as archive:
                self.labels[system] = archive["class_ids"].tolist()
                self.score_type[system] = str(archive["score_type"])
                self.arrays[system] = {
                    "row": archive["row"],
                    "representation": archive["representation"],
                    "control_representation": archive["control_representation"],
                    "representation_type": str(archive["representation_type"]),
                }
            rows = self.rows[system]
            arrays = self.arrays[system]
            ids = [int(r["row"]) for r in rows]
            if len(set(ids)) != len(ids) or not np.array_equal(arrays["row"], ids):
                raise ValueError("Prediction and representation row IDs must match exactly")
            if arrays["representation"].shape != arrays["control_representation"].shape or len(
                arrays["representation"]
            ) != len(rows):
                raise ValueError("Representation and control shapes do not match prediction rows")
            if not all(
                np.isfinite(arrays[k]).all() for k in ("representation", "control_representation")
            ):
                raise ValueError("Representations must be finite")
            for row in rows:
                if self.clean[system].get(row["target_file"], ())[:2] != (
                    row["fold"],
                    row["target_class"],
                ):
                    raise ValueError("Clean reference fold or target differs from evaluation")
        signatures = [
            [
                (
                    r["row"],
                    r["target_file"],
                    r["target_class"],
                    r["fold"],
                    r["condition"],
                    r["level_db"],
                    r["interferer"],
                    r["interferer_class"],
                    r["common_gain"],
                    r["activity_overlap"],
                )
                for r in self.rows[s]
            ]
            for s in self.systems
        ]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError("Systems must contain the same ordered trials")
        self.scaler_provenance = {}
        if "svm" in self.systems:
            directory = Path(svm_models or self.summary["systems"]["svm"]["model_dir"])
            scales = {}
            for fold in sorted({r["fold"] for r in self.rows["svm"]}):
                path = directory / f"svm_fold_{fold}.joblib"
                expected = (
                    self.summary["systems"]["svm"].get("checkpoint_sha256", {}).get(str(fold))
                )
                checksum = sha256_file(path)
                if expected and expected != checksum:
                    raise ValueError("SVM scaler checkpoint differs from the evaluated model")
                bundle = joblib.load(path)
                if bundle["test_fold"] != fold:
                    raise ValueError("SVM scaler belongs to a different held-out fold")
                scales[fold] = np.asarray(bundle["model"].named_steps["scale"].scale_)
                self.scaler_provenance[str(fold)] = {"path": str(path), "sha256": checksum}
            self.arrays["svm"]["training_scales"] = scales
            self.arrays["svm"]["row_folds"] = np.array([r["fold"] for r in self.rows["svm"]])
        self.levels = tuple(
            sorted({r["level_db"] for r in self.rows[self.systems[0]]}, reverse=True)
        )

    def subset(self, system, condition=None, level=None):
        return [
            r
            for r in self.rows[system]
            if (condition is None or r["condition"] == condition)
            and (level is None or r["level_db"] == level)
        ]

    def metrics(self, system, rows):
        return fold_metrics(
            [r["target_class"] for r in rows],
            [r["prediction"] for r in rows],
            [r["fold"] for r in rows],
            self.labels[system],
        )

    def clean_metrics(self, system, targets):
        clean = [self.clean[system][t] for t in targets]
        return fold_metrics(
            [c[1] for c in clean], [c[2] for c in clean], [c[0] for c in clean], self.labels[system]
        )

    def conditions(self):
        present = {r["condition"] for r in self.rows[self.systems[0]]}
        return [c for c in CONDITION_ORDER if c in present]


def _write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _layout(fig):
    axes = [axis for axis in fig.axes if axis.get_visible()]
    if len(axes) > 1:
        for index, axis in enumerate(axes):
            title = axis.get_title(loc="left")
            axis.set_title(
                f"({chr(97 + index)}) {title}", loc="left", fontsize=10, color=TEXT_PRIMARY
            )
    fig.tight_layout()


def _save(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.svg")
    fig.savefig(output_dir / f"{stem}.pdf")
    from PIL import Image, ImageOps

    with Image.open(output_dir / f"{stem}.png") as preview:
        ImageOps.grayscale(preview).save(output_dir / f"{stem}_grayscale.png", dpi=(600, 600))
    import matplotlib.pyplot as plt

    plt.close(fig)


def _db(level):
    return "0 dB" if float(level) == 0 else f"{float(level):+.0f} dB"


def _level_ticks(levels):
    return ["clean"] + [_db(level) for level in levels]


# ------------------------------------------------------------------ figure 1


def figure_macro_f1(data, output_dir):
    """Macro-F1 versus interference level, one panel per interference type."""
    plt = _matplotlib()
    conditions = data.conditions()
    ncols = 2 if len(conditions) > 1 else 1
    nrows = int(np.ceil(len(conditions) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.16 if ncols == 2 else 3.5, 2.9 * nrows), sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    records = []
    for axis, condition in zip(axes, conditions):
        end_labels = []
        for system in data.systems:
            targets = sorted({r["target_file"] for r in data.subset(system, condition)})
            clean = data.clean_metrics(system, targets)
            means, sds = [clean["macro_f1"]["mean"]], [clean["macro_f1"]["sample_std"] or 0]
            records.append(
                {
                    "condition": condition,
                    "system": system,
                    "level_db": "clean",
                    "n": len(targets),
                    "macro_f1_mean": means[0],
                    "macro_f1_sd": sds[0],
                }
            )
            for level in data.levels:
                rows = data.subset(system, condition, level)
                metric = data.metrics(system, rows)
                means.append(metric["macro_f1"]["mean"])
                sds.append(metric["macro_f1"]["sample_std"] or 0)
                records.append(
                    {
                        "condition": condition,
                        "system": system,
                        "level_db": level,
                        "n": len(rows),
                        "macro_f1_mean": means[-1],
                        "macro_f1_sd": sds[-1],
                    }
                )
            x = np.arange(len(means))
            end_labels.append((system, means[-1]))
            axis.errorbar(
                x[:1],
                means[:1],
                yerr=sds[:1],
                linestyle="none",
                color=SYSTEM_COLOUR[system],
                marker=SYSTEM_MARKER[system],
                markeredgecolor="white",
                capsize=2,
            )
            axis.errorbar(
                x[1:],
                means[1:],
                yerr=sds[1:],
                color=SYSTEM_COLOUR[system],
                marker=SYSTEM_MARKER[system],
                capsize=2,
                label=SYSTEM_LABEL[system],
                markeredgecolor="white",
                markeredgewidth=1,
            )
        # Direct labels at the right end, nudged apart when two systems nearly coincide.
        ordered = sorted(end_labels, key=lambda item: item[1])
        offsets = {}
        previous = None
        for system, value in ordered:
            shift = 0.0
            if previous is not None and value - previous < 0.05:
                shift = previous + 0.05 - value
            offsets[system] = value + shift
            previous = value + shift
        for system, value in end_labels:
            axis.text(
                len(data.levels) + 0.08,
                offsets[system],
                f"{100 * value:.0f}%",
                va="center",
                fontsize=8,
                color=TEXT_PRIMARY,
            )
        axis.set_title(CONDITION_LABEL.get(condition, condition), loc="left", color=TEXT_PRIMARY)
        axis.set_xticks(np.arange(len(data.levels) + 1))
        axis.set_xticklabels(_level_ticks(data.levels))
        axis.set_ylim(0, 1)
        axis.set_xlim(-0.3, len(data.levels) + 0.7)
        axis.axvline(0.5, color=GRID, linewidth=1, linestyle=":")
    for axis in axes[len(conditions) :]:
        axis.set_visible(False)
    for axis in axes[::ncols]:
        axis.set_ylabel("Macro-F1 (5-fold mean ± SD)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Figure 1. Macro-F1 under interference\nClean is a separate control; 0 dB is a stress test",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=10,
    )
    _layout(fig)
    _save(fig, output_dir, "fig1_macro_f1_vs_level")
    _write_csv(
        output_dir / "fig1_macro_f1_vs_level.csv",
        ["condition", "system", "level_db", "n", "macro_f1_mean", "macro_f1_sd"],
        records,
    )
    return records


# ------------------------------------------------------------------ figure 2


def figure_capture_rate(data, output_dir):
    """Distractor capture rate versus SIR: solid = different category, dashed = same."""
    plt = _matplotlib()
    semantic = [c for c in data.conditions() if c.startswith("semantic")]
    if not semantic:
        return []
    fig, axis = plt.subplots(figsize=(7.16, 3.6))
    records = []
    n_classes = len(data.labels[data.systems[0]])
    for system in data.systems:
        for condition in semantic:
            rates, counts = [], []
            for level in data.levels:
                rows = data.subset(system, condition, level)
                errors = [r for r in rows if not r["correct"]]
                captured = [r for r in errors if r["prediction"] == r["interferer_class"]]
                rate = len(captured) / len(errors) if errors else np.nan
                rates.append(rate)
                counts.append((len(rows), len(errors), len(captured)))
                records.append(
                    {
                        "system": system,
                        "condition": condition,
                        "level_db": level,
                        "n": len(rows),
                        "errors": len(errors),
                        "errors_equal_distractor": len(captured),
                        "capture_rate": rate,
                        "capture_fraction": len(captured) / len(rows) if rows else np.nan,
                    }
                )
            style = "-" if condition == "semantic_different" else "--"
            axis.plot(
                range(len(data.levels)),
                rates,
                style,
                color=SYSTEM_COLOUR[system],
                marker=SYSTEM_MARKER[system],
                markeredgecolor="white",
                markeredgewidth=1,
                label=f"{SYSTEM_LABEL[system]}, {'different' if condition == 'semantic_different' else 'same'} category",
            )
    axis.axhline(1 / (n_classes - 1), color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    axis.annotate(
        f"uniform wrong-class reference 1/{n_classes - 1}",
        (len(data.levels) - 1, 1 / (n_classes - 1)),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color=TEXT_SECONDARY,
    )
    axis.set_xticks(range(len(data.levels)))
    axis.set_xticklabels([_db(level) for level in data.levels])
    axis.set_xlabel("Signal-to-interference ratio")
    axis.set_ylabel("Errors predicted as the distractor")
    axis.set_ylim(0, min(1, max(0.5, np.nanmax([r["capture_rate"] for r in records]) * 1.15)))
    axis.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    axis.set_title(
        "Figure 2. Distractor capture rate\nPooled errors; 1/49 assumes uniform wrong labels, not an empirical null",
        loc="left",
        color=TEXT_PRIMARY,
        fontsize=10,
    )
    _layout(fig)
    _save(fig, output_dir, "fig2_capture_rate_vs_sir")
    _write_csv(
        output_dir / "fig2_capture_rate_vs_sir.csv",
        [
            "system",
            "condition",
            "level_db",
            "n",
            "errors",
            "errors_equal_distractor",
            "capture_rate",
            "capture_fraction",
        ],
        records,
    )
    return records


# ------------------------------------------------------------------ figure 3


def figure_margin(data, output_dir):
    """Target-minus-distractor score versus SIR, one panel per system (own score scale)."""
    plt = _matplotlib()
    semantic = [c for c in data.conditions() if c.startswith("semantic")]
    if not semantic:
        return []
    fig, axes = plt.subplots(1, len(data.systems), figsize=(7.16, 3.5))
    axes = np.atleast_1d(axes)
    records = []
    for axis, system in zip(axes, data.systems):
        positions, boxes, colours, hatches = [], [], [], []
        for i, level in enumerate(data.levels):
            for j, condition in enumerate(semantic):
                margins = [
                    r["margin"]
                    for r in data.subset(system, condition, level)
                    if r["margin"] is not None
                ]
                if not margins:
                    continue
                quartiles = np.percentile(margins, [25, 50, 75])
                records.append(
                    {
                        "system": system,
                        "score_type": data.score_type[system],
                        "condition": condition,
                        "level_db": level,
                        "n": len(margins),
                        "median": quartiles[1],
                        "q1": quartiles[0],
                        "q3": quartiles[2],
                        "fraction_negative": float(np.mean(np.array(margins) < 0)),
                    }
                )
                positions.append(i + (j - (len(semantic) - 1) / 2) * 0.32)
                boxes.append(margins)
                colours.append(SYSTEM_COLOUR[system])
                hatches.append(HATCH if condition == "semantic_same" else "")
        plot = axis.boxplot(
            boxes,
            positions=positions,
            widths=0.26,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": TEXT_PRIMARY, "linewidth": 1.2},
            whiskerprops={"color": TEXT_SECONDARY},
            capprops={"color": TEXT_SECONDARY},
        )
        for patch, colour, hatch in zip(plot["boxes"], colours, hatches):
            patch.set_facecolor(colour if not hatch else "white")
            patch.set_edgecolor(colour)
            patch.set_hatch(hatch)
            patch.set_linewidth(1.2)
        axis.axhline(0, color=TEXT_SECONDARY, linewidth=1)
        axis.set_xticks(range(len(data.levels)))
        axis.set_xticklabels([_db(level) for level in data.levels])
        axis.set_title(
            f"{SYSTEM_LABEL[system]}\n{data.score_type[system].replace('_', ' ')}",
            loc="left",
            color=TEXT_PRIMARY,
        )
    axes[0].set_ylabel("Target score − distractor score")
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=TEXT_SECONDARY, edgecolor=TEXT_SECONDARY, label="different category"),
        Patch(facecolor="white", edgecolor=TEXT_SECONDARY, hatch=HATCH, label="same category"),
    ]
    axes[-1].legend(
        handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2
    )
    fig.suptitle(
        "Figure 3. Target–distractor margin versus SIR\nBoxes: quartiles; whiskers: 1.5 IQR; outliers hidden; n in CSV; 0 dB stress test",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=10,
    )
    _layout(fig)
    _save(fig, output_dir, "fig3_margin_vs_sir")
    _write_csv(
        output_dir / "fig3_margin_vs_sir.csv",
        [
            "system",
            "score_type",
            "condition",
            "level_db",
            "n",
            "median",
            "q1",
            "q3",
            "fraction_negative",
        ],
        records,
    )
    return records


# ------------------------------------------------------------------ figure 4


def figure_same_vs_different(data, output_dir):
    """Paired-target relative F1 losses, with individual official folds visible."""
    plt = _matplotlib()
    semantic = [c for c in data.conditions() if c.startswith("semantic")]
    if len(semantic) < 2:
        return []
    fig, axes = plt.subplots(1, len(data.systems), figsize=(7.16, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    records, fold_records = [], []
    for axis, system in zip(axes, data.systems):
        for c_index, condition in enumerate(semantic):
            losses = []
            for i, level in enumerate(data.levels):
                target_sets = [
                    {r["target_file"] for r in data.subset(system, c, level)} for c in semantic
                ]
                targets = sorted(set.intersection(*target_sets))
                rows = [
                    r
                    for r in data.subset(system, condition, level)
                    if r["target_file"] in set(targets)
                ]
                clean = data.clean_metrics(system, targets)
                perturbed = data.metrics(system, rows)
                clean_f1, f1 = clean["macro_f1"]["mean"], perturbed["macro_f1"]["mean"]
                loss = (clean_f1 - f1) / clean_f1 if clean_f1 else np.nan
                losses.append(loss)
                records.append(
                    dict(
                        system=system,
                        condition=condition,
                        level_db=level,
                        n=len(rows),
                        clean_macro_f1=clean_f1,
                        macro_f1=f1,
                        relative_macro_f1_loss=loss,
                    )
                )
                values = []
                for fold, metric in perturbed["folds"].items():
                    reference = clean["folds"][fold]["macro_f1"]
                    value = 1 - metric["macro_f1"] / reference if reference else np.nan
                    values.append(value)
                    fold_records.append(
                        dict(
                            system=system,
                            condition=condition,
                            level_db=level,
                            fold=fold,
                            n=metric["n"],
                            relative_macro_f1_loss=value,
                        )
                    )
                offset = (c_index - 0.5) * 0.13
                axis.scatter(
                    i + offset + np.linspace(-0.035, 0.035, len(values)),
                    values,
                    s=13,
                    alpha=0.65,
                    color=SYSTEM_COLOUR[system],
                    marker=".",
                )
            axis.plot(
                np.arange(len(data.levels)) + (c_index - 0.5) * 0.13,
                losses,
                linestyle="--" if condition == "semantic_same" else "-",
                color=SYSTEM_COLOUR[system],
                marker=SYSTEM_MARKER[system],
                markerfacecolor="white" if condition == "semantic_same" else SYSTEM_COLOUR[system],
                label="same category" if condition == "semantic_same" else "different category",
            )
        axis.set_xticks(range(len(data.levels)), [_db(v) for v in data.levels])
        axis.set_title(SYSTEM_LABEL[system], loc="left")
        axis.set_xlabel("SIR (0 dB: stress test)")
        axis.set_ylim(
            min(0, min([r["relative_macro_f1_loss"] for r in fold_records] or [0]) - 0.03), 1
        )
    axes[0].set_ylabel("Relative macro-F1 loss vs clean")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Figure 4. Same- versus different-category interference\n"
        "Identical targets; markers: loss of fold-mean F1; small dots: individual folds",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    _layout(fig)
    _save(fig, output_dir, "fig4_same_vs_different")
    _write_csv(output_dir / "fig4_same_vs_different.csv", list(records[0]), records)
    _write_csv(output_dir / "fig4_fold_losses.csv", list(fold_records[0]), fold_records)
    return records


def displacement(system, arrays, rows):
    """Within-model displacement between each mixture and its gain-matched control.

    Handcrafted vectors are standardised per feature by the fold's training scaler before a
    normalised Euclidean distance; embeddings use cosine distance. Values are only
    comparable within a system.
    """
    mixture = np.asarray(arrays["representation"], dtype=np.float64)
    control = np.asarray(arrays["control_representation"], dtype=np.float64)
    if arrays["representation_type"] == "handcrafted_features":
        if "training_scales" not in arrays:
            raise ValueError("Handcrafted displacement requires training-fold scales")
        scale = np.stack([arrays["training_scales"][int(f)] for f in arrays["row_folds"]])
        if scale.shape != mixture.shape or np.any(scale <= 0) or not np.isfinite(scale).all():
            raise ValueError("Training feature scales must be positive and match the vectors")
        values = np.linalg.norm((mixture - control) / scale, axis=1) / np.sqrt(mixture.shape[1])
        kind = "normalised Euclidean distance of training-standardised features"
    else:
        norms = np.linalg.norm(mixture, axis=1) * np.linalg.norm(control, axis=1)
        if np.any(norms <= 0):
            raise ValueError("Cosine displacement requires nonzero embedding norms")
        cosine = np.sum(mixture * control, axis=1) / norms
        values = 1 - np.clip(cosine, -1, 1)
        kind = "cosine distance of embeddings"
    by_row = dict(zip(arrays["row"].tolist(), values.tolist()))
    return [by_row[int(r["row"])] for r in rows], kind


def figure_displacement(data, output_dir, level):
    """Representation displacement by condition and outcome at one level, per system."""
    plt = _matplotlib()
    conditions = data.conditions()
    if len(data.systems) > 2:
        fig, axes = plt.subplots(
            len(data.systems), 1, figsize=(7.16, 2.5 * len(data.systems) + 0.7)
        )
    else:
        fig, axes = plt.subplots(1, len(data.systems), figsize=(7.16, 3.8))
    axes = np.atleast_1d(axes)
    records = []
    for axis, system in zip(axes, data.systems):
        rows = data.subset(system, level=level)
        values, kind = displacement(system, data.arrays[system], rows)
        boxes, positions, colours, hatches = [], [], [], []
        for i, condition in enumerate(conditions):
            for j, correct in enumerate((1, 0)):
                selected = [
                    v
                    for v, r in zip(values, rows)
                    if r["condition"] == condition and r["correct"] == correct
                ]
                if not selected:
                    continue
                quartiles = np.percentile(selected, [25, 50, 75])
                records.append(
                    {
                        "system": system,
                        "displacement": kind,
                        "level_db": level,
                        "condition": condition,
                        "outcome": "correct" if correct else "incorrect",
                        "n": len(selected),
                        "median": quartiles[1],
                        "q1": quartiles[0],
                        "q3": quartiles[2],
                    }
                )
                boxes.append(selected)
                positions.append(i + (j - 0.5) * 0.34)
                colours.append(SYSTEM_COLOUR[system])
                hatches.append("" if correct else HATCH)
        plot = axis.boxplot(
            boxes,
            positions=positions,
            widths=0.28,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": TEXT_PRIMARY, "linewidth": 1.2},
            whiskerprops={"color": TEXT_SECONDARY},
            capprops={"color": TEXT_SECONDARY},
        )
        for patch, colour, hatch in zip(plot["boxes"], colours, hatches):
            patch.set_facecolor(colour if not hatch else "white")
            patch.set_edgecolor(colour)
            patch.set_hatch(hatch)
            patch.set_linewidth(1.2)
        axis.set_xticks(range(len(conditions)))
        axis.set_xticklabels(
            [
                {
                    "white": "white",
                    "demand": "DEMAND",
                    "semantic_same": "same\ncategory",
                    "semantic_different": "different\ncategory",
                }[c]
                for c in conditions
            ]
        )
        axis.set_title(f"{SYSTEM_LABEL[system]}", loc="left", color=TEXT_PRIMARY)
        axis.set_ylabel(
            "Training-standardised RMS shift" if system == "svm" else "Cosine distance", fontsize=8
        )
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=TEXT_SECONDARY, edgecolor=TEXT_SECONDARY, label="classified correctly"),
        Patch(facecolor="white", edgecolor=TEXT_SECONDARY, hatch=HATCH, label="misclassified"),
    ]
    axes[-1].legend(
        handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2
    )
    fig.suptitle(
        f"Figure 5. Representation displacement at {_db(level)} versus gain-matched controls\n"
        "Separate model scales; boxes: quartiles; whiskers: 1.5 IQR; outliers hidden; n in CSV",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=10,
    )
    _layout(fig)
    _save(fig, output_dir, "fig5_representation_displacement")
    _write_csv(
        output_dir / "fig5_representation_displacement.csv",
        ["system", "displacement", "level_db", "condition", "outcome", "n", "median", "q1", "q3"],
        records,
    )
    return records


# ------------------------------------------------------------------ figure 6


def figure_confusion_flow(data, output_dir, level, top=10):
    """Where captured predictions go: most frequent target → distractor flows at one level."""
    plt = _matplotlib()
    semantic = [c for c in data.conditions() if c.startswith("semantic")]
    if not semantic:
        return []
    fig, axes = plt.subplots(len(data.systems), 1, figsize=(7.16, 2.65 * len(data.systems) + 0.5))
    axes = np.atleast_1d(axes)
    records = []
    names = {}
    for system in data.systems:
        for r in data.rows[system]:
            names[r["target_class"]] = r["target_category"]
    for axis, system in zip(axes, data.systems):
        counter, totals, control_correct, new_captures = Counter(), Counter(), Counter(), Counter()
        for condition in semantic:
            for r in data.subset(system, condition, level):
                totals[(r["target_class"], r["interferer_class"])] += 1
                pair = (r["target_class"], r["interferer_class"])
                if r["control_prediction"] == r["target_class"]:
                    control_correct[pair] += 1
                    if r["prediction"] == r["interferer_class"]:
                        new_captures[pair] += 1
                if not r["correct"] and r["prediction"] == r["interferer_class"]:
                    counter[(r["target_class"], r["interferer_class"])] += 1
        flows = counter.most_common(top)
        labels, counts = [], []
        for (target, distractor), count in flows:
            labels.append(
                f"{str(names.get(target, target)).replace('_', ' ')} → "
                f"{str(names.get(distractor, distractor)).replace('_', ' ')}"
            )
            counts.append(count)
            records.append(
                {
                    "system": system,
                    "level_db": level,
                    "target_class": target,
                    "target_category": names.get(target, ""),
                    "distractor_class": distractor,
                    "distractor_category": names.get(distractor, ""),
                    "captured": count,
                    "pairs": totals[(target, distractor)],
                    "control_correct_pairs": control_correct[(target, distractor)],
                    "new_captures_from_correct_control": new_captures[(target, distractor)],
                }
            )
        y = np.arange(len(labels))[::-1]
        axis.barh(y, counts, color=SYSTEM_COLOUR[system], height=0.7)
        for yi, count, (pair, _) in zip(y, counts, flows):
            axis.annotate(
                f"{count}/{totals[pair]}",
                (count, yi),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=TEXT_PRIMARY,
            )
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_xlim(0, max(counts, default=1) * 1.18)
        axis.set_xlabel("Captured mixtures (count / pairs)")
        from matplotlib.ticker import MaxNLocator

        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis="x")
        axis.grid(False, axis="y")
        axis.set_title(SYSTEM_LABEL[system], loc="left", color=TEXT_PRIMARY)
    fig.suptitle(
        f"Figure 6. Most frequent target → distractor captures at {_db(level)} SIR\n"
        "Exploratory, selected by count; labels: captures / pairs; control transitions in CSV",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=10,
    )
    _layout(fig)
    _save(fig, output_dir, "fig6_confusion_flow")
    _write_csv(
        output_dir / "fig6_confusion_flow.csv",
        [
            "system",
            "level_db",
            "target_class",
            "target_category",
            "distractor_class",
            "distractor_category",
            "captured",
            "pairs",
            "control_correct_pairs",
            "new_captures_from_correct_control",
        ],
        records,
    )
    return records


# ------------------------------------------------------------------ figure 7


def figure_robustness_vs_cost(data, output_dir, costs, level, condition="semantic_different"):
    """Matched-target F1 versus CPU latency; distinguish checkpoint and parameter bytes."""
    plt = _matplotlib()
    fig, axis = plt.subplots(figsize=(7.16, 3.8))
    records, platforms = [], set()
    for system in data.systems:
        if system not in costs:
            continue
        cost = json.loads(Path(costs[system]).read_text())
        if cost.get("device", "cpu") != "cpu" or cost.get("batch_size", 1) != 1:
            raise ValueError("Cost comparison requires CPU and batch size one")
        if cost.get("numerical_threads", 1) != 1:
            raise ValueError("Cost comparison requires one numerical CPU thread")
        if cost.get("platform"):
            platforms.add(cost["platform"])
        milliseconds = cost["preprocessing_median_ms"] + cost["inference_median_ms"]
        if not np.isfinite(milliseconds) or milliseconds <= 0:
            raise ValueError("CPU latency must be positive and finite")
        rows = data.subset(system, condition, level)
        if not rows:
            continue
        targets = sorted({r["target_file"] for r in rows})
        clean = data.clean_metrics(system, targets)["macro_f1"]["mean"]
        f1 = data.metrics(system, rows)["macro_f1"]["mean"]
        records.append(
            dict(
                system=system,
                condition=condition,
                level_db=level,
                cpu_ms_per_clip=milliseconds,
                preprocessing_ms=cost["preprocessing_median_ms"],
                inference_ms=cost["inference_median_ms"],
                serialized_model_bytes=cost.get("model_bytes"),
                checkpoint_bytes=cost.get("checkpoint_bytes"),
                parameter_count=cost.get("parameter_count"),
                parameter_bytes=cost.get("parameter_bytes"),
                clean_macro_f1=clean,
                macro_f1=f1,
                platform=cost.get("platform"),
                runs=cost.get("runs"),
                warmup=cost.get("warmup"),
            )
        )
        colour, marker = SYSTEM_COLOUR[system], SYSTEM_MARKER[system]
        axis.plot([milliseconds] * 2, [clean, f1], color=colour, linewidth=1)
        axis.scatter(
            [milliseconds],
            [clean],
            facecolor="white",
            edgecolor=colour,
            marker=marker,
            s=60,
            zorder=3,
        )
        axis.scatter([milliseconds], [f1], color=colour, marker=marker, s=60, zorder=3)
        axis.annotate(
            SYSTEM_LABEL[system],
            (milliseconds, f1),
            xytext=(6, -4),
            textcoords="offset points",
            fontsize=8,
        )
    if len(platforms) > 1:
        raise ValueError("Cost files report different platforms")
    if not records:
        plt.close(fig)
        return []
    axis.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    upper = max(r["cpu_ms_per_clip"] for r in records)
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if t <= max(20, upper * 2)]
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_xlim(min(ticks) * 0.8, max(max(ticks) * 1.1, upper * 2.2))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Macro-F1")
    axis.set_xlabel("Single-clip CPU time: preprocessing + inference (ms, log scale)")
    axis.set_title(
        f"Figure 7. Robustness versus CPU cost at {_db(level)} SIR\n"
        "Different-category event; hollow: matched clean; filled: mixture",
        loc="left",
        fontsize=10,
    )
    _layout(fig)
    _save(fig, output_dir, "fig7_robustness_vs_cost")
    _write_csv(output_dir / "fig7_robustness_vs_cost.csv", list(records[0]), records)
    return records


def figure_overlap_strata(data, output_dir, level, edges=(0.0, 0.5, 0.8, 1.01)):
    """Accuracy on exactly paired targets; retain F1 and per-fold class coverage in CSV."""
    plt = _matplotlib()
    semantic = [c for c in data.conditions() if c.startswith("semantic")]
    if "demand" not in data.conditions() or not semantic:
        return []
    fig, axes = plt.subplots(
        len(data.systems),
        len(semantic),
        figsize=(7.16, 2.45 * len(data.systems) + 0.7),
        sharey=True,
        squeeze=False,
    )
    records, fold_records, pair_records, shift_records = [], [], [], []
    strata = list(zip(edges[:-1], edges[1:]))
    for si, system in enumerate(data.systems):
        demand = {r["target_file"]: r for r in data.subset(system, "demand", level)}
        all_rows = data.subset(system, level=level)
        distances, kind = displacement(system, data.arrays[system], all_rows)
        distance_by_row = {int(r["row"]): v for r, v in zip(all_rows, distances)}
        for ci, condition in enumerate(semantic):
            axis = axes[si, ci]
            sem_values, dem_values, labels = [], [], []
            for i, (lo, hi) in enumerate(strata):
                sem_rows = [
                    r
                    for r in data.subset(system, condition, level)
                    if lo <= float(r["activity_overlap"]) < hi and r["target_file"] in demand
                ]
                dem_rows = [demand[r["target_file"]] for r in sem_rows]
                stratum = f"[{lo}, {min(hi, 1.0)}{')' if hi <= 1 else ']'}"
                if not sem_rows:
                    sem_values.append(np.nan)
                    dem_values.append(np.nan)
                    labels.append(f"{lo:g}–{min(hi, 1):g}\nn=0")
                    continue
                sm, dm = data.metrics(system, sem_rows), data.metrics(system, dem_rows)
                sa, da = sm["pooled"]["accuracy"], dm["pooled"]["accuracy"]
                sem_values.append(sa)
                dem_values.append(da)
                labels.append(f"{lo:g}–{min(hi, 1):g}\nn={len(sem_rows)}")
                records.append(
                    dict(
                        system=system,
                        level_db=level,
                        semantic_condition=condition,
                        overlap_stratum=stratum,
                        n=len(sem_rows),
                        classes_present=len({r["target_class"] for r in sem_rows}),
                        semantic_macro_f1=sm["macro_f1"]["mean"],
                        demand_macro_f1_same_targets=dm["macro_f1"]["mean"],
                        semantic_accuracy=sa,
                        demand_accuracy_same_targets=da,
                    )
                )
                for fold, fm in sm["folds"].items():
                    df = dm["folds"][fold]
                    selected = [r for r in sem_rows if r["fold"] == fold]
                    fold_records.append(
                        dict(
                            system=system,
                            semantic_condition=condition,
                            overlap_stratum=stratum,
                            fold=fold,
                            n=fm["n"],
                            classes_present=len({r["target_class"] for r in selected}),
                            semantic_accuracy=fm["accuracy"],
                            demand_accuracy=df["accuracy"],
                            semantic_macro_f1=fm["macro_f1"],
                            demand_macro_f1=df["macro_f1"],
                        )
                    )
                    jitter = (fold - 3) * 0.018
                    axis.plot(
                        [i - 0.13 + jitter, i + 0.13 + jitter],
                        [fm["accuracy"], df["accuracy"]],
                        color=SYSTEM_COLOUR[system],
                        alpha=0.3,
                        linewidth=0.65,
                        marker=".",
                        markersize=3,
                    )
                sd = np.array([distance_by_row[int(r["row"])] for r in sem_rows])
                dd = np.array([distance_by_row[int(r["row"])] for r in dem_rows])
                shift_records.append(
                    dict(
                        system=system,
                        semantic_condition=condition,
                        overlap_stratum=stratum,
                        level_db=level,
                        n=len(sem_rows),
                        displacement=kind,
                        semantic_distance_median=float(np.median(sd)),
                        demand_distance_median=float(np.median(dd)),
                        paired_distance_difference_median=float(np.median(sd - dd)),
                        fraction_semantic_distance_smaller=float(np.mean(sd < dd)),
                        semantic_accuracy=sa,
                        demand_accuracy_same_targets=da,
                    )
                )
                for a, b in zip(sem_rows, dem_rows):
                    pair_records.append(
                        dict(
                            system=system,
                            semantic_condition=condition,
                            overlap_stratum=stratum,
                            fold=a["fold"],
                            target_file=a["target_file"],
                            target_class=a["target_class"],
                            semantic_row=a["row"],
                            demand_row=b["row"],
                            semantic_correct=a["correct"],
                            demand_correct=b["correct"],
                            semantic_distance=distance_by_row[int(a["row"])],
                            demand_distance=distance_by_row[int(b["row"])],
                        )
                    )
            x = np.arange(len(strata))
            colour = SYSTEM_COLOUR[system]
            axis.scatter(
                x - 0.13,
                sem_values,
                color=colour,
                marker=SYSTEM_MARKER[system],
                label="semantic event",
                s=42,
                zorder=4,
            )
            axis.scatter(
                x + 0.13,
                dem_values,
                facecolor="white",
                edgecolor=colour,
                marker=SYSTEM_MARKER[system],
                label="matched DEMAND",
                s=42,
                zorder=4,
            )
            for i, (a, b) in enumerate(zip(sem_values, dem_values)):
                axis.plot([i - 0.13, i + 0.13], [a, b], color=colour, linewidth=1.5)
            axis.set_xticks(x, labels)
            axis.set_ylim(0, 1)
            axis.set_xlim(-0.45, len(strata) - 0.55)
            axis.set_title(
                f"{SYSTEM_SHORT[system]} · {'same' if condition == 'semantic_same' else 'different'} category",
                loc="left",
            )
            if ci == 0:
                axis.set_ylabel("Accuracy on paired targets")
            if si == len(data.systems) - 1:
                axis.set_xlabel("Activity overlap stratum")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Figure 8. Semantic interference versus matched DEMAND at {_db(level)}\n"
        "Large markers: pooled accuracy; thin pairs: individual folds; class coverage in CSV",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    _layout(fig)
    _save(fig, output_dir, "fig8_overlap_strata")
    for filename, values in [
        ("fig8_overlap_strata", records),
        ("fig8_overlap_fold_metrics", fold_records),
        ("fig8_matched_trials", pair_records),
        ("representation_overlap_comparison", shift_records),
    ]:
        if values:
            _write_csv(output_dir / f"{filename}.csv", list(values[0]), values)
    return records


def make_figures(
    eval_dir,
    clean_dirs,
    output_dir,
    costs=None,
    level=5.0,
    additional_evaluations=(),
    svm_models=None,
):
    """Build every figure the evaluation supports; return a manifest of outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = EvaluationData(eval_dir, clean_dirs, additional_evaluations, svm_models)
    if level not in data.levels:
        raise ValueError(f"Level {level} is not in the evaluation: {data.levels}")
    produced = {
        "fig1_macro_f1_vs_level": len(figure_macro_f1(data, output_dir)),
        "fig2_capture_rate_vs_sir": len(figure_capture_rate(data, output_dir)),
        "fig3_margin_vs_sir": len(figure_margin(data, output_dir)),
        "fig4_same_vs_different": len(figure_same_vs_different(data, output_dir)),
        "fig5_representation_displacement": len(figure_displacement(data, output_dir, level)),
        "fig6_confusion_flow": len(figure_confusion_flow(data, output_dir, level)),
        "fig8_overlap_strata": len(figure_overlap_strata(data, output_dir, level)),
    }
    if costs:
        produced["fig7_robustness_vs_cost"] = len(
            figure_robustness_vs_cost(data, output_dir, costs, level)
        )
    meta = {
        "evaluation": str(Path(eval_dir)),
        "manifest_sha256": data.summary.get("manifest_sha256"),
        "systems": data.systems,
        "score_types": data.score_type,
        "input_evaluations": {k: str(v) for k, v in data.input_dirs.items()},
        "svm_training_scalers": data.scaler_provenance,
        "focus_level_db": level,
        "clean_dirs": {k: str(v) for k, v in clean_dirs.items()},
        "costs": {k: str(v) for k, v in (costs or {}).items()},
        "records_per_figure": produced,
        "palette": {
            "systems": SYSTEM_COLOUR,
            "note": "fixed per-system colours; condition contrast by fill vs hatch",
        },
        "source_sha256": {"figures.py": sha256_file(Path(__file__))},
        "input_sha256": {
            s: {
                name: sha256_file(directory / name)
                for name in ("summary.json", f"{s}_predictions.csv", f"{s}_scores.npz")
            }
            for s, directory in data.input_dirs.items()
        },
        "clean_sha256": {
            s: sha256_file(Path(clean_dirs[s]) / "predictions.csv") for s in data.systems
        },
        "caveats": [
            "0 dB is a stress test with ambiguous single-label ground truth",
            "Fold SD is descriptive, not a confidence interval",
            "Flows are exploratory examples selected by capture count",
            "Representation distances are comparable only within each system",
        ],
    }
    write_json(output_dir / "figures_meta.json", meta)
    return meta

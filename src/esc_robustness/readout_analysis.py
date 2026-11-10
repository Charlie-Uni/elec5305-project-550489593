"""Ranked readout of saved scores on semantic mixtures.

The single-label argmax forces one answer per mixture, so a mixture whose target and
distractor both score highly is counted as an error whenever the distractor edges
ahead. This module re-reads the same saved scores with a top-k rule and sorts every
semantic trial into four outcomes:

- ``target_only``: target in the top k, distractor not;
- ``both``: target and distractor both in the top k (two designated labels retrieved);
- ``distractor_only``: distractor in the top k, target not (the capture that survives
  a multi-label readout);
- ``neither``: neither in the top k.

The gain-matched clean control is read out the same way as a reference. Outcomes are
fractions of all semantic trials, not of errors, so they can be compared with the
error-conditioned capture rate without changing its definition. Ranks follow the
ordinal convention: descending score, ties broken by ascending class ID.
This diagnostic is not a calibrated multi-label detector or a test of perception.
"""

import csv
import json
from pathlib import Path

import numpy as np

from .analysis import read_predictions
from .artifacts import sha256_file, write_json

OUTCOMES = ("target_only", "both", "distractor_only", "neither")
TRIAL_FIELDS = (
    "region",
    "row",
    "target_file",
    "fold",
    "condition",
    "level_db",
    "target_class",
    "distractor_class",
    "correct",
    "argmax_captured",
    "target_rank",
    "distractor_rank",
    "control_target_rank",
    "control_distractor_rank",
)


def outcome(target_rank, distractor_rank, k):
    """Vectorised four-way outcome for ranks (NaN rank = unknown → None)."""
    target_rank = np.asarray(target_rank, dtype=float)
    distractor_rank = np.asarray(distractor_rank, dtype=float)
    labels = np.full(target_rank.shape, None, dtype=object)
    known = np.isfinite(target_rank) & np.isfinite(distractor_rank)
    t_in, d_in = target_rank <= k, distractor_rank <= k
    labels[known & t_in & ~d_in] = "target_only"
    labels[known & t_in & d_in] = "both"
    labels[known & ~t_in & d_in] = "distractor_only"
    labels[known & ~t_in & ~d_in] = "neither"
    return labels


def ordinal_ranks(scores):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite matrix")
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.argsort(order, axis=1, kind="stable") + 1


def trials_from_scores(region, eval_dir, system="clap"):
    """Exact ranks for mixtures and controls from clap_scores.npz + clap_predictions.csv."""
    eval_dir = Path(eval_dir)
    predictions = read_predictions(eval_dir / f"{system}_predictions.csv")
    for p in predictions:
        p["row"] = int(p["row"])
    with np.load(eval_dir / f"{system}_scores.npz") as archive:
        scores = archive["scores"]
        control = archive["control_scores"]
        class_ids = archive["class_ids"].tolist()
        rows = archive["row"].tolist()
    if len(set(rows)) != len(rows) or len({p["row"] for p in predictions}) != len(predictions):
        raise ValueError("Duplicate score or prediction row")
    if set(rows) != {p["row"] for p in predictions}:
        raise ValueError("Scores and predictions have different row IDs")
    if (
        class_ids != sorted(set(class_ids))
        or scores.shape != control.shape
        or scores.shape != (len(rows), len(class_ids))
    ):
        raise ValueError("Scores/classes have inconsistent shapes or ordering")
    ordinal, control_ordinal = ordinal_ranks(scores), ordinal_ranks(control)
    position = {r: i for i, r in enumerate(rows)}
    if system != "svm":
        for p in predictions:
            i = position[p["row"]]
            if (
                class_ids[int(np.argmin(ordinal[i]))] != p["prediction"]
                or class_ids[int(np.argmin(control_ordinal[i]))] != p["control_prediction"]
            ):
                raise ValueError("Stored prediction differs from score argmax")
    index = {int(c): i for i, c in enumerate(class_ids)}
    semantic = [p for p in predictions if p["interferer_class"] is not None]
    idx = np.array([position[int(p["row"])] for p in semantic])
    targets = np.array([index[p["target_class"]] for p in semantic])
    distractors = np.array([index[p["interferer_class"]] for p in semantic])
    trials = []
    tr = ordinal[idx, targets]
    dr = ordinal[idx, distractors]
    ctr = control_ordinal[idx, targets]
    cdr = control_ordinal[idx, distractors]
    for p, a, b, c, d in zip(semantic, tr, dr, ctr, cdr):
        trials.append(
            {
                "region": region,
                "row": int(p["row"]),
                "target_file": p["target_file"],
                "fold": p["fold"],
                "condition": p["condition"],
                "level_db": p["level_db"],
                "target_class": p["target_class"],
                "distractor_class": p["interferer_class"],
                "correct": p["correct"],
                "argmax_captured": int(p["prediction"] == p["interferer_class"]),
                "target_rank": int(a),
                "distractor_rank": int(b),
                "control_target_rank": int(c),
                "control_distractor_rank": int(d),
            }
        )
    return trials


def trials_from_competition_csv(csv_path, predictions):
    """Ranks from clap_competition_trials.csv; control target rank is exact only when the
    control was correct (rank 1), otherwise unknown (NaN)."""
    lookup = {}
    for region, path in predictions.items():
        for p in read_predictions(path):
            lookup[(region, int(p["row"]))] = p
    trials = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            key = (raw["region"], int(raw["row"]))
            p = lookup.get(key)
            if p is None:
                raise ValueError(f"No prediction row for {key}")
            control_rank = 1.0 if p["control_prediction"] == p["target_class"] else float("nan")
            trials.append(
                {
                    "region": raw["region"],
                    "row": int(raw["row"]),
                    "target_file": p["target_file"],
                    "fold": int(raw["fold"]),
                    "condition": raw["condition"],
                    "level_db": float(raw["level_db"]),
                    "target_class": int(raw["target_class"]),
                    "distractor_class": int(raw["distractor_class"]),
                    "correct": p["correct"],
                    "argmax_captured": int(p["prediction"] == p["interferer_class"]),
                    "target_rank": int(raw["target_rank"]),
                    "distractor_rank": int(raw["distractor_rank"]),
                    "control_target_rank": control_rank,
                    "control_distractor_rank": int(raw["control_distractor_rank"]),
                }
            )
    return trials


def load_trials(scores=None, competition_csv=None, predictions=None, system="clap"):
    trials = []
    for region, eval_dir in (scores or {}).items():
        trials.extend(trials_from_scores(region, eval_dir, system))
    if competition_csv is not None:
        if not predictions:
            raise ValueError("--competition-csv needs region=predictions pairs")
        trials.extend(trials_from_competition_csv(competition_csv, predictions))
    if not trials:
        raise ValueError("No trials loaded")
    if len({(t["region"], t["row"]) for t in trials}) != len(trials):
        raise ValueError("Duplicate readout trials")
    return trials


def summarise(trials, k=2):
    """Outcome fractions per region × condition × level, for mixtures and controls."""
    if type(k) is not int or not 1 <= k <= 50:
        raise ValueError("k must be an integer from 1 to 50")
    table = []
    keys = sorted(
        {(t["region"], t["condition"], t["level_db"]) for t in trials},
        key=lambda x: (x[0], x[1], -x[2]),
    )
    for region, condition, level in keys:
        selected = [
            t
            for t in trials
            if (t["region"], t["condition"], t["level_db"]) == (region, condition, level)
        ]
        n = len(selected)
        mixture = outcome(
            [t["target_rank"] for t in selected], [t["distractor_rank"] for t in selected], k
        )
        control = outcome(
            [t["control_target_rank"] for t in selected],
            [t["control_distractor_rank"] for t in selected],
            k,
        )
        known_control = int(np.sum([c is not None for c in control]))
        captured = [t for t in selected if t["argmax_captured"]]
        entry = {
            "region": region,
            "condition": condition,
            "level_db": level,
            "k": k,
            "n": n,
            "argmax_accuracy": float(np.mean([t["correct"] for t in selected])),
            "argmax_capture_fraction": len(captured) / n,
            "argmax_captured_target_rank_le_k": float(
                np.mean([t["target_rank"] <= k for t in captured])
            )
            if captured
            else None,
            "argmax_captured_target_rank_median": float(
                np.median([t["target_rank"] for t in captured])
            )
            if captured
            else None,
            "target_in_topk": float(np.mean([t["target_rank"] <= k for t in selected])),
            "distractor_in_topk": float(np.mean([t["distractor_rank"] <= k for t in selected])),
            "control_known": known_control,
            "argmax_captured_count": len(captured),
            "captured_target_in_topk_count": sum(t["target_rank"] <= k for t in captured),
        }
        for name in OUTCOMES:
            entry[f"mixture_{name}_count"] = int(np.sum(mixture == name))
            entry[f"control_{name}_count"] = int(np.sum(control == name))
            entry[f"mixture_{name}"] = float(np.mean(mixture == name))
            entry[f"control_{name}"] = (
                float(np.sum(control == name) / known_control) if known_control else None
            )
        entry["control_distractor_in_topk"] = float(
            np.mean([t["control_distractor_rank"] <= k for t in selected])
        )
        table.append(entry)
    return table


def analyse_readout(trials, output_dir, k=2):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = summarise(trials, k)
    with (output_dir / "readout_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(trials)
    with (output_dir / "readout_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    exact_controls = all(np.isfinite(t["control_target_rank"]) for t in trials)
    summary = {
        "k": k,
        "trials": len(trials),
        "regions": sorted({t["region"] for t in trials}),
        "control_ranks_exact": exact_controls,
        "rank_convention": "ordinal ranks for exact scores; score ties use ascending class IDs; legacy competition CSV uses competition ranks and is explicitly approximate",
        "outcome_definition": {
            "target_only": f"target rank <= {k}, distractor rank > {k}",
            "both": f"both ranks <= {k}",
            "distractor_only": f"distractor rank <= {k}, target rank > {k}",
            "neither": f"both ranks > {k}",
        },
        "table": table,
        "source_sha256": {"readout_analysis.py": sha256_file(Path(__file__))},
    }
    write_json(output_dir / "readout_summary.json", summary)
    summary["figure"] = figure_readout(table, output_dir, k)
    write_json(output_dir / "readout_summary.json", summary)
    return summary


def figure_readout(table, output_dir, k):
    """Stacked outcome fractions versus SIR; one panel per region × pairing."""
    from .figures import SYSTEM_COLOUR, TEXT_PRIMARY, TEXT_SECONDARY, _db, _matplotlib

    plt = _matplotlib()
    regions = sorted({r["region"] for r in table})
    conditions = [
        c
        for c in ("semantic_same", "semantic_different")
        if any(r["condition"] == c for r in table)
    ]
    fig, axes = plt.subplots(
        len(regions),
        len(conditions),
        figsize=(3.6 * len(conditions), 2.8 * len(regions)),
        squeeze=False,
        sharey=True,
    )
    base = SYSTEM_COLOUR["clap"]
    shades = {
        "target_only": "#bfe6d6",
        "both": base,
        "distractor_only": "#0b6b48",
        "neither": "#dddcd6",
    }
    labels = {
        "target_only": "target only",
        "both": "both in top-k",
        "distractor_only": "distractor only",
        "neither": "neither",
    }
    for i, region in enumerate(regions):
        for j, condition in enumerate(conditions):
            axis = axes[i][j]
            rows = sorted(
                [r for r in table if r["region"] == region and r["condition"] == condition],
                key=lambda r: -r["level_db"],
            )
            x = np.arange(len(rows))
            bottom = np.zeros(len(rows))
            for name in OUTCOMES:
                values = np.array([r[f"mixture_{name}"] for r in rows])
                axis.bar(
                    x,
                    values,
                    0.6,
                    bottom=bottom,
                    color=shades[name],
                    hatch={
                        "target_only": "",
                        "both": "//",
                        "distractor_only": "xx",
                        "neither": "..",
                    }[name],
                    edgecolor="white",
                    linewidth=0.8,
                    label=labels[name] if (i == 0 and j == 0) else None,
                )
                bottom += values
            # Clean-control reference for "both" and "distractor present".
            control_d = [r["control_distractor_in_topk"] for r in rows]
            axis.plot(
                x,
                control_d,
                "o--",
                color=TEXT_SECONDARY,
                markersize=4,
                linewidth=1,
                label="clean control: distractor in top-k" if (i == 0 and j == 0) else None,
            )
            for xi, r in zip(x, rows):
                axis.annotate(
                    f"{100 * r['mixture_both']:.0f}%",
                    (xi, r["mixture_target_only"] + r["mixture_both"] / 2),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )
            axis.set_xticks(x)
            axis.set_xticklabels([_db(r["level_db"]) for r in rows])
            axis.set_ylim(0, 1)
            axis.set_title(
                f"{region} power · {'same' if condition == 'semantic_same' else 'different'} category",
                loc="left",
                fontsize=9,
                color=TEXT_PRIMARY,
            )
            if j == 0:
                axis.set_ylabel("Fraction of semantic trials")
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Top-{k} rank inclusion of the two designated labels",
        x=0.01,
        ha="left",
        fontsize=9,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    stem = f"readout_top{k}_outcomes"
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return stem


def read_summary(path):
    return json.loads(Path(path).read_text())

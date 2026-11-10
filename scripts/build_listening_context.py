"""Write the context shown beside the website's listening examples.

The examples are illustrations, so the page states how often each outcome occurs in
the full experiment and what each selected target is meant to show. Every statement
written here is checked against the committed results before the file is produced.
"""

import argparse
import csv
import json
from pathlib import Path

from esc_robustness.artifacts import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("svm", "cnn", "clap")
SEMANTIC = ("semantic_same", "semantic_different")
LEVELS = (10.0, 5.0, 0.0)

# What each published target illustrates. `checks` are verified below.
TARGETS = {
    "1-101404-A-34.wav": {
        "tag": "a competing event takes first place",
        "summary": (
            "Opening a can while an alarm clock rings. Each model names the can when it is "
            "alone. At +10 and +5 dB white noise breaks the SVM and CNN but not CLAP, while "
            "the competing event takes first place from both the CNN and CLAP at every level."
        ),
        "checks": ("controls_correct", "white_breaks_svm_cnn_not_clap", "cnn_clap_always_captured"),
    },
    "5-210612-A-37.wav": {
        "tag": "all three models hold",
        "summary": (
            "An alarm clock rings while a can is opened, using different source recordings "
            "from the can-opening example. All three "
            "models keep the alarm clock under both competing events at every level. "
            "A competing event does not always win."
        ),
        "checks": ("controls_correct", "all_semantic_correct"),
    },
    "4-161100-A-47.wav": {
        "tag": "the target slips to second place",
        "summary": (
            "An aircraft passes while church bells ring. At +10 dB CLAP still answers "
            "airplane. At +5 dB the bells take first place and airplane drops to second, "
            "while the SVM and CNN keep airplane."
        ),
        "checks": ("controls_correct", "clap_slips_to_second"),
    },
    "1-100032-A-0.wav": {
        "tag": "two models are already wrong on the clean clip",
        "summary": (
            "A dog bark. The SVM and CNN mislabel this clip even without interference, so "
            "their mixture predictions say little about interference here. CLAP is right "
            "on the clean clip and is captured by the competing event at every level."
        ),
        "checks": ("svm_cnn_control_wrong", "clap_always_captured"),
    },
}


def read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def build(example_dirs, conditions_csv, readout_csv, output):
    examples = {}
    for directory in example_dirs:
        for row in json.loads((directory / "examples.json").read_text())["files"]:
            examples.setdefault(row["target_file"], {})[
                (row["condition"], float(row["level_db"]))
            ] = row
    if set(examples) != set(TARGETS):
        raise ValueError(
            f"Targets without a description, or the reverse: {set(examples) ^ set(TARGETS)}"
        )

    ranks = {
        int(r["row"]): (int(r["target_rank"]), int(r["distractor_rank"]))
        for r in read(readout_csv)
        if r["variant"] == "clap_baseline" and r["region"] == "own"
    }

    def right(row, system):
        return int(row["predictions"][system]["correct"]) == 1

    def control_right(row, system):
        return int(row["predictions"][system]["control_prediction"]) == int(row["target_class"])

    def captured(row, system):
        return int(row["predictions"][system]["prediction"]) == int(row["interferer_class"])

    def holds(name, rows):
        semantic = [rows[(c, level)] for c in SEMANTIC for level in LEVELS]
        if name == "controls_correct":
            return all(control_right(row, s) for row in rows.values() for s in SYSTEMS)
        if name == "svm_cnn_control_wrong":
            return all(
                not control_right(row, "svm")
                and not control_right(row, "cnn")
                and control_right(row, "clap")
                for row in rows.values()
            )
        if name == "white_breaks_svm_cnn_not_clap":
            return all(
                not right(rows[("white", level)], "svm")
                and not right(rows[("white", level)], "cnn")
                and right(rows[("white", level)], "clap")
                for level in (10.0, 5.0)
            )
        if name == "cnn_clap_always_captured":
            return all(captured(r, "cnn") and captured(r, "clap") for r in semantic)
        if name == "clap_always_captured":
            return all(captured(r, "clap") for r in semantic)
        if name == "all_semantic_correct":
            return all(right(r, s) for r in semantic for s in SYSTEMS)
        if name == "clap_slips_to_second":
            same = [rows[("semantic_same", level)] for level in LEVELS]
            return (
                right(same[0], "clap")
                and ranks[same[1]["row"]] == (2, 1)
                and right(same[1], "svm")
                and right(same[1], "cnn")
            )
        raise KeyError(name)

    for target, spec in TARGETS.items():
        for check in spec["checks"]:
            if not holds(check, examples[target]):
                raise ValueError(
                    f"Description of {target} is not supported by the results: {check}"
                )

    # How often each outcome occurs over all 2,000 targets (own-activity power).
    accuracy = {
        (r["system"], r["condition"]): (float(r["accuracy_mean"]), int(r["n"]))
        for r in read(conditions_csv)
        if r["region"] == "own"
    }
    captures = {}
    for r in read(readout_csv):
        if r["region"] == "own" and r["variant"].endswith("_baseline"):
            key = (r["variant"].split("_")[0], r["condition"], float(r["level_db"]))
            count, total = captures.get(key, (0, 0))
            captures[key] = (count + int(r["argmax_captured"]), total + 1)
    base_rates = {}
    for condition in ("white", "demand", *SEMANTIC):
        for level in LEVELS:
            cell = {}
            for system in SYSTEMS:
                value, n = accuracy[(system, f"{condition}@+{level:.0f}dB")]
                cell[system] = {"accuracy": value, "n": n}
                if condition in SEMANTIC:
                    count, total = captures[(system, condition, level)]
                    if total != n:
                        raise ValueError("Capture and accuracy tables cover different trials")
                    cell[system].update(captures=count, capture_fraction=count / total)
            base_rates[f"{condition}@{level:.0f}"] = cell

    clap_ranks = {
        str(row["row"]): dict(zip(("target_rank", "distractor_rank"), ranks[row["row"]]))
        for rows in examples.values()
        for (condition, _), row in rows.items()
        if condition in SEMANTIC
    }
    write_json(
        output,
        {
            "note": "Illustrative targets from the final own-activity manifest; rates use all 2,000 targets.",
            "sources": {
                str(Path(p).relative_to(ROOT)): sha256_file(p)
                for p in (conditions_csv, readout_csv, *[d / "examples.json" for d in example_dirs])
            },
            "targets": {t: {"tag": s["tag"], "summary": s["summary"]} for t, s in TARGETS.items()},
            "base_rates": base_rates,
            "clap_ranks": clap_ranks,
        },
    )
    print(
        json.dumps(
            {"targets": len(TARGETS), "cells": len(base_rates), "ranked_rows": len(clap_ranks)}
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--examples", type=Path, action="append", required=True)
    p.add_argument(
        "--conditions",
        type=Path,
        default=ROOT / "results/clap_full_20261109/all_conditions_verified.csv",
    )
    p.add_argument(
        "--readout", type=Path, default=ROOT / "results/closing_20261110/readout_trials.csv"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build([d.resolve() for d in a.examples], a.conditions.resolve(), a.readout.resolve(), a.output)

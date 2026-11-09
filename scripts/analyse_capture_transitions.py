"""Distinguish new distractor capture from errors already present in controls."""

import argparse
import csv
import json
from pathlib import Path

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.diagnostics import write_csv
from esc_robustness.extension_summary import variant_paths


def analyse(repo, runtime, run, output, core_only=False):
    output.mkdir(parents=True, exist_ok=True)
    records, sources = [], {}
    included = {"svm_clean", "cnn_clean_seed_5305", "clap_baseline", "clap_prompt_ensemble"}
    for variant, system, _, _, _, directories in variant_paths(repo, runtime, run):
        if core_only and variant not in included:
            continue
        for region, directory in directories.items():
            path = directory / f"{system}_predictions.csv"
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != (24000 if region == "own" else 23991):
                raise ValueError("Incomplete interference evaluation")
            sources[f"{region}/{variant}"] = sha256_file(path)
            for condition in ("semantic_same", "semantic_different"):
                for level in (10.0, 5.0, 0.0):
                    for stratum in ("all", "overlap_ge_0.8"):
                        subset = [
                            r
                            for r in rows
                            if r["condition"] == condition
                            and float(r["level_db"]) == level
                            and (stratum == "all" or float(r["activity_overlap"]) >= 0.8)
                        ]
                        if not subset or len({r["target_file"] for r in subset}) != len(subset):
                            raise ValueError("Empty or duplicate semantic subset")
                        n, error, capture, other = len(subset), 0, 0, 0
                        control_capture, new_capture, persistent_capture, lost_capture = 0, 0, 0, 0
                        control_correct, correct_to_capture = 0, 0
                        for r in subset:
                            target, distractor, p, control = [
                                int(r[k])
                                for k in (
                                    "target_class",
                                    "interferer_class",
                                    "prediction",
                                    "control_prediction",
                                )
                            ]
                            if target == distractor:
                                raise ValueError("Semantic target and distractor must differ")
                            error += p != target
                            capture += p == distractor
                            other += p not in (target, distractor)
                            control_capture += control == distractor
                            new_capture += p == distractor and control != distractor
                            persistent_capture += p == distractor and control == distractor
                            lost_capture += p != distractor and control == distractor
                            control_correct += control == target
                            correct_to_capture += control == target and p == distractor
                        if (
                            capture + other != error
                            or new_capture + persistent_capture != capture
                            or lost_capture + persistent_capture != control_capture
                        ):
                            raise ValueError("Capture transition accounting failed")
                        records.append(
                            dict(
                                variant=variant,
                                system=system,
                                region=region,
                                condition=condition,
                                level_db=level,
                                stratum=stratum,
                                n=n,
                                errors=error,
                                distractor_captures=capture,
                                other_wrong_labels=other,
                                capture_rate=capture / error if error else None,
                                capture_fraction=capture / n,
                                control_distractor_captures=control_capture,
                                new_captures=new_capture,
                                persistent_captures=persistent_capture,
                                lost_captures=lost_capture,
                                control_correct=control_correct,
                                correct_control_to_distractor=correct_to_capture,
                                correct_control_to_distractor_rate=correct_to_capture
                                / control_correct
                                if control_correct
                                else None,
                            )
                        )
    write_csv(output / "capture_transitions.csv", records)
    write_json(
        output / "audit.json",
        dict(
            verified=True,
            variants=len(sources) // 2,
            rows=len(records),
            scope="Core references and fixed prompt ensemble"
            if core_only
            else "All declared intervention variants",
            interpretation="Descriptive paired outcomes; conditioning on a correct control changes the population across classifiers",
            source_hashes=sources,
            table_sha256=sha256_file(output / "capture_transitions.csv"),
        ),
    )
    print(json.dumps(dict(variants=len(sources) // 2, cells=len(records)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "runtime", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    analyse(args.repo, args.runtime, args.run, args.output, args.core_only)

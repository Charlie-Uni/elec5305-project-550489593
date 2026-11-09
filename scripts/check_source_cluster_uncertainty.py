"""Post-hoc sensitivity of matched accuracy CIs to target-source clustering.

Resample original target source IDs globally, keeping every retained clip from a
sampled source together, including the rare source spanning official folds.
This conditions on fitted models and does not capture all shared-interferer or
training-set dependence. It is a resampling-unit sensitivity, not a new test used
to select models or replace the declared multiplicity families.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.diagnostics import write_csv
from esc_robustness.extension_summary import variant_paths


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def cluster_difference(rows, reference, sources, replicates=10000, seed=530510):
    rows = sorted(rows, key=lambda r: r["target_file"])
    originals = sorted({sources[r["target_file"]] for r in rows})
    index = {s: i for i, s in enumerate(originals)}
    clusters = np.array([index[sources[r["target_file"]]] for r in rows])
    y = np.array([int(r["target_class"]) for r in rows])
    a = np.array([int(r["prediction"]) for r in rows])
    b = np.array([int(reference[r["target_file"]]["prediction"]) for r in rows])
    difference = (a == y).astype(int) - (b == y).astype(int)
    n = len(originals)
    counts = np.bincount(clusters, minlength=n)
    sums = np.bincount(clusters, weights=difference, minlength=n)
    rng = np.random.default_rng(seed)
    sampled = np.empty(replicates)
    for start in range(0, replicates, 250):
        stop = min(start + 250, replicates)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=stop - start)
        denominator = weights @ counts
        if np.any(denominator == 0):
            raise ValueError("Empty source bootstrap sample")
        sampled[start:stop] = (weights @ sums) / denominator
    return dict(
        n_targets=len(rows),
        target_source_clusters=n,
        maximum_clips_per_source=int(counts.max()),
        accuracy_difference=float(difference.mean()),
        source_ci_low=float(np.quantile(sampled, 0.025)),
        source_ci_high=float(np.quantile(sampled, 0.975)),
        replicates=replicates,
        bootstrap_seed=seed,
    )


def run(repo, runtime, root, dataset, output):
    output.mkdir(parents=True, exist_ok=True)
    sources = {r["filename"]: r["src_file"] for r in read(dataset / "meta/esc50.csv")}
    selected = {
        "svm_clean",
        "cnn_clean_seed_5305",
        "cnn_clean_seed_5306",
        "cnn_clean_seed_5307",
        "clap_baseline",
        "clap_prompt_ensemble",
    }
    records, provenance = [], {}
    for variant, system, _, _, _, directories in variant_paths(repo, runtime, root):
        if variant not in selected:
            continue
        for region, directory in directories.items():
            file = directory / f"{system}_predictions.csv"
            rows = read(file)
            provenance[f"{region}/{variant}"] = sha256_file(file)
            demand = {
                r["target_file"]: r
                for r in rows
                if r["condition"] == "demand" and float(r["level_db"]) == 5
            }
            for condition in ("semantic_same", "semantic_different"):
                semantic = [
                    r
                    for r in rows
                    if r["condition"] == condition
                    and float(r["level_db"]) == 5
                    and float(r["activity_overlap"]) >= 0.8
                ]
                if len({r["target_file"] for r in semantic}) != len(semantic) or any(
                    r["target_class"] != demand[r["target_file"]]["target_class"] for r in semantic
                ):
                    raise ValueError("Invalid paired target set")
                values = cluster_difference(semantic, demand, sources)
                records.append(
                    dict(
                        variant=variant,
                        system=system,
                        region=region,
                        condition=condition,
                        level_db=5.0,
                        stratum="overlap_ge_0.8",
                        **values,
                    )
                )
    write_csv(output / "source_cluster_intervals.csv", records)
    write_json(
        output / "audit.json",
        dict(
            complete=True,
            scope="Exploratory target-source cluster sensitivity for clean-trained references and fixed prompt ensemble",
            contrasts=len(records),
            sampling="10,000 global multinomial resamples of original target source IDs; all clips of each source kept together, even across folds",
            estimand="Pooled retained-target accuracy difference: semantic minus matched DEMAND",
            limitations=[
                "Conditional on fitted models",
                "Target-source clustering does not capture all shared distractor/background dependence",
                "No new hypothesis-test family or model selection",
            ],
            source_hashes=provenance,
            metadata_sha256=sha256_file(dataset / "meta/esc50.csv"),
        ),
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "runtime", "run", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.repo, args.runtime, args.run, args.dataset, args.output)

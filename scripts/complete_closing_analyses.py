"""Close ranked-readout and acoustic matching analyses using existing evidence only."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from esc_robustness.analysis import read_predictions as _read_predictions
from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.diagnostics import write_csv
from esc_robustness.overlap_analysis import read_features
from esc_robustness.readout_analysis import summarise, trials_from_scores

SEED = 530510
REPLICATES = 2000
CALIPERS = {
    "close_primary": (0.10, 0.05, 0.10),
    "close_strict": (0.05, 0.025, 0.05),
    "close_loose": (0.20, 0.10, 0.20),
}
ACOUSTICS = ("target_activity_spectral_cosine", "masked_cell_fraction", "activity_overlap")


def read_predictions(path):
    rows = _read_predictions(path)
    for r in rows:
        r["row"] = int(r["row"])
    return rows


def bootstrap_means(values, sources, replicates=REPLICATES):
    """Source-cluster percentile draws; return means with random cluster sizes."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(sources) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid clustered values")
    unique, inverse = np.unique(sources, return_inverse=True)
    sums = np.zeros((len(unique), values.shape[1]))
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(SEED)
    result = np.empty((replicates, values.shape[1]))
    for start in range(0, replicates, 100):
        end = min(start + 100, replicates)
        weights = rng.multinomial(
            len(unique), np.full(len(unique), 1 / len(unique)), size=end - start
        )
        result[start:end] = (weights @ sums) / (weights @ counts)[:, None]
    return result


def interval(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite bootstrap draws")
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def readout(directories, sources, out):
    all_trials, all_summary, uncertainty, provenance = [], [], [], {}
    for variant, system, dirs in directories:
        trials = []
        for region, directory in dirs.items():
            part = trials_from_scores(region, directory, system)
            expected = 12000 if region == "own" else 11991
            if len(part) != expected:
                raise ValueError("Incomplete semantic readout")
            trials.extend(part)
            with np.load(directory / f"{system}_scores.npz") as archive:
                scores, controls = archive["scores"], archive["control_scores"]
                order = np.argsort(-scores, axis=1, kind="stable")
                co = np.argsort(-controls, axis=1, kind="stable")
                ties = {
                    str(k): int(
                        np.sum(
                            np.take_along_axis(scores, order[:, k - 1 : k], 1)
                            == np.take_along_axis(scores, order[:, k : k + 1], 1)
                        )
                    )
                    for k in (1, 2, 3, 5)
                }
                control_ties = {
                    str(k): int(
                        np.sum(
                            np.take_along_axis(controls, co[:, k - 1 : k], 1)
                            == np.take_along_axis(controls, co[:, k : k + 1], 1)
                        )
                    )
                    for k in (1, 2, 3, 5)
                }
                prediction_rows = read_predictions(directory / f"{system}_predictions.csv")
                pred_map = {r["row"]: r for r in prediction_rows}
                original = np.array([pred_map[int(row)]["prediction"] for row in archive["row"]])
                disagreements = int(np.sum(original != archive["class_ids"][order[:, 0]]))
            provenance[f"{variant}/{region}"] = {
                "predictions_sha256": sha256_file(directory / f"{system}_predictions.csv"),
                "scores_sha256": sha256_file(directory / f"{system}_scores.npz"),
                "boundary_ties_mixture": ties,
                "boundary_ties_control": control_ties,
                "score_argmax_vs_original_prediction_disagreements": disagreements,
            }
        for t in trials:
            t["source_id"] = sources[t["target_file"]]
        all_trials.extend({"variant": variant, "system": system, **t} for t in trials)
        for k in (1, 2, 3, 5):
            all_summary.extend(
                {"variant": variant, "system": system, **r} for r in summarise(trials, k)
            )
        keys = sorted({(t["region"], t["condition"], t["level_db"]) for t in trials})
        for region, condition, level in keys:
            selected = [
                t
                for t in trials
                if (t["region"], t["condition"], t["level_db"]) == (region, condition, level)
            ]
            src = [t["source_id"] for t in selected]
            tr, dr, ct, cd = [
                np.array([t[key] for t in selected])
                for key in (
                    "target_rank",
                    "distractor_rank",
                    "control_target_rank",
                    "control_distractor_rank",
                )
            ]
            captured = np.array([t["argmax_captured"] for t in selected], bool)
            values = []
            for k in (1, 2, 3, 5):
                ti, di, ci = tr <= k, dr <= k, ct <= k
                values.extend([ti, ti & di, (ti.astype(float) - ci), captured & ti])
            values.append(captured)
            v = np.column_stack(values)
            draws = bootstrap_means(v, src)
            for j, k in enumerate((1, 2, 3, 5)):
                point = v[:, j * 4 : j * 4 + 4].mean(0)
                point[3] /= captured.mean()
                b = draws[:, j * 4 : j * 4 + 4].copy()
                b[:, 3] /= draws[:, -1]
                for col, metric in enumerate(
                    (
                        "target_inclusion",
                        "joint_inclusion",
                        "target_inclusion_minus_control",
                        "target_inclusion_given_capture",
                    )
                ):
                    low, high = interval(b[:, col])
                    uncertainty.append(
                        dict(
                            variant=variant,
                            system=system,
                            region=region,
                            condition=condition,
                            level_db=level,
                            k=k,
                            metric=metric,
                            n=len(selected),
                            source_clusters=len(set(src)),
                            value=float(point[col]),
                            ci_low=low,
                            ci_high=high,
                        )
                    )
        print("Readout complete:", variant, flush=True)
    write_csv(out / "readout_trials.csv", all_trials)
    write_csv(out / "readout_outcomes.csv", all_summary)
    write_csv(out / "readout_uncertainty.csv", uncertainty)
    write_json(
        out / "readout_audit.json",
        dict(
            verified=True,
            trial_rows=len(all_trials),
            outcome_cells=len(all_summary),
            intervals=len(uncertainty),
            sources=provenance,
            replicates=REPLICATES,
            seed=SEED,
            interpretation="Exact rank inclusion, not calibrated detection or proof of source perception",
        ),
    )


def matching(features, predictions, sources, system, region):
    by_condition = {
        c: {r["target_file"]: r for r in predictions if r["condition"] == c and r["level_db"] == 5}
        for c in ("white", "demand", "semantic_same", "semantic_different")
    }
    summaries, trials = [], []
    for a, b in (
        ("semantic_same", "demand"),
        ("semantic_different", "demand"),
        ("semantic_same", "semantic_different"),
    ):
        names = sorted(set(by_condition[a]) & set(by_condition[b]))
        aa = [by_condition[a][t] for t in names]
        bb = [by_condition[b][t] for t in names]
        fa = [features[r["row"]] for r in aa]
        fb = [features[r["row"]] for r in bb]
        av = np.array([[r[k] for k in ACOUSTICS] for r in fa])
        bv = np.array([[r[k] for k in ACOUSTICS] for r in fb])
        delta = av - bv
        finite = np.isfinite(delta).all(1)
        strata = {
            "all": np.ones(len(names), bool),
            "both_overlap_ge_0.8": (av[:, 2] >= 0.8) & (bv[:, 2] >= 0.8),
        }
        for label, tolerances in CALIPERS.items():
            strata[label] = finite & (np.abs(delta) <= tolerances).all(1)
        ac = np.array([r["correct"] for r in aa], float)
        bc = np.array([r["correct"] for r in bb], float)
        acc = np.array([r["control_prediction"] == r["target_class"] for r in aa], float)
        bcc = np.array([r["control_prediction"] == r["target_class"] for r in bb], float)
        for i, t in enumerate(names):
            trials.append(
                dict(
                    region=region,
                    system=system,
                    comparison=a + "__" + b,
                    target_file=t,
                    source_id=sources[t],
                    target_class=aa[i]["target_class"],
                    a_row=aa[i]["row"],
                    b_row=bb[i]["row"],
                    a_correct=int(ac[i]),
                    b_correct=int(bc[i]),
                    a_control_correct=int(acc[i]),
                    b_control_correct=int(bcc[i]),
                    **{f"delta_{k}": float(delta[i, j]) for j, k in enumerate(ACOUSTICS)},
                    **{label: int(mask[i]) for label, mask in strata.items()},
                )
            )
        for label, mask in strata.items():
            n = int(mask.sum())
            selected = np.array(names)[mask]
            src = [sources[t] for t in selected]
            count = len(set(src))
            y = np.column_stack((ac, bc, acc - ac, bcc - bc, ac - bc, (acc - ac) - (bcc - bc)))[
                mask
            ]
            point = y.mean(0) if n else np.full(6, np.nan)
            low, high = [None, None]
            ci_loss = [None, None]
            if n >= 30 and count >= 30:
                draws = bootstrap_means(y[:, 4:], src)
                low, high = interval(draws[:, 0])
                ci_loss = interval(draws[:, 1])
            r = dict(
                region=region,
                system=system,
                comparison=a + "__" + b,
                a_condition=a,
                b_condition=b,
                stratum=label,
                level_db=5,
                n=n,
                total_pairs=len(names),
                retained_fraction=n / len(names),
                source_clusters=count,
                classes_present=len({r["target_class"] for r, m in zip(aa, mask) if m}),
                a_accuracy=float(point[0]) if n else None,
                b_accuracy=float(point[1]) if n else None,
                accuracy_difference=float(point[4]) if n else None,
                ci_low=low,
                ci_high=high,
                a_control_loss=float(point[2]) if n else None,
                b_control_loss=float(point[3]) if n else None,
                control_loss_difference=float(point[5]) if n else None,
                loss_ci_low=ci_loss[0],
                loss_ci_high=ci_loss[1],
            )
            for j, k in enumerate(ACOUSTICS):
                d = delta[mask, j]
                d = d[np.isfinite(d)]
                r[f"delta_{k}_median"] = float(np.median(d)) if len(d) else None
                r[f"abs_delta_{k}_max"] = float(np.abs(d).max()) if len(d) else None
            summaries.append(r)
    return summaries, trials


def acoustics(directories, sources, run, out):
    all_summary, all_trials, strata = [], [], []
    provenance = {}
    for region in ("own", "target"):
        fp = run / f"overlap_{region}/overlap_features.csv"
        features = read_features(fp)
        meta = json.loads(fp.with_name("overlap_features_meta.json").read_text())
        if len(features) != (24000 if region == "own" else 23991) or meta["demand_rows_skipped"]:
            raise ValueError("Incomplete acoustic features")
        for variant, system, dirs in directories:
            if variant == "clap_prompt_ensemble":
                continue
            directory = dirs[region]
            if (
                json.loads((directory / "summary.json").read_text())["manifest_sha256"]
                != meta["manifest_sha256"]
            ):
                raise ValueError("Different manifests")
            predictions = read_predictions(directory / f"{system}_predictions.csv")
            for r in predictions:
                f = features[r["row"]]
                if any(
                    str(r[k]) != str(f[k])
                    for k in ("condition", "target_file", "interferer", "level_db")
                ):
                    raise ValueError("Mismatched trial identities")
            summary, trials = matching(features, predictions, sources, system, region)
            all_summary.extend(summary)
            all_trials.extend(trials)
            for condition in ("white", "demand", "semantic_same", "semantic_different"):
                rows = [
                    r for r in predictions if r["condition"] == condition and r["level_db"] == 5
                ]
                for measure, edges in (
                    ("masked_cell_fraction", np.linspace(0, 1.000000001, 6)),
                    ("target_activity_spectral_cosine", np.linspace(0, 1.000000001, 6)),
                    ("activity_overlap", np.array([0, 0.5, 0.8, 1.000000001])),
                ):
                    for lo, hi in zip(edges[:-1], edges[1:]):
                        selected = [r for r in rows if lo <= features[r["row"]][measure] < hi]
                        n = len(selected)
                        if not n:
                            continue
                        src = [sources[r["target_file"]] for r in selected]
                        correct = np.array([r["correct"] for r in selected])
                        control = np.array(
                            [r["control_prediction"] == r["target_class"] for r in selected]
                        )
                        low = high = None
                        if n >= 30 and len(set(src)) >= 30:
                            low, high = interval(bootstrap_means(correct, src)[:, 0])
                        strata.append(
                            dict(
                                region=region,
                                system=system,
                                condition=condition,
                                measure=measure,
                                bin_low=float(lo),
                                bin_high=float(min(hi, 1)),
                                n=n,
                                source_clusters=len(set(src)),
                                classes_present=len({r["target_class"] for r in selected}),
                                accuracy=float(correct.mean()),
                                ci_low=low,
                                ci_high=high,
                                control_accuracy=float(control.mean()),
                                control_loss=float(np.mean(control - correct)),
                                capture_fraction=float(
                                    np.mean(
                                        [r["prediction"] == r["interferer_class"] for r in selected]
                                    )
                                )
                                if condition.startswith("semantic")
                                else None,
                            )
                        )
            provenance[f"{region}/{system}"] = sha256_file(directory / f"{system}_predictions.csv")
            print("Acoustic comparisons complete:", region, system, flush=True)
    write_csv(out / "acoustic_matched_comparisons.csv", all_summary)
    write_csv(out / "acoustic_matched_trials.csv", all_trials)
    write_csv(out / "acoustic_stratified.csv", strata)
    write_json(
        out / "acoustic_matching_audit.json",
        dict(
            verified=True,
            comparisons=len(all_summary),
            paired_trial_rows=len(all_trials),
            sources=provenance,
            replicates=REPLICATES,
            seed=SEED,
            calipers=CALIPERS,
            interpretation="Same-target selected-subset associations, not semantic causal effects",
        ),
    )


def main(args):
    repo = args.repo
    rt = args.runtime
    run = args.run
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    core = rt / "runs/clap_v4_20261109"
    dirs = []
    for system in ("svm", "cnn"):
        dirs.append(
            (
                system + "_baseline",
                system,
                {
                    region: repo / f"artifacts/interference/v3_20261109/eval_{region}"
                    for region in ("own", "target")
                },
            )
        )
    dirs.extend(
        [
            ("clap_baseline", "clap", {r: core / f"eval_{r}" for r in ("own", "target")}),
            (
                "clap_prompt_ensemble",
                "clap",
                {
                    r: rt / f"runs/extensions_20261109/clap_prompt_ensemble/eval_{r}"
                    for r in ("own", "target")
                },
            ),
        ]
    )
    with (rt / "data/ESC-50/meta/esc50.csv").open() as f:
        sources = {r["filename"]: r["src_file"] for r in csv.DictReader(f)}
    if args.part in ("readout", "all"):
        readout(dirs, sources, out)
    if args.part in ("acoustics", "all"):
        acoustics(dirs, sources, run, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("repo", "runtime", "run", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--part", choices=["readout", "acoustics", "all"], default="all")
    main(p.parse_args())

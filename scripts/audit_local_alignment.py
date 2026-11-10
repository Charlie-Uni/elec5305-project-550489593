"""Independently recompute alignment tables, references and primary paired intervals."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

READOUTS = ("single", "ensemble", "aligned_fixed", "aligned_calibrated")
TEMPERATURES = (16.0, 32.0, 64.0, 100.0)
BETAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def loaded(path):
    with np.load(path) as arrays:
        return {name: arrays[name] for name in arrays.files}


def metric(y, prediction, folds, classes=50):
    accuracy, f1 = [], []
    for fold in sorted(set(folds)):
        mask = folds == fold
        matrix = np.bincount(
            y[mask] * classes + prediction[mask], minlength=classes * classes
        ).reshape(classes, classes)
        denominator = matrix.sum(0) + matrix.sum(1)
        values = np.divide(
            2 * matrix.diagonal(), denominator,
            out=np.zeros(classes, dtype=float), where=denominator != 0,
        )
        accuracy.append(float(np.mean(y[mask] == prediction[mask])))
        f1.append(float(values.mean()))
    return {
        "accuracy_mean": float(np.mean(accuracy)),
        "accuracy_sd": float(np.std(accuracy, ddof=1)) if len(accuracy) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(f1)),
        "macro_f1_sd": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
    }


def unpack(packed):
    assert packed.shape[1] == 512 + 2 * 4 * 50
    blocks = packed[:, 512:].reshape(len(packed), 2, 4, 50).astype(float)
    return packed[:, :512].astype(float), blocks[:, 0], blocks[:, 1]


def fuse(base, text, audio, rule):
    t = TEMPERATURES.index(float(rule["temperature"]))
    return base + float(rule["beta_text"]) * text[:, t] + float(rule["beta_audio"]) * audio[:, t]


def rank(scores, y):
    order = np.argsort(-scores, axis=1, kind="stable")
    return 1 + np.argmax(order == y[:, None], axis=1)


def cluster_interval(reference, candidate, y, distractor, names, sources):
    keys = np.array([sources[name] for name in names])
    unique, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse)
    difference = (candidate == y).astype(int) - (reference == y).astype(int)
    capture = (candidate == distractor).astype(int) - (reference == distractor).astype(int)
    sums = np.bincount(inverse, weights=difference)
    capture_sums = np.bincount(inverse, weights=capture)
    weights = np.random.default_rng(5311).multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=2000
    )
    size = weights @ counts
    accuracy = 100 * (weights @ sums) / size
    capture_fraction = 100 * (weights @ capture_sums) / size
    return {
        "targets": len(y), "source_clusters": len(unique),
        "accuracy_difference_pp": float(100 * difference.mean()),
        "accuracy_ci_low_pp": float(np.quantile(accuracy, .025)),
        "accuracy_ci_high_pp": float(np.quantile(accuracy, .975)),
        "capture_fraction_difference_pp": float(100 * capture.mean()),
        "capture_fraction_ci_low_pp": float(np.quantile(capture_fraction, .025)),
        "capture_fraction_ci_high_pp": float(np.quantile(capture_fraction, .975)),
        "bootstrap_replicates": 2000, "seed": 5311,
        "scope": "pointwise target-source cluster interval; clean-calibration rules held fixed",
    }


def audit(run, single, ensemble, data, output):
    run, single, ensemble, data, output = map(Path, (run, single, ensemble, data, output))
    output.mkdir(parents=True, exist_ok=True)
    assert json.loads((run / "complete.json").read_text()) == {"complete": True, "limit": None}
    assert all(json.loads((run / name).read_text())["passed"] for name in
               ("preflight.json", "reference_check.json"))
    cache = json.loads((run / "encoding_cache.json").read_text())
    if cache["enabled"]:
        assert cache["unique_inputs"] + cache["cache_hits"] == 2000 + 2 * (24000 + 23991)
    texts = loaded(run / "text_embeddings.npz")
    assert tuple(texts["temperatures"]) == TEMPERATURES
    selection = json.loads((run / "selection.json").read_text())
    assert selection["candidates"] == 321 and set(selection["folds"]) == set("12345")
    metadata = rows(data / "meta/esc50.csv")
    sources = {row["filename"]: row["src_file"] for row in metadata}
    clean = loaded(run / "clean/decision_scores.npz")
    pooled, guided_text, guided_audio = unpack(clean["embeddings"])
    grid = [(0, 0.0, 0.0)] + [
        (t, a, b) for a in BETAS for b in BETAS for t in range(4) if a or b
    ]
    grid.sort(key=lambda v: (v[1] + v[2], v[2], v[0]))
    clean_sources = np.array([sources[name] for name in clean["filenames"]])
    for fold in range(1, 6):
        excluded = set(clean_sources[clean["folds"] == fold])
        mask = (clean["folds"] != fold) & ~np.isin(clean_sources, list(excluded))
        correct = np.array([
            np.sum((clean["scores"][mask] + a * guided_text[mask, t] +
                    b * guided_audio[mask, t]).argmax(1) == clean["targets"][mask])
            for t, a, b in grid
        ])
        t, a, b = grid[int(correct.argmax())]
        observed = selection["folds"][str(fold)]
        assert observed["choice"] == {"temperature": TEMPERATURES[t], "beta_text": a, "beta_audio": b}
        assert observed["calibration_clips"] == int(mask.sum())
    conditions = rows(run / "conditions.csv")
    assert len(conditions) == 100
    paired = json.loads((run / "summary.json").read_text())["paired_against_ensemble"]
    evidence, primary, comparisons = [], [], []
    maxima = {"single_reference_score": 0.0, "ensemble_reference_score": 0.0,
              "saved_fused_score": 0.0}

    def inspect(region, condition, level, base, packed, y, folds, names, ids,
                distractor=None, control_base=None, control_packed=None, saved=None):
        pooled, text, audio = unpack(packed)

        def readouts(base, pooled, text, audio):
            calibrated = np.empty_like(base)
            for fold in range(1, 6):
                mask = folds == fold
                calibrated[mask] = fuse(base[mask], text[mask], audio[mask],
                                        selection["folds"][str(fold)]["choice"])
            return {
                "single": pooled @ texts["single"].T, "ensemble": base,
                "aligned_fixed": fuse(base, text, audio, selection["fixed_rule"]),
                "aligned_calibrated": calibrated,
            }

        scores = readouts(base, pooled, text, audio)
        controls = None if control_base is None else readouts(
            control_base, *unpack(control_packed)
        )
        predictions = {key: value.argmax(1) for key, value in scores.items()}
        for readout, values in scores.items():
            assert np.isfinite(values).all()
            expected = next(row for row in conditions if row["region"] == region and
                            row["condition"] == condition and row["level_db"] == str(level) and
                            row["readout"] == readout)
            prediction = predictions[readout]
            for key, value in metric(y, prediction, folds).items():
                assert np.isclose(float(expected[key]), value, atol=1e-12), (region, condition, key)
            target_rank = rank(values, y)
            correct = prediction == y
            assert int(expected["correct"]) == int(correct.sum())
            assert int(expected["errors"]) == int((~correct).sum())
            assert int(expected["target_in_top2"]) == int((target_rank <= 2).sum())
            control_prediction = None if controls is None else controls[readout].argmax(1)
            if controls is not None:
                assert int(expected["control_correct"]) == int((control_prediction == y).sum())
            if distractor is not None:
                assert int(expected["captures"]) == int((prediction == distractor).sum())
                assert int(expected["other_errors"]) == int(((~correct) & (prediction != distractor)).sum())
            if saved is not None and readout.startswith("aligned"):
                for kind, actual in (("mixture", values), ("control", controls[readout])):
                    recorded = saved[f"{kind}_{readout}"]
                    difference = float(np.max(np.abs(recorded - actual)))
                    maxima["saved_fused_score"] = max(maxima["saved_fused_score"], difference)
                    assert difference < 1e-6
                    assert np.array_equal(recorded.argmax(1), actual.argmax(1))
            for i in range(len(y)):
                evidence.append({
                    "region": region, "condition": condition, "level_db": level,
                    "row": int(ids[i]), "filename": names[i], "fold": int(folds[i]),
                    "target": int(y[i]), "interferer": "" if distractor is None else int(distractor[i]),
                    "readout": readout, "prediction": int(prediction[i]),
                    "target_rank": int(target_rank[i]),
                    "control_prediction": "" if controls is None else int(control_prediction[i]),
                })
        if region != "clean":
            key = f"{region}/{condition}@{int(level):+d}"
            for readout in READOUTS[2:]:
                reference, prediction = predictions["ensemble"], predictions[readout]
                lost = int(np.sum((reference == y) & (prediction != y)))
                gained = int(np.sum((reference != y) & (prediction == y)))
                p = float(binomtest(lost, lost + gained).pvalue) if lost + gained else 1.0
                expected = paired[key][readout]
                assert (expected["lost"], expected["gained"]) == (lost, gained)
                assert np.isclose(expected["exact_mcnemar_p"], p)
                if region == "own" and level == 5 and distractor is not None:
                    primary.append({"condition": condition, "readout": readout, "lost": lost,
                                    "gained": gained, "exact_mcnemar_p": p,
                                    **cluster_interval(reference, prediction, y, distractor, names, sources)})
        return scores, predictions

    scores, predictions = inspect(
        "clean", "clean", "", clean["scores"], clean["embeddings"], clean["targets"],
        clean["folds"], clean["filenames"], np.arange(2000)
    )
    for name, directory in (("single", single), ("ensemble", ensemble / "clean")):
        reference = loaded(directory / "decision_scores.npz")
        assert np.array_equal(reference["filenames"], clean["filenames"])
        assert np.array_equal(reference["scores"].argmax(1), predictions[name])
        difference = float(np.max(np.abs(scores[name] - reference["scores"])))
        maxima[f"{name}_reference_score"] = difference
        assert difference <= 1e-4
    assert int((predictions["single"] == clean["targets"]).sum()) == 1635
    assert int((predictions["ensemble"] == clean["targets"]).sum()) == 1858
    for region, count in (("own", 24000), ("target", 23991)):
        directory = run / f"eval_{region}"
        records = rows(directory / "clap_local_predictions.csv")
        assert len(records) == count
        arrays = loaded(directory / "clap_local_scores.npz")
        fused = loaded(directory / "fused_scores.npz")
        assert np.array_equal(arrays["class_ids"], np.arange(50))
        ids = np.array([int(row["row"]) for row in records])
        assert np.array_equal(arrays["row"], ids) and np.array_equal(fused["row"], ids)
        references = {
            "single": rows(single.parent / f"eval_{region}/clap_predictions.csv"),
            "ensemble": rows(ensemble / f"eval_{region}/clap_predictions.csv"),
        }
        reference_arrays = {}
        for name, path in (
            ("single", single.parent / f"eval_{region}/clap_scores.npz"),
            ("ensemble", ensemble / f"eval_{region}/clap_scores.npz"),
        ):
            with np.load(path) as original:
                assert np.array_equal(original["row"], ids)
                assert np.array_equal(original["class_ids"], np.arange(50))
                reference_arrays[name] = {key: original[key] for key in ("scores", "control_scores")}
        control_single = unpack(arrays["control_representation"])[0] @ texts["single"].T
        for name, controls in (("single", control_single), ("ensemble", arrays["control_scores"])):
            expected = reference_arrays[name]["control_scores"]
            maximum = float(np.max(np.abs(controls - expected)))
            maxima[f"{name}_reference_score"] = max(maxima[f"{name}_reference_score"], maximum)
            assert maximum <= 1e-4
            assert np.array_equal(controls.argmax(1), expected.argmax(1))
        for reference in references.values():
            assert [int(row["row"]) for row in reference] == ids.tolist()
        condition_keys = [(row["condition"], int(float(row["level_db"]))) for row in records]
        for condition, level in sorted(set(condition_keys)):
            mask = np.array([key == (condition, level) for key in condition_keys])
            subset = [row for row, included in zip(records, mask) if included]
            y = np.array([int(row["target_class"]) for row in subset])
            folds = np.array([int(row["fold"]) for row in subset])
            names = np.array([row["target_file"] for row in subset])
            distractor = np.array([int(row["interferer_class"]) for row in subset]) if condition.startswith("semantic") else None
            values, predicted = inspect(
                region, condition, level, arrays["scores"][mask], arrays["representation"][mask],
                y, folds, names, ids[mask], distractor, arrays["control_scores"][mask],
                arrays["control_representation"][mask],
                {name: value[mask] for name, value in fused.items() if name.startswith(("mixture_", "control_"))},
            )
            for name, reference in references.items():
                expected = [row for row, included in zip(reference, mask) if included]
                assert np.array_equal(predicted[name], [int(row["prediction"]) for row in expected])
                difference = float(np.max(np.abs(values[name] - reference_arrays[name]["scores"][mask])))
                maxima[f"{name}_reference_score"] = max(maxima[f"{name}_reference_score"], difference)
                assert difference <= 1e-4
            comparisons.append({"region": region, "condition": condition, "level_db": level, "trials": len(y)})
    assert len(primary) == 4
    pvalues = np.array([row["exact_mcnemar_p"] for row in primary])
    order = np.argsort(pvalues, kind="stable")
    corrected = np.empty(4)
    corrected[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(4, 0, -1)))
    for result, value in zip(primary, corrected):
        result["holm_p"] = float(value)
        expected = paired[f"own/{result['condition']}@+5"][result["readout"]]["primary_holm_p"]
        assert np.isclose(expected, value)
    with (output / "primary_comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(primary[0]))
        writer.writeheader()
        writer.writerows(primary)
    with (output / "trial_evidence.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="trial_evidence.csv", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=list(evidence[0]))
                writer.writeheader()
                writer.writerows(evidence)
    clean_summary = json.loads((run / "clean/summary.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    assert clean_summary["clips"] == 2000 and clean_summary["seconds"] > 0
    setup = {
        "clean_clips": clean_summary["clips"],
        "clean_batch_size": clean_summary["batch_size"],
        "clean_encoding_seconds": clean_summary["seconds"],
        "calibration_on_cached_scores_seconds": selection["selection_seconds"],
        "text_preparation_seconds": provenance["one_off_text_preparation_seconds"],
        "text_preparation_kind": provenance["text_preparation_kind"],
        "historical_prompt_encoding_measured_here": False,
        "scope": "One-off setup, separate from per-query deployment latency; clean encoding includes all four temperature branches",
        "input_sha256": {name: sha(run / name) for name in
                         ("clean/summary.json", "provenance.json", "selection.json")},
    }
    (output / "setup_cost.json").write_text(json.dumps(setup, indent=2) + "\n")
    result = {
        "verified": True, "clean_targets": 2000, "own_rows": 24000, "target_rows": 23991,
        "readouts": list(READOUTS), "condition_rows_checked": len(conditions),
        "trial_readout_records": len(evidence), "primary_tests": len(primary),
        "query_slots_including_controls": 2000 + 2 * (24000 + 23991),
        "encoding_cache": cache,
        "source_exclusions_and_clean_selection_recomputed": True,
        "all_original_single_and_ensemble_predictions_identical": True,
        "max_abs_differences": maxima,
        "source_sha256": sha(Path(__file__)),
        "run_provenance_sha256": sha(run / "provenance.json"),
        "conditions_sha256": sha(run / "conditions.csv"),
        "primary_comparisons_sha256": sha(output / "primary_comparisons.csv"),
        "trial_evidence_sha256": sha(output / "trial_evidence.csv.gz"),
        "setup_cost_sha256": sha(output / "setup_cost.json"),
        "scope": "Local-alignment adaptation, not official or full PAT; calibrated results use clean labels",
    }
    (output / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "single-clean", "ensemble-run", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_dir, args.single_clean, args.ensemble_run, args.data_root, args.output)

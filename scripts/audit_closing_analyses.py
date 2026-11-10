"""Independent score-set/count audit and sampled SciPy acoustic reconstruction."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import stft as scipy_stft

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.mixing import renderer_from_manifest


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def close(x, y, atol=1e-10):
    if not np.isclose(float(x), float(y), atol=atol, rtol=0):
        raise ValueError(f"{x} != {y}")


def main(args):
    repo, rt, run, out = args.repo, args.runtime, args.run, args.output
    result = {
        "verified": False,
        "independent_score_trials": 0,
        "outcome_cells": 0,
        "cloud_feature_rows": {},
        "scipy_feature_checks": {},
        "paired_cells": 0,
    }
    trial = rows(out / "readout_trials.csv")
    summary = rows(out / "readout_outcomes.csv")
    by = {(r["variant"], r["region"], int(r["row"])): r for r in trial}
    for variant, system in [
        ("svm_baseline", "svm"),
        ("cnn_baseline", "cnn"),
        ("clap_baseline", "clap"),
        ("clap_prompt_ensemble", "clap"),
    ]:
        for region in ("own", "target"):
            d = (
                repo / f"artifacts/interference/v3_20261109/eval_{region}"
                if system != "clap"
                else (
                    rt / f"runs/clap_v4_20261109/eval_{region}"
                    if variant == "clap_baseline"
                    else rt / f"runs/extensions_20261109/clap_prompt_ensemble/eval_{region}"
                )
            )
            pred = rows(d / f"{system}_predictions.csv")
            lookup = {int(r["row"]): r for r in pred}
            with np.load(d / f"{system}_scores.npz") as z:
                ids = z["class_ids"]
                s = z["scores"]
                c = z["control_scores"]
                ident = z["row"]
            for i, row in enumerate(ident):
                p = lookup[int(row)]
                if not p["interferer_class"]:
                    continue
                t = by[variant, region, int(row)]
                y = int(p["target_class"])
                distractor = int(p["interferer_class"])
                # Python sorting and label lookup, separate from production inverse argsort.
                order = sorted(zip(s[i], ids), key=lambda a: (-a[0], a[1]))
                co = sorted(zip(c[i], ids), key=lambda a: (-a[0], a[1]))
                order = [int(a[1]) for a in order]
                co = [int(a[1]) for a in co]
                for field, value in [
                    ("target_rank", order.index(y) + 1),
                    ("distractor_rank", order.index(distractor) + 1),
                    ("control_target_rank", co.index(y) + 1),
                    ("control_distractor_rank", co.index(distractor) + 1),
                ]:
                    close(t[field], value)
                result["independent_score_trials"] += 1
    for s in summary:
        selected = [
            r
            for r in trial
            if all(r[k] == s[k] for k in ("variant", "region", "condition", "level_db"))
        ]
        k = int(s["k"])
        n = len(selected)
        close(s["n"], n)
        for prefix in ("mixture", "control"):
            tr = "target_rank" if prefix == "mixture" else "control_target_rank"
            dr = "distractor_rank" if prefix == "mixture" else "control_distractor_rank"
            counts = {key: 0 for key in ("target_only", "both", "distractor_only", "neither")}
            for r in selected:
                a, b = int(r[tr]) <= k, int(r[dr]) <= k
                key = (
                    "both"
                    if a and b
                    else ("target_only" if a else ("distractor_only" if b else "neither"))
                )
                counts[key] += 1
            assert sum(counts.values()) == n
            for key, value in counts.items():
                close(s[f"{prefix}_{key}_count"], value)
                close(s[f"{prefix}_{key}"], value / n)
        captured = [r for r in selected if int(r["argmax_captured"])]
        close(s["argmax_captured_count"], len(captured))
        close(
            s["argmax_captured_target_rank_le_k"],
            sum(int(r["target_rank"]) <= k for r in captured) / len(captured),
        )
        result["outcome_cells"] += 1
    # Independent framing uses scipy.signal.stft and explicitly rebuilt triangles.
    cfg = {
        "fs": 44100,
        "nperseg": 1024,
        "noverlap": 512,
        "nfft": 1024,
        "boundary": None,
        "padded": True,
    }
    mel_edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 22050 / 700), 66) / 2595) - 1)
    freqs = np.arange(513) * 44100 / 1024
    bank = np.stack(
        [
            np.maximum(0, np.minimum((freqs - a) / (b - a), (d - freqs) / (d - b)))
            for a, b, d in zip(mel_edges[:-2], mel_edges[1:-1], mel_edges[2:])
        ]
    )

    def power(x):
        _, _, z = scipy_stft(x, window="hann", **cfg)
        return (np.abs(z.T * 512) ** 2 / 1024) @ bank.T

    for region in ("own", "target"):
        features = rows(run / f"overlap_{region}/overlap_features.csv")
        fl = {int(r["row"]): r for r in features}
        supplied = rows(
            run / f"incoming/closing_analyses/results/overlap_{region}/overlap_features.csv"
        )
        maximum = 0.0
        legacy = (
            "spectral_correlation_db",
            "spectral_cosine",
            "masked_cell_fraction",
            "masked_energy_fraction",
            "interferer_energy_share",
        )
        for r in supplied:
            f = fl[int(r["row"])]
            assert all(f[k] == r[k] for k in ("condition", "target_file", "interferer", "level_db"))
            for k in legacy:
                delta = abs(float(f[k]) - float(r[k]))
                maximum = max(maximum, delta)
                close(f[k], r[k], 1e-8)
        result["cloud_feature_rows"][region] = {
            "n": len(supplied),
            "max_absolute_difference": maximum,
        }
        renderer, meta, mr = renderer_from_manifest(rt / f"runs/clap_v4_20261109/manifest_{region}")
        assert len(features) == len(mr)
        assert sum(r["condition"] == "demand" for r in features) == 6000
        sample = []
        for condition in ("white", "demand", "semantic_same", "semantic_different"):
            for level in (10.0, 5.0, 0.0):
                for fold in range(1, 6):
                    candidates = [
                        r
                        for r in mr
                        if r["condition"] == condition
                        and r["level_db"] == level
                        and r["fold"] == fold
                    ]
                    sample.append(candidates[len(candidates) // 2])
        max_error = 0.0
        for row in sample:
            audio = renderer.render(row)
            x, d = audio["clean_matched"], audio["interferer_scaled"]
            a, b = power(x), power(d)
            la, lb = a.sum(0), b.sum(0)
            ad, bd = [10 * np.log10(np.maximum(v, 1e-12)) for v in (la, lb)]
            norm = np.linalg.norm(la) * np.linalg.norm(lb)
            vals = {
                "spectral_correlation_db": np.corrcoef(ad, bd)[0, 1]
                if ad.std() > 0 and bd.std() > 0
                else 0.0,
                "spectral_cosine": la @ lb / norm if norm > 0 else 0.0,
                "masked_cell_fraction": np.mean(b[a >= a.max() * 1e-4] > a[a >= a.max() * 1e-4]),
                "masked_energy_fraction": a[b > a].sum() / a.sum(),
                "interferer_energy_share": b.sum() / (a.sum() + b.sum()),
            }
            nframes = a.shape[0]
            padded = np.pad(x, (0, 1024 + (nframes - 1) * 512 - len(x)))
            e = np.array([np.mean(padded[i * 512 : i * 512 + 1024] ** 2) for i in range(nframes)])
            active = e >= e.max() * 1e-4
            at, bi = a[active].sum(0), b[active].sum(0)
            norm = np.linalg.norm(at) * np.linalg.norm(bi)
            if norm > 0:
                vals["target_activity_spectral_cosine"] = np.clip(at @ bi / norm, 0, 1)
            dt, di = la / bank.sum(1), lb / bank.sum(1)
            dt = 10 * np.log10(np.maximum(dt / dt.max(), 1e-12))
            di = 10 * np.log10(np.maximum(di / di.max(), 1e-12))
            vals["density_spectral_correlation_db"] = np.corrcoef(dt, di)[0, 1]
            for key, value in vals.items():
                error = abs(float(fl[row["row"]][key]) - value)
                max_error = max(max_error, error)
                close(fl[row["row"]][key], value, 1e-8)
        result["scipy_feature_checks"][region] = {
            "n": len(sample),
            "max_absolute_difference": max_error,
            "features_sha256": sha256_file(run / f"overlap_{region}/overlap_features.csv"),
        }
    paired = rows(out / "acoustic_matched_trials.csv")
    for s in rows(out / "acoustic_matched_comparisons.csv"):
        selected = [
            r
            for r in paired
            if all(r[k] == s[k] for k in ("region", "system", "comparison"))
            and int(r[s["stratum"]])
        ]
        n = len(selected)
        close(s["n"], n)
        assert len({r["target_file"] for r in selected}) == n
        if n:
            a = np.array([int(r["a_correct"]) for r in selected])
            b = np.array([int(r["b_correct"]) for r in selected])
            ca = np.array([int(r["a_control_correct"]) for r in selected])
            cb = np.array([int(r["b_control_correct"]) for r in selected])
            close(s["accuracy_difference"], (a - b).mean())
            close(s["control_loss_difference"], ((ca - a) - (cb - b)).mean())
        result["paired_cells"] += 1

    # Independent implementation samples integer cluster indices rather than
    # multinomial weights. Its distinct RNG seed checks interval stability.
    def alternate_interval(values, source_ids, conditional=False):
        groups = sorted(set(source_ids))
        totals = np.array(
            [np.asarray(values)[np.array(source_ids) == g].sum(axis=0) for g in groups]
        )
        sizes = np.array([source_ids.count(g) for g in groups])
        rng = np.random.default_rng(530511)
        draws = []
        for _ in range(20):
            index = rng.integers(0, len(groups), size=(100, len(groups)))
            num = totals[index].sum(axis=1)
            if conditional:
                draws.extend(num[:, 0] / num[:, 1])
            else:
                draws.extend(num / sizes[index].sum(axis=1))
        return np.percentile(draws, [2.5, 97.5])

    interval_checks = []
    for cell in rows(out / "acoustic_matched_comparisons.csv"):
        if cell["stratum"] != "close_primary":
            continue
        selected = [
            r
            for r in paired
            if all(r[k] == cell[k] for k in ("region", "system", "comparison"))
            and int(r["close_primary"])
        ]
        alternative = alternate_interval(
            np.array([int(r["a_correct"]) - int(r["b_correct"]) for r in selected]),
            [r["source_id"] for r in selected],
        )
        delta = float(np.max(np.abs(alternative - [float(cell["ci_low"]), float(cell["ci_high"])])))
        if delta > 0.03:
            raise ValueError("Bootstrap interval check differs by more than 3 pp")
        interval_checks.append(delta)
    for cell in rows(out / "readout_uncertainty.csv"):
        if (
            cell["k"] != "2"
            or cell["level_db"] != "5.0"
            or not cell["variant"].startswith("clap")
            or cell["metric"] != "target_inclusion_given_capture"
        ):
            continue
        selected = [
            r
            for r in trial
            if all(r[k] == cell[k] for k in ("region", "variant", "condition", "level_db"))
        ]
        values = np.array(
            [
                [
                    int(r["argmax_captured"]) * (int(r["target_rank"]) <= 2),
                    int(r["argmax_captured"]),
                ]
                for r in selected
            ]
        )
        alternative = alternate_interval(
            values, [r["source_id"] for r in selected], conditional=True
        )
        delta = float(np.max(np.abs(alternative - [float(cell["ci_low"]), float(cell["ci_high"])])))
        if delta > 0.03:
            raise ValueError("Readout bootstrap interval check differs by more than 3 pp")
        interval_checks.append(delta)
    result["independent_interval_check"] = {
        "cells": len(interval_checks),
        "resamples": 2000,
        "seed": 530511,
        "max_endpoint_difference": max(interval_checks),
        "tolerance": 0.03,
        "method": "independent integer cluster draws; expected Monte Carlo differences",
    }
    result["verified"] = True
    result["bootstrap_scope"] = (
        "Point estimates and counts independently recomputed; 26 primary intervals additionally checked with independent integer cluster draws and a different fixed seed; remaining interval draws not independently regenerated."
    )
    result["output_hashes"] = {p.name: sha256_file(p) for p in out.glob("*.csv")}
    write_json(out / "independent_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "runtime", "run", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    main(p.parse_args())

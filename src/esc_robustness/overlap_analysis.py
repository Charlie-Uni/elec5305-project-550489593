"""Acoustic overlap between target and interferer, and how much of the condition effect
it is associated with.

Equal nominal power does not make interference types acoustically alike. For every
manifest row this module measures, from the rendered target and the scaled interferer:

- ``spectral_correlation_db``: Pearson correlation between the two long-term Mel spectra
  in decibels (64 bands summed over frames). 1 = the same spectral shape; near 0 or
  negative = opposite raw Mel-band trends. Unequal filter bandwidths affect this
  descriptor; it is not a literal absence-of-overlap or a perceptual measure.
- ``masked_cell_fraction``: among the target's active Mel time-frequency cells (within
  40 dB of its loudest cell), the fraction in which the interferer is stronger than the
  target. This joins spectral and temporal overlap into one masking proxy and counts
  cells, so weak but informative cells matter as much as peaks.
- ``masked_energy_fraction``: the energy-weighted version (share of target Mel energy in
  masked cells); dominated by the loudest cells, kept for reference.
- ``activity_overlap``: the manifest's temporal overlap of activity masks.
Linear ratios are invariant to common gain. The legacy raw-band dB correlation
uses an absolute floor; the added density correlation uses a relative floor.

The analysis stratifies accuracy by these measures, compares semantic distractors with
DEMAND on identical targets inside matched strata, and fits descriptive logistic
regressions of correctness on condition with and without the acoustic covariates.
Everything here is descriptive: it conditions on fitted models and reports how much
of the condition contrast survives acoustic matching, not a causal effect of meaning. Logistic coefficient changes are not fractions
of an effect explained.
"""

import csv
import json
from pathlib import Path

import numpy as np

from .analysis import read_predictions
from .artifacts import sha256_file, write_json
from .dsp import FeatureConfig, frame_audio, mel_filterbank, stft
from .mixing import (
    DemandCache,
    MixtureRenderer,
    interference_config_from_json,
    manifest_digest,
    read_manifest,
)

FEATURE_FIELDS = (
    "row",
    "condition",
    "ratio_type",
    "level_db",
    "fold",
    "target_file",
    "target_class",
    "target_coarse",
    "interferer",
    "interferer_class",
    "interferer_coarse",
    "common_gain",
    "activity_overlap",
    "spectral_correlation_db",
    "spectral_cosine",
    "masked_cell_fraction",
    "masked_energy_fraction",
    "interferer_energy_share",
    "target_activity_spectral_cosine",
    "density_spectral_correlation_db",
)
MASK_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)
CORRELATION_BINS = (-1.0, 0.0, 0.25, 0.5, 0.75, 1.0000001)
MEASURES = ("masked_cell_fraction", "spectral_correlation_db", "activity_overlap")
ACTIVE_CELL_DB = 40.0
CONDITIONS = ("white", "demand", "semantic_same", "semantic_different")


def mel_power(audio, config=FeatureConfig()):
    """Project DSP/CNN Mel power (frames x bands); not the CLAP internal frontend."""
    spectrum = stft(audio, config)
    return (np.abs(spectrum) ** 2 / config.n_fft) @ mel_filterbank(config).T


def overlap_measures(target, interferer, config=FeatureConfig()):
    """Spectral correlation (dB), masked cell/energy fractions and interferer energy share."""
    target_power = mel_power(target, config)
    interferer_power = mel_power(interferer, config)
    if target_power.sum() <= 0:
        raise ValueError("Target has no Mel energy")
    long_target, long_interferer = target_power.sum(axis=0), interferer_power.sum(axis=0)
    floor = config.log_floor
    target_db = 10 * np.log10(np.maximum(long_target, floor))
    interferer_db = 10 * np.log10(np.maximum(long_interferer, floor))
    if target_db.std() > 0 and interferer_db.std() > 0:
        correlation = float(np.corrcoef(target_db, interferer_db)[0, 1])
    else:
        correlation = 0.0
    norm = np.linalg.norm(long_target) * np.linalg.norm(long_interferer)
    cosine = float(long_target @ long_interferer / norm) if norm > 0 else 0.0
    active = target_power >= target_power.max() * 10 ** (-ACTIVE_CELL_DB / 10)
    masked_cells = float(np.mean(interferer_power[active] > target_power[active]))
    masked_energy = float(target_power[interferer_power > target_power].sum() / target_power.sum())
    share = float(interferer_power.sum() / (target_power.sum() + interferer_power.sum()))
    # A separate shape diagnostic conditions on target-active frames. Undefined
    # zero-energy interferers remain missing rather than being labelled orthogonal.
    energies = np.mean(frame_audio(target, config) ** 2, axis=1)
    active_frames = energies >= energies.max() * 10 ** (-ACTIVE_CELL_DB / 10)
    at = target_power[active_frames].sum(axis=0)
    ai = interferer_power[active_frames].sum(axis=0)
    active_norm = np.linalg.norm(at) * np.linalg.norm(ai)
    active_cosine = float(np.clip(at @ ai / active_norm, 0, 1)) if active_norm > 0 else None
    # Unequal Mel bandwidths make white-noise band energy increase with frequency.
    # Retain the supplied raw-band descriptor, and expose a density sensitivity.
    areas = mel_filterbank(config).sum(axis=1)
    dt, di = long_target / areas, long_interferer / areas
    dt = 10 * np.log10(np.maximum(dt / dt.max(), 1e-12))
    if di.max() > 0:
        di = 10 * np.log10(np.maximum(di / di.max(), 1e-12))
        density_r = float(np.corrcoef(dt, di)[0, 1]) if dt.std() > 0 and di.std() > 0 else None
    else:
        density_r = None
    return {
        "target_activity_spectral_cosine": active_cosine,
        "density_spectral_correlation_db": density_r,
        "spectral_correlation_db": correlation,
        "spectral_cosine": cosine,
        "masked_cell_fraction": masked_cells,
        "masked_energy_fraction": masked_energy,
        "interferer_energy_share": share,
    }


def renderer_for(manifest_dir, data_root=None, require_demand=False):
    """Renderer for a manifest; DEMAND rows are skipped when the cache is unavailable."""
    meta, rows = read_manifest(manifest_dir)
    config = interference_config_from_json(meta["interference_config"])
    feature_config = FeatureConfig(**meta["feature_config"])
    demand = None
    if meta.get("demand_cache"):
        try:
            demand = DemandCache(meta["demand_cache"])
        except FileNotFoundError:
            if require_demand:
                raise
    root = Path(data_root) if data_root is not None else Path(meta["data_root"])
    if sha256_file(root / "meta/esc50.csv") != meta["metadata_sha256"]:
        raise ValueError("Dataset metadata checksum differs from manifest")
    if require_demand and any(r["condition"] == "demand" for r in rows) and demand is None:
        raise ValueError("DEMAND is required but unavailable")
    if demand is not None and demand.meta["sample_rate"] != feature_config.sample_rate:
        raise ValueError("DEMAND sample rate differs from manifest")
    return MixtureRenderer(root, feature_config, config, demand), meta, rows


def compute_overlap_features(
    manifest_dir, output_dir, data_root=None, limit=None, require_demand=False
):
    """Write overlap_features.csv (one row per rendered mixture) and a provenance file."""
    manifest_dir, output_dir = Path(manifest_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    renderer, meta, rows = renderer_for(manifest_dir, data_root, require_demand)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped_demand = 0
    written = 0
    with (output_dir / "overlap_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        for i, row in enumerate(rows):
            if row["condition"] == "demand" and renderer.demand is None:
                skipped_demand += 1
                continue
            rendered = renderer.render(row)
            measures = overlap_measures(
                rendered["clean_matched"], rendered["interferer_scaled"], renderer.feature_config
            )
            record = {field: row[field] for field in FEATURE_FIELDS if field in row}
            record["interferer_class"] = (
                "" if row["interferer_class"] is None else row["interferer_class"]
            )
            record.update(measures)
            writer.writerow(record)
            written += 1
            if (i + 1) % 1000 == 0 or i + 1 == len(rows):
                print(f"Overlap features: {i + 1}/{len(rows)} rows", flush=True)
    write_json(
        output_dir / "overlap_features_meta.json",
        {
            "manifest": str(manifest_dir),
            "manifest_sha256": manifest_digest(manifest_dir),
            "power_region": meta["interference_config"].get("power_region", "own"),
            "rows_written": written,
            "features_sha256": sha256_file(output_dir / "overlap_features.csv"),
            "demand_rows_skipped": skipped_demand,
            "demand_available": renderer.demand is not None,
            "measures": {
                "spectral_correlation_db": (
                    "Pearson correlation of long-term 64-band Mel spectra in dB"
                ),
                "spectral_cosine": "cosine of long-term linear Mel power spectra (reference)",
                "masked_cell_fraction": (
                    f"fraction of target Mel TF cells within {ACTIVE_CELL_DB} dB of its loudest "
                    "cell where interferer power exceeds target power"
                ),
                "masked_energy_fraction": (
                    "share of target Mel TF energy in cells where interferer power exceeds target"
                ),
                "interferer_energy_share": "interferer / (target + interferer) Mel energy",
                "activity_overlap": "from the manifest",
                "target_activity_spectral_cosine": (
                    "cosine of linear Mel spectra averaged over target-active frames; "
                    "unwindowed frame energy within 40 dB of peak; blank if undefined"
                ),
                "density_spectral_correlation_db": (
                    "dB correlation after dividing each Mel band by its filter-weight sum; "
                    "each density spectrum normalised by its maximum with relative floor 1e-12"
                ),
            },
            "frontend": renderer.feature_config.as_dict(),
            "source_sha256": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in ("overlap_analysis.py", "mixing.py", "dsp.py")
            },
        },
    )
    return {"rows_written": written, "demand_rows_skipped": skipped_demand}


# ----------------------------------------------------------------- analysis


def read_features(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["row"] = int(row["row"])
        row["level_db"] = float(row["level_db"])
        for key in (
            "common_gain",
            "activity_overlap",
            "spectral_correlation_db",
            "spectral_cosine",
            "masked_cell_fraction",
            "masked_energy_fraction",
            "interferer_energy_share",
        ):
            row[key] = float(row[key])
    if len({r["row"] for r in rows}) != len(rows):
        raise ValueError("Duplicate acoustic-feature row")
    for row in rows:
        for key in ("target_activity_spectral_cosine", "density_spectral_correlation_db"):
            if key in row:
                row[key] = float(row[key]) if row[key] else float("nan")
    return {row["row"]: row for row in rows}


def _bin_index(value, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return None


def _quartiles(values):
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"n": len(values), "median": float(median), "q1": float(q1), "q3": float(q3)}


def measure_summary(features):
    """Median and quartiles of each measure per condition and level."""
    table = []
    for condition in CONDITIONS:
        levels = sorted(
            {f["level_db"] for f in features.values() if f["condition"] == condition}, reverse=True
        )
        for level in levels:
            selected = [
                f
                for f in features.values()
                if f["condition"] == condition and f["level_db"] == level
            ]
            entry = {"condition": condition, "level_db": level, "n": len(selected)}
            for key in (
                "masked_cell_fraction",
                "spectral_correlation_db",
                "activity_overlap",
                "masked_energy_fraction",
                "spectral_cosine",
                "interferer_energy_share",
            ):
                q = _quartiles([f[key] for f in selected])
                entry[f"{key}_median"] = q["median"]
                entry[f"{key}_q1"] = q["q1"]
                entry[f"{key}_q3"] = q["q3"]
            table.append(entry)
    return table


def stratified_accuracy(
    features, predictions, level, measure="masked_cell_fraction", edges=MASK_BINS
):
    """Accuracy (pooled) and capture fraction per condition inside bins of one measure."""
    table = []
    for condition in CONDITIONS:
        rows = [
            r
            for r in predictions
            if r["condition"] == condition and r["level_db"] == level and int(r["row"]) in features
        ]
        for i in range(len(edges) - 1):
            selected = [r for r in rows if _bin_index(features[int(r["row"])][measure], edges) == i]
            if not selected:
                continue
            entry = {
                "condition": condition,
                "level_db": level,
                "measure": measure,
                "bin_low": edges[i],
                "bin_high": min(edges[i + 1], 1.0),
                "n": len(selected),
                "accuracy": float(np.mean([r["correct"] for r in selected])),
                "classes_present": len({r["target_class"] for r in selected}),
            }
            if condition.startswith("semantic"):
                entry["capture_fraction"] = float(
                    np.mean([r["prediction"] == r["interferer_class"] for r in selected])
                )
            else:
                entry["capture_fraction"] = None
            table.append(entry)
    return table


def matched_semantic_vs_demand(features, predictions, level, measure="masked_cell_fraction"):
    """Semantic minus DEMAND accuracy on identical targets, by how the two mixtures differ
    in the acoustic measure (semantic masks less / about the same / more than DEMAND)."""
    demand = {
        r["target_file"]: r
        for r in predictions
        if r["condition"] == "demand" and r["level_db"] == level and int(r["row"]) in features
    }
    table = []
    for condition in ("semantic_same", "semantic_different"):
        pairs = []
        for r in predictions:
            if (
                r["condition"] != condition
                or r["level_db"] != level
                or int(r["row"]) not in features
            ):
                continue
            partner = demand.get(r["target_file"])
            if partner is None:
                continue
            delta = features[int(r["row"])][measure] - features[int(partner["row"])][measure]
            pairs.append((delta, r["correct"], partner["correct"]))
        if not pairs:
            continue
        deltas = np.array([p[0] for p in pairs])
        cuts = np.percentile(deltas, [100 / 3, 200 / 3])
        strata = (
            ("lower delta third", deltas < cuts[0]),
            ("middle delta third", (deltas >= cuts[0]) & (deltas <= cuts[1])),
            ("upper delta third", deltas > cuts[1]),
            ("all pairs", np.ones(len(deltas), dtype=bool)),
        )
        for name, mask in strata:
            if not mask.any():
                continue
            sem = np.array([p[1] for p in pairs])[mask]
            dem = np.array([p[2] for p in pairs])[mask]
            table.append(
                {
                    "condition": condition,
                    "level_db": level,
                    "measure": measure,
                    "stratum": name,
                    "n": int(mask.sum()),
                    "delta_measure_median": float(np.median(deltas[mask])),
                    "delta_measure_min": float(deltas[mask].min()),
                    "delta_measure_max": float(deltas[mask].max()),
                    "semantic_accuracy": float(sem.mean()),
                    "demand_accuracy": float(dem.mean()),
                    "semantic_minus_demand_pp": float(100 * (sem.mean() - dem.mean())),
                }
            )
    return table


def logistic_decomposition(features, predictions, level, baseline="demand"):
    """Descriptive logistic regressions of correctness on condition and acoustic covariates.

    Returns coefficients for three nested models: condition dummies only; acoustic
    covariates only; both. Covariates are standardised, so coefficients are per SD.
    Coefficient changes are not explained-effect proportions: logistic odds ratios are
    non-collapsible, observations are dependent, and support may be poor. No causal
    or inferential interpretation is attached; pseudo-R2 is in-sample fit only.
    """
    from sklearn.linear_model import LogisticRegression

    rows = [r for r in predictions if r["level_db"] == level and int(r["row"]) in features]
    present = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
    if len(present) < 2:
        raise ValueError("At least two conditions are needed for the decomposition")
    if baseline not in present:
        baseline = present[0]
    conditions = [c for c in present if c != baseline]
    covariates = MEASURES
    y = np.array([r["correct"] for r in rows], dtype=int)
    dummies = np.array([[1.0 if r["condition"] == c else 0.0 for c in conditions] for r in rows])
    acoustic = np.array([[features[int(r["row"])][k] for k in covariates] for r in rows])
    scale = acoustic.std(axis=0)
    scale[scale <= 0] = 1.0
    acoustic = (acoustic - acoustic.mean(axis=0)) / scale
    designs = {
        "condition_only": (dummies, list(conditions)),
        "acoustic_only": (acoustic, list(covariates)),
        "condition_plus_acoustic": (
            np.hstack([dummies, acoustic]),
            list(conditions) + list(covariates),
        ),
    }
    base_rate = y.mean()
    null_loglik = float(np.sum(y * np.log(base_rate) + (1 - y) * np.log(1 - base_rate)))
    result = {"level_db": level, "baseline_condition": baseline, "n": len(rows), "models": {}}
    for name, (x, names) in designs.items():
        model = LogisticRegression(C=1e6, max_iter=5000).fit(x, y)
        probabilities = np.clip(model.predict_proba(x)[:, 1], 1e-12, 1 - 1e-12)
        loglik = float(np.sum(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)))
        result["models"][name] = {
            "intercept": float(model.intercept_[0]),
            "coefficients": {n: float(c) for n, c in zip(names, model.coef_[0])},
            "mcfadden_pseudo_r2": 1 - loglik / null_loglik if null_loglik < 0 else None,
        }
    return result


def analyse_overlap(features_dir, prediction_dirs, output_dir, level=5.0):
    """Tables (CSV/JSON) and two figures for every system with predictions.

    ``prediction_dirs`` maps system name to an evaluation directory holding
    ``<system>_predictions.csv`` for the same manifest as the features.
    """
    features_dir, output_dir = Path(features_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = read_features(features_dir / "overlap_features.csv")
    meta = json.loads((features_dir / "overlap_features_meta.json").read_text())
    if meta.get("features_sha256") and meta["features_sha256"] != sha256_file(
        features_dir / "overlap_features.csv"
    ):
        raise ValueError("Acoustic feature checksum mismatch")
    summary = {
        "features": str(features_dir),
        "manifest_sha256": meta["manifest_sha256"],
        "power_region": meta.get("power_region"),
        "level_db": level,
        "demand_available": meta.get("demand_available"),
        "statistical_scope": "Exploratory pooled in-sample logistic fits; coefficient changes are not explained-effect proportions; no causal or significance claims",
        "measure_summary": measure_summary(features),
        "systems": {},
    }
    _write_csv(output_dir / "measure_summary.csv", summary["measure_summary"])
    strat_rows, matched_rows, logistic = [], [], {}
    for system, directory in prediction_dirs.items():
        predictions = read_predictions(Path(directory) / f"{system}_predictions.csv")
        evaluation_meta = json.loads((Path(directory) / "summary.json").read_text())
        if evaluation_meta["manifest_sha256"] != meta["manifest_sha256"]:
            raise ValueError("Prediction and acoustic manifest hashes differ")
        for r in predictions:
            if int(r["row"]) in features:
                f = features[int(r["row"])]
                for key in ("condition", "target_file", "interferer"):
                    if r[key] != f[key]:
                        raise ValueError("Acoustic and prediction identities differ")
                if r["level_db"] != f["level_db"]:
                    raise ValueError("Acoustic and prediction levels differ")
        for measure, edges in (
            ("masked_cell_fraction", MASK_BINS),
            ("spectral_correlation_db", CORRELATION_BINS),
        ):
            for entry in stratified_accuracy(features, predictions, level, measure, edges):
                strat_rows.append({"system": system, **entry})
        for entry in matched_semantic_vs_demand(features, predictions, level):
            matched_rows.append({"system": system, **entry})
        try:
            logistic[system] = logistic_decomposition(features, predictions, level)
        except ValueError as error:
            logistic[system] = {"error": str(error)}
    _write_csv(output_dir / "stratified_accuracy.csv", strat_rows)
    _write_csv(output_dir / "matched_semantic_vs_demand.csv", matched_rows)
    summary["systems"] = logistic
    write_json(output_dir / "overlap_analysis.json", summary)
    figures = {}
    if strat_rows:
        figures["accuracy_vs_masking"] = figure_accuracy_vs_measure(strat_rows, output_dir, level)
    if matched_rows:
        figures["matched_by_masking_difference"] = figure_matched(matched_rows, output_dir, level)
    summary["figures"] = figures
    write_json(output_dir / "overlap_analysis.json", summary)
    return summary


def _write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------- figures

CONDITION_STYLE = {
    "white": {"color": "#9a9890", "linestyle": ":", "marker": "x", "label": "white noise"},
    "demand": {"color": "#52514e", "linestyle": "--", "marker": "s", "label": "DEMAND background"},
    "semantic_same": {"linestyle": "-", "marker": "o", "label": "same-category event"},
    "semantic_different": {
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": "white",
        "label": "different-category event",
    },
}


def figure_accuracy_vs_measure(
    strat_rows, output_dir, level, measure="masked_cell_fraction", min_n=30
):
    """Bins with fewer than ``min_n`` trials are left to the CSV and not drawn."""
    from .figures import SYSTEM_COLOUR, SYSTEM_LABEL, SYSTEM_ORDER, TEXT_PRIMARY, _db, _matplotlib

    plt = _matplotlib()
    systems = [s for s in SYSTEM_ORDER if any(r["system"] == s for r in strat_rows)]
    fig, axes = plt.subplots(1, len(systems), figsize=(3.5 * len(systems), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, system in zip(axes, systems):
        for condition in CONDITIONS:
            rows = sorted(
                [
                    r
                    for r in strat_rows
                    if r["system"] == system
                    and r["condition"] == condition
                    and r["measure"] == measure
                    and r["n"] >= min_n
                ],
                key=lambda r: r["bin_low"],
            )
            if not rows:
                continue
            style = dict(CONDITION_STYLE[condition])
            style.setdefault("color", SYSTEM_COLOUR[system])
            if "markerfacecolor" not in style:
                style["markerfacecolor"] = style["color"]
            x = [(r["bin_low"] + r["bin_high"]) / 2 for r in rows]
            axis.plot(
                x,
                [r["accuracy"] for r in rows],
                markeredgecolor=style["color"],
                markersize=6,
                **style,
            )
        axis.set_title(SYSTEM_LABEL[system], loc="left", color=TEXT_PRIMARY)
        axis.set_xlabel("Masked fraction of target cells")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy (pooled)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, fontsize=7, loc="upper right")
    fig.suptitle(
        f"Accuracy versus acoustic masking at {_db(level)}: fraction of the target's active Mel cells where the interferer is stronger",
        x=0.01,
        ha="left",
        fontsize=8.5,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    stem = "overlap_accuracy_vs_masking"
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return stem


def figure_matched(matched_rows, output_dir, level):
    from .figures import (
        HATCH,
        SYSTEM_COLOUR,
        SYSTEM_LABEL,
        SYSTEM_ORDER,
        TEXT_PRIMARY,
        TEXT_SECONDARY,
        _db,
        _matplotlib,
    )

    plt = _matplotlib()
    systems = [s for s in SYSTEM_ORDER if any(r["system"] == s for r in matched_rows)]
    strata = ("lower delta third", "middle delta third", "upper delta third")
    fig, axes = plt.subplots(1, len(systems), figsize=(3.5 * len(systems), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.38
    for axis, system in zip(axes, systems):
        for j, condition in enumerate(("semantic_same", "semantic_different")):
            values, counts = [], []
            for stratum in strata:
                match = [
                    r
                    for r in matched_rows
                    if r["system"] == system
                    and r["condition"] == condition
                    and r["stratum"] == stratum
                ]
                values.append(match[0]["semantic_minus_demand_pp"] if match else np.nan)
                counts.append(match[0]["n"] if match else 0)
            x = np.arange(len(strata)) + (j - 0.5) * width
            same = condition == "semantic_same"
            axis.bar(
                x,
                values,
                width * 0.9,
                color="white" if same else SYSTEM_COLOUR[system],
                edgecolor=SYSTEM_COLOUR[system],
                hatch=HATCH if same else "",
                linewidth=1.2,
                label="same-category event" if same else "different-category event",
            )
            for xi, v, n in zip(x, values, counts):
                if v == v:
                    axis.annotate(
                        f"n={n}",
                        (xi, v),
                        xytext=(0, 3 if v >= 0 else -9),
                        textcoords="offset points",
                        ha="center",
                        fontsize=6,
                        color=TEXT_SECONDARY,
                    )
        axis.axhline(0, color=TEXT_SECONDARY, linewidth=1)
        axis.set_xticks(range(len(strata)))
        axis.set_xticklabels(
            ["lower delta\nthird", "middle delta\nthird", "upper delta\nthird"], fontsize=7
        )
        axis.set_title(SYSTEM_LABEL[system], loc="left", color=TEXT_PRIMARY)
    axes[0].set_ylabel("Semantic − DEMAND accuracy (pp), same targets")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"Semantic versus DEMAND on identical targets at {_db(level)}, by which mixture masks the target more",
        x=0.01,
        ha="left",
        fontsize=8.5,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    stem = "overlap_matched_by_masking_difference"
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return stem

"""PAT-style local audio-text alignment on frozen LAION-CLAP.

This is NOT a run of the official PAT release. That release (cs20s030/PAT at commit
8910ebb) targets MS-CLAP 2023 through a modified wrapper that is not public, and it
selects its fusion weights by maximising accuracy on the evaluated labels. This module

* keeps the project's fixed LAION-CLAP checkpoint and its fixed released-prompt ensemble,
* takes frame-level features from the same official forward pass as the pooled embedding
  and maps them into the joint space with the model's own audio projection,
* applies the released alignment arithmetic (frame/class softmax feedback, mean plus max
  pooling, three fused logits) in cosine units with an explicit attention temperature.

Two readouts are produced from the same cached branch scores:

``aligned_fixed``       weights declared before any result was seen; uses no labels.
``aligned_calibrated``  weights chosen for each outer test fold on labelled clean clips of
                        the other folds (sources shared with the test fold excluded). The
                        model stays frozen, but this readout is calibrated on labelled
                        data and must not be called zero-shot.

Selection on the evaluated labels, as the release does, is written to a separate file as
an optimistic reference: the best point of the searched grid, never a reported method.
The paper's dataset-level prompt weighting is absent from the released code and is not
used: prompts are averaged uniformly, as the release does. Fused scores are sums of
branch scores and are not cosine similarities.
"""

import argparse
import csv
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from .analysis import fold_metrics, read_predictions
from .artifacts import sha256_file, write_json
from .clap_system import CLAPSystem, load_laion_clap, unit_rows
from .data import load_audio, load_metadata
from .diagnostics import holm_adjust
from .dsp import FeatureConfig

SOURCE_COMMIT = "8910ebb756e33aec72e2cf6c1f02614bfa194e85"
TEMPERATURES = (16.0, 32.0, 64.0, 100.0)
BETAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
# Project heuristics declared before any alignment result existed. These values are
# not published PAT defaults or a verified scale-equivalent conversion from MS-CLAP.
FIXED_RULE = {"temperature": 64.0, "beta_text": 0.2, "beta_audio": 0.05}
READOUTS = ("single", "ensemble", "aligned_fixed", "aligned_calibrated")
SCORE_LABELS = {
    "single": "cosine similarity, one fixed template",
    "ensemble": "cosine similarity, uniform mean of the released templates",
    "aligned_fixed": "fused score, weights fixed in advance (no labels)",
    "aligned_calibrated": "fused score, weights calibrated on labelled clean non-test folds",
}
REFERENCE_TOLERANCE = 1e-4


def _softmax(values, axis):
    shifted = values - values.max(axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=axis, keepdims=True)


def alignment_components(pooled, frames, text, temperature):
    """Text-guided and audio-guided branch scores of the released inference, for a batch.

    ``pooled`` is (B, d), ``frames`` (B, c, d) and ``text`` (K, d). With un-normalised
    inputs and ``temperature=2`` this is exactly the arithmetic of the released script:
    ``logits1 = global @ (frames^T softmax_frames(A))`` and
    ``logits2 = (mean_f + max_f)(softmax_classes(A) text) @ text^T`` with ``A = 2 frames text^T``.
    """
    pooled, frames, text = (np.asarray(a, dtype=np.float64) for a in (pooled, frames, text))
    if frames.ndim != 3 or pooled.shape != (frames.shape[0], frames.shape[2]):
        raise ValueError("Expected pooled (B, d) and frames (B, c, d)")
    if text.ndim != 2 or text.shape[1] != frames.shape[2] or temperature <= 0:
        raise ValueError("Text must be (K, d) and the temperature positive")
    affinity = temperature * np.einsum("bcd,kd->bck", frames, text)
    over_frames = _softmax(affinity, axis=1)  # each class spreads its weight over frames
    over_classes = _softmax(affinity, axis=2)  # each frame spreads its weight over classes
    class_audio = np.einsum("bck,bcd->bkd", over_frames, frames)
    text_guided = np.einsum("bkd,bd->bk", class_audio, pooled)
    guided_frames = over_classes @ text
    audio_guided = (guided_frames.mean(axis=1) + guided_frames.max(axis=1)) @ text.T
    return text_guided, audio_guided


def pack_components(pooled, text_guided, audio_guided):
    """One float32 row per clip: pooled | text-guided per temperature | audio-guided."""
    parts = [np.asarray(pooled)] + list(text_guided) + list(audio_guided)
    return np.concatenate(parts, axis=1).astype(np.float32)


def unpack_components(packed, classes, temperatures):
    packed = np.asarray(packed, dtype=np.float64)
    n = len(temperatures)
    if packed.ndim != 2 or packed.shape[1] <= 2 * n * classes:
        raise ValueError("Packed components do not match the class and temperature counts")
    dimension = packed.shape[1] - 2 * n * classes
    blocks = packed[:, dimension:].reshape(len(packed), 2, n, classes)
    return packed[:, :dimension], blocks[:, 0], blocks[:, 1]


def fusion_grid(temperatures=TEMPERATURES, betas=BETAS):
    """Candidates (temperature index, beta_text, beta_audio), simplest first.

    The first entry is the fixed prompt ensemble (both weights zero); ties in the
    selection criterion therefore resolve toward the smallest departure from it.
    """
    if 0.0 not in betas or any(b < 0 for b in betas):
        raise ValueError("Fusion weights must be non-negative and include zero")
    grid = [(0, 0.0, 0.0)]
    grid += [
        (t, bt, ba)
        for bt in betas
        for ba in betas
        for t in range(len(temperatures))
        if bt > 0 or ba > 0
    ]
    return sorted(grid, key=lambda g: (g[1] + g[2], g[2], g[0]))


def fixed_choice(temperatures, rule=FIXED_RULE):
    matches = [i for i, t in enumerate(temperatures) if t == rule["temperature"]]
    if len(matches) != 1:
        raise ValueError("The fixed temperature must be one of the cached temperatures")
    return (matches[0], rule["beta_text"], rule["beta_audio"])


def describe_choice(choice, temperatures):
    return {
        "temperature": float(temperatures[choice[0]]),
        "beta_text": float(choice[1]),
        "beta_audio": float(choice[2]),
    }


def fused_scores(base, text_guided, audio_guided, choice):
    t, beta_text, beta_audio = choice
    return base + beta_text * text_guided[:, t] + beta_audio * audio_guided[:, t]


def grid_correct(base, text_guided, audio_guided, targets, class_ids, grid):
    """Number of correct top-1 decisions for every candidate."""
    targets, class_ids = np.asarray(targets), np.asarray(class_ids)
    return np.array(
        [
            int(
                np.sum(
                    class_ids[fused_scores(base, text_guided, audio_guided, g).argmax(1)] == targets
                )
            )
            for g in grid
        ]
    )


def calibration_mask(folds, sources, fold):
    """Clean clips outside the test fold whose source recording is absent from it."""
    folds, sources = np.asarray(folds), np.asarray(sources)
    return (folds != fold) & ~np.isin(sources, sources[folds == fold])


def select_cross_fold(base, text_guided, audio_guided, targets, folds, sources, class_ids, grid):
    """For each outer fold, the candidate with the most correct clean calibration clips."""
    folds = np.asarray(folds)
    selection = {}
    for fold in sorted(set(folds.tolist())):
        mask = calibration_mask(folds, sources, fold)
        if not mask.any():
            raise ValueError("Cross-fold calibration needs clips outside the test fold")
        correct = grid_correct(
            base[mask], text_guided[mask], audio_guided[mask], targets[mask], class_ids, grid
        )
        best = int(np.argmax(correct))  # first maximum: the simplest candidate wins ties
        selection[int(fold)] = {
            "choice": grid[best],
            "calibration_clips": int(mask.sum()),
            "excluded_shared_source_clips": int(np.sum(folds != fold) - mask.sum()),
            "calibration_accuracy": float(correct[best] / mask.sum()),
            "ensemble_calibration_accuracy": float(correct[0] / mask.sum()),
        }
    return selection


def apply_selection(base, text_guided, audio_guided, folds, selection):
    """Score every row with the rule of its own outer fold, whatever its condition."""
    folds = np.asarray(folds)
    if not np.isin(folds, list(selection)).all():
        raise ValueError("A row belongs to a fold without a selected fusion rule")
    scores = np.empty_like(base)
    for fold, chosen in selection.items():
        mask = folds == fold
        scores[mask] = fused_scores(
            base[mask], text_guided[mask], audio_guided[mask], tuple(chosen["choice"])
        )
    return scores


def collapse_repeated_frames(frames):
    """Undo the fixed-ratio frame repetition of HTSAT's fine-grained output (exact only)."""
    batch, steps, dimension = frames.shape
    for ratio in sorted((r for r in range(2, steps + 1) if steps % r == 0), reverse=True):
        blocks = frames.reshape(batch, steps // ratio, ratio, dimension)
        if bool((blocks == blocks[:, :, :1]).all()):
            return blocks[:, :, 0]
    return frames


def audio_frames(backend, prepared):
    """Unit pooled (B, d) and unit frame (B, c, d) embeddings from one official forward pass.

    A forward hook reads the audio tower's own output during the official embedding call,
    so the pooled embedding is the one every other CLAP result in this project uses. Two
    identities are asserted: projecting the tower's pooled latent reproduces the official
    embedding, and the frame latents average to that pooled latent. Each frame latent is
    then passed through the same audio projection as the pooled latent and unit-normalised.
    """
    import torch
    import torch.nn.functional as functional

    official = backend.module
    model = official.model
    captured = []
    handle = model.audio_branch.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        with torch.inference_mode():
            pooled = official.get_audio_embedding_from_data(
                x=torch.from_numpy(np.asarray(prepared, dtype=np.float32)), use_tensor=True
            )
            if len(captured) != 1:
                raise RuntimeError("Expected exactly one audio-tower forward pass")
            latent = captured[0]["embedding"]
            fine = collapse_repeated_frames(captured[0]["fine_grained_embedding"])
            projected = functional.normalize(model.audio_projection(latent), dim=-1)
            if not torch.allclose(projected, pooled, atol=1e-5):
                raise RuntimeError("Hooked latent does not reproduce the official embedding")
            if not torch.allclose(fine.mean(dim=1), latent, atol=1e-4, rtol=1e-4):
                raise RuntimeError("Frame latents do not average to the pooled latent")
            frames = functional.normalize(model.audio_projection(fine), dim=-1)
    finally:
        handle.remove()
    return (
        pooled.detach().cpu().numpy().astype(np.float64),
        frames.detach().cpu().numpy().astype(np.float64),
    )


class LocalAlignmentSystem(CLAPSystem):
    """Fixed prompt-ensemble scores plus cached alignment branch scores for later fusion.

    ``scores`` are the fixed-ensemble cosines, so the standard prediction files of a run
    reproduce the existing ensemble ablation. The ``representation`` row packs the pooled
    embedding with the text-guided and audio-guided scores of all classes at every
    temperature; mixtures and gain-matched controls are both encoded.
    """

    name = "clap_local"
    score_type = "cosine_similarity_fixed_prompt_ensemble"
    representation_type = "pooled_embedding_and_alignment_branch_scores"

    def __init__(
        self,
        module,
        class_names,
        ensemble_text,
        temperatures=TEMPERATURES,
        feature_config=FeatureConfig(),
        checkpoint=None,
        frame_source=audio_frames,
    ):
        super().__init__(module, class_names, feature_config, checkpoint=checkpoint)
        self.single_text_embeddings = self.text_embeddings
        self.single_prompts = self.prompts
        ensemble_text = unit_rows(ensemble_text)
        if ensemble_text.shape != self.single_text_embeddings.shape:
            raise ValueError("Ensemble text embeddings must be one row per class")
        if not temperatures or any(t <= 0 for t in temperatures):
            raise ValueError("Temperatures must be positive")
        self.text_embeddings = ensemble_text
        self.prompts = [
            f"Fixed PAT template ensemble: {self.class_names[int(i)]}" for i in self.class_ids
        ]
        self.temperatures = tuple(float(t) for t in temperatures)
        self.frame_source = frame_source

    def branch_scores(self, prepared, temperatures=None):
        pooled, frames = self.frame_source(self.module, prepared)
        pooled, frames = unit_rows(pooled), unit_rows(frames)
        components = [
            alignment_components(pooled, frames, self.text_embeddings, t)
            for t in (self.temperatures if temperatures is None else temperatures)
        ]
        return pooled, [c[0] for c in components], [c[1] for c in components]

    def evaluate_batch(self, folds, audios):
        if len(folds) != len(audios) or not audios:
            raise ValueError("Expected one fold per audio clip in a nonempty batch")
        prepared = np.stack([self.preprocess(audio) for audio in audios])
        pooled, text_guided, audio_guided = self.branch_scores(prepared)
        return pooled @ self.text_embeddings.T, pack_components(pooled, text_guided, audio_guided)

    def evaluate(self, fold, audio):
        scores, packed = self.evaluate_batch([fold], [audio])
        return scores[0], packed[0]

    def provenance(self):
        info = super().provenance()
        info.update(
            name="PAT-style local alignment on LAION-CLAP",
            not_official_PAT=(
                "Released script targets MS-CLAP 2023 via a non-public wrapper and selects "
                "fusion weights on evaluation labels; this port uses the project's LAION "
                "checkpoint, a preset rule and a clean-data calibrated rule"
            ),
            not_zero_shot=(
                "aligned_calibrated uses labelled clean clips of non-test folds; only "
                "single, ensemble and aligned_fixed use no ESC-50 labels"
            ),
            source=f"https://github.com/cs20s030/PAT/blob/{SOURCE_COMMIT}/inference_zs.py",
            frame_features=(
                "HTSAT fine-grained latents from the official forward pass, exact frame "
                "repetition removed, model audio projection per frame, unit norm"
            ),
            temperatures=list(self.temperatures),
            fixed_rule=FIXED_RULE,
            single_prompts=self.single_prompts,
            prompts=self.prompts,
        )
        return info


def _rank_of_target(scores, class_ids, targets):
    index = np.searchsorted(class_ids, targets)
    target_score = scores[np.arange(len(scores)), index]
    precedes = np.arange(len(class_ids))[None, :] < index[:, None]
    return 1 + np.sum(
        (scores > target_score[:, None]) | ((scores == target_score[:, None]) & precedes),
        axis=1,
    )


def cell_metrics(scores, class_ids, targets, folds, interferers=None, control_scores=None):
    """Fold-mean accuracy and macro-F1, error destinations and target rank for one cell."""
    predictions = class_ids[scores.argmax(1)]
    report = fold_metrics(targets, predictions, folds, class_ids)
    rank = _rank_of_target(scores, class_ids, targets)
    wrong = predictions != targets
    cell = {
        "n": int(len(targets)),
        "accuracy_mean": report["accuracy"]["mean"],
        "accuracy_sd": report["accuracy"]["sample_std"],
        "macro_f1_mean": report["macro_f1"]["mean"],
        "macro_f1_sd": report["macro_f1"]["sample_std"],
        "correct": int((~wrong).sum()),
        "errors": int(wrong.sum()),
        "target_in_top2": int((rank <= 2).sum()),
        "median_target_rank_on_errors": float(np.median(rank[wrong])) if wrong.any() else None,
    }
    if control_scores is not None:
        cell["control_correct"] = int(np.sum(class_ids[control_scores.argmax(1)] == targets))
    if interferers is not None:
        captured = wrong & (predictions == interferers)
        cell.update(
            captures=int(captured.sum()),
            other_errors=int((wrong & ~captured).sum()),
            capture_rate=float(captured.sum() / wrong.sum()) if wrong.any() else None,
            capture_fraction=float(captured.mean()),
        )
    return cell, predictions


def paired_change(reference, candidate, targets, interferers=None):
    """Candidate against reference on the same rows: exact McNemar test and differences."""
    lost = int(np.sum((reference == targets) & (candidate != targets)))
    gained = int(np.sum((reference != targets) & (candidate == targets)))
    result = {
        "lost": lost,
        "gained": gained,
        "accuracy_difference_pp": float(100 * (gained - lost) / len(targets)),
        "exact_mcnemar_p": float(binomtest(lost, lost + gained, 0.5).pvalue)
        if lost + gained
        else 1.0,
    }
    if interferers is not None:
        result["capture_difference"] = int(
            np.sum(candidate == interferers) - np.sum(reference == interferers)
        )
    return result


def _all_readouts(base, pooled, text_guided, audio_guided, folds, single_text, rules):
    return {
        "single": pooled @ single_text.T,
        "ensemble": base,
        "aligned_fixed": fused_scores(base, text_guided, audio_guided, rules["fixed"]),
        "aligned_calibrated": apply_selection(
            base, text_guided, audio_guided, folds, rules["calibrated"]
        ),
    }


def _write_rows(path, rows):
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def analyse(run_dir, regions=("own", "target")):
    """Fuse cached branch scores; write selection, condition tables and fused scores.

    Only clean calibration clips choose the calibrated rule, and one rule per outer fold
    serves every condition, level and power convention. Nothing here re-encodes audio.
    """
    start = perf_counter()
    run_dir = Path(run_dir)
    with np.load(run_dir / "text_embeddings.npz") as f:
        single_text, temperatures = f["single"], tuple(f["temperatures"].tolist())
    with np.load(run_dir / "clean_sources.npz") as f:
        source_names, source_ids = f["filenames"], f["source_ids"]
    with np.load(run_dir / "clean" / "decision_scores.npz") as f:
        clean = {
            k: f[k] for k in ("scores", "embeddings", "folds", "targets", "class_ids", "filenames")
        }
    if not np.array_equal(clean["filenames"], source_names):
        raise ValueError("Clean clips and source identifiers are in different orders")
    classes = len(single_text)
    class_ids, targets, folds = clean["class_ids"], clean["targets"], clean["folds"]
    grid = fusion_grid(temperatures)
    pooled, text_guided, audio_guided = unpack_components(
        clean["embeddings"], classes, temperatures
    )
    selection = select_cross_fold(
        clean["scores"], text_guided, audio_guided, targets, folds, source_ids, class_ids, grid
    )
    rules = {"fixed": fixed_choice(temperatures), "calibrated": selection}
    selection_seconds = perf_counter() - start
    write_json(
        run_dir / "selection.json",
        {
            "fixed_rule": FIXED_RULE,
            "calibration_criterion": (
                "top-1 correct count on clean clips outside the test fold, excluding "
                "sources shared with it; first maximum in simplest-first order"
            ),
            "one_rule_per_fold_for_all_conditions_levels_and_power_conventions": True,
            "temperatures": list(temperatures),
            "betas": sorted({g[1] for g in grid}),
            "candidates": len(grid),
            "selection_seconds": selection_seconds,
            "folds": {
                str(fold): {**chosen, "choice": describe_choice(chosen["choice"], temperatures)}
                for fold, chosen in selection.items()
            },
        },
    )
    rows, reference_rows, paired = [], [], {}

    def add_cell(region, condition, level, arrays, interferers=None, controls=None):
        base, emb, guided_text, guided_audio, row_folds, row_targets = arrays
        scores = _all_readouts(base, emb, guided_text, guided_audio, row_folds, single_text, rules)
        control = None if controls is None else _all_readouts(*controls, single_text, rules)
        predictions = {}
        for name in READOUTS:
            cell, predictions[name] = cell_metrics(
                scores[name],
                class_ids,
                row_targets,
                row_folds,
                interferers,
                None if control is None else control[name],
            )
            rows.append(
                {
                    "region": region,
                    "condition": condition,
                    "level_db": level,
                    "readout": name,
                    "score": SCORE_LABELS[name],
                    **cell,
                }
            )
        key = region if region == "clean" else f"{region}/{condition}@{level:+d}"
        paired[key] = {
            name: paired_change(
                predictions["ensemble"], predictions[name], row_targets, interferers
            )
            for name in ("aligned_fixed", "aligned_calibrated")
        }
        correct = grid_correct(base, guided_text, guided_audio, row_targets, class_ids, grid)
        best = int(np.argmax(correct))
        reference_rows.append(
            {
                "region": region,
                "condition": condition,
                "level_db": level,
                "label": "optimistic reference: best grid point on the evaluated labels",
                "n": int(len(row_targets)),
                "correct": int(correct[best]),
                "ensemble_correct": int(correct[0]),
                **describe_choice(grid[best], temperatures),
            }
        )
        return scores, control

    add_cell(
        "clean", "clean", "", (clean["scores"], pooled, text_guided, audio_guided, folds, targets)
    )
    for region in regions:
        directory = run_dir / f"eval_{region}"
        if not directory.is_dir():
            continue
        records = read_predictions(directory / "clap_local_predictions.csv")
        keys = ("scores", "control_scores", "representation", "control_representation")
        with np.load(directory / "clap_local_scores.npz") as f:
            arrays = {k: f[k] for k in keys + ("row", "class_ids")}
        if not np.array_equal(arrays["row"], [int(r["row"]) for r in records]):
            raise ValueError("Prediction and score rows differ")
        if not np.array_equal(arrays["class_ids"], class_ids):
            raise ValueError("Class order differs between clean and interference runs")
        mixture = unpack_components(arrays["representation"], classes, temperatures)
        control = unpack_components(arrays["control_representation"], classes, temperatures)
        row_folds = np.array([r["fold"] for r in records])
        row_targets = np.array([r["target_class"] for r in records])
        row_interferers = np.array(
            [-1 if r["interferer_class"] is None else r["interferer_class"] for r in records]
        )
        cell_keys = np.array([f"{r['condition']}@{r['level_db']:+.0f}" for r in records])
        fused = {
            f"{kind}_{name}": np.full(arrays["scores"].shape, np.nan, dtype=np.float32)
            for kind in ("mixture", "control")
            for name in ("aligned_fixed", "aligned_calibrated")
        }
        for key in sorted(set(cell_keys.tolist())):
            mask = cell_keys == key
            condition, level = key.split("@")
            scores, control_scores = add_cell(
                region,
                condition,
                int(level),
                (
                    arrays["scores"][mask],
                    *(a[mask] for a in mixture),
                    row_folds[mask],
                    row_targets[mask],
                ),
                row_interferers[mask] if condition.startswith("semantic") else None,
                (arrays["control_scores"][mask], *(a[mask] for a in control), row_folds[mask]),
            )
            for name in ("aligned_fixed", "aligned_calibrated"):
                fused[f"mixture_{name}"][mask] = scores[name]
                fused[f"control_{name}"][mask] = control_scores[name]
        np.savez_compressed(
            directory / "fused_scores.npz",
            row=arrays["row"],
            class_ids=class_ids,
            score_type=np.array("fused_score (sum of branch scores; not a cosine similarity)"),
            **fused,
        )
    _write_rows(run_dir / "conditions.csv", rows)
    _write_rows(run_dir / "label_selected_reference.csv", reference_rows)
    primary = [
        paired.get(f"own/{condition}@+5", {}).get(readout)
        for condition in ("semantic_same", "semantic_different")
        for readout in ("aligned_fixed", "aligned_calibrated")
    ]
    adjusted = holm_adjust([r["exact_mcnemar_p"] if r is not None else 1.0 for r in primary])
    for result, corrected in zip(primary, adjusted):
        if result is not None:
            result["primary_holm_p"] = float(corrected)
    summary = {
        "name": "PAT-style local alignment on LAION-CLAP",
        "readouts": SCORE_LABELS,
        "primary_contrast": (
            "aligned_fixed and aligned_calibrated against ensemble at own/+5 dB, "
            "semantic_same and semantic_different"
        ),
        "paired_against_ensemble": paired,
        "primary_test_family": {
            "tests": 4,
            "observed_tests": sum(r is not None for r in primary),
            "correction": "Holm across two readouts and two semantic conditions at own/+5 dB",
            "dependence_caveat": (
                "McNemar assumes independent target trials; shared source recordings "
                "and clean calibration introduce dependence. Report source-cluster "
                "intervals separately and condition interpretations on the fitted rules."
            ),
        },
        "label_selected_reference": (
            "label_selected_reference.csv is the best point of the searched grid on the "
            "evaluated labels; it is not a method and is excluded from conditions.csv"
        ),
        "cells": len(rows),
        "analysis_seconds": perf_counter() - start,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def verify_reference(system, data_root, single_clean, ensemble_clean=None, clips=None):
    """Check the new extraction path against cached pooled embeddings and baseline scores.

    ``single_clean`` is an existing ``evaluate-clap-clean`` directory and ``ensemble_clean``
    the clean directory of the fixed prompt-ensemble run. ``clips`` limits the check to
    evenly spaced clips for a preflight before the full encode.
    """
    with np.load(Path(single_clean) / "decision_scores.npz") as f:
        reference = {k: f[k] for k in ("scores", "embeddings", "text_embeddings", "filenames")}
    metadata = load_metadata(data_root)
    if [c.filename for c in metadata] != reference["filenames"].tolist():
        raise ValueError("Reference clips differ from the dataset metadata")
    index = np.arange(len(metadata))
    if clips is not None:
        index = np.unique(np.linspace(0, len(metadata) - 1, clips).round().astype(int))
    audio = [
        load_audio(Path(data_root) / "audio" / metadata[i].filename, FeatureConfig(), True)
        for i in index
    ]
    pooled = np.concatenate(
        [
            system.branch_scores(
                np.stack([system.preprocess(a) for a in audio[i : i + 8]]),
                system.temperatures[:1],
            )[0]
            for i in range(0, len(audio), 8)
        ]
    )
    single = pooled @ system.single_text_embeddings.T
    report = {
        "clips_checked": int(len(index)),
        "tolerance": REFERENCE_TOLERANCE,
        "single_text_max_abs_difference": float(
            np.max(np.abs(system.single_text_embeddings - reference["text_embeddings"]))
        ),
        "pooled_embedding_max_abs_difference": float(
            np.max(np.abs(pooled - reference["embeddings"][index]))
        ),
        "single_score_max_abs_difference": float(
            np.max(np.abs(single - reference["scores"][index]))
        ),
        "single_predictions_identical": bool(
            np.array_equal(single.argmax(1), reference["scores"][index].argmax(1))
        ),
    }
    checks = [v for k, v in report.items() if k.endswith("max_abs_difference")]
    if ensemble_clean is not None:
        with np.load(Path(ensemble_clean) / "decision_scores.npz") as f:
            expected = f["scores"][index]
        observed = pooled @ system.text_embeddings.T
        report["ensemble_score_max_abs_difference"] = float(np.max(np.abs(observed - expected)))
        report["ensemble_predictions_identical"] = bool(
            np.array_equal(observed.argmax(1), expected.argmax(1))
        )
        checks.append(report["ensemble_score_max_abs_difference"])
    report["passed"] = bool(
        max(checks) <= REFERENCE_TOLERANCE
        and report["single_predictions_identical"]
        and report.get("ensemble_predictions_identical", True)
    )
    return report


def compare_clean_run(run_dir, single_clean, ensemble_clean=None):
    """After the full clean pass: every cached pooled embedding and baseline score agrees."""
    run_dir = Path(run_dir)
    with np.load(run_dir / "text_embeddings.npz") as f:
        single_text, temperatures = f["single"], tuple(f["temperatures"].tolist())
    with np.load(run_dir / "clean" / "decision_scores.npz") as f:
        scores, packed, names = f["scores"], f["embeddings"], f["filenames"]
    pooled = unpack_components(packed, len(single_text), temperatures)[0]
    with np.load(Path(single_clean) / "decision_scores.npz") as f:
        if not np.array_equal(f["filenames"], names):
            raise ValueError("Reference clips are in a different order")
        report = {
            "clips_checked": int(len(names)),
            "tolerance": REFERENCE_TOLERANCE,
            "pooled_embedding_max_abs_difference": float(np.max(np.abs(pooled - f["embeddings"]))),
            "single_predictions_identical": bool(
                np.array_equal((pooled @ single_text.T).argmax(1), f["scores"].argmax(1))
            ),
        }
    passed = (
        report["pooled_embedding_max_abs_difference"] <= REFERENCE_TOLERANCE
        and report["single_predictions_identical"]
    )
    if ensemble_clean is not None:
        with np.load(Path(ensemble_clean) / "decision_scores.npz") as f:
            report["ensemble_score_max_abs_difference"] = float(
                np.max(np.abs(scores - f["scores"]))
            )
            report["ensemble_predictions_identical"] = bool(
                np.array_equal(scores.argmax(1), f["scores"].argmax(1))
            )
        passed = (
            passed
            and report["ensemble_score_max_abs_difference"] <= REFERENCE_TOLERANCE
            and report["ensemble_predictions_identical"]
        )
    report["passed"] = bool(passed)
    return report


def benchmark_alignment(system, audio_path, feature_config=FeatureConfig(), runs=50, warmup=5):
    """Single-clip deployment cost: pooled ensemble readout against one aligned readout."""
    audio = load_audio(audio_path, feature_config)
    choice = fixed_choice(system.temperatures)
    temperature = [system.temperatures[choice[0]]]
    frontend, pooled_ms, aligned_ms = [], [], []
    with threadpool_limits(limits=1):
        for i in range(warmup + runs):
            system.synchronize()
            start = perf_counter()
            prepared = system.preprocess(audio)
            ready = perf_counter()
            system.text_embeddings @ system.embed(prepared)
            system.synchronize()
            middle = perf_counter()
            pooled, text_guided, audio_guided = system.branch_scores(prepared[None], temperature)
            fused_scores(
                pooled @ system.text_embeddings.T,
                np.stack(text_guided, 1),
                np.stack(audio_guided, 1),
                (0, choice[1], choice[2]),
            )
            system.synchronize()
            end = perf_counter()
            if i >= warmup:
                frontend.append(1000 * (ready - start))
                pooled_ms.append(1000 * (middle - ready))
                aligned_ms.append(1000 * (end - middle))
    return {
        "scope": "single_clip_microbenchmark, batch one, one numerical thread",
        "runs": runs,
        "warmup": warmup,
        "frontend_median_ms": float(np.median(frontend)),
        "pooled_ensemble_readout_median_ms": float(np.median(pooled_ms)),
        "aligned_readout_median_ms": float(np.median(aligned_ms)),
        "aligned_includes": (
            "audio tower, frame capture and projection, alignment at one temperature, fusion"
        ),
        "excludes": ["disk I/O", "model loading", "one-off text preparation and calibration"],
        "device": system.device,
        "platform": platform.platform(),
    }


def build_system(checkpoint, templates_path, data_root, device="cpu"):
    from .prompt_ensemble import ensemble_embeddings

    templates_path = Path(templates_path)
    templates = [line.strip() for line in templates_path.read_text().splitlines() if line.strip()]
    class_names = {c.target: c.category for c in load_metadata(data_root)}
    backend = load_laion_clap(checkpoint, device=device)
    start = perf_counter()
    ensemble, _ = ensemble_embeddings(backend, class_names, templates)
    system = LocalAlignmentSystem(backend, class_names, ensemble, checkpoint=checkpoint)
    preparation = {
        "templates_sha256": sha256_file(templates_path),
        "templates_per_class": len(templates),
        "one_off_text_preparation_seconds": perf_counter() - start,
    }
    return system, preparation


def run(
    checkpoint,
    templates_path,
    data_root,
    output,
    manifests,
    single_clean,
    ensemble_clean=None,
    batch_size=8,
    device="cpu",
    limit=None,
    benchmark_audio=None,
):
    """Verify against cached references, encode everything once, then fuse and summarise."""
    from .clap_system import evaluate_clean
    from .interference_eval import evaluate_manifest

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new local-alignment run directory")
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        system, preparation = build_system(checkpoint, templates_path, data_root, device)
        preflight = verify_reference(system, data_root, single_clean, ensemble_clean, clips=50)
        write_json(output / "preflight.json", preflight)
        if not preflight["passed"]:
            raise RuntimeError("Preflight failed: the new path does not reproduce the cache")
        np.savez_compressed(
            output / "text_embeddings.npz",
            ensemble=system.text_embeddings,
            single=system.single_text_embeddings,
            single_prompts=np.array(system.single_prompts),
            temperatures=np.array(system.temperatures),
        )
        metadata = load_metadata(data_root)
        np.savez_compressed(
            output / "clean_sources.npz",
            filenames=np.array([c.filename for c in metadata]),
            source_ids=np.array([c.source_id for c in metadata]),
        )
        write_json(
            output / "provenance.json",
            {**system.provenance(), **preparation, "source_sha256": sha256_file(Path(__file__))},
        )
        evaluate_clean(system, data_root, output / "clean", batch_size=batch_size)
        full_check = compare_clean_run(output, single_clean, ensemble_clean)
        write_json(output / "reference_check.json", full_check)
        if not full_check["passed"]:
            raise RuntimeError("Clean pass does not reproduce the cached reference")
        for region, manifest in manifests.items():
            evaluate_manifest(
                manifest,
                output / f"eval_{region}",
                data_root,
                limit=limit,
                extra_systems=[system],
                batch_size=batch_size,
            )
        if benchmark_audio is not None:
            write_json(output / "cost.json", benchmark_alignment(system, benchmark_audio))
    summary = analyse(output, tuple(manifests))
    write_json(output / "complete.json", {"complete": True, "limit": limit})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        for option in ("checkpoint", "templates", "data-root", "single-clean"):
            command.add_argument(f"--{option}", type=Path, required=True)
        command.add_argument("--ensemble-clean", type=Path)
        command.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--own", type=Path, help="Own-activity manifest directory")
            command.add_argument("--target", type=Path, help="Target-activity manifest")
            command.add_argument("--batch-size", type=int, default=8)
            command.add_argument("--limit", type=int, help="First rows only (smoke test)")
            command.add_argument("--benchmark-audio", type=Path)
        else:
            command.add_argument("--clips", type=int, default=50)
    again = commands.add_parser("analyse", help="Recompute the fusion from cached scores")
    again.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "analyse":
        analyse(args.run_dir)
    elif args.command == "preflight":
        with threadpool_limits(limits=1):
            built, _ = build_system(args.checkpoint, args.templates, args.data_root, args.device)
            result = verify_reference(
                built, args.data_root, args.single_clean, args.ensemble_clean, args.clips
            )
        print(result)
        raise SystemExit(0 if result["passed"] else 1)
    else:
        chosen = {k: v for k, v in (("own", args.own), ("target", args.target)) if v}
        if not chosen:
            parser.error("Provide --own and/or --target")
        run(
            args.checkpoint,
            args.templates,
            args.data_root,
            args.output,
            chosen,
            args.single_clean,
            args.ensemble_clean,
            args.batch_size,
            args.device,
            args.limit,
            args.benchmark_audio,
        )

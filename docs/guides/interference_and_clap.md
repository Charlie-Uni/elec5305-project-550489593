# Interference, frozen CLAP and core figures

[Repository home](../../README.md) · [Documentation index](../README.md) · [Results index](../../results/README.md)

Detailed instructions and dated run records moved from the original README. Run commands from the repository root; use fresh output directories when repeating experiments. Historical measurements and test counts describe the runs recorded below.

## Contents

- [Controlled interference: noise and semantic distractors](#controlled-interference-noise-and-semantic-distractors)
- [Conditions and levels](#conditions-and-levels)
- [Power definition](#power-definition)
- [Manifest fields](#manifest-fields)
- [Commands](#commands)
- [Measured CNN interference result — 9 November 2026](#measured-cnn-interference-result--9-november-2026)
- [V3 power-region comparison — 9 November 2026](#v3-power-region-comparison--9-november-2026)
- [Frozen CLAP (zero-shot reference)](#frozen-clap-zero-shot-reference)
- [Figures](#figures)

## Controlled interference: noise and semantic distractors

The interference pipeline plans every mixture once, in a manifest, and rebuilds each
mixture deterministically whenever a system is evaluated. Nothing is stored as audio
except a few representative examples, and every system receives identical mixtures.

### Conditions and levels

| Condition | Interferer | Ratio | Levels |
| --- | --- | --- | --- |
| `white` | White Gaussian noise, one seeded realisation per target, shared across levels | SNR | +10, +5, 0 dB |
| `demand` | One 5-s segment of a DEMAND environment (`STRAFFIC`, `PSTATION`, `TMETRO`), channel 1, resampled 48 kHz → 44.1 kHz; environments are assigned to targets in a seeded, balanced order | SNR | +10, +5, 0 dB |
| `semantic_same` | Another ESC-50 clip from the same held-out fold and the same coarse category, different fine class and source recording | SIR | +10, +5, 0 dB |
| `semantic_different` | As above, from a different coarse category | SIR | +10, +5, 0 dB |

Pairing is seeded by `(seed, fold)` and uses the whole fold, so a diagnostic subset
receives exactly the distractors it has in the full manifest. Clips whose source
recording crosses official folds are never used as distractors. Target and distractor
keep their natural alignment (zero offset); the fraction of target-active samples at
which the distractor is also active is recorded as `activity_overlap`.

### Power definition

A component's power is its mean square over its **own active region**: the samples of
unwindowed 1,024-sample frames whose energy lies within 40 dB of that component's peak
frame. The interferer is scaled so that the active-power ratio equals the requested
level, and the achieved ratio is recomputed from the scaled components. Stationary
noise is active everywhere, so its active power equals its segment power; a sparse
event (door knock, glass breaking, keyboard typing) is measured only where it sounds.

The proposal described measuring both components on the target's active mask. During
implementation that definition amplified sparse distractors that are quiet inside the
target's region by factors of 20–40, forcing whole-mixture attenuation to ~0.05, so the
default (`"power_region": "own"`) measures the interferer in its own active region.
Because the choice changes how semantic distractors compare with background noise at a
nominal level, the proposal's definition is kept as a control: `"power_region":
"target"` builds a second manifest with the same pairs, scaled so that the interferer
power **inside the target's active region** meets the level (pairs with no distractor
energy there are dropped and listed in `manifest_meta.json`). Every row of either
manifest records both readings (`interferer_power`, `interferer_power_on_target_mask`)
and the activity overlap, so results can also be stratified by overlap. Both
definitions keep the 40 dB threshold and zero-offset alignment.

These definitions are not equivalent. The default own-region ratio does not guarantee
that interference types have equal power in the target's active region; the target-region
control enforces that equality. The common-target-region ratio can be recovered as
`10*log10(target_power/interferer_power_on_target_mask)` when the denominator is positive.
Interpret performance alongside this ratio and activity overlap, and treat this as a
documented protocol change rather than evidence isolating semantic meaning alone.
Row seeds depend on (seed, target, condition) only, so one white-noise realisation or
DEMAND segment serves all three levels of a target, as the semantic clip pairs do. (The
earlier 9 November CNN run used a rule in which noise draws varied with level; semantic
pairs and gains were identical.)

Mixtures are never hard clipped. When a mixture's peak exceeds 0.99 the whole mixture
receives one common gain (`common_gain`, which leaves the ratio unchanged). For every
row the **gain-matched clean control** (target × `common_gain`) is classified by the
same model, so level effects can be separated from interference effects.

### Manifest fields

`row, condition, ratio_type, level_db, fold, target_file, target_class, target_category,
target_coarse, interferer, interferer_class, interferer_category, interferer_coarse,
offset_samples, row_seed, interferer_gain, common_gain, target_power, interferer_power,
achieved_db, interferer_power_on_target_mask, target_active_fraction,
interferer_active_fraction, activity_overlap, peak_before_common_gain, target_sha256,
interferer_sha256`. `manifest_meta.json` records the configuration, rules, DEMAND
provenance and source hashes. Row seeds depend on (seed, target, condition) only.

### Commands

```bash
# DEMAND: download STRAFFIC_48k.zip, PSTATION_48k.zip and TMETRO_48k.zip from the
# official record, unzip to data/DEMAND/<ENV>/chNN.wav, then cache channel 1:
esc-baseline prepare-demand --demand-root data/DEMAND --output artifacts/noise/demand --config configs/interference.json
esc-baseline build-manifest --data-root data/ESC-50 --config configs/interference.json --demand-cache artifacts/noise/demand --output artifacts/interference/manifest
esc-baseline render-mixtures --manifest artifacts/interference/manifest --output artifacts/interference/examples --per-condition 2
esc-baseline evaluate-interference --manifest artifacts/interference/manifest --svm-dir artifacts/baselines/clean_svm --cnn-dir artifacts/baselines/clean_cnn --output artifacts/interference/eval_svm_cnn
```

`build-manifest` accepts `--limit-per-fold N` for diagnostic subsets; the full manifest
holds up to 2,000 targets × 4 conditions × 3 levels = 24,000 rows. The target-region
control excludes rows with exactly zero interferer energy on the target mask and
records those exclusions. Clean-reference losses use the targets retained in each
condition; cross-protocol comparisons additionally use their intersection. `evaluate-interference`
renders each row, classifies the mixture and its gain-matched clean control with the
model whose test fold contains the target, and writes `<system>_predictions.csv`
(prediction, correctness, whether the prediction equals the distractor class, target and
distractor scores and their margin) plus `<system>_scores.npz` (all class scores, the
92-dimensional handcrafted vector or 256-dimensional CNN embedding for every mixture
and control). `summary.json` tabulates accuracy, control accuracy, distractor capture
rate (`errors predicted as the distractor / errors`) and the unconditional capture
fraction per condition and level. CNN evaluation runs on the CPU with one thread.

Evaluation verifies audio/cache checksums and records model hashes and progress. Noise
conditions have no distractor label, so capture metrics are undefined for those rows.
`summarise-interference` reports loss relative to both the original clean baseline and
each trial's gain-matched control, plus transitions from correct controls to errors.
The uniform-wrong-class reference is explicitly hypothetical, not a measured random
baseline. The known cross-fold source overlap caveat still applies to target recordings.

### Measured CNN interference result — 9 November 2026

The original five clean-trained CNN checkpoints were evaluated on 2,000 targets,
four interference types and three levels (24,000 mixtures plus gain-matched controls).
All values below are percentages and five-fold means; each cell lists +10 / +5 / 0 dB.
The clean reference is 67.90% accuracy and 66.61% macro-F1. These runs use the
**separate-active-region power convention** documented above.

| Condition | Accuracy: +10 / +5 / 0 dB | Macro-F1: +10 / +5 / 0 dB |
| --- | ---: | ---: |
| White Gaussian noise | 18.55 / 12.30 / 7.00 | 13.67 / 7.83 / 3.36 |
| DEMAND background | 40.35 / 31.60 / 22.20 | 37.88 / 29.36 / 19.95 |
| Same-category event | 43.70 / 35.55 / 27.35 | 40.92 / 32.60 / 24.44 |
| Different-category event | 41.55 / 34.25 / 26.35 | 39.15 / 32.00 / 24.71 |

[Full table](../../results/interference_cnn_20261109/conditions.csv),
[verification](../../results/interference_cnn_20261109/verification.json), and
[power audit](../../results/interference_cnn_20261109/power_definition_audit.json) record the
results. All 24,000 predictions, margins and score/embedding shapes were verified;
five-fold accuracy, macro-F1 and control metrics were independently recomputed.
An exact 120-row manifest comparison confirms that integration checks preserve the
incoming patch's numerical mixing rules. That integration passed 61 tests, with one CUDA test skipped on this host; the
current v3 integration adds the power-region and matched-reference regression checks.

The power convention affects interpretation. At nominal +5 dB, the 95th percentile
of the ratio measured on the target mask is about +12.5 to +12.8 dB for semantic
mixtures. A supplementary diagnostic retains identical targets across all four
conditions, requiring target-mask ratios within 1 dB of nominal and both semantic
pairs to overlap at least 80% of the target-active region. At +5 dB this selects
826 targets: macro-F1 is 29.60% for DEMAND, 25.95% for same-category events and
26.90% for different-category events. This reverses the full-set DEMAND/semantic
ordering. The selected subset has a different class/difficulty distribution and
is exploratory; it is not a replacement for the full benchmark or proof isolating
semantic causation. See [sensitivity results](../../results/interference_cnn_20261109/matched_target_sensitivity.json).

Gain-matched CNN controls differ from the clean macro-F1 by -1.14 to +0.40 percentage
points across conditions. Control-relative loss and newly induced error transitions
are retained in the tables. White noise remains the most damaging condition in both
the full run and the selected-subset diagnostic. The supplied cloud SVM CSV contains
aggregates only; it was checked arithmetically, not independently reproduced from
cloud trial records. This run does not include frozen CLAP or SVM under DEMAND.

### V3 power-region comparison — 9 November 2026

The v3 protocol fixes the noise realization across levels and compares `own` (each
source's active region) with `target` (both powers measured on the target's active
region). The same SVM/CNN models receive the same manifest under each convention.
The own-region run has 24,000 rows per model; the target-region run has 23,991.
Three target–distractor pairs have exactly zero interferer energy on the target mask,
so their nine level-specific rows are excluded from the target-region run.

The table below uses the **intersection of retained targets within each condition**
for both power conventions. Values are five-fold mean macro-F1 percentages at +5 dB.
White/DEMAND each retain 2,000 targets; same-category events retain 1,998 and
different-category events retain 1,999. Full clean references are 47.96% for SVM
and 66.61% for CNN. Condition-specific matched clean references are in the CSV.

| Condition | SVM own | SVM target | CNN own | CNN target |
| --- | ---: | ---: | ---: | ---: |
| White noise | 2.03 | 2.03 | 7.76 | 7.76 |
| DEMAND | 16.13 | 16.15 | 29.46 | 29.33 |
| Same-category event | 22.49 | 21.14 | 32.54 | 31.70 |
| Different-category event | 21.36 | 20.87 | 32.01 | 30.93 |

[All levels, matched protocols](../../results/interference_v3_20261109/matched_protocol_conditions.csv),
[full-run tables](../../results/interference_v3_20261109/conditions.csv),
[overlap strata](../../results/interference_v3_20261109/overlap_strata.csv), and
[independent verification](../../results/interference_v3_20261109/verification.json) are retained.

Changing power definition modestly reduces semantic-condition F1, but does not reverse
the overall CNN semantic/DEMAND ordering. In the high-overlap stratum (overlap >= 0.8),
the paired comparison does reverse: at +5 dB, own-region CNN F1 is 24.42% for same-category
events versus 29.28% for DEMAND on those same targets, and 25.70% versus 29.07% for
different-category events. Under the target-region convention, the respective comparisons
are 24.19% versus 29.02%, and 25.84% versus 28.93%. These are descriptive within-stratum
comparisons, not evidence isolating semantic meaning from all acoustic differences.

The [accuracy decomposition](../../results/interference_v3_20261109/accuracy_decomposition.csv)
uses pooled accuracy on matched targets, which is additive across strata. For CNN at
+5 dB under the own-region convention, same-category low/middle/high-overlap strata
contribute +4.65 / +1.70 / -2.40 percentage points to the overall +3.95-point semantic
advantage over DEMAND. Low overlap is the largest positive contribution, but the middle
stratum also contributes. Macro-F1 itself is not an additive weighted average of strata.

Stratum macro-F1 retains all 50 labels, including labels absent from individual folds.
For example, same-category low-overlap folds contain only 26–34 classes. Their clean
macro-F1 of 40.81% should not be read as proof of greater baseline difficulty: clean
accuracy on those matched targets is 69.91%, versus 67.73% in the high-overlap group.
The output records class coverage and clean accuracy alongside all paired contrasts.

The local SVM was reconstructed from the original feature cache and recorded search
protocol because four original model files were iCloud placeholders. All 2,000 clean
predictions match the recorded baseline; the readable original fold-5 model's scaler
and SVM parameters match exactly. The historical cloud semantic aggregates match in
accuracy and capture rate; same-category 0 dB macro-F1 differs by 0.0031 percentage
points, and margin medians differ by at most 4.37e-7. These differences are retained in
[the historical comparison](../../results/interference_v3_20261109/historical_svm_comparison.csv);
cloud aggregate data alone cannot identify their per-trial cause. Current SVM/CNN
comparisons use the new local paired runs. All 12,000 own-region semantic CNN predictions
and reported scores match the earlier CNN run exactly.

The v3 integration passes 63 tests, with one CUDA test skipped. A regression test covers
matching each condition's clean baseline after exclusions. Independent verification
checks every row's scores, predictions, margins, finite representations and five-fold
metrics. Separate serial references cover all folds, conditions and levels: 60 mixtures
per power convention, with exact score, embedding and control equality for both models.
Evaluation used the unchanged evaluator on official-fold shards, then restored canonical
manifest order; its elapsed time is not a deployment-latency benchmark.

### Frozen CLAP (zero-shot reference)

`src/esc_robustness/clap_system.py` wraps the official `laion_clap` module as a third
system with the same interface as the SVM and CNN: fixed checkpoint
(`630k-audioset-best.pt`, HTSAT-tiny + RoBERTa, no fusion), 44.1 kHz → 48 kHz
resampling, the official int16 round trip, one prompt per class from the template
`the sound of {label}` (ESC-50 category names with underscores replaced by spaces), and
cosine similarity between the unit audio embedding and the 50 unit prompt embeddings
as the class score. CLAP is fold independent and is never trained or tuned. The
`laion_clap` package and the checkpoint are external research code and weights; they
use a **separate Python 3.11 environment**. LAION-CLAP requires NumPy 1.x; installing
it over the NumPy 2.x SVM/CNN environment would change the baseline runtime.
Keep `uv.lock` and the baseline environment unchanged. The tested Apple Silicon
dependencies are in `requirements-clap-macos.txt`; `configs/clap_reference.json`
records the fixed model and download URL. The loader checks the full SHA-256
before loading model weights. Tokenizer and RoBERTa assets are downloaded on first
use and can subsequently be used offline.

```bash
python3.11 -m venv /path/outside/icloud/clap-venv
/path/outside/icloud/clap-venv/bin/pip install -r requirements-clap-macos.txt
/path/outside/icloud/clap-venv/bin/pip install --no-deps .
```

Run the following commands with that environment's `esc-baseline`. Keep the
checkpoint, dataset, caches and long-running outputs outside iCloud-managed
directories, or explicitly retain them locally.

```bash
esc-baseline evaluate-clap-clean --data-root data/ESC-50 --clap-checkpoint models/clap/630k-audioset-best.pt --output artifacts/baselines/clean_clap
esc-baseline evaluate-interference --manifest artifacts/interference/manifest --clap-checkpoint models/clap/630k-audioset-best.pt --output artifacts/interference/eval_clap
esc-baseline summarise-interference --evaluation artifacts/interference/eval_clap --clap-clean artifacts/baselines/clean_clap
esc-baseline benchmark-clap --data-root data/ESC-50 --clap-checkpoint models/clap/630k-audioset-best.pt --audio data/ESC-50/audio/1-100032-A-0.wav --output artifacts/baselines/clean_clap/cost.json
```

`evaluate-clap-clean` writes the clean zero-shot reference in the baseline format
(predictions.csv, decision_scores.npz with audio and text embeddings, summary.json with
checkpoint hash and prompts). The interference run stores the 512-dimensional audio
embedding of every mixture and control for the displacement analysis. The CLAP cost
benchmark excludes prompt embedding (computed once) and reports preprocessing and
embedding + scoring separately. Resampling and int16 conversion count as
preprocessing; CLAP's internal Mel frontend remains part of embedding time. This
boundary differs from the handcrafted/CNN frontend boundary, so compare total
preprocessing-plus-inference cost as well as each component. Checkpoint file size
is reported separately from parameter count or resident memory.

`--clap-device cpu|mps|cuda` selects the inference device (default CPU).
`--batch-size N` batches clean clips or N interference rows plus their N gain-matched
controls. It does not change the pairing, prompts or model. A real four-clip check
gave the same top-1 predictions on CPU and MPS, with maximum cosine-score difference
below 4e-7. MPS was slower in this small local check; it is not assumed to accelerate
the full experiment. Cost comparisons should explicitly use CPU and batch size one.

The int16 conversion truncates toward zero exactly once. The backend uses the
official tensor entry point after conversion; the NumPy entry point would quantise
again. A real-model check matched the official single-quantisation path exactly.
The renderer validates recalculated gains, then uses the manifest's exact stored
gains. This avoids last-bit differences between NumPy 1.x and 2.x: all 240 sampled
mixtures and controls matched their original baseline waveform hashes.

The same checkpoint benefits from external pretraining for all five folds. A frozen
evaluation does not establish that ESC-50 recordings were absent from pretraining;
this is a comparison of deployment behaviour, not equal-data training efficiency.

The fixed-protocol clean run on 9 November 2026 evaluated all 2,000 clips on CPU
with batch size eight, without training, prompt selection or checkpoint selection.
Mean and sample SD across the five official folds are **81.75 ± 2.58% accuracy**
and **79.49 ± 3.28% macro-F1**. These are measured results for this project's
predeclared prompt and frontend, rather than a claim to reproduce a different
published CLAP setup. Audio/text cosine scores, predictions and per-fold metrics
were independently recomputed from the saved embeddings and confusion counts.
See [clean summary](../../results/clap_integration_20261109/clean/summary.json),
[predictions](../../results/clap_integration_20261109/clean/predictions.csv) and
[audit](../../results/clap_integration_20261109/clean_audit.json).

Real CLI integration checks also completed the first 24 rows of each `own` and
`target` manifest, covering all four interference types and all three levels.
The `own` check ran SVM, CNN and CLAP together; the sampled SVM/CNN predictions and
target scores exactly matched v3. Both checks wrote representations, controls,
capture/margin data and summary tables. These small checks validate the pipeline;
they are **not full CLAP interference results**. The subsequent full evaluations
completed on 9 November with 24,000 `own` rows and 23,991 `target` rows, using the
existing v3 manifests. Every trial, class score, control and condition metric was
independently checked for all three systems. Reconstructed CLAP cosine scores
agree with the saved scores to below 1e-10 absolute error.

At +5 dB, CLAP's five-fold mean macro-F1 is:

| Interference | Own active regions | Target active region |
| --- | ---: | ---: |
| White noise | 62.91% | 62.92% |
| DEMAND background | 71.49% | 71.64% |
| Same-category event | 48.86% | 47.76% |
| Different-category event | 45.16% | 42.81% |

These conditions use their retained targets; exact paired-cohort protocol
comparisons, all three systems, capture rates and control-relative losses are in
the [full result package](../../results/clap_full_20261109/README.md). The target protocol
omits only the nine previously identified zero-energy rows. At +5 dB, CLAP's
capture rate among errors is 66.95% / 60.30% for same-/different-category events
under `own`, and 68.74% / 61.27% under `target`. This describes where its errors go,
not a claim that it makes more errors overall than the supervised systems.

### Figures

`make-figures` draws the report figures from an evaluation directory and the clean
references, and writes the numbers behind each figure as CSV next to 600-DPI PNG,
SVG, PDF and grayscale PNG exports. The original own-region SVM/CNN figures were
checked against the supplied preview data. Complete three-system figures are now
available for [own](../../results/figures_three_systems_own_20261109/README.md) and
[target](../../results/figures_three_systems_target_20261109/README.md), with numerical,
layout, grayscale and embedded-font checks.

```bash
uv sync --locked --no-editable --extra dev --extra docs --extra cnn --extra figures
esc-baseline make-figures --evaluation artifacts/interference/v3_20261109/eval_own --output results/figures_svm_cnn_own_20261109 \
  --svm-clean results/clean_svm --cnn-clean results/clean_cnn \
  --svm-cost results/clean_svm/cost.json --cnn-cost results/clean_cnn/cost.json --level 5
```

| File | Content |
| --- | --- |
| `fig1_macro_f1_vs_level` | Macro-F1 (5-fold mean ± SD), clean as a separate control and +10/+5/0 dB connected, one panel per interference type |
| `fig2_capture_rate_vs_sir` | Errors predicted as the distractor versus SIR; dashed = same category, solid = different; dotted line = uniform wrong-class reference |
| `fig3_margin_vs_sir` | Target − distractor score per system on its own scale (OVR decision values, logits, cosines); boxes = quartiles |
| `fig4_same_vs_different` | Relative macro-F1 loss on identical same/different-condition targets; five individual fold values shown beside the aggregate markers |
| `fig5_representation_displacement` | Mixture versus gain-matched control: handcrafted RMS feature change standardised by the evaluated model's training-fold scales, or embedding cosine distance; split by outcome; separate scales |
| `fig6_confusion_flow` | Most frequent target → distractor captures, with counts over pairs; CSV also records captures that began with a correctly classified control |
| `fig7_robustness_vs_cost` | Macro-F1 under a semantic condition against single-clip CPU time (log axis); hollow marker = clean |
| `fig8_overlap_strata` | Accuracy under semantic interference versus DEMAND on identical targets within each overlap stratum; individual folds shown; CSV retains macro-F1 and class coverage |

Systems are read from the evaluation summary. CLAP runs separately in its own
environment, so append `--additional-evaluation /path/to/clap/eval_own`,
`--clap-clean /path/to/clap/clean` and `--clap-cost /path/to/clap/clean/cost.json`
to draw all three systems. The loader verifies manifest hashes and ordered trial
identities before combining them; it also rejects prediction/embedding row
misalignment. No copying or overwriting of v3 results is required.

Use `--svm-models /path/to/clean_svm_reconstructed` if model files have moved.
The saved model hashes are checked before using their training scalers; held-out
controls are never used to refit feature scaling. CLAP full evaluation completed
on 9 November using a frozen v4 source snapshot, independent of the figure edits.

Colours remain fixed per system (blue SVM, orange CNN, green CLAP), with marker,
line-style or fill/hatch cues. Long captions were shortened, panels labelled, and
PNG previews checked for clipping and overlap before vector export. Supplementary
CSVs contain fold losses, overlap-matched trials and paired representation shifts.
The cost table distinguishes serialized model bytes, checkpoint bytes and parameter
bytes, and validates CPU, batch-size-one and thread settings.

The CNN high-overlap result is supported on matched targets: at +5 dB, semantic
interference has smaller median embedding displacement but lower accuracy than
DEMAND. This shows that displacement magnitude alone is insufficient to explain
failure; it does not establish a causal mechanism. SVM does not show the same
high-overlap accuracy reversal. The two CNN 6/6 capture examples are exploratory:
only 4/6 can-opening controls and 1/6 water-drop controls were initially correct.
See [figure notes and audit](../../results/figures_svm_cnn_own_20261109/README.md).
CLAP's high-overlap semantic mixtures instead show both lower accuracy and larger
median embedding displacement than matched DEMAND at +5 dB. Interpret each
representation separately; the full three-system tables retain these contrasts
under both power definitions.

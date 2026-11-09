# When Background Noise Has Meaning

**Environmental Sound Classification under Semantic Interference**

ELEC5305 project · Qihan Zhu · SID 550489593

[Revised project proposal (PDF)](./ELEC5305_Project_proposal_550489593.pdf)

[Research report draft](results/research_report_20261109_draft/research_report.pdf)
· [10 November review against all 80 instructor recommendations](docs/teacher_review_20261110.md)
· [Interactive project page](index.html)
· [Attributed listening examples](examples/listening_own_20261109/README.md)
· [Video recording script](docs/video_narration_draft.txt)
· [Editable presentation draft](results/presentation_20261109_draft/research_presentation.pptx)
· [Exact evaluated mixture manifests](results/mixture_manifests_20261109/README.md)

The core study and expanded interventions are complete and independently checked:
14 variants, both power protocols, all three CNN seeds, 336 condition cells and
156 paired intervention contrasts. The report, figures and editable presentation
are prepared for author review. The recording script is preparation material,
not a finished video submission.

## Research question

At matched interference power, how do handcrafted features, a supervised log-Mel
CNN and a frozen audio-language model respond to ordinary noise versus another
recognisable sound event? When classification fails, does the model predict the
interfering event instead of the intended target?

For example, **dog bark + white noise** and **dog bark + siren** can have the same
target-to-interference power ratio while producing different errors. The project
studies those failure patterns and their computational cost.

## Experimental scope

| Component | Core design |
| --- | --- |
| Dataset | ESC-50, all 50 classes and five official folds |
| Classical system | Handcrafted temporal/spectral/MFCC features + SVM |
| Supervised system | 64-band log-Mel spectrogram + compact CNN |
| Pretrained reference | Frozen zero-shot LAION-CLAP; no fine-tuning |
| Acoustic controls | White Gaussian noise and held-out DEMAND environments |
| Semantic interference | A different ESC-50 class; one same-broad-category and one different-broad-category distractor per target |
| Levels | Clean, +10 dB, +5 dB and 0 dB; SNR for noise, SIR for event mixtures |
| Primary outcomes | Accuracy, macro-F1, relative degradation, distractor capture rate and target-distractor score margin |
| Further analysis | Within-model representation displacement, confusion patterns and deployment cost |

At 0 dB, both events have equal power under the chosen measurement convention.
This is a stress test with an ambiguous single target label. Equal power does not
control all spectral, temporal or perceptual differences between interference types.
DEMAND recordings may themselves contain recognisable events. The semantic
condition supplies an explicitly labelled competing ESC-50 class; it is not a
causal isolation of meaning from acoustic structure.

### Controls and reproducibility

- Outer test fold `k` uses validation fold `(k % 5) + 1`; the other three folds
  train candidate supervised models. Select on validation macro-F1, then refit on
  all four non-test folds. Fit all learned normalisation on training data only.
- Semantic test distractors come from the same held-out outer fold, with a
  different fine class and source recording. Keep each pair fixed across levels
  and models. The initial alignment is zero offset.
- Evaluate two explicit power conventions: `own` measures each component on its
  own active samples; `target` measures both components on the target's active
  samples, as originally proposed. Activity uses frames within 40 dB of each
  relevant signal's peak frame energy. The default `own` convention avoids very
  large gains for sparse distractors that barely overlap the target. Neither
  convention removes differences in temporal overlap, so report both and stratify
  by overlap. Verify achieved ratios numerically and record gains, peak control,
  offsets, labels, categories, checksums, folds and seeds in the manifest.
- Never hard-clip mixtures. Use identical canonical mixtures for every system.
  Retain gain-matched clean controls when common attenuation is needed.
- CLAP uses its official frontend, including 48 kHz input resampling, a fixed
  checkpoint and the fixed prompt `the sound of [class]`. Record checkpoint
  provenance and possible pretraining overlap; frozen inference does not establish
  that ESC-50 was absent from external pretraining.
- The core CNN uses seed 5305. The completed extension evaluates seeds 5305–5307
  and separate white-noise and semantic-mixture training, with clean validation
  and training-partition-only distractors. See the [extension protocol](docs/extension_protocol_20261109.md).

**Distractor capture rate** is `predictions equal to distractor / incorrect
predictions`, with an undefined value when there are no errors. Also report the
unconditional capture fraction. **Margin** is `target score - distractor score`;
analyse it within each model because score scales differ. A negative margin alone
does not establish that the distractor is the overall top prediction.

## Implementation milestones

1. **Implemented and evaluated:** clean SVM baseline, official folds, validated custom STFT, 92-dimensional
   handcrafted features, training-only scaling and out-of-fold predictions.
2. **Implemented and evaluated:** compact CNN on the same folds, with validation-based duration selection.
3. **Implemented:** shared white-noise and DEMAND mixing with achieved-SNR checks and a
   reproducible manifest (`build-manifest`, see below).
4. **Implemented:** reproducible semantic pairing and SIR-controlled mixtures in the same
   manifest; `evaluate-interference` classifies every mixture with the fold-matched SVM and
   CNN. Measured interference results and power-definition controls are reported below.
5. **Implemented and evaluated:** frozen CLAP inference on all clean clips and both
   full interference manifests (24,000 own-region / 23,991 target-region mixtures),
   cached audio/text embeddings, class scores and a separate cost benchmark.
6. **Completed:** three-system robustness, semantic capture, representation and
   computational-cost analysis, with eight audited figures for each power protocol.
   See [full results and verification](results/clap_full_20261109/README.md).
7. **Expanded study:** three-seed clean CNN replication and a fixed CLAP prompt
   ensemble are complete and verified. Separate white-noise/semantic augmentation
   for CNN and SVM is complete under the declared shared settings.
   Paired bootstrap intervals, individual feature groups, 50-class CLAP ranking,
   displacement/error association and the course MATLAB STFT comparison are
   [completed extensions](results/extensions_20261109/).

## Clean SVM baseline

The first baseline implements custom DSP, a 92-feature representation and
validation-selected RBF SVMs. The compact CNN is described below. The interference
pipeline and measured results are described after the CNN, followed by the frozen
CLAP integration and its separate execution environment.

### First measured result

The complete clean ESC-50 run on 7 October 2026 produced the following results.
Values are percentages; the last row gives the mean and sample standard deviation
across official test folds.

| Test fold | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| 1 | 42.50 | 42.02 |
| 2 | 48.00 | 46.65 |
| 3 | 53.75 | 52.74 |
| 4 | 49.00 | 47.89 |
| 5 | 51.25 | 50.51 |
| **Mean ± SD** | **48.90 ± 4.21** | **47.96 ± 4.07** |

[Full result and provenance](results/clean_svm/summary.json),
[held-out predictions](results/clean_svm/predictions.csv),
[confusion matrix](results/clean_svm/confusion_matrix.csv), and
[STFT validation](results/clean_svm/stft_validation.json) are checked into the
results directory. The source-overlap caveat below applies to these scores.
The expanded pipeline passed **84 tests**, with one CUDA hardware test skipped
on the local Mac. This includes training-partition leakage checks, cached
evaluation, prompt-ensemble aggregation and paired-comparison alignment.
No model settings were changed using these
outer-test results.

A [single-clip CPU microbenchmark](results/clean_svm/cost.json) measured median
feature processing at 3.66 ms and SVM inference at 7.44 ms for one five-second
recording, with 100 runs after ten warm-ups. This is an initial timing check,
not a dataset-wide latency estimate.

### Install

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). Dependencies are recorded in
`pyproject.toml` and pinned in `uv.lock`.

```bash
uv sync --locked --no-editable --extra dev --extra docs
source .venv/bin/activate
```

The non-editable installation also works in environments that disable `.pth`
processing. After modifying package code, reinstall it before using the CLI:

```bash
uv sync --locked --no-editable --extra dev --extra docs --reinstall-package elec5305-semantic-interference
```

### Download and validate ESC-50

```bash
python scripts/download_esc50.py --destination data/ESC-50
esc-baseline inspect-data --data-root data/ESC-50
esc-baseline validate-stft --config configs/baseline.json --output artifacts/validation/stft.json
pytest -q
```

The downloader pins an official repository revision, records the archive checksum
and preserves the dataset license. Expect roughly 600 MB of downloads and about
1.5 GB including the archive and extracted files. Existing data can be used by
pointing `--data-root` at a directory containing `audio/` and `meta/esc50.csv`.
Data, feature caches and model binaries are excluded from Git.

`inspect-data` verifies 2,000 files, 50 classes, eight examples per class per fold,
unique filenames and source-recording overlap. Feature extraction additionally
checks 44.1 kHz mono audio and five-second duration. The CLI rejects incomplete
datasets rather than silently reporting subset scores as full ESC-50 results.

The pinned official metadata contains four source IDs that cross folds under
different labels: `131943`, `134049`, `209698`, `234879` (nine clips in total).
The loader reports these exceptions and preserves the official folds; their full
file lists are saved in the feature cache and evaluation summary. The resulting
benchmark is not perfectly source-disjoint. Use `inspect-data --strict-sources`
to make any such overlap an error. Training-only scaling and model selection
still exclude all rows of the designated test fold.

### Extract features and evaluate all five folds

```bash
esc-baseline extract-features --data-root data/ESC-50 --config configs/baseline.json --output artifacts/features/esc50_handcrafted.npz
esc-baseline train-svm --features artifacts/features/esc50_handcrafted.npz --config configs/baseline.json --output artifacts/baselines/clean_svm
```

The declared grid is `C = [1, 10, 100]`, `gamma = [scale, 0.001, 0.1]`. Select by
validation macro-F1; ties retain the first candidate in scikit-learn's
`ParameterGrid` order. A fresh scaler and SVM are fitted to all four non-test
folds after selection. All 50 classes are included in macro-F1. Report mean and
**sample** standard deviation across the five outer folds; pooled out-of-fold
scores are saved separately.

Choose a new output directory for each run. A nonempty run directory is rejected
to preserve its models and provenance. The feature cache contains configuration,
ordered feature names, audio hashes, metadata hash and source-code hashes.

| Output | Contents |
| --- | --- |
| `summary.json` | Fold means/std, per-class metrics, versions, configuration and source hashes |
| `fold_1.json` ... `fold_5.json` | Every validation candidate, chosen parameters and test metrics |
| `svm_fold_1.joblib` ... `svm_fold_5.joblib` | Fitted scaler + SVM, class names and feature configuration |
| `predictions.csv` | Exactly one held-out prediction for every clip |
| `decision_scores.npz` | Per-class OVR scores with explicit class IDs and filenames |
| `confusion_matrix.csv` | Pooled confusion matrix with class labels |

### Classify audio and measure cost

```bash
esc-baseline predict --model artifacts/baselines/clean_svm/svm_fold_1.joblib --audio path/to/mono_44100Hz.wav
esc-baseline benchmark --model artifacts/baselines/clean_svm/svm_fold_1.joblib --audio path/to/mono_44100Hz.wav --output artifacts/baselines/clean_svm/cost.json
```

The demo accepts finite, nonempty mono audio at 44.1 kHz, including clips shorter
than five seconds. Its top scores are decision scores, not calibrated
probabilities. Each saved model excludes its named outer test fold; use the
corresponding model when checking a dataset clip. The benchmark measures one
clip with ten warm-ups and 100 timed runs at batch size one, reporting feature
processing and SVM inference separately. This microbenchmark does not estimate
latency across all classes or compare the other planned systems.

### DSP conventions and verification

- Periodic Hann; one-sided unnormalised FFT; no centre padding. Preserve a final
  incomplete frame with right zero padding. No pre-emphasis or per-clip amplitude
  normalisation.
- RMS and ZCR use unwindowed frames. Exact zeros are nonnegative for ZCR.
  Centroid, spread and 85% roll-off use `abs(STFT)**2 / n_fft`. Spectral flux is
  the L2 change in L1-normalised magnitude spectra, with zero for the first frame.
- MFCCs use 64 HTK-spaced triangular Mel filters without area normalisation,
  natural-log power with a `1e-12` floor, and orthonormal DCT-II. Keep coefficients
  0 through 19; deltas use a +/-2-frame regression with replicated boundaries.
- Concatenate all 46 means, then all 46 population standard deviations. These
  are project-specific conventions, not an exact reproduction of Piczak's
  historical 39.6% MFCC/ZCR baseline.
- Tests cover SciPy STFT agreement, a direct cosine-transform MFCC reference,
  tonal descriptors, silence, tail padding, official split isolation, scaler
  fitting and a synthetic end-to-end train/save/predict pipeline.

SciPy STFT validation and an actual MATLAB cross-check against the course
`MyStft.m` are complete. Five cases include silence, an impulse, random input,
a sinusoid and real ESC-50 audio; window, leading padding, frame alignment and
one-sided output are explicitly reconciled. Maximum absolute disagreement is
`2.93e-14`; see the [numerical audit](results/extensions_20261109/course_stft/validation.json).
Synthetic test scores are never reported as ESC-50 benchmark results.

## Clean compact CNN baseline

The CNN shares the custom STFT and 64-band natural-log Mel frontend with the
handcrafted pipeline. Each five-second clip produces a `64 x 430` spectrogram.
The cache contains **unnormalised** log-Mel values, filenames, official folds,
audio hashes, metadata hash and frontend source hashes. The measured SVM and CNN
caches use identical recordings and fold assignments.

### First measured CNN result

The full clean ESC-50 run on 8 October 2026 used all 2,000 clips and all five
official folds, with seed 5305. Values below are percentages; SD is the sample
standard deviation across folds, not variability across repeated seeds.

| Test fold | Selected epochs | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| 1 | 32 | 65.25 | 64.05 |
| 2 | 22 | 64.00 | 62.12 |
| 3 | 39 | 66.75 | 65.52 |
| 4 | 33 | 71.75 | 70.91 |
| 5 | 38 | 71.75 | 70.46 |
| **Mean ± SD** | — | **67.90 ± 3.65** | **66.61 ± 3.91** |

For comparison, the clean SVM obtained 48.90% accuracy and 47.96% macro-F1 under
the same outer splits. These are clean-condition measurements, not evidence about
noise or semantic-interference robustness. The official source-overlap caveat
also applies to the CNN. No architecture or training settings were changed in
response to these outer-test scores.

[Full result and provenance](results/clean_cnn/summary.json),
[held-out predictions](results/clean_cnn/predictions.csv),
[confusion matrix](results/clean_cnn/confusion_matrix.csv), and
[artifact verification](results/clean_cnn/verification.json) are included in the
results directory. Reloading every saved model on CPU reproduced all 2,000 MPS
class predictions; the maximum absolute logit difference was 0.0000229.
The selected durations and training-only input statistics were independently
checked against the stored histories and spectrograms.

The [single-clip CPU measurement](results/clean_cnn/cost.json) on the same
five-second recording used for the SVM benchmark gave median preprocessing
**1.55 ms** and inference **1.05 ms**,
with 100 runs after ten warm-ups. The checkpoint occupies 156,597
bytes. This does not estimate latency over the full dataset.

### Model and selection protocol

- Three `3 x 3` convolution / batch-normalisation / ReLU / max-pooling blocks,
  with 16, 32 and 64 channels. The first convolution has temporal stride two.
  Pooling uses ceiling mode so short demonstration clips remain supported.
- Adaptive pooling retains four frequency bins and averages time, producing a
  256-dimensional embedding. Dropout (0.3) and a linear 50-class head give
  **36,258 trainable parameters**. Saved class scores are logits, not calibrated
  probabilities. The embedding is saved for subsequent interference analysis.
- Fit one mean and standard deviation per Mel band on selection-training clips
  only. These statistics are model buffers and are applied identically during
  training and inference. Batch-normalisation statistics also update only during
  training; validation and test inference use evaluation mode.
- The declared protocol in `configs/cnn.json` uses AdamW, learning rate 0.001,
  weight decay 0.0001, batch size 32, maximum 40 epochs and patience 8. Select the
  epoch with highest validation macro-F1; ties keep the earliest epoch.
- Discard selection weights and optimizer. Reset seed 5305, fit new input
  statistics on all four non-test folds, then train a new model for exactly the
  selected number of epochs. The final refit never uses test metrics for stopping.
- One fixed seed per fold; clean training only. No augmentation, pretrained
  weights, learning-rate search or architecture search is used in this baseline.

Tests change both the test-fold inputs and labels and verify that selection,
normalisation and final CPU model weights are unchanged. Other tests cover early
stopping/ties, learning a synthetic frequency-separation task, cache validation,
five-fold coverage, saved-model prediction parity and the CPU timing command.
Synthetic diagnostic runs are explicitly labelled and never counted as ESC-50
results.

### Run the CNN

PyTorch is optional for the SVM workflow. Install the additional dependency with:

```bash
uv sync --locked --no-editable --extra dev --extra docs --extra cnn
source .venv/bin/activate
esc-baseline extract-log-mel --data-root data/ESC-50 --config configs/cnn.json --output artifacts/features/esc50_log_mel.npz
esc-baseline train-cnn --spectrograms artifacts/features/esc50_log_mel.npz --config configs/cnn.json --device auto --output artifacts/baselines/clean_cnn
```

`auto` selects CUDA, then Apple MPS, then CPU according to availability. The local
M3 run uses MPS; CPU remains available with `--device cpu`. Random seeds, package
versions, backend and deterministic settings are recorded. MPS and CUDA use deterministic
algorithms in warning mode because backend support varies; a fixed seed does not
promise bitwise equality across platforms or PyTorch versions. See the official
[MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) and
[reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) notes.

The portability review enabled warning mode for CUDA as well: adaptive pooling's
CUDA backward pass otherwise raises under strict deterministic mode. CPU remains
strict. This change leaves the measured CPU/MPS paths unchanged. The recorded
training-source hashes refer to baseline commit `932d445`, immediately before this
portability fix; the result files are historical run records, not rewritten hashes.
CUDA's policy is unit-tested here, while its hardware smoke test requires a CUDA host.

The extractor rejects an existing cache and training rejects a nonempty run
folder. To repeat an experiment, choose a new run folder and reuse the existing
cache. Reinstall the package after editing source when using `--no-editable`.

```bash
esc-baseline predict-cnn --model artifacts/baselines/clean_cnn/cnn_fold_1.pt --audio data/ESC-50/audio/1-100032-A-0.wav
esc-baseline benchmark-cnn --model artifacts/baselines/clean_cnn/cnn_fold_1.pt --audio data/ESC-50/audio/1-100032-A-0.wav --output artifacts/baselines/clean_cnn/cost.json
```

The demo accepts nonempty, finite 44.1 kHz mono audio. Each checkpoint excludes
its named outer test fold; use the matching checkpoint for dataset clips.
Benchmarking always uses CPU, one numerical thread, batch size one, ten warm-ups
and 100 timed runs, with preprocessing and inference measured separately.
Normalisation is included in inference. Disk access and model loading are
excluded. A single-clip benchmark is not a dataset-wide latency estimate.

| Output | Contents |
| --- | --- |
| `run_config.json` | Frozen settings, input/source hashes, software versions and backend |
| `progress.json` | Most recent fold/phase/epoch, then final completion status |
| `fold_1.json` ... `fold_5.json` | Validation/refit histories, selected duration, both normalisations and test metrics |
| `cnn_fold_1.pt` ... `cnn_fold_5.pt` | CPU-loadable weights, input normalisation, labels and frontend configuration |
| `predictions.csv` | One held-out prediction for every recording |
| `decision_scores.npz` | Out-of-fold logits and 256-dimensional embeddings, with filenames and class IDs |
| `confusion_matrix.csv` | Pooled 50-class confusion matrix |
| `summary.json` | Five-fold means/sample SD, pooled metrics, class report and provenance |

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

[Full table](results/interference_cnn_20261109/conditions.csv),
[verification](results/interference_cnn_20261109/verification.json), and
[power audit](results/interference_cnn_20261109/power_definition_audit.json) record the
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
semantic causation. See [sensitivity results](results/interference_cnn_20261109/matched_target_sensitivity.json).

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

[All levels, matched protocols](results/interference_v3_20261109/matched_protocol_conditions.csv),
[full-run tables](results/interference_v3_20261109/conditions.csv),
[overlap strata](results/interference_v3_20261109/overlap_strata.csv), and
[independent verification](results/interference_v3_20261109/verification.json) are retained.

Changing power definition modestly reduces semantic-condition F1, but does not reverse
the overall CNN semantic/DEMAND ordering. In the high-overlap stratum (overlap >= 0.8),
the paired comparison does reverse: at +5 dB, own-region CNN F1 is 24.42% for same-category
events versus 29.28% for DEMAND on those same targets, and 25.70% versus 29.07% for
different-category events. Under the target-region convention, the respective comparisons
are 24.19% versus 29.02%, and 25.84% versus 28.93%. These are descriptive within-stratum
comparisons, not evidence isolating semantic meaning from all acoustic differences.

The [accuracy decomposition](results/interference_v3_20261109/accuracy_decomposition.csv)
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
[the historical comparison](results/interference_v3_20261109/historical_svm_comparison.csv);
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
See [clean summary](results/clap_integration_20261109/clean/summary.json),
[predictions](results/clap_integration_20261109/clean/predictions.csv) and
[audit](results/clap_integration_20261109/clean_audit.json).

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
the [full result package](results/clap_full_20261109/README.md). The target protocol
omits only the nine previously identified zero-energy rows. At +5 dB, CLAP's
capture rate among errors is 66.95% / 60.30% for same-/different-category events
under `own`, and 68.74% / 61.27% under `target`. This describes where its errors go,
not a claim that it makes more errors overall than the supervised systems.

### Figures

`make-figures` draws the report figures from an evaluation directory and the clean
references, and writes the numbers behind each figure as CSV next to 600-DPI PNG,
SVG, PDF and grayscale PNG exports. The original own-region SVM/CNN figures were
checked against the supplied preview data. Complete three-system figures are now
available for [own](results/figures_three_systems_own_20261109/README.md) and
[target](results/figures_three_systems_target_20261109/README.md), with numerical,
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
See [figure notes and audit](results/figures_svm_cnn_own_20261109/README.md).
CLAP's high-overlap semantic mixtures instead show both lower accuracy and larger
median embedding displacement than matched DEMAND at +5 dB. Interpret each
representation separately; the full three-system tables retain these contrasts
under both power definitions.

### Expanded training and diagnostics

The [extension protocol](docs/extension_protocol_20261109.md) fixes all three
CNN seeds and separate WGN/semantic training settings before their results are
examined. Its dated analysis amendment adds direct semantic-versus-WGN contrasts
before semantic results, while disclosing prior inspection of some WGN results.
All 156 final intervention tests share one Holm correction. This is a post-baseline
extension, not a retrospective preregistration. Clean validation selects model
settings; all three seeds are retained.

```bash
# Use the same full clean spectrogram/feature caches as the baseline.
# Substitute explicit local paths and a config with the declared CNN seed.
esc-baseline train-augmented-cnn --spectrograms artifacts/features/esc50_log_mel.npz \
  --data-root data/ESC-50 --bank-root artifacts/augmentation_banks --mode wgn \
  --config configs/cnn.json --device mps --output artifacts/cnn_wgn_seed_5305
esc-baseline train-augmented-svm --features artifacts/features/esc50_handcrafted.npz \
  --spectrograms artifacts/features/esc50_log_mel.npz --data-root data/ESC-50 \
  --bank-root artifacts/augmentation_banks --mode wgn --output artifacts/svm_wgn
```

Repeat with `--mode semantic` and CNN seeds 5306/5307 in separately saved
configurations. Reuse identical banks across optimisation seeds. The evaluation
module `python -m esc_robustness.extension_eval --help` caches the deterministic
frontends once, then evaluates fold-matched supervised checkpoints. The full
original CNN own-region cache check reproduces all 24,000 mixture and control
predictions; score differences from MPS arithmetic are below 0.00006.

`python -m esc_robustness.extension_summary --help` describes the fixed run layout
and paired summaries. `scripts/plot_extensions.py` requires all declared variants
before final intervention plots. Seed SD, fold SD and paired bootstrap intervals
describe different uncertainties and are labelled separately. The completed
[diagnostic extension](results/extensions_20261109/README.md) includes feature
groups, full-class CLAP ranking, paired uncertainty and actual course MATLAB STFT
validation. All three clean CNN seeds reproduce the negative high-overlap
semantic-minus-DEMAND contrast. Both added seeds pass the separate declared
12-comparison Holm family; the original seed does not. The
[replication table](results/extensions_20261109/clean_replication/cnn_overlap_replication.csv)
reports every seed separately rather than pooling repeated targets.

The evidence ZIP also provides a portable `trial_evidence/` layout. From its
extracted root, the locked base environment can regenerate the main extension
tables without audio, model checkpoints, Torch or CLAP inference:

```bash
uv run --locked --no-dev python -m esc_robustness.extension_summary \
  --repo . --runtime . --run . --trial-evidence trial_evidence \
  --output reproduced_tables
uv run --locked --no-dev python scripts/audit_extension_results.py \
  --repo . --runtime . --run . --trial-evidence trial_evidence \
  --input reproduced_tables
```

This recomputes fold metrics, paired bootstrap intervals and corrected tests from
the saved trial predictions. It does not retrain models or reproduce embedding
displacements from the excluded large arrays. Canonical results are in
`results/clap_full_20261109/` and `results/extensions_20261109/`; older pipeline
directories are retained as historical evidence and should not be mixed with
these final manifests.

The fixed released PAT prompt-list ablation is implemented in
`python -m esc_robustness.prompt_ensemble --help`. It reuses the exact cached
audio embeddings, applies all templates to every class and performs no label-based
prompt search. The audited 300-entry list contains 275 unique templates, with
released multiplicities preserved. Clean accuracy is **92.90 ± 0.65%**, and
macro-F1 is **92.53 ± 0.59%** (five-fold mean ± SD). Both interference protocols
are complete; the audio embeddings are unchanged. It is explicitly **not full PAT**.
See the [independent audit](results/extensions_20261109/prompt_ensemble_audit.json).

### Verified augmentation outcomes

Own-activity protocol, +5 dB macro-F1 (%). CNN values average all three
seeds’ five-fold means; SVM values are five-fold means. The report supplies
fold/seed SD and per-seed paired intervals separately.

| Model / training | Clean | White | DEMAND | Same category | Different category |
|---|---:|---:|---:|---:|---:|
| SVM / clean | 47.96 | 2.03 | 16.13 | 22.47 | 21.34 |
| SVM / wgn | 45.99 | 41.14 | 23.72 | 22.60 | 22.19 |
| SVM / semantic | 49.27 | 1.91 | 19.92 | 26.83 | 24.96 |
| CNN / clean | 65.78 | 7.48 | 30.92 | 32.99 | 31.41 |
| CNN / wgn | 64.65 | 43.95 | 33.80 | 33.46 | 32.10 |
| CNN / semantic | 61.70 | 9.91 | 33.62 | 36.05 | 33.73 |

WGN training mainly protects against broadband noise. Semantic training yields
larger semantic point-estimate gains but reduces clean CNN performance. None
of the 12 full-test-set CNN semantic-condition comparisons between semantic and WGN training
passes the common 156-test Holm correction; the modest advantage is uncertain.
SVM passes three of its four corresponding comparisons. See the
[independent audit](results/extensions_20261109/interventions/independent_audit.json)
and [all paired effects](results/extensions_20261109/interventions/paired_intervention_effects.csv).

The deployment benchmark retains two model orders and 10,500 timed queries.
Timing drift is visible even in identical CNN frontends, so both passes are
reported and no causal augmentation-cost claim is made. See the
[repeatability check](results/extensions_20261109/cost_repeatability/audit.json).

## Repository layout

```text
configs/baseline.json        Fixed feature settings and SVM search grid
configs/cnn.json             CNN architecture and training protocol
configs/interference.json    Interference conditions, levels, seed and DEMAND environments
src/esc_robustness/          DSP, caches, metadata validation, SVM, CNN, mixing, evaluation and CLI
tests/                      Numerical, leakage and end-to-end tests
scripts/download_esc50.py    Pinned official dataset download
docs/proposal.tex            Editable proposal in the original LaTeX format
scripts/build_proposal.py    LaTeX compilation and word-limit check
data/                       Local dataset; ignored by Git
artifacts/                  Local features, predictions and models; ignored by Git
results/clean_svm/           Measured SVM results and reproducibility records
results/clean_cnn/           Measured CNN results and reproducibility records
```

The PDF retains the original proposal's LaTeX article format: Latin Modern fonts,
11-point body text, 25 mm margins, numbered sections, bulleted student information,
a compact timeline table, hanging references and centred page numbers. Edit
`docs/proposal.tex`, then rebuild with `python scripts/build_proposal.py`.

The builder requires [Tectonic](https://tectonic-typesetting.github.io/) or
`pdflatex` from TeX Live. On macOS, install the former with `brew install tectonic`.
A compiler in `.venv/bin/tectonic` is also detected; `--compiler /path/to/tectonic`
selects an explicit binary. The first Tectonic build downloads its TeX packages.

The proposal follows the course guide's eight sections and 700-1000-word limit.
The builder counts visible source text including headings, timeline and references;
automatically generated numbering and hyperlink destinations are excluded. Detailed
implementation conventions live here to keep the proposal concise.

## References and external code

- [ESC-50 official dataset, folds, taxonomy, license and caveats](https://github.com/karolpiczak/ESC-50).
  Its source preprocessing can be class dependent, which limits generalisation claims.
- [Piczak's environmental sound CNN](https://doi.org/10.1109/MLSP.2015.7324337),
  a historical reference rather than an architecture to reproduce exactly.
- [LAION-CLAP official code and checkpoints](https://github.com/LAION-AI/CLAP)
  and [paper](https://arxiv.org/abs/2211.06687).
- [PAT, NAACL 2025](https://aclanthology.org/2025.naacl-long.616/), an optional
  training-free audio-text alignment extension.
- [Mix-CLAP, ICASSP 2026](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=4206),
  context for noise-aware and efficient audio-language models.
- [RSA-Bench, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.913/),
  context for structured interference in realistic acoustic scenes.

The project's DSP, data handling and evaluation code are implemented in this
repository with AI-assisted development and checking. NumPy/SciPy/scikit-learn/PyTorch are external libraries; pretrained
CLAP and optional PAT remain external research implementations with attribution.
CLAP's external pretraining is not counted as local inference cost or treated as
equivalent to the supervised systems' training budget.

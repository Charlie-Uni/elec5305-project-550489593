# Clean SVM and CNN baselines

[Repository home](../../README.md) · [Documentation index](../README.md) · [Results index](../../results/README.md)

Detailed instructions and dated run records moved from the original README. Run commands from the repository root; use fresh output directories when repeating experiments. Historical measurements and test counts describe the runs recorded below.

## Contents

- [Clean SVM baseline](#clean-svm-baseline)
- [First measured result](#first-measured-result)
- [Install](#install)
- [Download and validate ESC-50](#download-and-validate-esc-50)
- [Extract features and evaluate all five folds](#extract-features-and-evaluate-all-five-folds)
- [Classify audio and measure cost](#classify-audio-and-measure-cost)
- [DSP conventions and verification](#dsp-conventions-and-verification)
- [Clean compact CNN baseline](#clean-compact-cnn-baseline)
- [First measured CNN result](#first-measured-cnn-result)
- [Model and selection protocol](#model-and-selection-protocol)
- [Run the CNN](#run-the-cnn)

## Clean SVM baseline

The first baseline implements custom DSP, a 92-feature representation and
validation-selected RBF SVMs. The compact CNN is described below. The separate
[interference and CLAP guide](interference_and_clap.md) covers measured interference
results and the frozen reference model's execution environment.

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

[Full result and provenance](../../results/clean_svm/summary.json),
[held-out predictions](../../results/clean_svm/predictions.csv),
[confusion matrix](../../results/clean_svm/confusion_matrix.csv), and
[STFT validation](../../results/clean_svm/stft_validation.json) are checked into the
results directory. The source-overlap caveat below applies to these scores.
The expanded pipeline passed **84 tests**, with one CUDA hardware test skipped
on the local Mac. This includes training-partition leakage checks, cached
evaluation, prompt-ensemble aggregation and paired-comparison alignment.
No model settings were changed using these
outer-test results.

A [single-clip CPU microbenchmark](../../results/clean_svm/cost.json) measured median
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
`2.93e-14`; see the [numerical audit](../../results/extensions_20261109/course_stft/validation.json).
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

[Full result and provenance](../../results/clean_cnn/summary.json),
[held-out predictions](../../results/clean_cnn/predictions.csv),
[confusion matrix](../../results/clean_cnn/confusion_matrix.csv), and
[artifact verification](../../results/clean_cnn/verification.json) are included in the
results directory. Reloading every saved model on CPU reproduced all 2,000 MPS
class predictions; the maximum absolute logit difference was 0.0000229.
The selected durations and training-only input statistics were independently
checked against the stored histories and spectrograms.

The [single-clip CPU measurement](../../results/clean_cnn/cost.json) on the same
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

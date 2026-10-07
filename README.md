# When Background Noise Has Meaning

**Environmental Sound Classification under Semantic Interference**

ELEC5305 project · Qihan Zhu · SID 550489593

[Revised project proposal (PDF)](./ELEC5305_Project_proposal_550489593.pdf)

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

### Controls and reproducibility

- Outer test fold `k` uses validation fold `(k % 5) + 1`; the other three folds
  train candidate supervised models. Select on validation macro-F1, then refit on
  all four non-test folds. Fit all learned normalisation on training data only.
- Semantic test distractors come from the same held-out outer fold, with a
  different fine class and source recording. Keep each pair fixed across levels
  and models. The initial alignment is zero offset.
- Measure both component powers over the same target-derived active sample mask
  (frames within 40 dB of peak target energy). Verify the achieved SNR/SIR after
  mixing. Record component gains, any common peak-control gain, offsets, labels,
  broad categories, checksums, fold and random seed in a manifest.
- Never hard-clip mixtures. Use identical canonical mixtures for every system.
  Retain gain-matched clean controls when common attenuation is needed.
- CLAP uses its official frontend, including 48 kHz input resampling, a fixed
  checkpoint and the fixed prompt `the sound of [class]`. Record checkpoint
  provenance and possible pretraining overlap; frozen inference does not establish
  that ESC-50 was absent from external pretraining.
- Start the CNN with one fixed seed per fold. Three seeds, generic-noise
  augmentation, semantic augmentation and PAT are optional follow-up work.

**Distractor capture rate** is `predictions equal to distractor / incorrect
predictions`, with an undefined value when there are no errors. Also report the
unconditional capture fraction. **Margin** is `target score - distractor score`;
analyse it within each model because score scales differ. A negative margin alone
does not establish that the distractor is the overall top prediction.

## Implementation milestones

1. **Implemented and evaluated:** clean SVM baseline, official folds, validated custom STFT, 92-dimensional
   handcrafted features, training-only scaling and out-of-fold predictions.
2. **Implemented and evaluated:** compact CNN on the same folds, with validation-based duration selection.
3. Shared white-noise and DEMAND mixing with achieved-SNR checks.
4. Reproducible semantic pairing and SIR-controlled mixtures.
5. Frozen CLAP inference with cached audio/text embeddings and class scores.
6. Robustness, semantic capture, representation and computational-cost analysis.
7. If time permits, choose generic-noise augmentation or released PAT inference.

## Clean SVM baseline

The first baseline implements custom DSP, a 92-feature representation and
validation-selected RBF SVMs. The compact CNN is described below. Mixing and CLAP
experiments remain subsequent milestones; no results are claimed for them yet.

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
With the CNN extra installed, the current code passes 44 tests on the local Mac;
one CUDA hardware test is skipped. No model settings were changed using these
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

SciPy STFT validation is implemented. The course `MyStft.m` file has not been
provided, so its cross-check remains pending. Synthetic test scores are never
reported as ESC-50 benchmark results.

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

## Repository layout

```text
configs/baseline.json        Fixed feature settings and SVM search grid
configs/cnn.json             CNN architecture and training protocol
src/esc_robustness/          DSP, caches, metadata validation, SVM, CNN and CLI
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

The project's DSP, data handling and evaluation code will be developed in this
repository. NumPy/SciPy/scikit-learn/PyTorch are external libraries; pretrained
CLAP and optional PAT remain external research implementations with attribution.
CLAP's external pretraining is not counted as local inference cost or treated as
equivalent to the supervised systems' training budget.

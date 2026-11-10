# Frozen CLAP integration — 9 November 2026

The CLAP evaluator and CLI integration is complete on local branch `feature/interference`.
The work has not been committed or pushed. This folder contains measured clean results
and small real-model integration checks; it does not contain the full CLAP interference
experiment.

## Completed

- Applied the CLAP increment to the reviewed local v3 implementation.
- Wired `evaluate-clap-clean`, `evaluate-interference --clap-checkpoint`,
  `summarise-interference --clap-clean` and `benchmark-clap`.
- Added explicit CPU/MPS/CUDA selection and batch-size control. Mixture batches include
  the gain-matched clean controls. SVM/CNN evaluation remains supported in the same call.
- Fixed int16 truncation to match the official implementation; used its tensor path to
  avoid a second quantisation. An actual model check matched the official path exactly.
- Checked the full checkpoint SHA-256 before loading; froze all model parameters.
- Validated recalculated mixing gains, then used manifest gains exactly. All 240
  sampled raw mixtures and gain-matched controls matched the original NumPy 2.x
  baseline hashes in the NumPy 1.x CLAP environment.
- Installed CLAP in a separate environment. Copied verified data and available model
  artifacts to `/Users/leaf/Projects/elec5305-local`, outside iCloud-managed storage.
- Passed 71 tests, with one CUDA-only test skipped, and passed Ruff checks.

## Measured clean result

Fixed checkpoint: `630k-audioset-best.pt`, SHA-256
`8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037`.
HTSAT-tiny, RoBERTa, no fusion, one fixed `the sound of {label}` prompt per class;
underscores become spaces. Float64 polyphase resampling from 44.1 to 48 kHz, then
one official int16 round trip. No training, prompt search or checkpoint selection.

| Official fold | Clips | Accuracy (%) | Macro-F1 (%) |
| --- | ---: | ---: | ---: |
| 1 | 400 | 83.75 | 82.04 |
| 2 | 400 | 77.25 | 73.78 |
| 3 | 400 | 83.00 | 81.13 |
| 4 | 400 | 82.25 | 80.08 |
| 5 | 400 | 82.50 | 80.42 |
| Mean ± sample SD | 2,000 | 81.75 ± 2.58 | 79.49 ± 3.28 |

The CPU run used eight clean clips per batch and took 248 seconds excluding model
loading and prompt embedding. `clean_audit.json` independently checks saved audio/text
cosines, predictions and per-fold metrics calculated from confusion counts.
The benchmark timing is separate: 100 single-clip CPU repetitions after ten warm-ups,
5.78 ms median resampling/quantisation plus 92.35 ms embedding/scoring. This is a
single-clip timing check, not a dataset latency distribution. CLAP's internal Mel
frontend is included in embedding time. The full audio/text model has 158,334,169
parameters (633,336,676 parameter bytes); the 1,863,587,645-byte checkpoint is not
an inference-memory measurement.

The official repository's example uses a different prompt and preprocessed data.
These project results should not be presented as reproducing its published score.
CLAP's external pretraining is also different from the local supervised training;
frozen evaluation does not prove the absence of ESC-50 recordings from pretraining.

## Real CLI checks

`smoke_own_all_systems` contains the first 24 `own` rows evaluated together by SVM,
CNN and CLAP. `smoke_target` contains 24 `target` rows evaluated by CLAP. Each
includes four interference types and three levels, but only two targets in fold 1.
These are integration checks, not statistical robustness estimates. The condition
macro-F1 values include all 50 labels despite incomplete class coverage.

Every stored CLAP cosine, control cosine, prediction, semantic margin and capture
indicator was independently checked. The sampled SVM/CNN predictions and target
scores exactly matched their v3 results. Both summary CLI invocations completed,
including the three-system summary with the corresponding clean references.

CPU/MPS testing on four clips gave identical top-1 predictions, with maximum score
difference 3.56e-7. CPU scalar/batch differences were below 1.64e-7. The small local
execution benchmark did not show an MPS speed advantage. Production defaults and
CPU cost measurement retain one numerical thread; the multi-thread timing experiment
did not change the recorded clean evaluation protocol.

## Remaining experiment

Full CLAP interference evaluation remains to run:

1. `manifest_own`: 24,000 rows plus gain-matched controls.
2. `manifest_target`: 23,991 rows plus controls; the nine previously excluded rows remain excluded.
3. Summarise each with this clean CLAP reference, then combine with the existing v3
   SVM/CNN outputs for paired-target and overlap-stratified comparisons and figures.

The relocated manifest CSVs are byte-identical to v3. Only local data/cache paths
in their metadata changed. A ready-to-run local script is at
`/Users/leaf/Projects/elec5305-local/runs/clap_v4_20261109/run_full_interference.sh`.
It writes separate full-run outputs and refuses to overwrite existing nonempty
output directories. It has not been executed as part of the integration checks.

The source patches are exported separately from the results. Apply the increment
on reviewed v3 or the full source patch on `main@1ba8f4f`; do not apply both.
Both are checked by applying to reconstructed base trees and comparing every source
file byte for byte. Result files are included in the review archive alongside the
patches.

Official API reference: https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/hook.py

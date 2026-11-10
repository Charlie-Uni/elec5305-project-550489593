# Completed closing analyses — 10 November 2026

These are the two final exploratory analyses, using existing recordings, manifests,
scores and models. Core results and previous intervention results are unchanged.
No new model, dataset, training or pairing is introduced. The original cloud package
is retained locally; the completed and corrected results are in this directory.

## Coverage and verification

| Output | Coverage |
|---|---:|
| Own-power acoustic features | 24,000 rows, including 6,000 DEMAND |
| Target-power acoustic features | 23,991 rows, including 6,000 DEMAND |
| Exact semantic readout | 23,991 trials × 4 variants = 95,964 |
| Top-k summary cells | 192, for k = 1, 2, 3, 5 |
| Same-target acoustic comparisons | 90, across 3 models, 2 power definitions, 3 contrasts, 5 selections |
| Final figures | 7, each with CSV, PNG, SVG and PDF |

`independent_audit.json` verifies all 95,964 rank records against independently
sorted raw scores, all 192 state summaries and all 90 paired comparison estimates.
The 35,991 available cloud feature rows agree within 5.6e-15. A separate SciPy
framing and independently constructed Mel filterbank checks 120 acoustic rows,
including DEMAND (maximum difference 1.2e-15). Twenty-six primary confidence
intervals were regenerated with an independent cluster-draw algorithm and seed;
the largest endpoint difference was 1.17 percentage points. Other interval draws
were not independently regenerated. Hashes identify the exact audited tables.

The incoming target feature table contained 17,991 rows, not 18,000: nine semantic
rows have undefined target-region power and were already excluded upstream.
Nine own-power rows additionally have undefined target-active spectral cosine
because no interferer energy falls in the target-active frames. They remain
missing for that descriptor, while the original classification trials are retained.

## What the results support

At +5 dB under own power, single-template CLAP retains the target at rank two in
236/632 same-category captures (37.34%) and 203/612 different-category captures
(33.17%). Both designated labels appear in the top two in 26.05% / 21.00% of all
semantic mixtures, versus 5.40% / 1.70% of gain-matched clean controls. But 23.35% /
24.05% of mixtures return the distractor without the target. The existing prompt
ensemble retains the target in 47.60% / 38.60% of captures. Target-power results
show the same partial-recovery pattern. Top-k is ranked retrieval, not a calibrated
multi-label detector or proof of subjective hearing. Larger k relaxes the task.

Raw white-noise dB Mel correlation has median −0.728 at +5 dB, but the median is
−0.008 after dividing band energies by filter-weight sums. Unequal Mel bandwidths
therefore matter to this descriptor. This frontend is the project's DSP/CNN
frontend, not CLAP's. A time–frequency masking fraction is an acoustic proxy,
not a validated model of perceptual masking.

Primary same-target matching requires absolute differences no greater than 0.10
in target-active spectral cosine, 0.05 in masked-cell fraction and 0.10 in temporal
activity overlap. These audio-only rules select the same targets for every model.
Own-power same-/different-category versus DEMAND subsets retain 210/178 targets;
target-power retains 205/187, about 9–10.5% of eligible targets. Class coverage is
47–49. The full CSV reports original source counts, strict/loose sensitivities,
gain-control-referenced losses and achieved descriptor differences.

| Power | Model | Same minus DEMAND, accuracy pp [95% CI] | Different minus DEMAND, pp [95% CI] |
|---|---|---:|---:|
| Own | SVM | +1.90 [−4.42, +8.13] | +8.43 [+2.78, +14.61] |
| Own | CNN | +3.81 [−3.33, +11.79] | +2.25 [−3.37, +7.96] |
| Own | CLAP | −25.71 [−32.71, −19.05] | −19.10 [−26.74, −11.80] |
| Target | SVM | +2.44 [−3.81, +8.96] | +8.56 [+3.21, +14.05] |
| Target | CNN | +3.90 [−3.94, +11.77] | +3.74 [−2.12, +9.19] |
| Target | CLAP | −23.41 [−30.30, −16.19] | −16.04 [−23.28, −8.95] |

Intervals are pointwise percentile intervals from 2,000 original target-source
bootstrap resamples (seed 530510), conditional on fitted models and observed
pairings. They retain clips sharing a target source, but do not fully capture
shared interferer, background and training-set dependence. No new p-value family
is claimed. Fewer than 30 targets or source clusters yields no interval.

CLAP retains a semantic disadvantage in these selected populations; measured
acoustic proxies do not remove it. CNN intervals include zero. The selected
population differs from the original high-overlap group, so this is not a causal
decomposition. Same-minus-different primary intervals span zero for all systems.
Unmeasured acoustics and salience remain plausible explanations; DEMAND itself
can contain meaningful events. Logistic coefficient shrinkage is not a percentage
explained, and in-sample pseudo-R² is not explained causal variance. The supplied
fits remain secondary diagnostics in `descriptive_own/` and `descriptive_target/`.

## Files and reproducibility

- `readout_trials.csv`: exact target/distractor ranks and controls for four variants.
- `readout_outcomes.csv`, `readout_uncertainty.csv`: counts, four states and intervals.
- `overlap_own/`, `overlap_target/`: complete acoustic features and original execution metadata.
- `executed_features/`: source snapshot matching the feature metadata's hashes;
  current source additionally includes validation, documentation and plot cleanup.
- `acoustic_matched_trials.csv`, `acoustic_matched_comparisons.csv`: paired evidence.
- `acoustic_stratified.csv`: descriptive bins with counts and class coverage;
  target compositions can differ between conditions in a bin.
- `figures/`: final exports, numerical inputs, layout audit and provenance.
- `completion_audit.json`: preservation and final delivery checks.

Recorded execution commands from the local repository, with the existing runtime layout.
Feature extraction requires an empty output directory; use fresh run/output paths to
reproduce it. Existing complete outputs are intentionally protected from overwrite:

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=../mpl-cache
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
../baseline-venv/bin/python -m esc_robustness.cli overlap-features --manifest ../runs/clap_v4_20261109/manifest_own --data-root ../data/ESC-50 --require-demand --output ../runs/closing_20261110/overlap_own
../baseline-venv/bin/python -m esc_robustness.cli overlap-features --manifest ../runs/clap_v4_20261109/manifest_target --data-root ../data/ESC-50 --require-demand --output ../runs/closing_20261110/overlap_target
../baseline-venv/bin/python scripts/complete_closing_analyses.py --repo . --runtime .. --run ../runs/closing_20261110 --output results/closing_20261110 --part all
../baseline-venv/bin/python scripts/audit_closing_analyses.py --repo . --runtime .. --run ../runs/closing_20261110 --output results/closing_20261110
../baseline-venv/bin/python scripts/plot_closing_analyses.py --data results/closing_20261110 --run ../runs/closing_20261110 --output results/closing_20261110/figures --export
../baseline-venv/bin/python scripts/build_final_report.py
```

The feature commands locate DEMAND through the manifest's recorded cache path;
when relocating data, restore that cache or regenerate a verified manifest.
Full rank recomputation requires the saved NPZ score arrays, deliberately excluded
from the compact evidence archive; included ranks and counts allow the summary
tables to be independently recomputed. Original audio is likewise not duplicated.
The independent cloud comparison additionally expects the supplied archive extracted
under `runs/closing_20261110/incoming/closing_analyses/`.

The [protocol](../../docs/closing_protocol_20261110.md) documents the exploratory
status and stopping rule. These two analyses close experimental work. Author
review, final recording and course submission remain delivery actions. The
[delivery synchronization](../delivery_sync_20261110/README.md) subsequently updated
the 16-slide presentation, exact narration and publicly deployed Pages; the
report and original numerical experiment results are unchanged.

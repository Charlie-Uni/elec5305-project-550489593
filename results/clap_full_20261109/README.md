# Full three-system interference results — 9 November 2026

Both frozen CLAP runs are complete: 24,000 own-region mixtures and 23,991
target-region mixtures, each with a gain-matched clean control. The corresponding
v3 SVM/CNN runs use the identical manifest within each protocol. All three systems
were independently checked from their saved trial-level scores and predictions.

## Results at +5 dB

Values below are macro-F1 averaged across the five official test folds, using all
50 labels in each fold. Each condition uses its retained targets; for exact
own/target cohort comparisons see [protocol_comparison.csv](protocol_comparison.csv).
The target protocol removes three pairs at three levels (nine rows), because the
interferer has no energy inside the target activity mask.

| Power region | Interference (+5 dB) | SVM macro-F1 | CNN macro-F1 | CLAP macro-F1 |
| --- | --- | ---: | ---: | ---: |
| own | white | 2.03% | 7.76% | 62.91% |
| own | demand | 16.13% | 29.46% | 71.49% |
| own | semantic_same | 22.47% | 32.60% | 48.86% |
| own | semantic_different | 21.34% | 32.00% | 45.16% |
| target | white | 2.03% | 7.76% | 62.92% |
| target | demand | 16.15% | 29.33% | 71.64% |
| target | semantic_same | 21.14% | 31.70% | 47.76% |
| target | semantic_different | 20.87% | 30.93% | 42.81% |

The clean references are SVM 47.96%, CNN 66.61%, and CLAP 79.49% macro-F1.
CLAP has extensive external pretraining: these are deployment comparisons,
not equal-data or equal-training-cost comparisons.

## Distractor capture

Capture rate is the fraction of **incorrect predictions** equal to the distractor
label. It is not the fraction of all trials and does not itself measure accuracy.

| Power region | Semantic condition (+5 dB) | SVM | CNN | CLAP |
| --- | --- | ---: | ---: | ---: |
| own | semantic_same | 18.53% | 31.26% | 66.95% |
| own | semantic_different | 15.89% | 27.53% | 60.30% |
| target | semantic_same | 18.73% | 31.85% | 68.74% |
| target | semantic_different | 16.32% | 28.01% | 61.27% |

The 1/49 line in figure 2 assumes uniform choice among wrong classes; it is not an
empirical null or a significance test. The larger CLAP capture rates describe the
destination of its errors, not a claim that it makes more errors overall.

## High activity overlap (at least 80%), +5 dB

| Region | System | Semantic condition | n | Semantic accuracy | Matched DEMAND accuracy | Semantic median distance | DEMAND median distance |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| own | svm | semantic_same | 1464 | 19.67% | 18.51% | 0.748 | 0.824 |
| own | svm | semantic_different | 1465 | 19.25% | 17.75% | 0.756 | 0.835 |
| own | cnn | semantic_same | 1464 | 29.10% | 32.38% | 0.110 | 0.170 |
| own | cnn | semantic_different | 1465 | 28.40% | 31.33% | 0.119 | 0.169 |
| own | clap | semantic_same | 1464 | 48.70% | 74.25% | 0.325 | 0.261 |
| own | clap | semantic_different | 1465 | 44.98% | 73.52% | 0.358 | 0.261 |
| target | svm | semantic_same | 1464 | 19.60% | 18.44% | 0.748 | 0.823 |
| target | svm | semantic_different | 1465 | 19.18% | 17.61% | 0.756 | 0.837 |
| target | cnn | semantic_same | 1464 | 29.03% | 32.38% | 0.111 | 0.170 |
| target | cnn | semantic_different | 1465 | 28.46% | 31.33% | 0.119 | 0.169 |
| target | clap | semantic_same | 1464 | 48.77% | 74.45% | 0.326 | 0.261 |
| target | clap | semantic_different | 1465 | 44.71% | 73.65% | 0.360 | 0.260 |

Each semantic/DEMAND comparison uses the same target cohort. Distances use the
model-specific definition in figure 5 and must not be ranked across models.
The earlier CNN observation (smaller semantic displacement despite lower
accuracy) must not be assumed to hold for CLAP; the table reports its actual
within-model distances separately.

## Protocol and interpretation

- `own`: each source is measured on its own active region. `target`: both powers
  are measured inside the target activity mask. The target remains the labelled
  source; 0 dB is an ambiguous single-label stress test.
- Both protocols retain 40 dB activity thresholding, zero offset, deterministic
  same-/different-coarse-category pairings, and fixed white/DEMAND realisations
  across the three interference levels. Manifest ratios were independently
  recomputed using the corresponding stored power field and match within 1e-9 dB.
- The target protocol can require strong attenuation for sparse interferers;
  gain-matched controls, control-relative losses and correct-to-error transitions
  are retained in each system's table. Do not attribute a control error to the
  added interferer.
- Figure 8 compares semantic and DEMAND accuracy on the **same targets** within
  overlap strata; CSVs retain fold counts, macro-F1 and class coverage. Sparse
  strata can lack classes, so their all-50-class macro-F1 is not an unqualified
  comparison of difficulty across strata.
- Score margins and representation distances are interpreted within each model.
  SVM distances use the evaluated training-fold scaler; embeddings use cosine
  distance to the gain-matched control. Absolute distances are not comparable
  across architectures.
- Equal nominal power does not make interference acoustics identical. These
  controlled results and capture diagnostics support representation-specific
  failure analysis, not proof that semantic identity alone causes the differences.
- Capture flows are selected exploratory examples. Counts and the number of
  initially correct controls are provided; small 6/6 examples are not population
  estimates.

## Figures and audit trail

- [Own: eight figures, all formats and CSVs](../figures_three_systems_own_20261109/README.md)
- [Target: eight figures, all formats and CSVs](../figures_three_systems_target_20261109/README.md)
- [Independently verified conditions](all_conditions_verified.csv)
- [Own audit](own/numerical_audit.json), [target audit](target/numerical_audit.json),
  [manifest and held-out-pair audit](manifest_audit.json)
- Trial CSVs and summary tables are mirrored in `own/` and `target/`. Large
  embedding archives stay outside iCloud; [locations and hashes](raw_artifact_locations.json).

Each figure has 600-DPI PNG, editable SVG, embedded-font PDF, grayscale PNG and
numeric CSV. Error bars are sample SD over five folds, not confidence intervals;
boxplots show quartiles and 1.5-IQR whiskers with outliers hidden and counts in CSV.
All eight previews in each protocol were inspected before final vector export.
Generic PDF font-check warnings were resolved by inspecting embedded TrueType
streams in the Type0 fonts' CIDFontType2 descendants.

The fixed full-run snapshot, checkpoint hash and launch metadata are retained.
No new CLAP training, prompt tuning, model selection or supervised retraining was
performed for this result package.

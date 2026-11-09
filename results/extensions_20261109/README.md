# Extended analysis and intervention experiments

These are extensions motivated by the completed baseline experiment. Training
settings and the original contrasts were fixed before the new results. The direct
semantic-versus-WGN comparisons were added at 22:45 AEDT, before semantic results
were available but after some WGN results had been inspected; all 156 final tests
share one Holm correction. This is not a retrospective preregistration. See the
[dated protocol amendment](../../docs/extension_protocol_20261109.md).

## Completed and checked

- [Paired uncertainty](diagnostics/paired_uncertainty.csv): 2,000 target-clip
  bootstrap replicates, stratified by official fold. Models remain fixed.
- [Eight feature groups](diagnostics/feature_group_displacement.csv): training-fold
  standardisation and dimension-normalised displacement from gain-matched controls.
- [CLAP ranking](diagnostics/clap_rank_summary.csv) and
  [all competing-class diagnostics](diagnostics/clap_competition_trials.csv).
- [Displacement/error association](diagnostics/displacement_error_auc.csv):
  fold-specific and pooled AUC, interpreted descriptively.
- [MATLAB course STFT comparison](course_stft/validation.json): actual execution of
  unmodified teaching code, with explicit matching window/frame conventions.
  Relative error on the real audio example is approximately 2.6e-16.
- [Four supplementary figures](figures/figure_metadata.json), with PNG, SVG, PDF
  and source-data CSV exports. The previews passed numerical, layout and visual checks.
- [Three-seed CNN overlap replication](clean_replication/cnn_overlap_replication.csv),
  with both full interference protocols checked for seeds 5305, 5306 and 5307.
  [Paired replication figure](replication_figures/17_cnn_overlap_replication.pdf).
- [Fixed CLAP prompt ensemble audit](prompt_ensemble_audit.json): all 2,000 clean
  clips, 24,000 own and 23,991 target trials. Audio embeddings are bit-identical
  to the single-prompt baseline; every score is recomputed from the text vectors.

## What the uncertainty analysis changes

At +5 dB and overlap at least 0.8, compare semantic interference with DEMAND on
the **same targets**. Differences below are accuracy percentage points; intervals
are paired 95% bootstrap intervals conditioned on the fitted models.

| System | Own / same category | Own / different category | Target / same category | Target / different category |
|---|---:|---:|---:|---:|
| SVM | +1.16 [-1.44, +3.69] | +1.50 [-0.96, +4.03] | +1.16 [-1.23, +3.69] | +1.57 [-0.96, +3.89] |
| CNN | -3.28 [-6.01, -0.75] | -2.94 [-5.26, -0.34] | -3.35 [-6.08, -0.75] | -2.87 [-5.26, -0.41] |
| CLAP | -25.55 [-28.28, -22.95] | -28.53 [-31.20, -25.73] | -25.68 [-28.28, -23.16] | -28.94 [-31.60, -26.14] |

The CLAP differences are large. The CNN direction is consistent across power
conventions, but its exact McNemar tests do **not** pass 0.05 after Holm adjustment
across the 12 primary comparisons (adjusted p approximately 0.12–0.15). Do not
describe this core seed alone as firmly established. The completed replication
below addresses additional training variability. Clip-level intervals do not account for all
dependence between clips sharing a source recording or for model refitting.

For CLAP under the own convention, the same-category distractor's median rank
moves from 17 in the matched control to 4, 3 and 2 at +10, +5 and 0 dB. This is a
50-class ranking result, not the fraction of errors captured by the distractor.
The two metrics have different denominators.

White noise causes much larger standardised shifts in ZCR, centroid, spread and
roll-off than the other conditions at +5 dB. Representation displacement is
associated with failure within models, but the distances do not share a common
scale across architectures and do not establish a causal semantic mechanism.

## Completed replication and prompt ablation

For clean CNN seeds 5306 and 5307, accuracy is 66.40% for each, with macro-F1
65.19% and 65.53%. The core seed's macro-F1 is 66.61%. At +5 dB and overlap ≥0.8,
the new seeds' semantic-minus-matched-DEMAND accuracy differences range from
−4.92 to −6.83 percentage points across both protocols/category relationships.
All eight new-seed comparisons pass the declared 12-comparison Holm family
(adjusted p <0.002); seed 5305 does not (adjusted p ≈0.060 in this replication
family). The original three-system family and the replication family are different,
so their adjusted p values differ. No repeated target is counted as independent.

The fixed prompt ensemble uses all 300 released entries (275 unique templates),
including their released multiplicities. Clean accuracy is **92.90 ±0.65%** and
macro-F1 **92.53 ±0.59%**, with SD across folds. No prompts were selected on these
test labels. This is **not full PAT**.

| Power | White +5 dB | DEMAND +5 dB | Same category +5 dB | Different category +5 dB |
|---|---:|---:|---:|---:|
| Own | 75.72 | 84.32 | 55.50 | 53.69 |
| Target | 75.71 | 84.15 | 54.10 | 51.64 |

Cells are five-fold macro-F1 percentages. The ensemble improves class readout
without changing the audio embeddings, while a large semantic-interference gap
persists. Error-conditioned capture rates remain high and must be read alongside
unconditional capture counts. See [own conditions](clap_prompt_ensemble/own_conditions.json)
and [target conditions](clap_prompt_ensemble/target_conditions.json).

## Completed training interventions

All 20 training banks have passed an independent full-file and row audit:
84,000 views, all 2,000 source-audio hashes, source-code hashes, achieved ratios,
peak gains and finite feature/Mel tensors. Selection banks exclude validation
and test audio; refit banks exclude test audio. Semantic partners also exclude
cross-fold source IDs. See [bank audit](augmentation_banks_audit.json).

- Separate WGN and semantic-mixture CNNs at all three seeds, with the same
  augmentation probability, view schedule and maximum epoch budget. Actual
  refit durations can differ because clean validation selects each one.
- Deterministic SVM augmentation using the same training banks. Both modes and
  both interference protocols are complete and independently verified.
- Both final own and target manifests for every supervised variant, followed by
  independent row/score/hash verification. New results are separate from the core runs.

All 22 supervised evaluation runs are complete. Together with the original SVM
and two frozen CLAP readouts, the final study contains 14 variants and 336
condition cells. [Independent recomputation](interventions/independent_audit.json)
checks all metrics, 156 paired effects, 12 replication contrasts and 72 CNN seed
aggregates. [Seven extension figures](intervention_figures/figure_metadata.json)
(including the separate [cost panel](cost_figures/18_balanced_deployment_cost.pdf))
retain numerical source tables. The cost study has two independently audited
passes, with identical models and audio but reversed model order; all 10,500
predictions match the formal clean evaluations.

For the CNN, WGN training raises own/+5-dB white-noise macro-F1 from 7.48% to
43.95%, while same/different semantic scores move from 32.99/31.41% to
33.46/32.10%. Semantic training reaches 36.05/33.73%, with clean macro-F1 falling
from 65.78% to 61.70%. None of the 12 full-test-set semantic-condition CNN accuracy contrasts
passes the common Holm family. These are modest positive point estimates with
uncertain direct advantages, not proof of superiority. All three seeds are kept.
The 12 contrasts are two semantic pairing types × two power protocols × three
seeds. White-noise and DEMAND contrasts remain in the full 156-test family.

## Additional interpretation checks

- [Capture transitions](capture_transitions_core/capture_transitions.csv) separate
  new distractor captures from predictions already naming the distractor in the
  gain-matched clean control. Own-protocol CLAP at +5 dB has 632/612 captures
  (same/different category) with the single prompt and 645/645 with the ensemble,
  despite total errors falling from 944/1,015 to 834/876. Conditional capture rate
  therefore changes with both the numerator and the error denominator.
- [Prompt multiplicity](prompt_multiplicity/sensitivity.csv) is an exploratory
  equal-weight check on the 275 unique templates. It changes clean accuracy from
  92.90% to 92.80%; the largest absolute macro-F1 change across all 25 cells is
  0.47 percentage points. The declared 300-entry result remains the intervention.
- [Target-source uncertainty](source_cluster_sensitivity/source_cluster_intervals.csv)
  resamples original target recording IDs jointly, using 10,000 replicates.
  All 12 clean-CNN pointwise intervals remain negative for high-overlap
  semantic-minus-DEMAND accuracy; SVM intervals span zero. CLAP ensemble contrasts
  remain about −31 to −34 points. This sensitivity does not replace the original
  corrected tests or account for every dependency from shared distractors.

The [editable presentation](../presentation_20261109_draft/research_presentation.pptx)
contains native charts, speaker notes and two embedded audio examples. It is an
author-review draft; final narration and course submission have not occurred.

# Extended analysis and intervention experiments

These are extensions motivated by the completed baseline experiment. Settings
were fixed before inspecting the new intervention results; this is not a
retrospective preregistration. See the [protocol](../../docs/extension_protocol_20261109.md).

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
describe the CNN effect as firmly established; the three-seed extension addresses
additional training variability. Clip-level intervals do not account for all
dependence between clips sharing a source recording or for model refitting.

For CLAP under the own convention, the same-category distractor's median rank
moves from 17 in the matched control to 4, 3 and 2 at +10, +5 and 0 dB. This is a
50-class ranking result, not the fraction of errors captured by the distractor.
The two metrics have different denominators.

White noise causes much larger standardised shifts in ZCR, centroid, spread and
roll-off than the other conditions at +5 dB. Representation displacement is
associated with failure within models, but the distances do not share a common
scale across architectures and do not establish a causal semantic mechanism.

## Running extensions

- Clean CNN seeds 5306 and 5307, in addition to the completed seed-5305 baseline.
- Separate WGN and semantic-mixture CNNs at all three seeds, with matched exposure.
- Deterministic SVM WGN and semantic augmentation using the same training banks.
- Fixed LAION-CLAP prompt-ensemble ablation using the complete released PAT list;
  this is **not full PAT** and does not search parameters on evaluation labels.
- Both final own and target manifests for every supervised variant, followed by
  independent row/score/hash verification. New results are separate from the core runs.

Do not quote augmentation improvements until the saved results and comparisons
have been verified. Training/evaluation state is recorded locally under
`/Users/leaf/Projects/elec5305-local/runs/extensions_20261109`.

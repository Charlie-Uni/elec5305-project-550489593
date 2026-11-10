# Extension protocol — 9 November 2026

This extension is specified after inspecting the completed baseline robustness
results. It is not a retrospective preregistration. Training settings and the
original comparisons were fixed before examining the new intervention results;
the later, explicitly dated analysis amendment is recorded below. Existing baseline runs
remain separately identifiable by their manifests and hashes.

## Research questions and primary comparisons

1. Does white Gaussian noise augmentation transfer to unseen DEMAND backgrounds
   and competing ESC-50 events?
2. Does semantic-mixture augmentation improve semantic robustness more than
   white-noise augmentation at comparable training exposure, and at what clean
   performance cost?
3. Are the observed CNN differences stable across seeds 5305, 5306 and 5307?
4. How do individual handcrafted feature groups, the complete 50-class CLAP
   ranking, and representation displacement explain failures?

The primary level is +5 dB. +10 dB measures weaker interference and 0 dB is an
ambiguous-label stress test. Preserve both own-activity and target-activity power
protocols; report high-overlap (at least 0.8) comparisons on matched targets.

## Supervised interventions

- Keep the existing compact CNN architecture, optimiser, maximum epochs,
  early-stopping rule and clean-only validation metric unchanged. All five
  official test folds are retained. Reuse the completed clean seed-5305 run,
  then run clean seeds 5306 and 5307.
- Train separate white-noise and semantic-mixture CNNs at all three seeds.
  Use 50% clean / 50% augmented examples during training. Augmentation levels
  are +5, +10 and +15 dB; exclude equal-power semantic targets from training.
- Use a deterministic bank of three augmented views per training clip, with
  banks fixed across optimisation seeds. Random per-epoch sampling and model
  initialisation follow the declared optimisation seed. Match the augmentation
  probability, number of views, level schedule and maximum epoch budget between
  interventions. Actual refit epochs are selected separately on clean validation
  and can differ; total training-example presentations are therefore not identical.
- Use own-active-region power, the existing 40-dB activity threshold and logged
  common attenuation. Semantic partners come only from the current training
  partition, have a different fine class/source recording, and balance same-
  and different-coarse-category pairings. Exclude source IDs crossing official
  folds from distractor eligibility. During selection, never use validation or
  test audio as a training distractor. Rebuild the bank for the four-fold refit
  partition, where the former validation fold is now training data; the outer
  test fold remains excluded.
- Estimate Mel normalisation on clean training clips only, as in the baseline.
  Use clean validation for duration selection; do not choose settings using
  any outer-test interference scores.
- White-noise augmentation is deliberately synthetic. The existing DEMAND test
  recording segments are never used for augmentation, so DEMAND remains a
  transfer condition. Call this intervention WGN augmentation, not DEMAND
  augmentation.
- Include an SVM WGN/semantic augmentation comparison using the same validated
  banks. Fit each SVM with every clean training clip plus one augmented view per
  clip (view index = partition position modulo 3), giving equal clean/augmented
  example counts and balanced levels. Fit its scaler on these training examples
  only and select the original C/gamma grid using clean validation. This static
  SVM exposure differs from the CNN's stochastic per-epoch view sampling.

## Statistics and diagnostics

- Paired target-level bootstrap, stratified by official fold, 2,000 replicates,
  fixed seed 530509; retain paired predictions and all 50 F1 labels. Report
  percentile 95% intervals for differences, alongside five-fold sample SD.
  These intervals condition on trained models and do not capture training
  uncertainty; the three-seed results address that separately.
- Treat the +5 dB intervention tests (all targets plus high-overlap semantic
  subsets) as one Holm family. Report the 12 clean-CNN overlap replications
  (three seeds, two protocols and two category relationships) as a separate
  declared family. Accuracy differences pool matched target counts; macro-F1
  differences average the five official folds with all 50 labels retained.
- The unit of resampling is the target clip, retaining its paired conditions;
  do not treat multiple mixtures of one clip as independent observations.
- Separate eight feature groups: RMS, ZCR, centroid, spread, roll-off, flux,
  MFCC and delta-MFCC. Standardise with each evaluated SVM's training-fold
  scales and normalise group Euclidean displacement by sqrt(group dimension).
- Report target/distractor ranks among 50 classes, top competing classes other
  than target/distractor, and score changes relative to gain-matched controls.
- Report representation-displacement discrimination of errors descriptively,
  including fold/condition-specific AUC, without treating correlation as proof
  of a semantic mechanism.
- Verify course MyStft.m numerically after explicitly reconciling window,
  padding, frame alignment and one-sided/full-spectrum conventions.

### Analysis amendment recorded at 22:45 AEDT, 9 November 2026

The stated second research question requires a direct semantic-augmentation
versus WGN-augmentation contrast, beyond comparisons with clean training. The
initial analysis implementation omitted those direct contrasts. They are added
before any completed semantic-intervention result is available, after some WGN
results have already been inspected. Training settings and saved evaluations
are unchanged. The final +5-dB family includes 108 clean-reference contrasts and
48 direct semantic-versus-WGN contrasts (SVM plus three matched CNN seeds, both
power definitions, four full conditions and two high-overlap semantic subsets).
All 156 exact tests share one Holm correction. This amendment is disclosed;
it is not presented as a preregistration before every intervention result.

## Modern alignment extension

Inspect the official PAT release before claiming a reproduction. Preserve the
frozen single-prompt CLAP reference. Any prompt-ensemble-only or LAION-specific
adaptation must have a distinct name. Never perform the released code's
evaluation-label beta search on the outer test folds. If local-feature access or
published settings cannot be reproduced faithfully, document the exact limitation
and report the verified controlled ablation instead of relabelling it as PAT.

The first alignment ablation uses every nonblank template in the official PAT
`promt_list.txt` at commit `8910ebb756e33aec72e2cf6c1f02614bfa194e85`, identically
for all 50 classes. Average unit text embeddings and re-normalise the mean. Reuse
the exact cached LAION audio embeddings. No prompt subset or fusion coefficient
is selected using clean or interference labels. This is a fixed prompt-ensemble
ablation, not a reproduction of PAT's local-feature mutual alignment.

## Post-hoc timing repeat

After the first balanced CPU panel showed different frontend times for CNNs
with identical processing, a second pass reversed the model order. Both passes
use the same 250 class/fold-balanced recordings, three repetitions, one CPU
thread, batch size one, model hashes and software versions. All 10,500 timed
predictions match the formal clean evaluations. Both passes are retained;
neither is selected as the preferred latency result. This is a descriptive
order sensitivity check added after observing drift, not a randomised hardware
study or evidence that augmentation changes operation count. See
`results/extensions_20261109/cost_repeatability/` for the recorded design and audit.

## Delivery

Provide reproducible commands, manifests/provenance, uncertainty and diagnostic
tables, publication-quality figures, refreshed final-manifest listening examples,
an updated project site, written report and video/demo material. Literature must
distinguish this environmental-event classification study from speech/reasoning
benchmarks such as RSA-Bench. No result or run is described as complete until its
saved outputs have been checked.

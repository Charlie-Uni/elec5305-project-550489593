# Experiment design

[Repository home](../../README.md) · [Documentation index](../README.md) · [Results index](../../results/README.md)

Detailed instructions and dated run records moved from the original README. Run commands from the repository root; use fresh output directories when repeating experiments. Historical measurements and test counts describe the runs recorded below.

## Contents

- [Research question](#research-question)
- [Experimental scope](#experimental-scope)
- [Controls and reproducibility](#controls-and-reproducibility)
- [Implementation milestones](#implementation-milestones)

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
  and training-partition-only distractors. See the [extension protocol](../../docs/extension_protocol_20261109.md).

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
   See [full results and verification](../../results/clap_full_20261109/README.md).
7. **Expanded study:** three-seed clean CNN replication and a fixed CLAP prompt
   ensemble are complete and verified. Separate white-noise/semantic augmentation
   for CNN and SVM is complete under the declared shared settings.
   Paired bootstrap intervals, individual feature groups, 50-class CLAP ranking,
   displacement/error association and the course MATLAB STFT comparison are
   [completed extensions](../../results/extensions_20261109/).

# Results and evidence index

[Repository home](../README.md) · [Documentation](../docs/README.md) · [Interactive project page](https://charlie-uni.github.io/elec5305-project-550489593/)

## Current deliverables

| Material | Canonical file |
|---|---|
| **Main report** | **[11-page PDF](final_submission_20261110/final_report.pdf)** |
| Presentation | [12-slide PDF](final_submission_20261110/presentation.pdf) · [PowerPoint](final_submission_20261110/presentation.pptx) |
| Real code execution | [Video](final_submission_20261110/code_demo.mp4) · [local run guide](../docs/code_demo.md) |
| Narration and delivery checks | [Script](../docs/video_narration_final.txt) · [delivery evidence](final_submission_20261110/README.md) |

The short execution video has no author narration. See the
[report follow-up](../docs/report_followup_20261110.md) and
[submission checklist](../docs/final_submission_checklist_20261110.md) for remaining
editorial and author tasks. GitHub publication is not a Canvas submission.

## Canonical experimental evidence

| Evidence | Location | Coverage |
|---|---|---|
| Clean classical baseline | [SVM summary](clean_svm/summary.json) | Official five folds and out-of-fold clean results |
| Clean supervised baseline | [CNN summary](clean_cnn/summary.json) | Compact CNN, core seed 5305 |
| Frozen zero-shot reference | [CLAP integration](clap_integration_20261109/README.md) | Clean baseline, provenance and inference checks |
| Canonical interference | [Verified three-system results](clap_full_20261109/README.md) | Both own/target power definitions, white noise, DEMAND and semantic events |
| Mixture definitions | [Exact manifests](mixture_manifests_20261109/README.md) | Compressed manifests, metadata and checksums |
| Core figures | [Own power](figures_three_systems_own_20261109/README.md) · [target power](figures_three_systems_target_20261109/README.md) | Eight figures per protocol, with numerical inputs |
| Training and other extensions | [Extension evidence](extensions_20261109/README.md) | Three CNN seeds, augmentation, prompt ensemble, diagnostics and cost |
| Closing explanatory analyses | [Rank and acoustic results](closing_20261110/README.md) | Exact rank readout, spectral/temporal overlap and matched comparisons |
| Same-checkpoint local feedback | [Alignment results](extensions_20261111/clap_local_alignment/README.md) | Preset and clean-calibrated readouts, 100 checked conditions; no confirmed primary gain |
| Representative sounds | [All four targets](../examples/README.md) | Source-derived audio, matched controls, predictions and licences |

## Historical and verification records

These are retained at their original paths for traceability. They are not additional
versions of the main submission to choose between.

| Directory or record | Why it is retained |
|---|---|
| `interference_cnn_20261109/` | Earlier CNN interference measurements and diagnostic cohorts |
| `interference_v3_20261109/` | SVM/CNN power-definition comparison before full CLAP integration |
| `figures_svm_cnn_own_20261109/` | Earlier two-system figure exports and visual checks |
| `report_figure_assets_20261109/` | Figure inputs/exports used to build the detailed report |
| `presentation_20261109_draft/` | Earlier presentation, source data, narration and audit trail |
| [Report review](report_review_20261110/audit.json) | Recomputed figure claims, gain controls and other-class competition |
| [Delivery synchronisation](delivery_sync_20261110/README.md) | Checks of the earlier report, presentation, narration and public files |
| [Listening-page checks](listening_site_20261110/README.md) | Four-target reconstruction, predictions, rates and audio checks |
| [Repository integration](repository_integration_20261110.json) | Preservation of pre-existing results and example files during integration |
| [Earlier delivery audit](delivery_verification_20261109.json) · [site review](site_review_20261109.json) | Dated checks of their recorded versions |

## How to use these records

Match the cohort, power convention, seed, readout and aggregation to the claim.
A filename containing “full” does not make an intermediate table the final result.
Capture per error and capture per trial have different denominators; fold means
and pooled metrics are also distinct.

Repeated files inside standalone result packages are intentional: they preserve
plot inputs, model-specific provenance, executed source snapshots and reproducible
exports. This cleanup has not deleted or relocated original experiment files.
Historical audits continue to refer to the file versions and hashes they recorded.

Full dataset audio, trained weights and large embedding caches are external to Git;
their preparation, hashes and commands are documented in the
[reproduction guides](../docs/README.md#reproduce-the-study).

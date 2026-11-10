# Results and evidence index

[Repository home](../README.md) · [Documentation](../docs/README.md) · [Interactive project page](https://charlie-uni.github.io/elec5305-project-550489593/)

## Start with the current delivery

- [Main report: 10 pages](final_submission_20261110/final_report.pdf)
- [Technical supplement: 26 pages, unchanged](research_report_20261109_draft/research_report.pdf)
- [Presentation: 12 slides](final_submission_20261110/presentation.pptx) and [PDF](final_submission_20261110/presentation.pdf)
- [Actual code execution video](final_submission_20261110/code_demo.mp4), [run guide](../docs/code_demo.md) and [narration script](../docs/video_narration_final.txt)
- [Current delivery evidence](final_submission_20261110/README.md), [remaining report revisions](../docs/report_followup_20261110.md) and [submission checklist](../docs/final_submission_checklist_20261110.md)

The short execution video contains no author narration. It is one component of the final presentation, not confirmation of a complete narrated submission.

## Trace the experimental evidence

| Evidence | Location | Coverage |
|---|---|---|
| Clean classical baseline | [SVM summary](clean_svm/summary.json) | Official five folds and out-of-fold clean results |
| Clean supervised baseline | [CNN summary](clean_cnn/summary.json) | Compact CNN, core seed 5305 |
| Frozen zero-shot reference | [CLAP integration](clap_integration_20261109/README.md) | Clean baseline, provenance and inference checks |
| Canonical interference | [Verified three-system results](clap_full_20261109/README.md) | Own/target power, white noise, DEMAND and semantic events |
| Mixture definitions | [Exact manifests](mixture_manifests_20261109/README.md) | Compressed manifests, metadata and checksums |
| Core figures | [Own power](figures_three_systems_own_20261109/README.md) · [target power](figures_three_systems_target_20261109/README.md) | Eight figures per protocol, with numerical inputs |
| Training and other extensions | [Extension evidence](extensions_20261109/README.md) | Three CNN seeds, augmentation, prompt ensemble, diagnostics and cost |
| Final explanatory analyses | [Rank and acoustic results](closing_20261110/README.md) | Exact rank readout, spectral/temporal overlap, matched comparisons |
| Representative sounds | [Listening examples](../examples/listening_own_20261109/README.md) | Real source-derived audio, predictions and licences |

## Version boundaries

`presentation_20261109_draft/`, `delivery_sync_20261110/`, intermediate figure directories and historical reports are retained as dated evidence. The root README identifies the current primary deliverables. Do not substitute an intermediate table merely because its filename contains “full”: cohort, power convention, seed and aggregation must match the claim.

Dataset files, model weights and large embedding arrays remain outside Git. Their sources, hashes and reproduction instructions are recorded. Repository integration does not rerun experiments or replace original numerical results. Historical hash audits refer to their recorded versions; the [repository-integration record](repository_integration_20261110.json) separately checks preservation of all pre-existing result and example files.

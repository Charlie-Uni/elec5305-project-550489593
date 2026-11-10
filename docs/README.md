# Documentation index

[Repository home](../README.md) · [Results index](../results/README.md) · [Project website](https://charlie-uni.github.io/elec5305-project-550489593/)

## Reports and demonstration

| Purpose | Document |
|---|---|
| Main manuscript | [11-page PDF](../results/final_submission_20261110/final_report.pdf) · [editable Word](../results/final_submission_20261110/final_report.docx) · [LaTeX source](final_report.tex) · [generated tables](final_report_tables.tex) |
| Presentation | [12-slide PDF](../results/final_submission_20261110/presentation.pdf) · [editable PowerPoint](../results/final_submission_20261110/presentation.pptx) |
| Code demonstration | [Run guide](code_demo.md) · [actual execution video](../results/final_submission_20261110/code_demo.mp4) · [recording script](video_narration_final.txt) |
| Current report edits | [Prioritised follow-up and verified capture counts](report_followup_20261110.md) |
| Course submission | [Final checklist](final_submission_checklist_20261110.md) · [rubric review](rubric_review_20261111.md) · [80 instructor suggestions](teacher_review_20261110.md) |

The execution clip has no author narration. Experimental completion, manuscript
readiness and Canvas submission are separate states; see the current follow-up
and checklist before submitting.

## Reproduce the study

Run commands in these guides from the repository root. The detailed instructions
were moved out of the main README; dated measurements remain tied to their
original runs and cohorts.

| Step | Guide |
|---|---|
| 1. Understand the design | [Research question, folds, controls, power definitions and metrics](guides/experiment_design.md) |
| 2. Set up data and clean baselines | [Install, download ESC-50, validate DSP, train SVM/CNN and measure cost](guides/baselines.md) |
| 3. Reproduce interference and CLAP | [DEMAND, manifests, own/target evaluation, frozen CLAP and figures](guides/interference_and_clap.md) |
| 4. Reproduce extensions | [Augmentation, seeds, prompts and closing analyses](guides/extensions.md) |
| 5. Inspect implementation | [Source-code map](../src/esc_robustness/README.md) · [scripts by task](../scripts/README.md) |

## Protocols and maintenance

- [Augmentation, seeds and prompt protocol](extension_protocol_20261109.md)
- [PAT-style local alignment protocol](local_alignment_protocol_20261111.md) · [verified results and limits](../results/extensions_20261111/clap_local_alignment/README.md)
- [Ranked-readout and acoustic-overlap protocol](closing_protocol_20261110.md)
- [Verified Pages publication](pages_deployment.md)
- [Repository organisation, duplicate policy and cleanup record](repository_maintenance.md)

## Historical material and sources

- [Original proposal PDF](../ELEC5305_Project_proposal_550489593.pdf) · [source](proposal.tex) · [rebuild instructions and references](guides/proposal_and_references.md)
- [Earlier narration](video_narration_draft.txt) and [earlier presentation](../results/presentation_20261109_draft/research_presentation.pptx)
- `report_*.tex` files are build inputs for the detailed report; they are retained with its source.
- Earlier drafts and audit files record their own versions. Their dates and hashes are not a statement of the current deployment or submission status.

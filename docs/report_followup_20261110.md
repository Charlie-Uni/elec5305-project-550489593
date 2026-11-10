# Report revision record and remaining author tasks

The reporting revisions below are complete in the 11-page report, 12-slide presentation, 823-word narration and Pages. The original experiment results, execution video are unchanged; the older detailed report has been removed at the author's request. The separately approved local-feedback evaluation adds no model weights or dataset; original experiments are preserved. This is evidence of completed edits, not a predicted grade or a claim that personal submission is complete.

## Completed editorial changes

- Added introduction citations and a specific research gap limited to the reviewed methods.
- Added pooled capture rates and counts to Results 4.1, slide 6 and the website, with errors as the explicit denominator. Kept all-trial listening-example percentages separate.
- Added the original-seed high-overlap CNN contrasts, target counts and Holm qualification; retained the distinction from acoustically close subsets.
- Added published numerical context from the official ESC-50 record and the revised CLAP paper. Different pipelines, evaluation exclusions and checkpoint/prompt choices prevent controlled superiority claims.
- Added BC learning and distinguished its proportional-label objective from designated-target augmentation.
- Added CLAP margin and representation-displacement results, in the report with descriptive interpretation.
- Corrected the course title, separated conclusion from implementation/attribution and removed the production-process sentence.
- Added a first-person AI acknowledgement: OpenAI Codex, GPT-6 Astra (confirmed by the author); Anthropic Claude contributions, model version not recorded. The declaration explicitly includes substantial code, analyses and document preparation, beyond language editing.

Primary references: [ESC-50 record](https://github.com/karolpiczak/ESC-50), [CLAP revised v4, Table 4 / Section 4.3 / Appendix H](https://arxiv.org/html/2211.06687v4), and [BC learning](https://arxiv.org/abs/1711.10282). CLAP reports 88.0–91.0% across selected models after overlap exclusions; the paper does not identify those rows as this project's exact checkpoint/protocol. The project's 81.75% and 92.90% use all 2,000 clips.

## Remaining author actions

Review the final acknowledgement and contribution statements, rehearse and record the narrated video with an explanation of real code execution, update the report/site to link the complete video, and upload to Canvas with a receipt. The existing 40.8-second execution clip has no author narration and is not a complete video report.

## Verified capture-rate inputs

Own power, +5 dB, core clean-trained systems, all 2,000 targets per condition. These are pooled counts across folds, not a mean of fold-specific ratios.

| System | Same category: captures / errors | Different category: captures / errors |
|---|---|---|
| SVM | 283/1527 = 18.53% | 246/1548 = 15.89% |
| CNN | 403/1289 = 31.26% | 362/1315 = 27.53% |
| CLAP | 632/944 = 66.95% | 612/1015 = 60.30% |

Source: [original capture table](../results/figures_three_systems_own_20261109/fig2_capture_rate_vs_sir.csv). The separate unconditional capture fraction divides by all trials and must not be substituted here.

For the core CNN seed 5305, own/+5 dB/high-overlap: semantic-minus-DEMAND accuracy is −3.28 percentage points on 1,464 same-category targets and −2.94 points on 1,465 different-category targets. Both Holm-adjusted p-values are approximately 0.0604 in the declared 12-comparison family. The two additional seeds support directional replication; do not conflate these populations with the smaller acoustic-closeness subsets. Source: [replication table](../results/extensions_20261109/clean_replication/cnn_overlap_replication.csv).

## AI acknowledgement requirements

The University's [Current Students guidance](https://www.sydney.edu.au/students/responsible-ai-use.html) asks for the tools and versions, publishers, URLs, and a brief explanation of use, alongside any course-specific requirements. Here AI assistance included substantial code implementation/revision, analysis orchestration and checking, figure/report/slide generation and text drafting. It should not be described merely as spelling correction. The author confirmed Codex GPT-6 Astra; the Claude model version was not recorded. The author should check the final statement for completeness. Keep drafts, prompts/outputs and research materials; do not publish private conversation logs indiscriminately.

## Verification and preservation

The [current editorial audit](../results/final_submission_20261110/single_report_delivery_audit.json) checks report values, native slide charts/tables, exact narration, website selector states and preserved experiment/example files against repository commit `8cec874`. Earlier dated audits remain records of their respective versions. The source builders regenerate the delivery from unchanged evidence; the newly added capture/margin/displacement data are shared across the report, deck and page.

## Single-document revision — 11 November 2026

The formal report is now 11 pages with five figures. Capture/margin and representation-displacement diagnostics are in the main report. The new [rubric review](rubric_review_20261111.md) checks all eight criteria. Two submitted patches were applied, then wording was corrected for limited augmentation transfer, outcome-conditioned medians, power-protocol sensitivity and unconfirmed model-version details.

## Same-checkpoint local feedback — 11 November 2026

The full own/target runs and independent 100-condition audit are complete. The report now adds a compact four-readout table; slide 11, exact narration and Pages use the same audited values. Preset/calibrated own/+5-dB accuracy changes versus the ensemble are −0.05/−0.10 pp for same-category and +0.20/+0.75 pp for different-category interference. No primary test passes Holm correction (minimum adjusted p = 0.7346). This is an inconclusive result for a LAION adaptation, not official/full PAT or evidence of equivalence. Both same-category target-top-two percentages decline descriptively.

A separate 4,500-query balanced panel preserves both orders and bypasses input caching. New reference/preset/calibrated medians are 59.16/61.74/61.85 ms forward and 60.34/63.01/63.03 ms reversed. Setup and external pretraining are excluded. The author also confirmed representative listening. Author narration and Canvas submission remain.

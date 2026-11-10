# Report revision record and remaining author tasks

The reporting revisions below are complete in the 10-page report, 12-slide presentation, 822-word narration and Pages. The original experiment results, execution video and 26-page supplement are unchanged. No model, dataset or experiment was added. This is evidence of completed edits, not a predicted grade or a claim that personal submission is complete.

## Completed editorial changes

- Added introduction citations and a specific research gap limited to the reviewed methods.
- Added pooled capture rates and counts to Results 4.1, slide 6 and the website, with errors as the explicit denominator. Kept all-trial listening-example percentages separate.
- Added the original-seed high-overlap CNN contrasts, target counts and Holm qualification; retained the distinction from acoustically close subsets.
- Added published numerical context from the official ESC-50 record and the revised CLAP paper. Different pipelines, evaluation exclusions and checkpoint/prompt choices prevent controlled superiority claims.
- Added BC learning and distinguished its proportional-label objective from designated-target augmentation.
- Added CLAP margin and representation-displacement results, with supplement pointers and descriptive interpretation.
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

The [current editorial audit](../results/final_submission_20261110/editorial_delivery_audit.json) checks report values, native slide charts/tables, exact narration, website selector states and preserved experiment/example files against repository commit `8cec874`. Earlier dated audits remain records of their respective versions. The source builders regenerate the delivery from unchanged evidence; the newly added capture/margin/displacement data are shared across the report, deck and page.

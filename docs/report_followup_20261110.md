# Remaining report revisions after repository integration

The experiments and evidence are complete; these are reporting changes, not new modelling work. The existing 10-page report, 26-page supplement, decks and original data have not been rewritten during this repository-integration step. The 26-page supplement stays unchanged. Do not describe the current manuscript as meeting every grading descriptor.

## Prioritised edits

1. **Final narrated video.** Record the author's explanation of signal processing and actual execution, then update the report/site video links and submit via Canvas. The current 40.8-second clip is genuine execution with audio examples but no narration.
2. **Error destinations in Results 4.1.** Add the compact capture-rate table below, retaining the denominator as errors. Include a numerical high-overlap CNN contrast, its target set and multiplicity qualification.
3. **Introduction.** Add relevant citations and state the specific gap in the introduction. Avoid the universal claim that no earlier work has compared these phenomena; limit the statement to the reviewed methods and the controlled comparison undertaken here.
4. **Literature and numerical context.** Verify published ESC-50 benchmarks against primary sources and distinguish accuracy from macro-F1, data/training differences and human evaluation conditions. Add Tokozume et al.'s between-class learning and explain how its mixture-label training objective differs from this project's designated-target augmentation.
5. **CLAP comparison.** Identify the exact published checkpoint, prompt and protocol before comparing a reported score to 81.75%. Checkpoint/prompt differences are possible explanations, not demonstrated causes. The fixed prompt ensemble's measured clean accuracy is 92.90%.
6. **Method/result balance.** Give brief results or explicit supplement pointers for the score margin and representation displacement already defined in Methods.
7. **Format and declaration.** Correct the manuscript's course title to “Acoustics, Speech and Signal Processing”; separate conclusion from implementation/demo/attribution; remove production-process sentences. Replace the generic AI note with a truthful statement identifying tools, versions, publishers, URLs and actual scope. Do not invent an unknown Claude version or claim personal verification that the author has not performed.

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

The University's [Current Students guidance](https://www.sydney.edu.au/students/responsible-ai-use.html) asks for the tools and versions, publishers, URLs, and a brief explanation of use, alongside any course-specific requirements. Here AI assistance included substantial code implementation/revision, analysis orchestration and checking, figure/report/slide generation and text drafting. It should not be described merely as spelling correction. The author must confirm the exact tools/versions used and the final statement. Keep drafts, prompts/outputs and research materials; do not publish private conversation logs indiscriminately.

## Scope of this integration

The default repository branch now carries the working implementation and existing reviewed deliverables. Previous feature-branch links, reports, figures and audit files are retained. Dated audits refer to the files/commits they checked; this checklist records subsequent open editorial items. Canvas submission and final personal narration are separate actions.

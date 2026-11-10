# Final submission checklist

2026-11-10. The deliverables have been shortened and synchronised against the final project instructions and rubric supplied by the user. The experimental scope is closed: no further models, datasets or training will be added. The repository has been integrated into `main`; the [report revision record](report_followup_20261110.md) now records the completed editorial changes. Personal review, narrated video and formal submission remain. See the [rubric review](rubric_review_20261110.md) for item-by-item evidence.

## Materials ready for review

- [Formal report](../results/final_submission_20261110/final_report.pdf): 10 pages, including the contents, literature review, methods, selected results, discussion and 13 references. An [editable Word companion](../results/final_submission_20261110/final_report.docx) has the same content; the PDF fixes the submission layout.
- [Technical supplement](../results/research_report_20261109_draft/research_report.pdf): the original 26-page detailed report, preserved byte for byte.
- [Editable presentation](../results/final_submission_20261110/presentation.pptx) and [PDF](../results/final_submission_20261110/presentation.pdf): 12 slides, with editable charts, notes and two real audio clips.
- [Final video narration](video_narration_final.txt): 822 English words, estimated at 6.0–6.8 minutes including planned demonstration pauses; actual speaking speed and loading may increase this. The course instructions give no specific duration limit.
- [Actual code execution video](../results/final_submission_20261110/code_demo.mp4) and [run guide](code_demo.md): reload source audio, mix it, verify SIR, run SVM/CNN/CLAP and check against historical results. The video includes actual audio examples but no author narration.
- [Pages](https://charlie-uni.github.io/elec5305-project-550489593/): the main report, supplement, presentation, narration, actual execution clip, interactive results and code links.
- [New delivery verification record](../results/final_submission_20261110/editorial_delivery_audit.json); the [two original closing analyses](../results/closing_20261110/README.md) and original experimental results are retained.

## Remaining author tasks

1. **Review and rehearse**: confirm understanding and contribution attribution, listen to the examples, and be able to explain own/target, capture denominators, top-2 limitations and the acoustically matched subset. Check the disclosure against the course AI rules.
2. **Record the complete brief video**: explain the project using the synchronised narration and 12-slide presentation, then switch to actual code execution or clearly identify playback of the recorded execution clip. Retain the demonstration and audio; do not substitute result screenshots for running code. Check the complete exported video.
3. **Follow the actual Canvas assignment page**: the deadline, closing date and permitted PDF, MP4 and ZIP formats have been checked in the current account. Personal page records remain local, and personal extension information is not published. File size limits, applicable AI rules and differences in weight/page guidance between the outline and assignment instructions still need confirmation.
4. **Complete formal submission**: update the report and Pages with the final narrated video link, upload the report and video to Canvas, and retain the receipt. GitHub/Pages publication does not replace Canvas upload. The complete code and existing deliverables have been integrated into `main` through PR #1; the feature branch is retained so that older links remain valid.

## Statistical interpretations to preserve

- Own, +5 dB, single-template CLAP: the target remains second in 236/632 = 37.34% and 203/612 = 33.17% of captures; the denominator is the number of captures.
- Across all semantic trials, both labels appear in the top two in 26.05% / 21.00%; only the distractor appears in 23.35% / 24.05%. Ranking is not calibrated event detection or a direct measurement of subjectively “hearing” an event.
- Within the acoustically matched subsets under both power definitions, CLAP is about 16–26 accuracy percentage points below DEMAND; the CNN intervals include zero. Only about 9–10.5% of targets are retained, so this cannot be generalised as a causal decomposition.
- The median raw Mel correlation for white noise is about −0.728, and about −0.008 after bandwidth normalisation. Regression coefficient changes cannot be interpreted as a “percentage explained by a mechanism”.
- The high temporal overlap and acoustically matched subsets contain different target cohorts; all results are retained. The fixed prompt ensemble is not full PAT, and Mix-CLAP was not reproduced.

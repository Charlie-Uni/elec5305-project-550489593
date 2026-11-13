# Final submission checklist

Updated 2026-11-13. The final report and video are synchronised against the project instructions and rubric supplied by the user. The experimental scope is closed: no further models, datasets or training will be added. The final subtitled video is linked in the report and embedded on the project website. Personal review and formal Canvas submission remain. See the [rubric review](rubric_review_20261111.md) for item-by-item evidence.

## Materials ready for review

- [Formal report](../results/final_submission_20261110/final_report.pdf): 11 pages, including the contents, literature review, methods, selected results, discussion and 17 references. The review includes Mix-CLAP, RSA-Bench and AIP-Speech (2026), with their task and protocol differences stated. An [editable Word companion](../results/final_submission_20261110/final_report.docx) has the same content; the PDF fixes the submission layout.
- [Editable presentation](../results/final_submission_20261110/presentation.pptx) and [PDF](../results/final_submission_20261110/presentation.pdf): 12 slides, with editable charts, notes and two real audio clips.
- [Final video narration](video_narration_final.txt): 823 English words, estimated at 6.0–6.8 minutes including planned demonstration pauses; actual speaking speed and loading may increase this. The course instructions give no specific duration limit.
- [Final project video](../results/final_submission_20261110/project_video.mp4): the author's supplied v13 export, 5 min 50.4 s, 1920 × 1080, H.264/AAC, with subtitles and a short code demonstration. The earlier script is a preparation document; the exported video is the final recording.
- [Actual code execution video](../results/final_submission_20261110/code_demo.mp4) and [run guide](code_demo.md): reload source audio, mix it, verify SIR, run SVM/CNN/CLAP and check against historical results. The video includes actual audio examples but no author narration.
- [Pages](https://charlie-uni.github.io/elec5305-project-550489593/): the single formal report, presentation, narration, actual execution clip, interactive results and code links.
- [New delivery verification record](../results/final_submission_20261110/single_report_delivery_audit.json); the [two original closing analyses](../results/closing_20261110/README.md) and original experimental results are retained.

The author confirmed representative listening on 11 November: sounds are recognisable without obvious distortion. The separately approved local-feedback extension is complete and audited; none of its four primary contrasts passes Holm correction.

## Remaining author tasks

1. **Review and rehearse**: confirm understanding and contribution attribution, and be able to explain own/target, capture denominators, top-2 limitations and the acoustically matched subset. Check the disclosure against the course AI rules.
2. **Review the final exported video**: confirm its speech, subtitles, demonstrations and sound examples are clear. The final video has been supplied; recording a replacement is not an outstanding task.
3. **Follow the actual Canvas assignment page**: the deadline, closing date and permitted PDF, MP4 and ZIP formats have been checked in the current account. Personal page records remain local, and personal extension information is not published. File size limits, applicable AI rules and differences in weight/page guidance between the outline and assignment instructions still need confirmation.
4. **Complete formal submission**: upload the final report and video to Canvas and retain the receipt. The report and Pages provide the final video link; GitHub/Pages publication does not replace Canvas upload. The feature branch is retained so that older links remain valid.

## Statistical interpretations to preserve

- Own, +5 dB, single-template CLAP: the target remains second in 236/632 = 37.34% and 203/612 = 33.17% of captures; the denominator is the number of captures.
- Across all semantic trials, both labels appear in the top two in 26.05% / 21.00%; only the distractor appears in 23.35% / 24.05%. Ranking is not calibrated event detection or a direct measurement of subjectively “hearing” an event.
- Within the acoustically matched subsets under both power definitions, CLAP is about 16–26 accuracy percentage points below DEMAND; the CNN intervals include zero. Only about 9–10.5% of targets are retained, so this cannot be generalised as a causal decomposition.
- The median raw Mel correlation for white noise is about −0.728, and about −0.008 after bandwidth normalisation. Regression coefficient changes cannot be interpreted as a “percentage explained by a mechanism”.
- The high temporal overlap and acoustically matched subsets contain different target cohorts; all results are retained. The fixed prompt ensemble is not full PAT, and Mix-CLAP was not reproduced.

The new [alignment evidence](../results/extensions_20261111/clap_local_alignment/README.md) retains both full power protocols and separate balanced query costs. Preset feedback is label-free; clean-calibrated feedback is not zero-shot. No further expansion is planned.

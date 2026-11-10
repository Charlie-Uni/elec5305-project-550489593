# Final concise delivery — 10 November 2026

Start with [the 10-page report](final_report.pdf), [12-slide editable presentation](presentation.pptx) or its [PDF](presentation.pdf). The [728-word recording script](../../docs/video_narration_final.txt) matches the slide notes exactly. The [unchanged 26-page report](../research_report_20261109_draft/research_report.pdf) remains the technical supplement.

[The 40.8-second execution video](code_demo.mp4) shows real source reconstruction and fresh SVM/CNN/CLAP inference, followed by playback of the control and mixture. It has sound examples but no author narration. [Run instructions](../../docs/code_demo.md) explain the required environments/data/checkpoints and comparison outputs. The video is not a claim that the author-recorded full presentation or Canvas submission is complete.

The main report selects three figures and three tables. PNG/PDF/SVG, grayscale previews and exact source-row CSVs are in `figures/`. Existing detailed figures and original experiment results are preserved. An old 22.49% SVM same-category value used a common retained-target intersection; the new full-2,000-target main figure and corresponding prose consistently use 22.47%.

Validation includes 96 passed tests (one CUDA skip), three real model executions, 20 native chart values, 30 table cells, six editable interval lines, 12 exact narration segments and four website selector states. All 676 pre-existing result/example files match their baseline Git blobs; the original 26-page PDF has SHA-256 `e5ebcef99033065fdaea048b1a2debfdb3ad90c58710ad45b80adc98cd693e35`.

- [Public deployment hash verification](public_deployment_audit.json)
- [Numerical and preservation audit](delivery_audit.json)
- [Rendering/visual audit](visual_audit.json)
- [Execution-video metadata](code_demo_audit.json)
- [Rubric and formal requirements](../../docs/rubric_review_20261110.md)
- [Remaining submission checklist](../../docs/final_submission_checklist_20261110.md)

Source authority: the user's supplied final project requirements and marking rubric. The report's approximately ten-page guidance is now met. The source says “brief” for video, Week 13 end for deadline and 30% of the course mark; an exact video duration, deadline time, file format/size and AI policy were not provided. These are not inferred from the earlier proposal guide. No new experiments, models or datasets are planned.

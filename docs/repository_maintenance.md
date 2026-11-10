# Repository organisation and preservation

[Repository home](../README.md) · [Documentation](README.md) · [Results index](../results/README.md)

## Where to start

- The root README is the short assessment entry point: main report, website, code, presentation, execution video and supplement.
- `docs/guides/` holds detailed setup, experimental design and reproduction instructions previously embedded in the long root README. Commands still run from the repository root.
- [The source-code map](../src/esc_robustness/README.md) follows the signal-processing and evaluation pipeline.
- [The script index](../scripts/README.md) groups supporting tools by task, separating current delivery builders from historical ones.
- [The results index](../results/README.md) distinguishes canonical evidence from earlier runs and verification records.
- [The listening index](../examples/README.md) covers both audio collections and their attribution.

## Canonical deliverables

Keep one canonical file for each deliverable and link to it from the indexes:

| Deliverable | Stable location |
|---|---|
| Main report | [10-page PDF](../results/final_submission_20261110/final_report.pdf) |
| Detailed supplement | [Original 26-page PDF](../results/research_report_20261109_draft/research_report.pdf) |
| Presentation | [PDF](../results/final_submission_20261110/presentation.pdf) · [editable deck](../results/final_submission_20261110/presentation.pptx) |
| Actual execution clip | [Video](../results/final_submission_20261110/code_demo.mp4) |
| Recording script | [Final narration](video_narration_final.txt) |

The word `draft` in the supplement's historical directory is retained to preserve
existing report links and audit references. Its role is the technical supplement;
it is not the primary manuscript. No duplicate submission folder is needed.

## What was cleaned on 10 November 2026

The long root README was split into five topic guides, with relative links adjusted.
The main README and directory indexes now identify primary deliverables first.
The generic ignore template was replaced with project-relevant rules for Python,
external research assets, generated runs, editor files and typesetting caches.

73 ignored local files (2,009,542 bytes) were moved out of the working tree into a
recoverable archive under the local runtime's
`runs/repository_cleanup_20261110/removed_local_files/`. They comprised old Python
build copies, test/lint/bytecode caches, temporary report build files, an OS metadata
file and a generated report log. The temporary PDF was verified byte-identical to
the retained technical supplement before removal from the working tree.
Local model artifacts, environments and experiment caches were kept in place.
This local housekeeping does not remove files from repository history.

## Duplicates that are evidence

Identical bytes are not sufficient grounds for deletion. These copies are retained:

- Matched-control WAVs belong to specific manifest rows; the recorded demonstration also carries its own verified audio inputs.
- Plot CSVs accompany exported figures, while the corresponding canonical tables remain in their result packages.
- Dataset and benchmark-panel provenance appears in each model's standalone output.
- `executed_features/` records the source used for an analysis, even when it matches the current implementation.
- Earlier presentation data and dated audits describe the deliverables they verified.

Every pre-existing result file other than the navigation `results/README.md`, all
example files, model/source code and the report/presentation/video artifacts are
preserved byte for byte in this cleanup. Historical hash records continue to refer
to their recorded versions; they are not rewritten to make old audits describe a
new README or deployment.

## Future maintenance

Use fresh output directories for new runs. Keep full datasets, weights and large
caches outside Git; retain the configurations and provenance needed to reconstruct
them. Store scratch work under ignored `tmp/` or outside the checkout. Do not add
blanket ignore rules for `results/`, `examples/`, PDFs, WAVs or CSVs.

Before publishing, check local document links and compare protected artifact hashes.
The [Pages workflow](pages_deployment.md) packages the selected public resources and
verifies all 96 listening WAVs. Publish source changes through `main`; retain stable
result URLs and keep Canvas submission as a separate author action.

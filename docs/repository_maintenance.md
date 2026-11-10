# Repository organisation and preservation

[Repository home](../README.md) · [Documentation](README.md) · [Results index](../results/README.md)

## Where to start

- The root README is the short assessment entry point: formal report, website, code, presentation and execution video.
- `docs/guides/` holds detailed setup, experimental design and reproduction instructions previously embedded in the long root README. Commands still run from the repository root.
- [The source-code map](../src/esc_robustness/README.md) follows the signal-processing and evaluation pipeline.
- [The script index](../scripts/README.md) groups supporting tools by task, separating current delivery builders from historical ones.
- [The results index](../results/README.md) distinguishes canonical evidence from earlier runs and verification records.
- [The listening index](../examples/README.md) covers both audio collections and their attribution.

## Canonical deliverables

Keep one canonical file for each deliverable and link to it from the indexes:

| Deliverable | Stable location |
|---|---|
| Sole formal report | [11-page PDF](../results/final_submission_20261110/final_report.pdf) · [editable Word](../results/final_submission_20261110/final_report.docx) |
| Presentation | [PDF](../results/final_submission_20261110/presentation.pdf) · [editable deck](../results/final_submission_20261110/presentation.pptx) |
| Actual execution clip | [Video](../results/final_submission_20261110/code_demo.mp4) |
| Recording script | [Final narration](video_narration_final.txt) |

On 11 November the author requested removal of the retired 26-page report. Its
PDF, dedicated LaTeX inputs, build records, builders and obsolete delivery checker
were removed. Navigation and the current preservation audit now reflect that
decision; the 11-page report and its Word companion are the sole current manuscript.
Original numerical results, figures, model evidence, audio and presentation remain.
Earlier report versions are recoverable through Git history, which was not rewritten.

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
the then-retained detailed report before that temporary duplicate was removed.
Local model artifacts, environments and experiment caches were kept in place.
This local housekeeping does not remove files from repository history.

## Duplicates that are evidence

Identical bytes are not sufficient grounds for deletion. These copies are retained:

- Matched-control WAVs belong to specific manifest rows; the recorded demonstration also carries its own verified audio inputs.
- Plot CSVs accompany exported figures, while the corresponding canonical tables remain in their result packages.
- Dataset and benchmark-panel provenance appears in each model's standalone output.
- `executed_features/` records the source used for an analysis, even when it matches the current implementation.
- Earlier presentation data and dated audits describe the deliverables they verified.

The 10 November cleanup preserved pre-existing results and examples byte for byte.
The later report retirement is a separate, explicit exception: four obsolete report
outputs were removed and navigation was updated. Historical hash records continue
to refer to their recorded versions; they are not rewritten to describe a new
README or deployment. The current audit lists the authorised removals and preserves
all other protected experiment and example files.

## Future maintenance

Use fresh output directories for new runs. Keep full datasets, weights and large
caches outside Git; retain the configurations and provenance needed to reconstruct
them. Store scratch work under ignored `tmp/` or outside the checkout. Do not add
blanket ignore rules for `results/`, `examples/`, PDFs, WAVs or CSVs.

Before publishing, check local document links and compare protected artifact hashes.
The [Pages workflow](pages_deployment.md) packages the selected public resources and
verifies all 96 listening WAVs. Publish source changes through `main`; retain stable
result URLs and keep Canvas submission as a separate author action.

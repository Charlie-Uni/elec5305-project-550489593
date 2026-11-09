# v3 own-region SVM/CNN figures — 9 November 2026

These eight figures use the real v3 own-region evaluation: **24,000 mixtures per
system, 2,000 targets, four interference types, three levels and five official
folds**. They do not yet include full CLAP interference results. CLAP's own/target
runs were started separately using the frozen v4 evaluator snapshot.

The supplied figure patch was applied and run unchanged first. All eight original
CSV tables reproduced the supplied preview package (absolute numerical tolerance
1e-10). The reviewed figures below incorporate the methodological and display
changes described here. No SVM/CNN predictions or mixture manifests were changed.

## Figure files

Each stem has a 600-DPI PNG, a grayscale PNG, an SVG, a PDF and its data CSV. Figures
are 7.16 inches wide; SVG text remains editable and PDF TrueType fonts are embedded.

| Figure | Main question | Files |
| --- | --- | --- |
| 1 | How does macro-F1 change with interference level? | [PNG](fig1_macro_f1_vs_level.png) · [PDF](fig1_macro_f1_vs_level.pdf) · [CSV](fig1_macro_f1_vs_level.csv) |
| 2 | What fraction of errors name the distractor? | [PNG](fig2_capture_rate_vs_sir.png) · [PDF](fig2_capture_rate_vs_sir.pdf) · [CSV](fig2_capture_rate_vs_sir.csv) |
| 3 | How does the target–distractor score margin shift? | [PNG](fig3_margin_vs_sir.png) · [PDF](fig3_margin_vs_sir.pdf) · [CSV](fig3_margin_vs_sir.csv) |
| 4 | How do same/different-category losses compare on identical targets? | [PNG](fig4_same_vs_different.png) · [PDF](fig4_same_vs_different.pdf) · [CSV](fig4_same_vs_different.csv) |
| 5 | How far does each representation move from its gain-matched control? | [PNG](fig5_representation_displacement.png) · [PDF](fig5_representation_displacement.pdf) · [CSV](fig5_representation_displacement.csv) |
| 6 | Which class pairs account for the most observed captures? | [PNG](fig6_confusion_flow.png) · [PDF](fig6_confusion_flow.pdf) · [CSV](fig6_confusion_flow.csv) |
| 7 | How does semantic robustness compare with CPU inference cost? | [PNG](fig7_robustness_vs_cost.png) · [PDF](fig7_robustness_vs_cost.pdf) · [CSV](fig7_robustness_vs_cost.csv) |
| 8 | Does temporal overlap change the paired semantic/DEMAND comparison? | [PNG](fig8_overlap_strata.png) · [PDF](fig8_overlap_strata.pdf) · [CSV](fig8_overlap_strata.csv) |

## Interpretation supported by matched targets

At +5 dB and overlap >= 0.8, the following CNN comparisons use exactly the same
targets for semantic interference and DEMAND:

| Pairing | Targets | Semantic accuracy | Matched DEMAND accuracy | Semantic median cosine distance | Matched DEMAND median distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same category | 1,464 | 29.10% | 32.38% | 0.1098 | 0.1698 |
| Different category | 1,465 | 28.40% | 31.33% | 0.1192 | 0.1686 |

The semantic distance is smaller on 71.58% and 66.55% of individual paired targets,
respectively. This supports the conclusion that **displacement magnitude alone is
insufficient to explain CNN failure**. It does not prove that semantic meaning is
the sole causal factor: spectral structure, temporal structure and the mixture
protocol also differ. The same high-overlap accuracy reversal is not present for
SVM: its semantic accuracies are 19.67% / 19.25%, versus matched DEMAND 18.51% / 17.75%.
See `representation_overlap_comparison.csv` and `fig8_matched_trials.csv`.

The CNN capture examples are real but require a control qualification:

| Target → distractor, +5 dB | Captured mixtures / pairs | Correct gain-matched controls | New captures from a correct control |
| --- | ---: | ---: | ---: |
| can_opening → clock_alarm | 6/6 | 4/6 | 4 |
| water_drops → crickets | 6/6 | 1/6 | 1 |

These were selected after observing the results and have small sample counts. They
are illustrative examples, not evidence that every original classification was
correct or that the corresponding classes are universally captured.

## Changes from the incoming preview

- **Figure 1:** clean is a separate control point, avoiding an implied finite SNR
  for clean audio. The three corrupted levels remain connected. Error bars are
  descriptive sample SD across five folds, not confidence intervals.
- **Figure 4:** individual fold losses are visible. Same/different comparisons use
  the intersection of retained targets. Aggregate markers retain loss of fold-mean
  F1; individual dots are per-fold losses. These are different summaries and are
  labelled explicitly.
- **Figure 5:** the 92 handcrafted features use each outer model's training scaler,
  rather than estimating a new scale from all held-out controls. The measure is
  `sqrt(mean(((mixture - matched_control) / training_scale)^2))`. CNN/CLAP use
  cosine distance. Distances are never compared numerically across models.
- **Figure 6:** the CSV adds correctly classified controls and new captures. Bar
  heights remain all observed distractor captures, matching the original definition.
- **Figure 7:** model-file bytes, checkpoint bytes and parameter bytes are separate
  columns. The plot uses total preprocessing-plus-inference time, because frontend
  timing boundaries differ across systems. Existing cost files are single-clip,
  one-thread CPU microbenchmarks on the same platform, not dataset latency distributions.
- **Figure 8:** the display uses paired-target accuracy and shows individual folds.
  Macro-F1 and per-fold class coverage remain in CSV. Low-overlap strata have missing
  classes in some folds, so cross-stratum macro-F1 alone is not a difficulty measure.
- Shorter captions, aligned panel labels, visible legends and fixed final-size
  canvases replace oversized titles that widened several original exports.

All 0-dB semantic mixtures are stress tests with ambiguous single-label ground
truth. The 1/49 line in Figure 2 assumes uniform choice among wrong classes; it is
not an empirical null distribution or a significance test. No significance tests
or causal-effect estimates are claimed.

## Validation and reproducibility

- **75 tests passed, one CUDA-only test skipped; Ruff and `uv lock --check` passed.**
- `uv.lock` adds figure dependencies without changing any existing package version.
- `numerical_audit.json`: independent checks of all eight tables, including F1 from
  confusion counts, margins from class-score arrays, training-scaled distances,
  control transitions, cost arithmetic and overlap-matched target sets.
- `visual_qa.json`: no missing-glyph, clipping or tick-overlap findings in the final
  eight previews. Colour and grayscale exports were visually reviewed.
- `pdf_font_audit.json`: all PDF fonts are embedded and all canvases are 7.16 inches
  wide. The generic checker warned about parent Type0 fonts; inspection of their
  descendant font descriptors confirmed embedded font streams.
- `figures_meta.json`: input file hashes, model-scaler hashes, source hash and run settings.
- `data_profile_summary.json`: group counts and field interpretation. Class IDs and
  fold IDs are identifiers, not continuous quantities; blank distractor scores in
  white/DEMAND rows are expected.

To regenerate the two-system figures, use the `make-figures` example in the main
README. To add CLAP, pass its completed run with `--additional-evaluation`, alongside
`--clap-clean` and `--clap-cost`. The loader requires the identical manifest hash
and trial identities; it does not silently combine unmatched experiments. Use
`--svm-models` if the original model directory has moved.

The CLAP process logs and progress files are under
`/Users/leaf/Projects/elec5305-local/runs/clap_v4_20261109/`.
This figure delivery does not mark the CLAP experiment, publication or final report
as complete.

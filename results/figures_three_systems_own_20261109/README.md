# Three systems — own power region

Complete own evaluation: SVM, compact CNN and frozen zero-shot CLAP.

[Full results, methods and interpretation](../clap_full_20261109/README.md)

| Figure | Preview | Vector | Numbers |
| --- | --- | --- | --- |
| Macro-F1 versus level | [PNG](fig1_macro_f1_vs_level.png) | [PDF](fig1_macro_f1_vs_level.pdf) · [SVG](fig1_macro_f1_vs_level.svg) | [CSV](fig1_macro_f1_vs_level.csv) |
| Distractor capture rate | [PNG](fig2_capture_rate_vs_sir.png) | [PDF](fig2_capture_rate_vs_sir.pdf) · [SVG](fig2_capture_rate_vs_sir.svg) | [CSV](fig2_capture_rate_vs_sir.csv) |
| Target–distractor margin | [PNG](fig3_margin_vs_sir.png) | [PDF](fig3_margin_vs_sir.pdf) · [SVG](fig3_margin_vs_sir.svg) | [CSV](fig3_margin_vs_sir.csv) |
| Same- versus different-category degradation | [PNG](fig4_same_vs_different.png) | [PDF](fig4_same_vs_different.pdf) · [SVG](fig4_same_vs_different.svg) | [CSV](fig4_same_vs_different.csv) |
| Representation displacement | [PNG](fig5_representation_displacement.png) | [PDF](fig5_representation_displacement.pdf) · [SVG](fig5_representation_displacement.svg) | [CSV](fig5_representation_displacement.csv) |
| Exploratory capture flows | [PNG](fig6_confusion_flow.png) | [PDF](fig6_confusion_flow.pdf) · [SVG](fig6_confusion_flow.svg) | [CSV](fig6_confusion_flow.csv) |
| Robustness versus CPU cost | [PNG](fig7_robustness_vs_cost.png) | [PDF](fig7_robustness_vs_cost.pdf) · [SVG](fig7_robustness_vs_cost.svg) | [CSV](fig7_robustness_vs_cost.csv) |
| Overlap-stratified paired accuracy | [PNG](fig8_overlap_strata.png) | [PDF](fig8_overlap_strata.pdf) · [SVG](fig8_overlap_strata.svg) | [CSV](fig8_overlap_strata.csv) |

Figures 5–8 focus on +5 dB. Figures 1–4 include +10/+5/0 dB. The source
figures and raw tables have been preserved; these outputs occupy a new directory.

- [Numerical verification](numerical_audit.json)
- [Layout checks](visual_qa.json)
- [Export and embedded-font checks](export_audit.json)
- [Input hashes and plotting provenance](figures_meta.json)

Use the vector files at their intended report width. Grayscale previews retain
system markers and same/different or correct/incorrect line and fill cues.

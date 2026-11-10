# Scripts by task

[Repository home](../README.md) · [Source-code map](../src/esc_robustness/README.md) · [Reproduction guides](../docs/README.md#reproduce-the-study)

Use the guides for the required environment, inputs and invocation order. Training and evaluation commands are primarily exposed through `esc-baseline`; this directory contains supporting scripts. Some delivery builders write to their canonical output directories, so review their output settings before rebuilding a published artifact. Ordinary browsing and the recorded demonstration need no rebuild.

For a live demonstration, start with [the run guide](../docs/code_demo.md). For publication, use [the Pages guide](../docs/pages_deployment.md).

## Data and the actual demonstration

| Script | Purpose |
|---|---|
| [download_esc50.py](download_esc50.py) | Download a pinned official ESC-50 snapshot (roughly 600 MB) outside Git tracking. |
| [restore_manifests.py](restore_manifests.py) | Restore the exact evaluated mixture CSVs and relocate local data paths. |
| [run_code_demo.py](run_code_demo.py) | Recompute one published mixture and its predictions, then check saved evidence. |
| [serve_code_demo.py](serve_code_demo.py) | Serve a local recording interface that launches actual inference on demand. |
| [validate_course_stft.py](validate_course_stft.py) | Run the unmodified course STFT in MATLAB, then compare aligned complex bins. |

## Current report, presentation and website

| Script | Purpose |
|---|---|
| [build_final_docx.py](build_final_docx.py) | Create the editable Word companion from the same concise LaTeX source using Pandoc and python-docx. |
| [build_final_report.py](build_final_report.py) | Build the concise submission report from unchanged, audited experiment tables. |
| [build_final_presentation.cjs](build_final_presentation.cjs) | Build the 12-slide editable final presentation. |
| [build_final_narration.py](build_final_narration.py) | Keep the concise recording guide identical to the deck's speaker notes. |
| [build_local_alignment_delivery.py](build_local_alignment_delivery.py) | Add independently verified alignment values to report tables, presentation data and website data. |
| [update_local_alignment_deck.mjs](update_local_alignment_deck.mjs) | Author the native alignment table with Artifact Tool, preserve original charts/audio and update exact narration. Run after generating the base deck, not before. |
| [build_site_data.py](build_site_data.py) | Generate the project site's numbers from saved research outputs. |
| [build_listening_examples.py](build_listening_examples.py) | Export final-manifest examples and matched controls with source attribution. |
| [build_listening_context.py](build_listening_context.py) | Write the context shown beside the website's listening examples. |
| [prepare_pages.py](prepare_pages.py) | Package tracked public files and verify the complete listening collection. |
| [package_research_evidence.py](package_research_evidence.py) | Package verified tables, per-trial predictions and author-review deliverables. |

## Numerical and delivery verification

| Script | Purpose |
|---|---|
| [audit_single_report.py](audit_single_report.py) | Independently verify the added diagnostic figures and single-report claims. |
| [audit_final_delivery.py](audit_final_delivery.py) | Check the report, slide charts, narration and website against saved evidence. |
| [audit_report_claims.py](audit_report_claims.py) | Recheck the core figure data and expose controls and third-class competition. |
| [audit_closing_analyses.py](audit_closing_analyses.py) | Independent score-set/count audit and sampled SciPy acoustic reconstruction. |
| [audit_extension_results.py](audit_extension_results.py) | Recalculate intervention tables directly from per-trial CSVs with sklearn. |
| [audit_augmentation_banks.py](audit_augmentation_banks.py) | Check every training view against official metadata and held-out partitions. |
| [audit_prompt_ensemble.py](audit_prompt_ensemble.py) | Independently recompute the prompt ablation from saved vectors and predictions. |
| [audit_local_alignment.py](audit_local_alignment.py) | Recompute all alignment conditions, calibration exclusions, old reference identities and four primary paired tests. |
| [audit_cost_panel.py](audit_cost_panel.py) | Verify the balanced CPU benchmark and export comparable total-query costs. |
| [inspect_report_pdf.py](inspect_report_pdf.py) | Render every report page and check text bounds before visual inspection. |

## Extensions, diagnostics and figures

| Script | Purpose |
|---|---|
| [run_extension_evaluations.py](run_extension_evaluations.py) | Consume completed extension models and caches, then verify saved predictions. |
| [finalize_extensions.py](finalize_extensions.py) | Wait for declared runs, summarise, render previews and measure idle CPU cost. |
| [complete_closing_analyses.py](complete_closing_analyses.py) | Close ranked-readout and acoustic matching analyses using existing evidence only. |
| [analyse_capture_transitions.py](analyse_capture_transitions.py) | Distinguish new distractor capture from errors already present in controls. |
| [check_prompt_multiplicity.py](check_prompt_multiplicity.py) | Descriptive post-hoc sensitivity to duplicate released prompt templates. |
| [check_source_cluster_uncertainty.py](check_source_cluster_uncertainty.py) | Post-hoc sensitivity of matched accuracy CIs to target-source clustering. |
| [benchmark_panel.py](benchmark_panel.py) | Balanced 250-clip CPU latency panel, run after training/evaluation is idle. |
| [benchmark_local_alignment.py](benchmark_local_alignment.py) | Compare ensemble, preset and calibrated readouts on the same 250 clips in both orders, bypassing input caches. |
| [repeat_cost_panel.py](repeat_cost_panel.py) | Post-hoc latency sensitivity: repeat the identical panel in reverse model order. |
| [compare_cost_passes.py](compare_cost_passes.py) | Verify identical benchmark inputs across the primary and reversed-order pass. |
| [plot_closing_analyses.py](plot_closing_analyses.py) | Render final readout/overlap diagnostics without changing earlier figures. |
| [plot_cost_panel.py](plot_cost_panel.py) | Balanced CPU latency and semantic robustness with explicit interval meanings. |
| [plot_diagnostics.py](plot_diagnostics.py) | Four supplementary research figures from verified diagnostic tables. |
| [plot_extensions.py](plot_extensions.py) | Intervention figures from complete, audited runs; no best-seed selection. |

## Proposal, earlier presentation and shared inputs

| Script | Purpose |
|---|---|
| [build_proposal.py](build_proposal.py) | Compile the editable LaTeX proposal in the original document's typography. |
| [render_report_figures.py](render_report_figures.py) | Re-render core plots without baked-in figure numbers for report captions. |
| [build_presentation_data.py](build_presentation_data.py) | Collect verified result values for the editable research presentation. |
| [build_presentation.cjs](build_presentation.cjs) | Build the earlier detailed editable presentation. |
| [build_video_narration.py](build_video_narration.py) | Synchronise the recording script with the deck's exact speaker notes. |
| [delivery_claims.py](delivery_claims.py) | One audited numerical source for slides, narration and the public project page. |

The retired detailed report's source, builders and delivery checker were removed at the author's request. Dated audit records remain historical evidence tied to their original commits; use `audit_final_delivery.py` for current deliverables. The sole current manuscript is the 11-page report and its editable Word companion.

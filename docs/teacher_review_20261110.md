# Review of the instructor's 80 suggestions

Review date: 2026-11-10 (Sydney time).

The main experiments and the extensions declared in this revision are complete. The 80 suggestions include research motivation, interpretive advice, explicitly optional methods and delivery requirements; they should not be treated as 80 equally weighted tasks when calculating completion. Full PAT was not reproduced. Human listening checks, the author-recorded video, final report revisions and course submission still need to be completed; public Pages has been published and verified. The user's final requirements have now been checked: a report typically around 10 pages, a brief execution video, a GitHub Site and Canvas submission. See the [item-by-item rubric review](rubric_review_20261110.md). The exact deadline, formats and AI rules still need to be checked in Canvas.

The project studies why background sounds at the same intensity cause different classifiers to fail in different ways, and whether ordinary noise training protects them against competition from another meaningful event. The three systems are a handcrafted-feature SVM, a compact log-Mel CNN and frozen CLAP. Current conclusions concern differences in failure under controlled interference; they do not establish a causal effect of “semantics” independent of spectrum, temporal structure and salience.

## Revisions and verification in this review

- Recomputed 348 figure records for core own/target Figures 1–6 and 8 from the corresponding original predictions, scores and representations, with a tolerance of 1e-10. Figure 7 retains the historical single-clip microbenchmark definition; the complete cost analysis uses two separate rounds of measurements on balanced samples.
- Independently recomputed 14 variants, 336 condition results and 156 paired comparisons again; the core conclusions are unchanged.
- Corrected the scope of the “12 comparisons”: two semantic conditions × two power definitions × three CNN seeds, excluding white noise and DEMAND.
- Clarified that the three statistical test families are corrected separately, while pointwise confidence intervals are not adjusted for multiple comparisons. Dependence from shared sources, distractors and training sets remains a limitation.
- Added a gain-control table. Across all core conditions, control−clean macro-F1 changes range from about −1.74 to +0.49 percentage points; they cannot be described as having no effect.
- Added CLAP prediction counts for target/distractor/other classes and changes in the strongest other-class cosine score, strengthening item 48.
- Completed captions with SD, confidence intervals, sample sizes and capture denominators; clarified that confusion figures include pre-existing clean-control confusion.
- Moved the main figures closer to their corresponding sections; retained original figure data and experimental predictions rather than selecting more favourable results.

## Experiment and delivery status

| Item | Status |
|---|---|
| Core three systems, five folds, complete own/target evaluations | Completed and verified |
| Three CNN seeds and both augmentation types for SVM/CNN | Completed and verified |
| CLAP fixed prompt ensemble, feature groups, all-class competition, source-clustering sensitivity and repeated cost measurements | Completed and verified |
| Full PAT local audio–text mutual feedback | Not implemented; explicitly optional in the instructor's advice. The fixed prompt ensemble is not full PAT |
| Report, both three-system figure sets, extension figures, presentation, narration and interactive website | Both closing analyses are synchronised; the formal report has 10 pages, the original 26-page version is unchanged, and the 12-slide presentation agrees with the 728-word narration; an actual code execution clip has been recorded |
| Human listening check and final author-recorded video | Still require the author |
| GitHub feature branch and draft PR | Provided; review revisions continue on the same branch |
| Public Pages deployment | Published from gh-pages; all four selector states and hashes of six public deliverables verified |
| main merge and course submission | PR #1 is integrated into main; Canvas submission is still pending |

## Two closing analyses (added in this revision; complete)

- All 24,000 own and 23,991 target acoustic records are complete, including 6,000 DEMAND records each. Original models, predictions and figure data are unchanged.
- Exact top-k covers SVM, CNN, CLAP and the existing prompt ensemble, totalling 95,964 semantic readouts. Top-2 is primary, with 1/3/5 as sensitivity checks and complete gain-matched controls.
- At +5 dB under the own definition, about 33–37% of CLAP captures still rank the target second. This does not establish that the model “heard both”, nor does it justify reclassifying every capture as correct.
- In acoustically matched subsets of the same targets, CLAP remains about 16–26 percentage points below DEMAND, while the CNN intervals include zero. The subsets retain only about 9–10.5% of targets and cannot be treated as a causal decomposition of the full results.
- Corrected two overly strong interpretations in the original package: Mel bandwidth affects white-noise correlation, and shrinkage of logistic regression coefficients is not a “percentage explained by a mechanism”.
- Updated report Section 6, the conclusion and supplementary figures, adding 7 traceable figures. See the [closing analysis results](../results/closing_20261110/README.md) for the protocol and complete verification.

The current experimental scope is closed. Optional full PAT, Mix-CLAP, new models and new data will not be pursued. Final human listening, author recording, the main merge and course submission remain delivery matters and must not be described as complete prematurely. The presentation, narration and public Pages have been synchronised with the closing analyses.

## Item-by-item comparison

| Item | Status | Recommendation | Assessment |
|---:|---|---|---|
| 1 | Implemented | Preserve the methodological strengths of the original proposal | Official folds, paired mixtures, training isolation, macro-F1 and cost measurements are all retained. |
| 2 | Implemented | Revise the main research question | Compare failures under nonspecific noise and labelled competing events, and test augmentation transfer. |
| 3 | Implemented | Revise the title | Use When Background Noise Has Meaning with a subtitle on semantic interference. |
| 4 | Discussed | Real environments contain multiple meaningful sounds | Used as the research motivation; the experiments remain controlled two-source mixtures. |
| 5 | Discussed | RSA-Bench research context | Cited in the report, with its focus on speech perception and reasoning tasks noted; it is not treated as an ESC classification baseline. |
| 6 | Discussed | Noise-aware audio-language models | Mix-CLAP is discussed in the literature review; no reproduction is claimed. |
| 7 | Implemented | Add a modern reference model | Frozen LAION-CLAP has completed clean evaluation and both full interference evaluations. |
| 8 | Implemented | Do not train CLAP | Weights are fixed, with no fine-tuning. |
| 9 | Implemented | Retain handcrafted DSP and SVM | The 92-dimensional features include RMS, ZCR, centroid, spread, roll-off, flux, MFCCs and deltas. |
| 10 | Implemented | Retain a compact log-Mel CNN | The supervised baseline is retained, with 36,258 parameters. |
| 11 | Implemented | Compare three generations of representations | SVM, a task-trained CNN and a pretrained audio-language representation are all retained. |
| 12 | Implemented | Avoid framing the study as a best-model contest or a fair training comparison | Study failure mechanisms and deployment behaviour, and explain CLAP's external pretraining advantage. |
| 13 | Implemented | Distinguish three types of interference | White noise, DEMAND backgrounds and labelled ESC events; the last group is split into same- and different-category distractors. |
| 14 | Discussed | Explain what distinguishes semantic interference | Both sounds may correspond to labels; DEMAND may also contain recognisable events. |
| 15 | Implemented | Check whether power alone explains failure | Compare failures at fixed nominal relative power without treating this as causal isolation of semantics. All 47,991 spectral/temporal descriptor records and acoustically matched subsets of the same targets are complete; residual differences still do not establish a causal semantic effect. |
| 16 | Implemented | Distinguish SNR and SIR | Use SNR for noise and SIR for competing events. |
| 17 | Implemented | Systematically mix different fine classes | Use the same held-out fold, different fine classes and source recordings, at +10, +5 and 0 dB. |
| 18 | Implemented | Interpret 0 dB cautiously | Explicitly treat equal-power mixtures as a stress test with single-label ambiguity. Add the four top-2 outcomes and top-1/3/5 sensitivity checks while retaining the original single-label stress test. |
| 19 | Implemented | Emphasise +5 and +10 dB | The main paired analysis uses +5 dB; complete curves retain all three levels. |
| 20 | Implemented | Same- and different-category interference | Both pairing types are complete; coarse categories are not a quantified perceptual distance. |
| 21 | Implemented | Use fixed pairing rather than exhaustive combinations | Each target has one fixed pair per semantic condition; pairing randomness can be examined in future replication. |
| 22 | Implemented | Keep distractor recordings out of supervised training folds | Test distractors come from the target's held-out fold; source recordings spanning folds are also excluded as distractors. |
| 23 | Implemented | Complete mixture manifest | Identity, class, category group, power, offset, gain, seed, hashes and overlap are all recorded. |
| 24 | Implemented | Analyse error destinations | Distinguish target, distractor and other classes, and add changes from gain-matched control to mixture. Exact top-k readout and clean gain-matched controls distinguish first-place capture from target rank retention. |
| 25 | Implemented | Conditional capture rate | The denominator is the number of errors; also report the capture fraction across all trials and raw counts. Separately report the fraction of captures retaining the target in the top two, with its denominator distinguished from all trials. |
| 26 | Implemented | CLAP target and distractor text scores | Save cosine scores for all 50 classes and examine target–distractor competition. |
| 27 | Implemented | Define the target-minus-distractor margin | Use SVM decision scores, CNN logits and CLAP cosines; do not compare absolute scales across models. |
| 28 | Implemented | Plot margin against SIR | Boxplots and numerical tables are complete for all three levels. |
| 29 | Diagnostic completed | Psychometric-style transition trend | Margin distribution trends are shown; three SIR points are insufficient to fit a precise decision-flip threshold. |
| 30 | Implemented | Retain ordinary-noise controls | WGN and all three DEMAND environments are complete. |
| 31 | Implemented | Match interference power carefully | Retain both own and the original proposal's target definitions in full; nominal power is not complete acoustic matching. Add joint selection using spectral overlap, time-frequency masking and temporal overlap; disclose that about 9–10.5% of targets are retained. |
| 32 | Implemented | Active-region power | Use a 40 dB active-frame definition and record active-region overlap; do not attribute the own-region change to an explicit instructor instruction. Include temporal overlap and spectra within target-active frames; no alignment changes or new time-shift experiments were performed. |
| 33 | Implemented | Verify the achieved ratios | Recompute after generation and store in the manifest; explicitly exclude zero-energy pairs with undefined ratios. |
| 34 | Implemented | Official five folds | Retain all 2,000 clips, 50 classes and five folds. |
| 35 | Discussed | ESC-50 dataset caveats | Some original preprocessing is class-dependent; also disclose source IDs spanning folds and uncertainty about external pretraining. |
| 36 | Implemented | Control the CNN training workload | One seed in the core experiment; the extension completes seeds 5305, 5306 and 5307. |
| 37 | Implemented | Prioritise failure analysis over architecture tuning for leaderboard performance | Retain the small CNN and focus extensions on interference, augmentation transfer and diagnostics. |
| 38 | Implemented | Treat ordinary augmentation as an extension | Complete clean-trained robustness first, followed by separate WGN-augmented training. |
| 39 | Implemented | Test transfer from ordinary augmentation to semantic interference | Evaluate SVM/CNN under both power definitions. |
| 40 | Implemented | Treat augmentation transfer as a hypothesis to test | Observed transfer is limited; retain uncertainty and negative findings. |
| 41 | Implemented | Optional semantic-mixture augmentation | Both augmentation types are complete for SVM and for all three CNN seeds. |
| 42 | Implemented | Control the extension scope | Retain core results and name/report extensions separately. |
| 43 | Implemented | Displacement by feature group | Analyse 8 feature groups separately, using the corresponding training-fold standard deviations. |
| 44 | Implemented | Is displacement magnitude sufficient to explain failure? | Compare displacement and classification for matched targets; retain uncontrolled explanations such as direction and class boundaries. |
| 45 | Implemented | CNN representation displacement | Save 256-dimensional penultimate-layer vectors and use cosine distance. |
| 46 | Implemented | CLAP representation displacement | Save 512-dimensional vectors, including AUC for the association between displacement and errors. |
| 47 | Implemented | Do not compare absolute distances across architectures | Use separate panels and explain the different representation geometries. |
| 48 | Implemented | Inspect text scores for other competing classes | Save the top 3 other classes; this revision adds changes in the strongest other-class score and a table of the three prediction destinations. Exact top-1/2/3/5 from all 50 class scores is complete, including the existing fixed prompt ensemble. |
| 49 | Implemented | Control prompt wording | Use a fixed template in the core experiment; the separate fixed prompt ensemble is not selected using test labels. |
| 50 | Optional; not fully reproduced | PAT as an optional modern extension | Only reuse the released template list for fixed averaging; full mutual feedback and local audio alignment are not implemented. |
| 51 | Reviewed | Public PAT code | Pin the source commit and template provenance, and report differences from full PAT. |
| 52 | Partly answered | Does better audio–text alignment help? | Changes from fixed text readout have been evaluated; full PAT's alignment mechanism has not. |
| 53 | Followed | Do not train PAT | PAT was not trained, and the prompt-ensemble ablation is not described as full PAT. |
| 54 | Discussed | Mix-CLAP as research context | Cited but not reproduced; the instructor explicitly did not require reproduction. |
| 55 | Implemented | Macro-F1 against interference level | Four interference panels and three systems; both own/target figure sets are in the repository. |
| 56 | Implemented | Capture-rate figure | Conditions and denominators are explicit; 1/49 is only a reference for uniformly distributed incorrect labels. |
| 57 | Implemented | Margin figure | Retain each model's own score scale. |
| 58 | Implemented | Same- versus different-category comparison | Relative F1 loss uses the same retained targets; distinguish the ratio of means from foldwise ratios. |
| 59 | Implemented | Representation-displacement figure | Compare gain-matched controls and mixtures, split by correct/incorrect predictions; make no causal claim. |
| 60 | Implemented | Confusion-flow figure | Select examples by observed counts; captions distinguish existing confusion from newly induced capture. |
| 61 | Implemented | Robustness versus cost figure | Retain the original microbenchmark and state its definition; add two rounds of cost measurements on balanced samples. |
| 62 | Implemented | Fair deployment-cost boundaries | Use the same machine, one thread and batch 1; explain preprocessing/inference boundaries and list one-off text preparation separately. |
| 63 | Implemented | Do not include CLAP pretraining in local inference cost | Explain the external training advantage; do not claim a fair comparison of lifetime costs. |
| 64 | Implemented | Official ESC-50 data | Pin the official snapshot and metadata. |
| 65 | Method compared | Historical Piczak CNN reference | Cite the log-Mel CNN approach; do not reproduce the historical architecture layer by layer, which the instructor did not require. |
| 66 | Implemented | Official LAION-CLAP implementation | Pin the checkpoint, use the official tensor path and frozen inference, and record hashes and environments. |
| 67 | Optional; not fully reproduced | Official PAT implementation | Reviewed and used for an explicitly named prompt-ensemble ablation; the full algorithm has not been reproduced. |
| 68 | Completed | Milestone 1: handcrafted SVM | Clean five-fold evaluation, custom STFT and the course MATLAB comparison are complete. |
| 69 | Completed | Milestone 2: small CNN | Clean five-fold evaluation and the three-seed extension are complete. |
| 70 | Completed | Milestone 3: ordinary interference | WGN/DEMAND mixing and evaluation on identical inputs are complete. |
| 71 | Awaiting author listening check | Milestone 4: semantic mixing and listening | Engineering verification and 24 examples are complete; successful audio loading is not presented as a human listening assessment. |
| 72 | Completed | Milestone 5: frozen CLAP | Clean and full own/target evaluations are complete, with scores and vectors cached. |
| 73 | Completed | Milestone 6: all diagnostics | F1, accuracy, degradation, capture, margin and representation displacement are complete. |
| 74 | Completed | Milestone 7: choose one extension | Ordinary augmentation is complete, with semantic augmentation and a fixed prompt ensemble added. |
| 75 | Implemented | Expand the modern literature review | A separate review is included in the 10-page formal report, connecting course material on features/STFT/hearing with the strengths and weaknesses of the methods. |
| 76 | Implemented | 2026 literature context | Primary sources for Mix-CLAP and RSA-Bench have been checked, and task boundaries are noted. |
| 77 | Completed | The instructor's core project scope | Three systems, five folds, all interference conditions and all metrics are covered. |
| 78 | Followed | Avoid unnecessary large-scale training | No CLAP training/fine-tuning, Mix-CLAP reproduction or exhaustive pairing was performed. |
| 79 | Branch and Pages available | GitHub content requirements | Code, results, figures and examples are in feature/interference and the PR; Pages was published from gh-pages and verified. PR #1 has now integrated the work into main; the feature branch is retained. |
| 80 | Addressed | Overall research objective | Study different failure modes under acoustic corruption and label competition; the experiments do not causally isolate semantics. Both closing analyses are complete; retain the interpretive boundaries between readout, acoustics and unmeasured salience, with no further models or datasets. |

## Evidence locations

- [Cross-deliverable numerical checks and public deployment record](../results/delivery_sync_20261110/README.md)
- [Final submission checklist](final_submission_checklist_20261110.md)
- [10-page formal report](../results/final_submission_20261110/final_report.pdf); [26-page supplement](../results/research_report_20261109_draft/research_report.pdf)
- [Figure recomputation audit for this review](../results/report_review_20261110/audit.json)
- [Machine-readable record of all 80 items](../results/report_review_20261110/teacher_80_review.json)
- [Gain-control values](../results/report_review_20261110/gain_control_effects.csv)
- [Other competing-class values](../results/report_review_20261110/clap_other_competitors.csv)
- [Independent recomputation of interference and augmentation comparisons](../results/extensions_20261109/interventions/independent_audit.json)
- [Complete interference results and data definitions](../results/clap_full_20261109/README.md)
- [Extension protocol and post hoc revision notes](extension_protocol_20261109.md)
- [Draft PR for code and deliverables](https://github.com/Charlie-Uni/elec5305-project-550489593/pull/1)

## Directions retained only as future work (not pursued in this study)

1. **Repeated pairing replication.** Freeze existing models and use multiple prespecified pairing seeds to give each target different held-out distractors, with a paired design for background segments as well. Results from one pairing do not establish behaviour under arbitrary competing scenes. Declare primary contrasts and aggregation rules before running; do not select retained seeds based on significance.
2. **Acoustic controls that better isolate semantics.** Create versions of the same distractor with different recognisability while matching spectrum, envelope and activity overlap as closely as possible, and verify recognisability. Phase randomisation does not automatically mean “no semantics”; it may also change temporal structure. This would most directly advance the core scientific question.
3. **Time and SIR sweeps for fixed sound pairs.** Keep the sound pair fixed and vary only relative onset and a denser SIR grid, recomputing achieved power and checking clipping. This would support estimating flip thresholds and move from between-group overlap differences to controlled within-pair experiments.
4. **More independent backgrounds or external-data validation.** There are currently only three DEMAND recordings and one benchmark dataset, which does not establish real-world multisource generalisation. Define shared labels, evaluation splits and primary metrics first.
5. **Full PAT.** Implement only when local audio representations can be obtained according to the original method and all selection rules can be controlled; test labels must not be used to search for fusion coefficients. This has lower priority than the robustness and mechanism checks for the current key conclusions above.

The [original PAT paper](https://aclanthology.org/2025.naacl-long.616/) includes mutual feedback between text prompts and local audio, beyond this project's fixed template averaging. The [original RSA-Bench paper](https://aclanthology.org/2026.findings-acl.913/) provides related motivation from speech tasks in complex backgrounds; it is not an ESC classification experiment reproduced in this project.

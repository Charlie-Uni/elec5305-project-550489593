# Review of the instructor's 80 suggestions

Review updated: 2026-11-11 (Sydney time).

The 80 suggestions combine motivation, methodology, interpretation, optional methods
and delivery requirements. They are not 80 equally weighted implementation tasks.
The core three-system experiment, augmentation, repeated CNN seeds, fixed prompts,
ranked readout and acoustic diagnostics are complete. A separately named PAT-style
local-feedback adaptation is complete and independently verified; it is not
an official or full PAT reproduction. The author has now confirmed representative
listening. The narrated video and Canvas submission remain author actions.

The project asks why equal nominal interference power causes different failures,
and whether ordinary noise training protects against a competing labelled event.
The evidence does not causally isolate semantic meaning from acoustics or salience.
See the [current rubric review](rubric_review_20261111.md) and
[submission checklist](final_submission_checklist_20261110.md).

## Experiment and delivery status

| Item | Status |
|---|---|
| Three systems, five official folds, full own/target evaluations | Completed and independently verified |
| Three CNN seeds; ordinary and semantic augmentation for SVM/CNN | Completed; uncertainty and negative results retained |
| Fixed prompt ensemble, feature groups, all-class scores, cost and source sensitivity | Completed and verified |
| Top-k and spectral–temporal diagnostics, including all DEMAND rows | Completed; ranking and observational matching limits disclosed |
| PAT-style local audio–text feedback on the same LAION checkpoint | Complete: 24,000 own and 23,991 target rows; all four primary tests inconclusive; not official/full PAT |
| Formal report and presentation | One 11-page report; 12-slide editable deck; originals and historical 26-page report preserved |
| Human listening | Author confirmed representative website controls/mixtures on 11 November; not an exhaustive listening study |
| Repository and Pages | Integrated into main; published through the verified GitHub Actions workflow |
| Narrated video and Canvas | Pending; the 40.8-second real execution clip has no author narration |

## Verification and interpretation

- Core figure data, extension metrics and paired comparisons were independently
  recomputed. SD, IQR, pointwise intervals and corrected tests retain distinct meanings.
- Capture per error, capture per trial and rank-two retention among captures use
  different denominators. Exact top-k covers all four original readouts and controls.
- Gain controls have small nonzero effects. Both own and proposal-target power
  protocols remain available; zero-energy pairs are explicitly excluded.
- All 47,991 spectral/temporal records include DEMAND. Acoustically close subsets
  retain only about 9–10.5% of targets, with inconclusive CNN contrasts; they do not
  identify a causal semantic mechanism.
- Three-seed replication and augmentation retain declared test families and negative
  findings. Larger high-overlap cohorts must not be conflated with acoustic matching.
- Raw Mel correlation depends on band width. Logistic coefficient shrinkage is not
  a percentage of mechanism explained, and top-two rank is not calibrated detection.

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
| 50 | Adapted extension completed | PAT as an optional modern extension | Preset and clean-calibrated local feedback use the same frozen LAION checkpoint. This is an adaptation of released arithmetic with uniform prompts, not official or full PAT. |
| 51 | Reviewed and pinned | Public PAT code | Release commit 8910ebb and template hashes are recorded; the required modified MS-CLAP interface is not supplied in the release. The LAION extraction path is independently gated against original embeddings and scores. |
| 52 | Answered for this adaptation | Does better audio–text alignment help? | Neither preset nor calibrated feedback establishes a primary own/+5-dB accuracy change relative to the fixed ensemble; all four corrected tests are inconclusive. This does not establish equivalence or refute published PAT. |
| 53 | Frozen adaptation; disclosed departure | Do not train PAT; use released methods | No model weights are trained or fine-tuned. Released feedback arithmetic is ported rather than the official MS-CLAP system reproduced. Clean-label calibration is explicitly not zero-shot. |
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
| 67 | Adapted extension completed | Official PAT implementation as an optional reference | Code and paper were examined; the uniform-prompt LAION port is separately named. Official/full PAT, including published prompt weighting and model-specific details, is not claimed. |
| 68 | Completed | Milestone 1: handcrafted SVM | Clean five-fold evaluation, custom STFT and the course MATLAB comparison are complete. |
| 69 | Completed | Milestone 2: small CNN | Clean five-fold evaluation and the three-seed extension are complete. |
| 70 | Completed | Milestone 3: ordinary interference | WGN/DEMAND mixing and evaluation on identical inputs are complete. |
| 71 | Completed | Milestone 4: semantic mixing and listening | Engineering checks are complete. On 11 November the author confirmed that representative clean and mixed website examples were recognisable with no obvious distortion. This is a human confirmation, not a listening-accuracy experiment or an assessment of every example. |
| 72 | Completed | Milestone 5: frozen CLAP | Clean and full own/target evaluations are complete, with scores and vectors cached. |
| 73 | Completed | Milestone 6: all diagnostics | F1, accuracy, degradation, capture, margin and representation displacement are complete. |
| 74 | Completed | Milestone 7: choose one extension | Ordinary augmentation is complete, with semantic augmentation and a fixed prompt ensemble added. |
| 75 | Implemented | Expand the modern literature review | A separate review is included in the 11-page formal report, connecting course material on features/STFT/hearing with the strengths and weaknesses of the methods. |
| 76 | Implemented | 2026 literature context | Primary sources for Mix-CLAP and RSA-Bench have been checked, and task boundaries are noted. |
| 77 | Completed | The instructor's core project scope | Three systems, five folds, all interference conditions and all metrics are covered. |
| 78 | Followed | Avoid unnecessary large-scale training | No CLAP training/fine-tuning, Mix-CLAP reproduction or exhaustive pairing was performed. |
| 79 | Implemented | GitHub content requirements | Code, results, figures, attributed examples and deliverables are integrated into main. Pages publishes through GitHub Actions; the old feature and gh-pages branches remain historical. |
| 80 | Addressed | Overall research objective | Study different failure modes under acoustic corruption and label competition; the experiments do not causally isolate semantics. Both closing analyses are complete; retain the interpretive boundaries between readout, acoustics and unmeasured salience, with no further models or datasets. |

## Evidence locations

- [Cross-deliverable numerical checks and public deployment record](../results/delivery_sync_20261110/README.md)
- [Final submission checklist](final_submission_checklist_20261110.md)
- [11-page formal report](../results/final_submission_20261110/final_report.pdf); [26-page supplement](../results/research_report_20261109_draft/research_report.pdf)
- [Figure recomputation audit for this review](../results/report_review_20261110/audit.json)
- [Machine-readable record of all 80 items](../results/report_review_20261110/teacher_80_review.json)
- [Gain-control values](../results/report_review_20261110/gain_control_effects.csv)
- [Other competing-class values](../results/report_review_20261110/clap_other_competitors.csv)
- [Independent recomputation of interference and augmentation comparisons](../results/extensions_20261109/interventions/independent_audit.json)
- [Complete interference results and data definitions](../results/clap_full_20261109/README.md)
- [Extension protocol and post hoc revision notes](extension_protocol_20261109.md)
- [Integrated PR for code and deliverables](https://github.com/Charlie-Uni/elec5305-project-550489593/pull/1)

## Directions retained only as future work (not pursued in this study)

1. **Repeated pairing replication.** Freeze existing models and use multiple prespecified pairing seeds to give each target different held-out distractors, with a paired design for background segments as well. Results from one pairing do not establish behaviour under arbitrary competing scenes. Declare primary contrasts and aggregation rules before running; do not select retained seeds based on significance.
2. **Acoustic controls that better isolate semantics.** Create versions of the same distractor with different recognisability while matching spectrum, envelope and activity overlap as closely as possible, and verify recognisability. Phase randomisation does not automatically mean “no semantics”; it may also change temporal structure. This would most directly advance the core scientific question.
3. **Time and SIR sweeps for fixed sound pairs.** Keep the sound pair fixed and vary only relative onset and a denser SIR grid, recomputing achieved power and checking clipping. This would support estimating flip thresholds and move from between-group overlap differences to controlled within-pair experiments.
4. **More independent backgrounds or external-data validation.** There are currently only three DEMAND recordings and one benchmark dataset, which does not establish real-world multisource generalisation. Define shared labels, evaluation splits and primary metrics first.
5. **Full published PAT.** The same-checkpoint local-feedback adaptation is complete. Reproducing dataset-dependent prompt weighting and the published encoder-specific implementation remains outside this study. It would require a separate, prespecified protocol without selecting coefficients on evaluated labels.

The [original PAT paper](https://aclanthology.org/2025.naacl-long.616/) includes mutual feedback between text prompts and local audio, beyond fixed template averaging. This project separately tests a uniform-prompt local-feedback adaptation, whose result is not a full PAT reproduction. The [original RSA-Bench paper](https://aclanthology.org/2026.findings-acl.913/) provides related motivation from speech tasks in complex backgrounds; it is not an ESC classification experiment reproduced in this project.

# Review of the single final report

Reviewed on 11 November 2026 against the user's final project instructions and eight-criterion rubric. This review checks coverage and evidence, not a predicted mark. The formal report is one 11-page document with five figures, four numbered result tables and 13 references. The author supplied a 12-page maximum; the detailed assignment text describes a typical length of about ten pages. The locally downloaded file named `ELEC5305_Project_Report_Guidelines.pdf` actually contains a proposal guide and its separate rubric; those proposal weights and word limits were not used here.

| Criterion | Weight | Evidence in the report | Status and author responsibility |
|---|---:|---|---|
| Originality and personal contribution | 10% | Sections 1, 6 and 7 identify the controlled experiment and failure analysis; distinguish project code, official models/data, prompt resources and substantial AI assistance | Written attribution is present. The author must check the description and explain the work personally. The confirmed Codex model is recorded; the Claude version remains unrecorded rather than guessed. |
| Command of subject | 10% | Sections 2–3 connect course features, window/hop/FFT, MFCCs, learned representations, active power, scoring and leakage control; actual course STFT validation is reported | Covered through both theory and implementation. |
| Introduction | 5% | Section 1 states one main question, two subordinate tests, motivation, a bounded literature gap, scope and principal findings | Covered. Claims are limited to the tested protocol. |
| Literature review | 15% | Section 2 compares classical features/CNN, augmentation/BC learning, CLAP, PAT and structured-scene work; gives historical and current numerical context and method-selection rationale | Covered. Supervised leaderboard scores and the overlap-filtered CLAP paper are explicitly distinguished from this frozen zero-shot protocol. |
| Methodology | 15% | Section 3 describes official folds, source caveats, preprocessing/models, mixing equation, own/target conventions, held-out distractors, matched controls, metrics, resampling, augmentation and prespecified local feedback/calibration | Covered. Code/configurations provide operational detail; exploratory choices are disclosed. |
| Results | 10% | Section 4 contains robustness curves, capture counts and margins, displacement by outcome, three-seed augmentation, top-two readout, acoustic matching, a four-readout CLAP comparison and CPU costs | Covered. SD, IQR and confidence intervals are distinguished; sample counts and denominators are explicit. |
| Discussion and conclusion | 25% | Sections 5–6 directly answer the main question and subordinate tests, explain expected/unexpected results with literature, preserve corrected non-significant findings, identify confounds and propose specific future checks | Covered. Observational acoustic matching is not presented as semantic causal identification. |
| Presentation, format and references | 10% | Contents, numbered sections, five figures with captions, tables, 13 references, and clickable site/code/video links | PDF and Word are 11 pages in the verified render. The video link still identifies an execution clip without author narration. |

The weights sum to 100% of the project rubric. The supplied rubric bands are Fail below 50, Pass 50–74, Credit 75–84 and Distinction 85 or above; this document does not assign a band.

## Why the five figures are retained

1. Robustness curves answer whether equal nominal power implies equal failure.
2. Capture rate and score margins show where errors go and how class preference changes.
3. Representation displacement connects performance with changes in each representation.
4. Top-two readout tests whether a captured first-place decision retains the target at rank two.
5. Acoustic matching tests a narrower comparison while preserving uncertainty and selection limits.

The local-feedback table retains a negative/inconclusive result: none of four primary comparisons passes Holm correction, while calibrated readout uses non-test clean labels and is not zero-shot. This is a LAION adaptation, not official/full PAT.

The augmentation table tests an actual intervention; the cost table answers the deployment question. Detailed individual seeds, both full power-protocol curves and secondary diagnostics remain reproducible repository evidence rather than another report required to understand the main argument.

## Corrections made during this review

- Recomputed every added figure value from the original committed tables: 18 capture rates, 18 median/IQR margin records and 24 outcome-conditioned displacement records.
- Verified CLAP same-category capture rises from 56.25% to 71.45%; clarified the 1/49 line as uniform allocation over wrong labels, not an empirical chance test.
- Changed the claim about greater error displacement to a statement about group medians. Outcome distributions overlap and the comparison does not establish causation.
- Replaced absolute augmentation non-transfer with limited transfer under the tested CNN conditions.
- Added +5 dB power sensitivity: semantic macro-F1 differs by 0.47–2.35 percentage points; retained cohorts differ slightly, so this is not an isolated causal power-mask effect.
- Clarified the official pretrained CLAP architecture/weights are frozen while the project supplies inference integration.
- Removed unsupported Claude version names and retained the confirmed/unrecorded information. The author-role sentence refers to approving scope and requesting analyses rather than claiming sole methodological authorship.
- Connected the new figure builder to the report build and corrected stale page-count assumptions in the delivery checks.

## Readability review against the supplied comments

The seven compressed passages were revised without changing experimental outputs:

| Location | Revision |
|---|---|
| Introduction opening | Motivation now precedes the dataset list; ESC-50/Freesound, DEMAND and generated noise remain explicitly sourced. |
| Introduction scope | Connected prose explains identical mixtures, gain-matched controls, scope and power sensitivity. Controls distinguish new errors from errors already present; they do not make every error interference-caused. |
| Section 3.4 uncertainty | Explains shared source recordings and identifies which analyses use source-cluster resampling. Earlier overlap intervals remain clip-based; the 156-test family amendment is disclosed. |
| Section 3.4 local feedback | Explains 32 frames, the model projection, both attention directions, score fusion, preset coefficients, fold-local calibration and the four-test family. Labelled calibration is not called zero-shot. |
| Section 4.2 overlap | States the whole-set CNN macro-F1 ordering, defines the 80% temporal-overlap filter, and separates subset accuracy from whole-set macro-F1 and three-seed evidence. |
| Section 4.2 readouts | Replaces slash-packed values with explicit comparisons, denominators and corrected inference. Ensembling reduces other-class errors while capture counts do not fall; rank-two retention is scoped to the original single prompt. |
| Section 4.4 cost | Summarises approximately 2.6–2.7 ms extra per query; both orders and raw timings remain in repository evidence. |

The discussion now distinguishes the different populations selected by temporal overlap and acoustic matching, explains clean-calibration objective mismatch as a hypothesis, and separates aligned recording starts from audio–text alignment. The conclusion restores key results, a deployment implication and the unresolved causal question. Repeated implementation and delivery detail was shortened; the formal PDF and Word remain 11 pages at the existing body sizes.

The PAT comparison was checked against [the primary paper](https://aclanthology.org/2025.naacl-long.616.pdf), Table 1 (PDF page 6): LAION-CLAP ESC-50 accuracy is 89.00% without PAT and 93.00% with PAT. These provide unmatched literature context. Our ensemble's 92.90% clean accuracy is **not** evidence of little remaining headroom under interference. Clean calibration has no constructed-mixture target-retention objective; clean ESC-50 clips are not asserted to contain only one audible event. Top-two inclusion is not described as calibrated event detection.

The AI acknowledgement retains substantial implementation, analysis and writing assistance and limits the author's stated actions to confirmed scope approval, analysis requests and representative listening. The video remains labelled as lacking author narration. This readability revision changed no experiments, original audio or model/data scope. The obsolete detailed report was subsequently removed at the author's request.

## Submission still requires the author

Review the final attribution and recorded-version details, rehearse the signal-processing explanation, record the full brief narrated video with actual execution or clearly identified execution playback, update the final video links and description in report/site, upload to Canvas and retain the receipt. The existing 40.8-second clip contains genuine execution and audio but no author narration. It remains accurately labelled.

The earlier detailed PDF and its dedicated source/build pipeline were removed at the author's request. The 11-page report and its Word companion are the sole current manuscript; earlier versions are recoverable from Git history. The separately approved same-checkpoint alignment extension is complete. No further model, dataset or experimental expansion is planned.

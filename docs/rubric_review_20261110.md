# Item-by-item course rubric review

Basis: the final project instructions and rubric supplied by the user; reviewed on 2026-11-10. This is a review of evidence and gaps, not a predicted grade. The subsequent editorial gaps are now addressed: introduction citations, capture-rate counts, numerical literature context, BC learning, margin/displacement diagnostics and an explicit AI acknowledgement. See the [revision record](report_followup_20261110.md). The table below records available evidence and does not imply that every higher-band criterion has been fully met. The formal version is the [10-page report](../results/final_submission_20261110/final_report.pdf); the original [26-page detailed version](../results/research_report_20261109_draft/research_report.pdf) is retained unchanged as a supplement. Research data, original predictions and the 26-page supplement are preserved; the current concise deliverables have been revised.

## Eight criteria (100% in total)

| Criterion | Weight | Focus of the higher-band requirements | Implementation and evidence in this revision | Remaining action |
|---|---:|---|---|---|
| Originality and personal contribution | 10% | Accurate attribution, acknowledgement of others' work, and a contribution grounded in credible research | Pages 2 and 9 identify controlled interference, reproducible DSP and failure analysis as the contribution; distinguish project implementation from external CLAP/PAT, data and libraries; list AI assistance and audio licences | The author must confirm contributions and AI disclosure against course rules and explain the implementation personally; automated checks cannot replace this |
| Discipline knowledge | 10% | Apply theory to research and compare the strengths and weaknesses of representations | Pages 2–4 and 8–9 connect Lecture 3 features, Lecture 4 STFT and Lecture 6 auditory weighting; compare handcrafted, supervised and pretrained representations; the course MATLAB STFT validation was actually run | Clearly explain window/FFT lengths, SIR masks, training splits and score readout in the recording |
| Introduction | 5% | Identify the literature gap and justify a clear scope | Page 2 sets one main research question, with augmentation transfer and error destinations as subquestions; acknowledges unequal prior training | Content added; final author read-through required |
| Literature review | 15% | Organise and analyse the literature to motivate method choices | Pages 2–3 provide a separate review with numerical context: course DSP → Piczak CNN → augmentation/BC learning → CLAP → PAT/modern structured interference; discuss strengths, limitations and selection rationale, and explain new concepts | Content added; the author must check understanding of the literature and avoid calling the fixed prompt ensemble full PAT |
| Methods | 15% | Derive the approach from prior work's strengths and weaknesses and provide enough detail to reproduce it | Pages 3–5 cover splits, features/models, mixing equations, the own/target deviation, controls, exclusions, statistical units and extensions; complete configurations/commands are in the supplement and repository | No additional experiments required; the demonstration environment has been run, but another machine still needs the external data/weights |
| Results | 10% | Reliable data, appropriate analysis, and clear figures and explanations of error sources | Pages 5–7 select three figures and three tables: core curves and pooled capture counts, augmentation, top-2, the acoustically matched subset and cost; specify SD/CI, denominators and sample sizes; complete remaining results are in the supplement | Content added; do not generalise the small-subset findings to the full population |
| Discussion and conclusion | 25% | Interpret findings through theory, anticipate criticism and propose ways to address limitations | Pages 8–9 answer the main question and connect features, Piczak, Salamon/Bello and CLAP/PAT; retain non-significant findings; discuss shared sources, limited backgrounds, pretraining and unmeasured acoustics, with corresponding future checks | Content added; do not expand this study further; the author should prepare to explain these limitations |
| Writing, formatting and references | 10% | Contents, coherent structure, accurate citations and clear expression | The formal report is exactly 10 pages, including contents, figures/tables and 13 references; it contains clickable site, code, video and supplement links; the presentation has 12 slides and the narration has 822 words | Check Canvas file requirements and update the video link after the personal narrated video is complete |

The original rubric bands are **Fail <50, Pass 50–74, Credit 75–84, Distinction ≥85**. They are retained as written rather than replaced with the conventions of another university/course. The eight weights sum to 100% within the project; the project's contribution to the overall course is discussed below.

## Mandatory deliverables and format requirements

| Original requirement | Current status | Notes |
|---|---|---|
| Working code downloadable from GitHub, with sufficient run instructions and example results | Provided and actually executed | Default `main` branch (`feature/interference` retained for older links), README and [code demonstration guide](code_demo.md); prepare external models and data as documented |
| Research report, typically about 10 pages including figures | Provided | The new formal report is exactly 10 pages; “typically” describes the usual length, and the original wording gives no explicit hard limit |
| Brief video explaining the signal-processing code and demonstrating its execution | Partly complete | Actual execution and two sound examples have been recorded; the final author-narrated version has not. The clip without narration is not presented as a complete personal video report |
| GitHub Site hosting the report and linking working code | Updated | Pages retains interactive results, complete figures and downloadable materials; local files alone are not treated as a live deployment |
| Clickable GitHub Site link in the report | Provided | Pages 1 and 9 |
| Video linked from both the report and website | Links to the actual execution clip are provided | Update/add the complete video link in both places once the author exports the final video |
| Report and video uploaded to Canvas | Not submitted | GitHub/Pages publication is not Canvas submission; final upload and a receipt are required |
| Deadline and file types | Checked against the current account's Canvas assignment page | Follow the actual deadline/closing date on the personal page; PDF, MP4 and ZIP are allowed |

The supplied instructions state that the project contributes **30%** of the course, which differs from the 35% mentioned earlier in the conversation. This review follows the attachment; the current Canvas assignment/course outline still needs to be checked. The instructions describe the video only as “brief”, without specifying duration, file format/size, a face-camera requirement or subtitles. The current 6.0–6.8 minute duration is an estimate from the narration, not a course rule. A separate PPTX upload was not required in the original text, but one was prepared at the user's request. AI acknowledgement is not a separate weighted criterion; it supports truthful attribution under the originality criterion and the University guidance. The author must still consult the course's specific AI use and disclosure policy; a declaration in the report does not itself establish compliance.

## How the literature and course material inform the report

- Lecture 3: frame and clip features, energy/ZCR, spectral shape and MFCCs; supports the trade-off that features are interpretable but temporal aggregation loses event structure.
- Lecture 4: the roles of window length, hop and FFT length, the time/frequency trade-off, and why zero padding does not increase true resolution; supports the custom frontend and MyStft comparison.
- Lecture 6: frequency-dependent auditory sensitivity; supports avoiding a direct equivalence between relative RMS power and perceptual dominance.
- Piczak, Salamon/Bello, BC learning, CLAP, PAT, Mix-CLAP and RSA-Bench: organised by research question and method rather than as a list of papers; the actual reproduction scope and task differences are explicit.

Course materials are cited by lecture and page only; course PDFs have not been publicly uploaded. No personal classroom discussions or reading experiences have been invented. The author should explain the concepts they understand in their own words.

## Remaining submission gaps

1. The author must review contributions, listen to the examples and rehearse, then record the final brief video with their own narration; the verified execution clip can be used directly.
2. Check the actual Canvas deadline, permitted formats/sizes, project weighting and AI rules; the current attachment alone does not establish all these details.
3. Synchronise the final video link in the report and website, upload the formal report and video to Canvas, and retain the receipt. PR #1 has been integrated into default `main`; the public feature branch is retained.

The main experiments and declared extensions have no remaining mandatory runs. Full PAT, Mix-CLAP and other items among the instructor's 80 suggestions are not all mandatory. The [80-suggestion review](teacher_review_20261110.md) and this document address research advice and formal assessment requirements, respectively.

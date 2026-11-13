# When Background Noise Has Meaning

**Environmental Sound Classification under Semantic Interference**

ELEC5305 — Acoustics, Speech and Signal Processing

Qihan Zhu · SID 550489593

## Start here

| Material | Direct link | What it contains |
|---|---|---|
| **Main report** | **[11-page PDF](https://charlie-uni.github.io/elec5305-project-550489593/results/final_submission_20261110/final_report.pdf)** · [editable Word](https://charlie-uni.github.io/elec5305-project-550489593/results/final_submission_20261110/final_report.docx) · [LaTeX source](docs/final_report.tex) | Research question, methods, selected results and discussion |
| **Project website** | **[Open the live website](https://charlie-uni.github.io/elec5305-project-550489593/)** | Interactive results, figures, downloads and four target sounds to explore |
| **Working code** | **[Source-code map](src/esc_robustness/README.md)** · [reproduction guides](docs/README.md#reproduce-the-study) | DSP, SVM, CNN, frozen CLAP, controlled mixing and analysis |
| **Presentation** | **[12-slide PDF](results/final_submission_20261110/presentation.pdf)** · [editable PowerPoint](results/final_submission_20261110/presentation.pptx) | Selected findings, speaker notes and embedded audio |
| **Project video** | **[Watch the final video](https://charlie-uni.github.io/elec5305-project-550489593/results/final_submission_20261110/project_video.mp4)** | 5 min 50 s, with subtitles, project findings and a short code demonstration |
| **Code demonstration** | [Standalone execution clip](results/final_submission_20261110/code_demo.mp4) · [run it locally](docs/code_demo.md) | Source reconstruction, SIR verification and fresh three-model inference |
| **Evidence** | **[Results index](results/README.md)** · [recording script](docs/video_narration_final.txt) | Original results, numerical checks and narration |

The final project video is linked in the report and embedded on the website.
The earlier 40.8-second execution clip is retained as separate demonstration evidence.
Review the [submission checklist](docs/final_submission_checklist_20261110.md)
before uploading the final report and video to Canvas; website publication does not
replace formal submission.

## What the project studies

At matched interference power, do classifiers fail differently when a background
is ordinary noise or another recognisable sound event? When they fail, do they
predict the competing event instead of the intended target?

The study compares handcrafted features with an SVM, a compact supervised log-Mel
CNN and frozen zero-shot LAION-CLAP on all 2,000 ESC-50 clips using the five official
folds. White noise, DEMAND backgrounds and same-/different-category competing events
are evaluated at +10, +5 and 0 dB, using identical mixtures across systems.

Both `own` and `target` active-region power definitions are retained. The analyses
measure macro-F1, accuracy, distractor capture, score margins, representation
changes and computational cost. Extensions cover three CNN seeds, noise/semantic
augmentation, a fixed CLAP prompt ensemble, ranked readout and spectral–temporal
overlap. A further [PAT-style local-alignment protocol](docs/local_alignment_protocol_20261111.md)
uses the same frozen LAION checkpoint, separating label-free presets from calibration
on non-test clean labels. See the [design](docs/guides/experiment_design.md),
[core results](results/clap_full_20261109/README.md) and
[closing analyses](results/closing_20261110/README.md) and [verified local-feedback results](results/extensions_20261111/clap_local_alignment/README.md).

Equal power does not isolate semantics from acoustic structure, and 0 dB has
single-label ambiguity. Top-2 retention is a ranking diagnostic, not proof that a
model “heard both” events. CLAP benefits from external pretraining; the fixed
prompt ensemble and the separately named local-feedback adaptation are not official
or full PAT reproductions. Frozen weights do not make the calibrated readout zero-shot.

## Run and reproduce

Use Python 3.11 and `uv`. From a fresh checkout:

```bash
git clone https://github.com/Charlie-Uni/elec5305-project-550489593.git
cd elec5305-project-550489593
uv sync --locked --no-editable --extra dev
uv run --locked --no-editable --extra dev esc-baseline --help
uv run --locked --no-editable --extra dev pytest -q
```

This installs the base SVM/DSP environment. CNN and figure dependencies are optional;
CLAP uses a separate environment. Dataset audio, trained weights and large caches
are external to Git. The guides provide their preparation steps and provenance:

- [Clean SVM/CNN: setup, data, DSP checks and training](docs/guides/baselines.md)
- [Interference, manifests, frozen CLAP and core figures](docs/guides/interference_and_clap.md)
- [Augmentation, repeated seeds, ranked readout and overlap analysis](docs/guides/extensions.md)
- [Actual local code demonstration and expected outputs](docs/code_demo.md)
- [Scripts for builds, audits and publication](scripts/README.md)

To preview the static website from the repository root, run
`python3 -m http.server 8000 --bind 127.0.0.1` and open
[localhost:8000](http://127.0.0.1:8000/). The website displays saved experimental
results; the separate local demonstration runs actual inference.

## Repository map

```text
README.md                 Main report, code, presentation and website entry points
src/esc_robustness/        Project implementation and source-code map
configs/                  Fixed model, feature and interference settings
tests/                    Numerical, leakage and pipeline checks
scripts/                  Reproduction, delivery and verification tools
  README.md               Script directory by task
docs/                     Report sources, protocols and submission documentation
  guides/                 Detailed setup and reproduction instructions
results/                  Original results, published deliverables and dated audits
  README.md               Canonical results versus historical evidence
  final_submission_20261110/      Main report, presentation and execution video
examples/                 Four attributed target examples and matched controls
demo/                     Local interface for actual model execution
index.html + site-data.json       Public project website
.github/workflows/        Verified GitHub Pages publication
```

Experiment result and audio paths remain stable. The retired 26-page report and
its dedicated build pipeline were removed at the author's request; the current
11-page PDF and its Word companion are the only final report. Repeated CSVs, provenance files and audio inside
self-contained result packages are intentional. [Repository maintenance](docs/repository_maintenance.md)
explains the preservation and temporary-file policy.

## Attribution and project status

The DSP, experiment control and evaluation implementation is project code developed
with AI assistance. NumPy, SciPy, scikit-learn and PyTorch are external libraries;
LAION-CLAP supplies pretrained code and weights. See the
[references and external-code notes](docs/guides/proposal_and_references.md),
[audio attribution](examples/README.md) and
[formal submission review](docs/rubric_review_20261111.md).

`main` contains the integrated project. The older `feature/interference` branch
remains available for existing links. Pages publishes from the verified `main`
workflow; publication does not constitute Canvas submission.

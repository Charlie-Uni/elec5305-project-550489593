# When Background Noise Has Meaning

**Environmental Sound Classification under Semantic Interference**

ELEC5305 — Acoustics, Speech and Signal Processing

Qihan Zhu · SID 550489593

## Start here

| Material | Direct link | What it contains |
|---|---|---|
| **Main report** | **[10-page PDF](results/final_submission_20261110/final_report.pdf)** · [LaTeX source](docs/final_report.tex) | Research question, methods, selected results and discussion |
| **Project website** | **[Open the live website](https://charlie-uni.github.io/elec5305-project-550489593/)** | Interactive results, figures, downloads and four target sounds to explore |
| **Working code** | **[Source-code map](src/esc_robustness/README.md)** · [reproduction guides](docs/README.md#reproduce-the-study) | DSP, SVM, CNN, frozen CLAP, controlled mixing and analysis |
| **Presentation** | **[12-slide PDF](results/final_submission_20261110/presentation.pdf)** · [editable PowerPoint](results/final_submission_20261110/presentation.pptx) | Selected findings, speaker notes and embedded audio |
| **Code demonstration** | **[Watch actual execution](https://charlie-uni.github.io/elec5305-project-550489593/results/final_submission_20261110/code_demo.mp4)** · [run it locally](docs/code_demo.md) | Source reconstruction, SIR verification and fresh three-model inference |
| **Technical supplement** | **[Original 26-page PDF](results/research_report_20261109_draft/research_report.pdf)** | Detailed diagnostics and both power conventions; preserved unchanged |
| **Evidence** | **[Results index](results/README.md)** · [recording script](docs/video_narration_final.txt) | Original results, numerical checks and narration |

The 40.8-second execution clip includes audio but no author narration. The
[submission checklist](docs/final_submission_checklist_20261110.md) and
[report follow-up](docs/report_followup_20261110.md) distinguish completed
experiments from remaining manuscript, narrated-video and Canvas submission tasks.

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
overlap. See the [design](docs/guides/experiment_design.md),
[core results](results/clap_full_20261109/README.md) and
[closing analyses](results/closing_20261110/README.md).

Equal power does not isolate semantics from acoustic structure, and 0 dB has
single-label ambiguity. Top-2 retention is a ranking diagnostic, not proof that a
model “heard both” events. CLAP benefits from external pretraining; the fixed
prompt ensemble is not full PAT.

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
  research_report_20261109_draft/ Preserved 26-page technical supplement
examples/                 Four attributed target examples and matched controls
demo/                     Local interface for actual model execution
index.html + site-data.json       Public project website
.github/workflows/        Verified GitHub Pages publication
```

Existing result and audio paths are stable so that report links and historical
hash records continue to resolve. Repeated CSVs, provenance files and audio inside
self-contained result packages are intentional. [Repository maintenance](docs/repository_maintenance.md)
explains the preservation and temporary-file policy.

## Attribution and project status

The DSP, experiment control and evaluation implementation is project code developed
with AI assistance. NumPy, SciPy, scikit-learn and PyTorch are external libraries;
LAION-CLAP supplies pretrained code and weights. See the
[references and external-code notes](docs/guides/proposal_and_references.md),
[audio attribution](examples/README.md) and
[formal submission review](docs/rubric_review_20261110.md).

`main` contains the integrated project. The older `feature/interference` branch
remains available for existing links. Pages publishes from the verified `main`
workflow; publication does not constitute Canvas submission.

# Source-code map

[Repository home](../../README.md) · [Reproduction guides](../../docs/README.md#reproduce-the-study) · [Scripts](../../scripts/README.md)

The `esc-baseline` entry point is implemented in [cli.py](cli.py). Use
`esc-baseline --help` to list commands and `esc-baseline <command> --help` for
arguments. Run commands from the repository root after installing the package.

## Follow the signal path

| Stage | Main modules | Role |
|---|---|---|
| Dataset and folds | [data.py](data.py), [artifacts.py](artifacts.py) | Validate ESC-50 metadata, isolate official folds and record cache provenance |
| Signal processing | [dsp.py](dsp.py), [validation.py](validation.py) | Custom framing/STFT, Mel frontend and numerical checks |
| Handcrafted system | [features.py](features.py), [svm.py](svm.py) | 92 features, training-only scaling, validation selection and SVM inference |
| Supervised system | [spectrograms.py](spectrograms.py), [cnn.py](cnn.py) | Log-Mel cache, compact CNN training and fold-matched inference |
| Pretrained reference | [clap_system.py](clap_system.py) | Frozen external LAION-CLAP, fixed prompts, resampling and cosine scores |
| Controlled interference | [mixing.py](mixing.py), [interference_eval.py](interference_eval.py) | Reproducible pairing/mixing, verified SNR/SIR, gain-matched controls and shared evaluation |
| Core analysis | [analysis.py](analysis.py), [figures.py](figures.py) | Fold metrics, capture/margins and figures with traceable data |

## Extensions and explanatory analysis

| Modules | Role |
|---|---|
| [augmentation.py](augmentation.py), [extension_eval.py](extension_eval.py), [extension_summary.py](extension_summary.py) | Training-only augmentation, repeated-seed evaluation and paired comparisons |
| [prompt_ensemble.py](prompt_ensemble.py) | Fixed released prompt-list ablation; not the full PAT algorithm |
| [local_alignment.py](local_alignment.py) | PAT-style local alignment ported to frozen LAION-CLAP: preset and clean-data calibrated fusion from cached branch scores; not the official PAT release |
| [diagnostics.py](diagnostics.py) | Representation, feature-group and ranking diagnostics |
| [readout_analysis.py](readout_analysis.py) | Exact top-k outcomes from stored scores, including matched clean controls |
| [overlap_analysis.py](overlap_analysis.py) | Spectral and temporal descriptors and descriptive conditional analyses |

Numerical conventions and leakage controls are tested in [tests/](../../tests/).
[configs/](../../configs/) contains the declared settings. External datasets,
checkpoints and large caches are not bundled in the code repository.

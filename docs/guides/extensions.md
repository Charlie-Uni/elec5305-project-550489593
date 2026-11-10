# Extensions and closing analyses

[Repository home](../../README.md) · [Documentation index](../README.md) · [Results index](../../results/README.md)

Detailed instructions and dated run records moved from the original README. Run commands from the repository root; use fresh output directories when repeating experiments. Historical measurements and test counts describe the runs recorded below.

## Contents

- [Expanded training and diagnostics](#expanded-training-and-diagnostics)
- [Verified augmentation outcomes](#verified-augmentation-outcomes)
- [Ranked readout and spectral–temporal overlap: completed closing analyses](#ranked-readout-and-spectraltemporal-overlap-completed-closing-analyses)

## Expanded training and diagnostics

The [extension protocol](../../docs/extension_protocol_20261109.md) fixes all three
CNN seeds and separate WGN/semantic training settings before their results are
examined. Its dated analysis amendment adds direct semantic-versus-WGN contrasts
before semantic results, while disclosing prior inspection of some WGN results.
All 156 final intervention tests share one Holm correction. This is a post-baseline
extension, not a retrospective preregistration. Clean validation selects model
settings; all three seeds are retained.

```bash
# Use the same full clean spectrogram/feature caches as the baseline.
# Substitute explicit local paths and a config with the declared CNN seed.
esc-baseline train-augmented-cnn --spectrograms artifacts/features/esc50_log_mel.npz \
  --data-root data/ESC-50 --bank-root artifacts/augmentation_banks --mode wgn \
  --config configs/cnn.json --device mps --output artifacts/cnn_wgn_seed_5305
esc-baseline train-augmented-svm --features artifacts/features/esc50_handcrafted.npz \
  --spectrograms artifacts/features/esc50_log_mel.npz --data-root data/ESC-50 \
  --bank-root artifacts/augmentation_banks --mode wgn --output artifacts/svm_wgn
```

Repeat with `--mode semantic` and CNN seeds 5306/5307 in separately saved
configurations. Reuse identical banks across optimisation seeds. The evaluation
module `python -m esc_robustness.extension_eval --help` caches the deterministic
frontends once, then evaluates fold-matched supervised checkpoints. The full
original CNN own-region cache check reproduces all 24,000 mixture and control
predictions; score differences from MPS arithmetic are below 0.00006.

`python -m esc_robustness.extension_summary --help` describes the fixed run layout
and paired summaries. `scripts/plot_extensions.py` requires all declared variants
before final intervention plots. Seed SD, fold SD and paired bootstrap intervals
describe different uncertainties and are labelled separately. The completed
[diagnostic extension](../../results/extensions_20261109/README.md) includes feature
groups, full-class CLAP ranking, paired uncertainty and actual course MATLAB STFT
validation. All three clean CNN seeds reproduce the negative high-overlap
semantic-minus-DEMAND contrast. Both added seeds pass the separate declared
12-comparison Holm family; the original seed does not. The
[replication table](../../results/extensions_20261109/clean_replication/cnn_overlap_replication.csv)
reports every seed separately rather than pooling repeated targets.

The evidence ZIP also provides a portable `trial_evidence/` layout. From its
extracted root, the locked base environment can regenerate the main extension
tables without audio, model checkpoints, Torch or CLAP inference:

```bash
uv run --locked --no-dev python -m esc_robustness.extension_summary \
  --repo . --runtime . --run . --trial-evidence trial_evidence \
  --output reproduced_tables
uv run --locked --no-dev python scripts/audit_extension_results.py \
  --repo . --runtime . --run . --trial-evidence trial_evidence \
  --input reproduced_tables
```

This recomputes fold metrics, paired bootstrap intervals and corrected tests from
the saved trial predictions. It does not retrain models or reproduce embedding
displacements from the excluded large arrays. Canonical results are in
`results/clap_full_20261109/` and `results/extensions_20261109/`; older pipeline
directories are retained as historical evidence and should not be mixed with
these final manifests.

The fixed released PAT prompt-list ablation is implemented in
`python -m esc_robustness.prompt_ensemble --help`. It reuses the exact cached
audio embeddings, applies all templates to every class and performs no label-based
prompt search. The audited 300-entry list contains 275 unique templates, with
released multiplicities preserved. Clean accuracy is **92.90 ± 0.65%**, and
macro-F1 is **92.53 ± 0.59%** (five-fold mean ± SD). Both interference protocols
are complete; the audio embeddings are unchanged. It is explicitly **not full PAT**.
See the [independent audit](../../results/extensions_20261109/prompt_ensemble_audit.json).

## Verified augmentation outcomes

Own-activity protocol, +5 dB macro-F1 (%). CNN values average all three
seeds’ five-fold means; SVM values are five-fold means. The report supplies
fold/seed SD and per-seed paired intervals separately.

| Model / training | Clean | White | DEMAND | Same category | Different category |
|---|---:|---:|---:|---:|---:|
| SVM / clean | 47.96 | 2.03 | 16.13 | 22.47 | 21.34 |
| SVM / wgn | 45.99 | 41.14 | 23.72 | 22.60 | 22.19 |
| SVM / semantic | 49.27 | 1.91 | 19.92 | 26.83 | 24.96 |
| CNN / clean | 65.78 | 7.48 | 30.92 | 32.99 | 31.41 |
| CNN / wgn | 64.65 | 43.95 | 33.80 | 33.46 | 32.10 |
| CNN / semantic | 61.70 | 9.91 | 33.62 | 36.05 | 33.73 |

WGN training mainly protects against broadband noise. Semantic training yields
larger semantic point-estimate gains but reduces clean CNN performance. None
of the 12 full-test-set CNN semantic-condition comparisons between semantic and WGN training
passes the common 156-test Holm correction; the modest advantage is uncertain.
SVM passes three of its four corresponding comparisons. See the
[independent audit](../../results/extensions_20261109/interventions/independent_audit.json)
and [all paired effects](../../results/extensions_20261109/interventions/paired_intervention_effects.csv).

The deployment benchmark retains two model orders and 10,500 timed queries.
Timing drift is visible even in identical CNN frontends, so both passes are
reported and no causal augmentation-cost claim is made. See the
[repeatability check](../../results/extensions_20261109/cost_repeatability/audit.json).

## Ranked readout and spectral–temporal overlap: completed closing analyses

The final two analyses use existing scores and reconstructed mixtures only. Original
predictions, core figures and intervention results are preserved. Experimental work
is closed: no new models, datasets or training runs are required for this delivery.
The [fixed exploratory protocol](../../docs/closing_protocol_20261110.md),
[verified results and reproduction commands](../../results/closing_20261110/README.md),
[seven new figures](../../results/closing_20261110/figures) and updated
[research report](../../results/research_report_20261109_draft/research_report.pdf) are available.

- **Exact top-k readout** (`analyse-readout`): complete mixture and gain-matched-control
  scores give four exhaustive states—target only, both, distractor only, neither.
  Top-2 is primary; 1/3/5 are sensitivities. All 23,991 semantic trials per variant
  are included for SVM, CNN, frozen CLAP and its existing prompt ensemble.
  At +5 dB under own power, the single-template CLAP target remains second in
  37.34% / 33.17% of same-/different-category argmax captures. Yet 23.35% / 24.05%
  of all semantic trials include only the distractor among the two designated labels.
  Rank inclusion is not proof of hearing both sources or a calibrated detection result.
- **Spectral and temporal overlap** (`overlap-features`, `analyse-overlap`): all
  24,000 own and 23,991 target rows are complete, including 6,000 DEMAND rows each.
  These use the project's 64-band DSP/CNN frontend, not CLAP's internal frontend.
  The raw white-noise dB Mel correlation median changes from −0.728 to −0.008 after
  normalising by Mel-filter bandwidth; raw anticorrelation cannot be read as a
  physical anti-spectrum effect. Time–frequency masking remains a proxy.
- **Same-target acoustic-closeness comparisons**: under both power conventions,
  CLAP's semantic-minus-DEMAND accuracy is about −16 to −26 percentage points in
  the primary selected subsets. CNN intervals include zero. Only about 9–10.5% of
  targets meet the three calipers, so these are selected-population associations,
  not a decomposition of the original contrast or proof of semantic causation.
  Intervals resample original target-source clusters; counts and class coverage
  accompany every comparison. Logistic coefficient attenuation is **not** reported
  as a percentage explained.

The score-file route is exact; competition-only CSVs cannot recover all control
ranks and are not used for the final results. `--system svm|cnn|clap` selects the
score format (default CLAP). Example:

```bash
esc-baseline analyse-readout --k 2 --scores own=artifacts/interference/eval_own_clap target=artifacts/interference/eval_target_clap --output results/readout_top2
esc-baseline overlap-features --manifest artifacts/interference/manifest_own --data-root data/ESC-50 --require-demand --output results/overlap_own
```

The manifest metadata locates the verified DEMAND cache. `--require-demand`
prevents a partial acoustic run from being mistaken for completion. The closing
results README supplies the full local orchestration and independent-audit commands.

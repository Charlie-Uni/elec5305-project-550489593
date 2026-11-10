# PAT-style local alignment on LAION-CLAP: protocol

Written on 11 November 2026, before any alignment result existed. It covers the
instructor's optional items 50, 51, 52 and 67 (PAT as a modern extension; whether better
audio-text alignment preserves the target when another meaningful sound is present).
Code: `src/esc_robustness/local_alignment.py`; tests: `tests/test_local_alignment.py`.

## Name and status

The arm is called **PAT-style local alignment on LAION-CLAP**. It is not "official PAT"
and not "full PAT". Item 53 asks for PAT "as released"; the release cannot be run on this
project's model (see the next section), so the released inference arithmetic is ported to
the same frozen LAION-CLAP checkpoint. The model is frozen in every readout, but frozen
is not the same as zero-shot:

| Readout | Labels used | Name in tables |
|---|---|---|
| `single` | none | single prompt (zero-shot) |
| `ensemble` | none | fixed prompt ensemble (zero-shot) |
| `aligned_fixed` | none; weights declared below | PAT-style alignment, preset weights |
| `aligned_calibrated` | clean labels of non-test folds | PAT-style alignment, clean-data calibrated |

`aligned_calibrated` is a calibrated readout and must never be grouped with the zero-shot
results. Comparing it with `aligned_fixed` separates the effect of the alignment mechanism
from the effect of calibration; both come from the same cached branch scores.

## What the release does, what the paper says, what is used here

| Aspect | Paper (NAACL 2025) | Released `inference_zs.py` @ `8910ebb` | This project |
|---|---|---|---|
| Base model | six ALMs incl. LAION-CLAP | MS-CLAP 2023 only | LAION-CLAP `630k-audioset-best.pt` (same checkpoint as every other CLAP result) |
| Frame features | frame-level features before pooling; extraction per encoder not specified | requires a modified `msclap` interface not supplied in the release | HTSAT fine-grained latents captured by a forward hook during the official embedding call; exact frame repetition removed (32 frames per 10 s input); the model's own `audio_projection` applied per frame; unit norm |
| Prompts | score-weighted ensemble over the dataset's audio | uniform mean of the prompt list | uniform mean of the released list (the existing fixed ensemble) |
| Affinity | frames x text, no scale stated | frames x text, multiplied by 2, on un-normalised MS-CLAP features | unit-norm features with an explicit temperature in {16, 32, 64, 100} |
| Pooling of text-guided frames | mean | mean + max | mean + max (as released) |
| Fusion | base + b1 audio-guided + b2 text-guided | same three terms | same three terms |
| Weights | grid search reported on evaluation datasets | 200 x 200 grid, best accuracy on the evaluated labels | preset rule, and a rule calibrated on clean non-test folds |

`alignment_components(..., temperature=2)` on un-normalised inputs reproduces the released
per-clip loop to 1e-10 (unit test). The two asserted identities of the hook are: projecting
the tower's pooled latent reproduces the official embedding, and the frame latents average
to the pooled latent. Projecting each frame through the pooled-embedding MLP is this
project's choice; the paper only states that both modalities pass through same-size fully
connected layers.

## Rules fixed in advance

1. **Preset rule (`aligned_fixed`)**: temperature 64, text-guided weight 0.2, audio-guided
   weight 0.05. Reason: the release multiplies by 2 on features whose final LayerNorm gives
   norms near 32, i.e. about 64 on cosines; the midpoints of its two search ranges, converted
   to cosine units the same way, are 0.176 and 0.047, rounded to the grid.
2. **Grid**: temperatures {16, 32, 64, 100}; each weight in
   {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2}; 321 candidates. Zero weights are the fixed
   ensemble, so calibration can return it.
3. **Calibration metric**: number of correct top-1 decisions on clean clips.
4. **Calibration set for test fold k**: clean clips of the other four folds, excluding any
   clip whose source recording also occurs in fold k.
5. **Ties**: candidates are ordered by total weight, then audio-guided weight, then
   temperature; the first maximum wins, so ties go to the smallest departure from the
   ensemble.
6. **One rule per outer fold** for all conditions, levels and both power conventions. No
   rule is chosen or changed using interference results.
7. **Primary contrast**: each aligned readout against the fixed ensemble at own/+5 dB,
   same-category and different-category; exact McNemar test on top-1 correctness, with
   the capture-count difference reported beside it. Other cells are descriptive.
8. **Test-label selection** (what the release does) is written only to
   `label_selected_reference.csv`, labelled an optimistic reference: the best point of the
   searched grid on the evaluated labels. It is not a method, is not in the main table and
   is not used to choose anything.

### Mac review amendment before real-checkpoint evaluation

The supplied preset values are retained unchanged. Their scale-conversion rationale is
heuristic: the required norm equivalence has not been verified across the two encoders,
so these are project presets, not published PAT defaults or an exact MS-CLAP conversion.
The four primary contrasts (two readouts by two semantic conditions at own/+5 dB) form
one Holm-corrected family. Other cells remain descriptive. Exact McNemar results assume
independent trials; separately report pointwise 95% target-source-cluster bootstrap
intervals (2,000 resamples, seed 5311), conditional on the selected clean-calibration
rules and observed sound pairs. The interval analysis does not refit calibration inside
each bootstrap. Target ranks break score ties in ascending class-ID order, consistently
with the argmax decision. These reporting and tie-handling clarifications precede any
real-checkpoint alignment results and do not alter the declared parameter grid or preset.

The source-cluster interval addresses repeated target recordings, not all dependence:
some trials share a distractor and calibration sets overlap across folds. It conditions
on these observed partners and chosen rules rather than estimating variability from
new pairing seeds, new calibration sets or repeated training.

The Mac runner additionally supports `--text-cache` pointing to the existing fixed
ensemble run. It checks the checkpoint, template bytes and order, class order and cache
file SHA256, and recomputes the uniform mean from the saved per-prompt vectors. Its
preparation timing then measures verified cache loading, not historical text encoding.
`--cache-encodings` reuses scores only for byte-identical resampled/quantised float32
inputs within the same frozen system and temperature grid. This avoids repeated controls
and identical inputs across protocols; `encoding_cache.json` records unique inputs and
hits. Deployment benchmarks bypass this input cache. Neither option changes the grid,
preset, calibration population or mixture construction.

## Run (Mac, CLAP environment)

```bash
# Run from the current repository root, including when the venv holds an older package.
export PYTHONPATH="$PWD/src"
CKPT=<630k-audioset-best.pt>                 # sha256 8053c977...
TEMPLATES=<promt_list.txt from cs20s030/PAT@8910ebb>   # sha256 2decfa74...
DATA=<ESC-50 root>
RUNS=/Users/leaf/Projects/elec5305-local/runs
SINGLE=$RUNS/clap_v4_20261109/clean          # existing single-prompt clean run
ENSEMBLE=<fixed prompt-ensemble run>/clean   # existing ensemble clean rescoring
OUT=$RUNS/clap_local_20261111

# 1. Preflight on 50 evenly spaced clean clips with the real model (a few minutes,
#    mostly the one-off text ensemble of 300 templates x 50 classes).
python -m esc_robustness.local_alignment preflight --checkpoint $CKPT \
  --templates $TEMPLATES --data-root $DATA --single-clean $SINGLE --ensemble-clean $ENSEMBLE

# 2. Smoke run: all clean clips, first 120 manifest rows.
python -m esc_robustness.local_alignment run --checkpoint $CKPT --templates $TEMPLATES \
  --data-root $DATA --single-clean $SINGLE --ensemble-clean $ENSEMBLE \
  --own $RUNS/clap_v4_20261109/manifest_own --limit 120 --output ${OUT}_smoke

# 3. Full run (about one hour per power convention on the M3; controls are encoded too).
python -m esc_robustness.local_alignment run --checkpoint $CKPT --templates $TEMPLATES \
  --data-root $DATA --single-clean $SINGLE --ensemble-clean $ENSEMBLE \
  --own $RUNS/clap_v4_20261109/manifest_own --target $RUNS/clap_v4_20261109/manifest_target \
  --benchmark-audio $DATA/audio/1-100032-A-0.wav --output $OUT

# Fusion only, from cached branch scores (seconds; no audio is re-encoded):
python -m esc_robustness.local_alignment analyse --run-dir $OUT
```

On the Mac the existing text cache can be passed with
`--text-cache /Users/leaf/Projects/elec5305-local/runs/extensions_20261109/clap_prompt_ensemble`;
use `--cache-encodings` for the verified identical-input cache. The frozen inference
module at the full-run commit is retained unchanged while the model is running.
The independent verification command is `scripts/audit_local_alignment.py --help`.
`scripts/benchmark_local_alignment.py` additionally reuses the original 250-clip
class/fold-balanced panel, three repetitions and both readout orders. Its query costs
exclude input-cache hits and setup; calibration-grid timing is conditional on cached
clean branch scores, and clean encoding time is separately recorded by the clean run.

`run` repeats the preflight and stops before encoding if it fails; after the clean pass
it compares all 2,000 pooled embeddings and baseline scores with the cache
(`reference_check.json`) and stops before the manifests if they differ. Every row
receives mixture and gain-matched control scores. Without `--cache-encodings` both
inputs are encoded again; with it, only byte-identical prepared inputs reuse an
earlier frozen-model encoding. Logical trial counts and evaluation denominators do
not change.

## Acceptance checks before any number is used

1. `preflight.json` and `reference_check.json`: `passed` is true; pooled-embedding,
   single-score and ensemble-score differences are at most 1e-4 and predictions are
   identical.
2. `conditions.csv`, clean row: `single` accuracy 81.75 %, `ensemble` 92.90 % (macro-F1
   92.53 %).
3. `conditions.csv`, own/+5 dB: `single` same-category 632 captures of 944 errors and
   different-category 612 of 1,015; `ensemble` same-category 645 captures and 189 other
   errors, accuracy 58.30 %; white 76.15 %, DEMAND 85.05 %, different-category 56.20 %.
4. `eval_own/clap_local_predictions.csv` agrees with the existing ensemble predictions
   row for row (the standard files of this run are the ensemble readout).
5. `selection.json`: five folds; calibration clips about 1,600 less the shared-source
   exclusions; the chosen rule is the same object for every condition.
6. `pytest` passes (102 passed, 1 skipped at the time of writing).

If check 1-4 fails, stop and report; do not tune anything to make them pass.

## Outputs

Large arrays stay in the run directory. Commit only `preflight.json`, `reference_check.json`, `provenance.json`,
`selection.json`, `conditions.csv`, `label_selected_reference.csv`, `summary.json` and
`cost.json` under `results/extensions_20261111/clap_local_alignment/`.

- `eval_*/clap_local_scores.npz`: ensemble cosines plus, for mixture and control, the
  pooled embedding and the text-guided and audio-guided scores of all 50 classes at every
  temperature.
- `eval_*/fused_scores.npz`: fused scores of all 50 classes for both aligned readouts.
  These are sums of branch scores; they are labelled fused scores, not cosine
  similarities, and their margins are not comparable with cosine margins.
- `conditions.csv`: per region, condition, level and readout - fold-mean accuracy and
  macro-F1 with SD, correct, errors, captures, other errors, capture rate (of errors) and
  fraction (of trials), target in top two, median target rank on errors, control correct.
- `summary.json`: paired differences of each aligned readout against the ensemble
  (lost, gained, accuracy difference, exact McNemar p, capture difference) for every cell.
- `cost.json`: per-clip medians for the frontend, the pooled ensemble readout and the
  aligned readout (audio tower, frame capture and projection, alignment at one
  temperature, fusion). One-off costs are separate: text preparation seconds in
  `provenance.json`, calibration seconds in `selection.json`.

## Reporting

One small table at own/+5 dB with four rows (single, ensemble, preset alignment,
calibrated alignment) and columns clean accuracy, same-category accuracy, captures, other
errors and target-in-top-two, plus one paragraph. State whichever of these the data show:

- alignment keeps more targets than the ensemble (fewer captures), or
- alignment changes accuracy mainly through other errors while captures stay, or
- alignment makes the distractor more prominent, or
- calibration returns the ensemble (zero weights), i.e. clean data give no reason to align.

All four are results. Then replace "Full PAT is outside the implemented scope" in the
conclusion and the corresponding sentences in Sections 2.2, 4.2, 5.2 and 7.2 with the
accurate description: the released inference arithmetic was ported to LAION-CLAP; the
official release was not run; the calibrated readout uses labelled clean data.

Not to be claimed: reproduction of PAT or of its published numbers; that the calibrated
readout is zero-shot; that the label-selected reference is achievable; any causal
statement about semantics.

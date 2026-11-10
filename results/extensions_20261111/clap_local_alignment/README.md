# PAT-style local feedback on frozen LAION-CLAP

[Project home](../../../README.md) · [Frozen protocol](../../../docs/local_alignment_protocol_20261111.md) · [Code](../../../src/esc_robustness/local_alignment.py)

This is a project adaptation of the released PAT feedback arithmetic to the same
LAION-CLAP checkpoint. It is **not official or full PAT**: uniform released prompts
and a project choice of per-frame projection replace paper-specific components.
No model weights are trained or fine-tuned. The calibrated readout uses non-test
clean labels and is **not zero-shot**.

## Main readout comparison

Own activity, +5 dB, same-category interference; 2,000 targets. Accuracy and
macro-F1 are five-fold means (%); target top-two is pooled over all trials.
Captures and other errors are raw counts, not percentages among errors.

| Readout | Clean acc. | Clean F1 | Same acc. | Same F1 | Captures | Other errors | Target top-two |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single prompt | 81.75 | 79.49 | 52.80 | 48.86 | 632 | 312 | 70.00 |
| Fixed ensemble | 92.90 | 92.53 | 58.30 | 55.50 | 645 | 189 | 78.20 |
| Local / preset | 92.90 | 92.53 | 58.25 | 55.46 | 644 | 191 | 76.90 |
| Local / calibrated | 92.85 | 92.42 | 58.20 | 55.47 | 645 | 191 | 76.10 |

Neither added readout establishes a primary accuracy improvement over the fixed
ensemble. The four declared own/+5-dB contrasts have Holm-adjusted p-values of
0.7346–1.0000, and every pointwise source-cluster interval includes zero. This is
an inconclusive result, not evidence of equivalence or a refutation of published PAT.
Same-category target top-two retention also falls descriptively.

| Primary contrast | Accuracy change (pp) | Pointwise 95% interval (pp) | Holm p |
|---|---:|---:|---:|
| Local / preset / different | +0.20 | [-0.50, +0.88] | 1.0000 |
| Local / calibrated / different | +0.75 | [-0.30, +1.75] | 0.7346 |
| Local / preset / same | -0.05 | [-0.69, +0.60] | 1.0000 |
| Local / calibrated / same | -0.10 | [-1.03, +0.81] | 1.0000 |

Intervals resample 1,524 original target sources (2,000 replicates, seed 5311),
holding clean-calibration rules and observed partners fixed. Exact McNemar tests
assume independent trials; shared distractors and overlapping calibration sets
are not fully modelled. Both power protocols and all levels remain descriptive
outside the four primary contrasts.

## Verification and provenance

- [Independent audit](independent_audit.json): all 100 condition records and all 199,964
  trial/readout records recomputed; source exclusions, selected rules, ranks, F1,
  raw captures, controls and saved fused scores checked independently.
- [All conditions](conditions.csv) and [primary comparisons](primary_comparisons.csv).
- [Compressed per-trial evidence](trial_evidence.csv.gz) retains predictions, target ranks
  and control predictions for all four readouts; full embeddings/branch scores remain
  in the external runtime cache documented by the run provenance.
- [Real-checkpoint preflight](preflight.json), [full clean reference check](reference_check.json)
  and [smoke acceptance](smoke_acceptance.json). All original single/ensemble predictions
  and control predictions reproduce exactly; maximum old/new score difference is
  below 2.4e-7. Saved fused-score recomputation has zero difference.
- [Frozen rules](frozen_protocol.json), [chosen clean-fold rules](selection.json) and
  [executed provenance](provenance.json) record hashes and the unmodified full-run source.
- [Cache counters](encoding_cache.json): 97,982 logical queries including controls;
  54,972 unique prepared inputs and 43,010 exact-byte cache hits. No trial is removed.
- [Test-label-selected reference](label_selected_reference.csv) is a deliberately optimistic
  grid reference, **not a method**, excluded from the main table and method selection.
- [Author listening confirmation](author_listening_confirmation.json) covers representative
  website examples, not all 48 choices or a human accuracy experiment.

The frozen provenance describes a “non-public wrapper”; the narrower verified fact is
that the modified MS-CLAP interface expected by the released script is not supplied
in that release. The protocol records this distinction without changing executed files.

## Cost boundaries

The [separate setup record](setup_cost.json) measures 125.89 s for all 2,000 clean
encodings at batch four, 0.277 s for the calibration search on cached branch scores,
and 1.10 s to verify/load the existing text cache. That load time does not measure
historical prompt encoding. Local deployment timing excludes all of these and external
pretraining. The balanced 250-clip panel bypasses input caching and retains both readout
orders. All 4,500 queries reproduce reference decisions (maximum score difference
below 2.5e-7). The clean pass is an end-to-end job timer, not a per-query kernel measurement.

| Readout | Forward median (ms) | Reverse median (ms) |
|---|---:|---:|
| Fixed ensemble | 59.16 | 60.34 |
| Local / preset | 61.74 | 63.01 |
| Local / calibrated | 61.85 | 63.03 |

[All timing trials](cost_balanced/trials.csv) · [summary](cost_balanced/summary.csv) · [timing audit](cost_balanced/audit.json). Compare readouts within this new panel; the original three-system latency panel is preserved.

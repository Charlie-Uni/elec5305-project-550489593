# Closing analyses — fixed scope, 10 November 2026

These are exploratory analyses of already inspected results, not a preregistration.
The supplied cloud package and an earlier top-2 check have been inspected. This
document fixes the local completion before examining its DEMAND results. Original
models, predictions, core figures and intervention analyses remain unchanged.
No new model, dataset, pairing or threshold search is part of this closure.

## Ranked readout

Use complete saved mixture and gain-matched-control scores. Top-2 is the primary
diagnostic; top-1, top-3 and top-5 are fixed sensitivity checks. Include the frozen
single-prompt CLAP, its existing fixed prompt ensemble, and the core SVM/CNN as
ranking diagnostics. The supervised references are not calibrated multi-label
detectors. No cosine threshold is selected using clean or mixed labels.

Report four exhaustive states: target only, both designated labels, distractor
only, neither. Report target inclusion, joint inclusion and target inclusion
conditional on an argmax capture. Preserve raw counts, exact control states,
folds and target IDs. Use deterministic descending score order, ties broken by
ascending class ID, for exactly k labels. An SVM argmax of decision scores may
differ from its stored voting prediction; retain voting for original correctness
and capture, and explicitly record any disagreement.

The two designated classes are not an exhaustive annotation of everything audible.
Top-k membership means rank inclusion, not demonstrated perception of both sources.
Controls can already rank a nominal distractor highly before it is mixed in.
Pointwise 95% intervals use 2,000 bootstrap resamples of original target source IDs
(seed 530510), keeping repeated conditions/levels together in a source cluster.
They condition on fitted models, do not cover shared distractor/background/training
dependencies, and are descriptive, without new significance-test families.

## Acoustic and temporal overlap

Reconstruct all 24,000 own and 23,991 target rows, including all 6,000 DEMAND rows
in each protocol. Verify row identities, original manifest hashes and source
checksums. The incoming target feature file actually has 17,991 rows, not 18,000;
the nine omitted rows are undefined semantic pairs, not missing DEMAND rows.

Retain the incoming raw-band dB correlation, linear-band cosine, target-cell
masking proxy, energy-weighted masking and activity overlap. Their frontend is
the project's 64-band DSP/CNN frontend, not CLAP's internal Mel frontend. Add
target-active-frame spectral cosine (active frame energy within 40 dB of peak)
and bandwidth-normalised spectral-density dB correlation as named sensitivities.
The latter checks how unequal Mel filter widths affect the white-noise descriptor.
Zero-energy shape comparisons remain missing, not artificial zero similarities.
Masking here is only the fraction of target-active TF cells with larger interferer
power; it is not a validated perceptual masking model.

Primary level: +5 dB, with both power conventions retained. Descriptive fixed bins
show all four conditions, with n, class coverage and source-bootstrap intervals.
These plots compare different target compositions and cannot establish matched
effects. Capture rates are defined only for the two labelled semantic conditions.

The main paired comparisons use the exact same target in semantic vs DEMAND,
and same-category vs different-category mixtures. For each pair report:

1. All paired targets.
2. Both conditions with activity overlap at least 0.8.
3. Primary acoustic closeness: absolute difference in target-active spectral
   cosine <= 0.10, masked-cell fraction <= 0.05, and activity overlap <= 0.10.
4. Fixed stricter (0.05, 0.025, 0.05) and looser (0.20, 0.10, 0.20) sensitivities.

Select subsets once using acoustics only, identically for all three core systems.
Report retained n, source counts, class coverage, covariate differences and both
mixture accuracies. Also report the difference in gain-control-referenced accuracy
loss on those same targets. Source-bootstrap intervals require at least 30 targets
and 30 source clusters; smaller cells retain counts and point estimates only.
Empty or weak common support is a result, not grounds for widening a caliper.

The supplied regressions remain descriptive in-sample fits. Logistic coefficient
attenuation is not a proportion of a mechanism explained; non-collapsibility,
limited support, model specification and repeated targets prevent that reading.
Tertiles of masking differences are named lower/middle/upper delta third, never
"similar masking" merely because they are the middle third. Neither residual
condition associations nor their disappearance identify semantic causation.

## Verification, figures and stopping rule

Independently recompute all readout states from raw score ordering, compare the
cloud's available feature rows, and verify selected acoustic rows with a separate
framing/filterbank calculation. Export CSV, PNG, SVG and PDF plus provenance.
Update the report with numerical results, retained-support limits and exact
interpretation. These two analyses close the experimental work; additional models,
datasets and interventions are future work only.

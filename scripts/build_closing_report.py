"""Build the closing report insert from independently audited numerical tables."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_closing_report():
    directory = ROOT / "results/closing_20261110"
    audit = json.loads((directory / "independent_audit.json").read_text())
    if not audit["verified"]:
        raise ValueError("Closing report requires the independent numerical audit")
    for name, digest in audit["output_hashes"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Closing report input changed after independent audit: {name}")
    with (directory / "readout_outcomes.csv").open() as f:
        outcomes = list(csv.DictReader(f))
    with (directory / "acoustic_matched_comparisons.csv").open() as f:
        comparisons = list(csv.DictReader(f))
    text = r"""\subsection{Closing analysis: a ranked two-label readout}
These final analyses use existing predictions and audio only. They are exploratory:
the protocol was fixed after inspecting the cloud top-two and partial acoustic
results, before the local DEMAND completion. No model, prompt or dataset was
added. The original single-label results remain unchanged.

An argmax capture does not establish that the target has disappeared from the
ranking. We therefore select exactly the top $k$ labels, with $k=2$ primary and
$k\in\{1,3,5\}$ as sensitivity checks. Equal scores use ascending class ID; no
ties occurred at the inspected boundaries. Both the target and distractor ranks
are recovered from all 50 saved scores, including gain-matched clean controls.
For each designated pair the four states are target only, both, distractor only
and neither. Fractions use all semantic trials, whereas target retention among
captures conditions on the original prediction equalling the distractor.
There are 23,991 semantic trials per system or text-readout variant across both
power definitions. This is ranked retrieval, not a calibrated multi-label
detector: two labels are always returned, and a rank does not demonstrate
subjective hearing or complete source recovery.

\begin{table}[htbp]\centering\small
\caption{CLAP top-two states at +5 dB (percent of all semantic trials). Last column: target in top two among original argmax captures, a different denominator. Own uses 2,000 targets per type; target uses 1,998 same-category and 1,999 different-category targets.}
\label{tab:top2}\begin{tabular}{lllrrrrr}\toprule
Power & Text readout & Type & Target only & Both & Distractor only & Neither & Retained\\\midrule
"""
    for region in ("own", "target"):
        for variant, label in (("clap_baseline", "Single"), ("clap_prompt_ensemble", "Ensemble")):
            for condition, typ in (("semantic_same", "Same"), ("semantic_different", "Different")):
                matches = [
                    r
                    for r in outcomes
                    if r["region"] == region
                    and r["variant"] == variant
                    and r["condition"] == condition
                    and r["k"] == "2"
                    and float(r["level_db"]) == 5
                ]
                if len(matches) != 1:
                    raise ValueError("Missing or duplicate readout cell")
                row = matches[0]
                values = [
                    100 * float(row["mixture_" + s])
                    for s in ("target_only", "both", "distractor_only", "neither")
                ]
                values.append(100 * float(row["argmax_captured_target_rank_le_k"]))
                text += (
                    f"{region.capitalize()} & {label} & {typ} & "
                    + " & ".join(f"{v:.2f}" for v in values)
                    + r" \\"
                    + "\n"
                )
        if region == "own":
            text += r"\midrule" + "\n"
    text += r"""\bottomrule\end{tabular}\end{table}

At +5 dB under own power, the single-template target remains second in
236/632 same-category captures (37.34\%) and 203/612 different-category captures
(33.17\%). Joint top-two inclusion occurs in 26.05\% and 21.00\% of all mixtures,
compared with 5.40\% and 1.70\% of their clean controls. Yet 23.35\% and 24.05\%
contain only the distractor among the two designated labels. The existing prompt
ensemble retains the target in 47.60\% and 38.60\% of captures. The target-power
analysis gives a similar partial recovery (Table~\ref{tab:top2}). Thus some
single-label failures retain useful target ranking, while the top-two readout
does not remove the interference problem. Increasing $k$ mechanically makes
inclusion easier; Figure~\ref{fig:retention} reports that trade-off without
reclassifying the original errors as correct. Even the two designated source
labels need not exhaust every audible event in a recording.

\closingfig{20_readout_top2_single}{Exact top-two states for frozen single-template CLAP. Each SIR has gain-matched control (left) and mixture (right). Four states partition every designated target--distractor pair; the states are not perceptual detection claims. Counts and all-system sensitivities accompany the figure.}{fig:top2}

\subsection{Closing analysis: spectral and temporal overlap}
All 24,000 own-power and 23,991 target-power rows now have acoustic descriptors,
including 6,000 DEMAND rows in each protocol. The legacy long-term dB Mel
correlation, linear spectral cosine, target time--frequency masking fraction,
energy-weighted masking fraction and temporal activity overlap are retained.
We add a cosine between spectra averaged only over target-active frames and a
dB correlation after dividing each Mel band by its summed filter weights.
The latter is a bandwidth-normalised density diagnostic. All use the project's
64-band DSP/CNN frontend, not CLAP's internal frontend. A target-active frame
has unwindowed energy within 40 dB of its peak; a target-active time--frequency
cell is within 40 dB of peak Mel-cell power. The masked-cell fraction counts
those cells where scaled interference power exceeds target power. It is a
computational proxy, not a validated measure of perceptual masking. Undefined
target-active cosines are left missing (nine own-power rows), not replaced by zero.

This frontend detail changes the interpretation of the preliminary analysis.
At +5 dB the white-noise median raw dB Mel correlation is $-0.728$; after
bandwidth normalisation it is $-0.008$ (Figure~\ref{fig:acoustic-distributions}).
White-noise power per unequal-width Mel band is not flat, even when its power
density is flat. The apparent anticorrelation is therefore not evidence that
white noise physically avoids the target spectrum. Median masked-cell fractions
are 0.319 for white noise, 0.146 for DEMAND, 0.121 for same-category events and
0.120 for different-category events under own power. These marginal summaries
describe different spectral and temporal structures, not equivalent acoustic
difficulty or semantic causality.

\closingfig{22_acoustic_distributions_own}{Acoustic descriptor distributions at +5 dB, own-power protocol. Each box represents 2,000 target mixtures; whiskers extend to 1.5 interquartile ranges. Bandwidth normalisation changes the white-noise correlation interpretation. Masking is a time--frequency proxy; each descriptor has its own scale. Target-power distributions are exported separately.}{fig:acoustic-distributions}

\FloatBarrier
\subsection{Same-target comparisons with close acoustic descriptors}
The primary +5-dB comparison retains a target only when its two mixtures differ
by at most 0.10 in target-active spectral cosine, 0.05 in masked-cell fraction,
and 0.10 in temporal overlap. Selection uses audio descriptors alone and is
identical for SVM, core CNN seed 5305 and single-template CLAP. Strict
$(0.05,0.025,0.05)$ and loose $(0.20,0.10,0.20)$ versions accompany it, as do
all-target and high-temporal-overlap comparisons. These calipers improve
agreement on measured descriptors; they do not make the waveforms identical.

The primary same-/different-category versus DEMAND subsets retain 210/178
targets under own power (199/168 source clusters; 49/48 classes) and 205/187
under target power (195/178 clusters; 49/47 classes). Only about 9--10.5\% of
the benchmark is retained. Accuracy differences compare the same targets within
each selected subset. Pointwise 95\% percentile intervals use 2,000 bootstrap
resamples of original target-recording IDs, keeping clips from one source
together (seed 530510). They are conditional on the fitted models and observed
pairings; shared interferers, backgrounds and training-set dependence are not
fully captured. No new family of significance tests is introduced. Intervals
are omitted when either target or source-cluster count is below 30.

\begin{table}[htbp]\centering\small
\caption{Accuracy difference in the primary acoustic-closeness subsets at +5 dB: semantic minus DEMAND, percentage points [pointwise 95\% source-bootstrap interval]. Positive favours the semantic mixture. Each column compares identical targets within its own subset; subset sizes are given in the text.}
\label{tab:acoustic-matched}\begin{tabular}{llrr}\toprule
Power & Model & Same category minus DEMAND & Different category minus DEMAND\\\midrule
"""
    for region in ("own", "target"):
        for system in ("svm", "cnn", "clap"):
            values = []
            for condition in ("semantic_same", "semantic_different"):
                matches = [
                    r
                    for r in comparisons
                    if r["region"] == region
                    and r["system"] == system
                    and r["comparison"] == condition + "__demand"
                    and r["stratum"] == "close_primary"
                ]
                if len(matches) != 1:
                    raise ValueError("Missing or duplicate acoustic cell")
                r = matches[0]
                values.append(
                    f"{100 * float(r['accuracy_difference']):+.2f} [{100 * float(r['ci_low']):+.2f}, {100 * float(r['ci_high']):+.2f}]"
                )
            text += (
                region.capitalize()
                + " & "
                + system.upper()
                + " & "
                + " & ".join(values)
                + r" \\"
                + "\n"
            )
        if region == "own":
            text += r"\midrule" + "\n"
    text += r"""\bottomrule\end{tabular}\end{table}

CLAP's semantic disadvantage remains about 16--26 percentage points in these
selected populations (Table~\ref{tab:acoustic-matched}); its strict and loose
semantic-minus-DEMAND intervals also remain below zero. Measured spectral and
temporal proxies therefore do not eliminate the observed CLAP gap on their
available common support. This is residual association, not proof of an
independent causal effect of meaning: onset detail, modulation, event
recognisability and salience remain uncontrolled, and DEMAND itself can contain
meaningful events.

The CNN point estimates in the close-descriptor subsets are instead positive,
with all four intervals including zero. Its original high-overlap semantic
disadvantage is still present in the earlier, larger population. Because target
composition changes, the difference between those results cannot quantify how
much acoustics explains. SVM same-category intervals also include zero;
different-category contrasts remain positive. Direct same-minus-different
contrasts on their own primary matched subsets have intervals spanning zero
for all three models in both power protocols. They do not establish an
independent coarse-category effect.

\closingfig{23_acoustic_matched_contrasts}{Same-target semantic-minus-DEMAND accuracy at +5 dB. Grey: all eligible targets; open markers: both mixtures have temporal overlap at least 0.8; filled markers: primary acoustic-closeness subset. Error bars are pointwise 95\% target-source bootstrap intervals; displayed counts refer only to the close subset. The three selections are different populations, not sequential estimates of a causal mediation effect.}{fig:acoustic-matched}

Descriptive bin summaries and logistic fits remain available as secondary
diagnostics. In particular, a change in a logistic condition coefficient after
adding covariates is not a percentage of the effect explained: odds ratios are
non-collapsible, and support and model specification also matter
\cite{noncollapsibility}. In-sample pseudo-$R^2$ is not explained causal
variance. The middle third of a masking-difference distribution is labelled as
a third, not as an acoustically matched group. Accordingly, the regressions
are descriptive sensitivity checks and do not separate semantic from acoustic causes.
"""
    (ROOT / "docs/report_closing_analyses.tex").write_text(text)


if __name__ == "__main__":
    build_closing_report()

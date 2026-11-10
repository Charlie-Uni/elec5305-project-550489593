# Delivery synchronization — 10 November 2026

The existing 26-page report, 16-slide editable presentation, exact speaker notes,
recording script and project site now share the completed closing analyses.
Original numerical experiment outputs and the report PDF remain unchanged.

## Checks

- `consistency_audit.json`: the seven native chart caches (54 values), intervention
  table (30 values), six editable confidence-interval lines, 16 narration segments,
  unchanged embedded audio and four rendered website selector states agree with
  the underlying verified CSV files. Website interval limits and capture
  numerators/denominators are checked as well as point estimates.
- `finalizer_receipt.json`: slide package, layout, fonts, editable chart workbooks
  and first-party import pass structural validation. Native PowerPoint playback
  was not tested; the author must test the audio in the recording application.
- `pptx_validation.txt`: independent OOXML schema validation passes.
- `tests.txt`: 96 tests pass, one CUDA-only test is skipped on this Mac.
- `public_deployment_audit.json`: six public downloads match the local SHA-256
  values, with the Pages build and source branch recorded. The live
  [project page](https://charlie-uni.github.io/elec5305-project-550489593/) is served
  from `gh-pages`; `main` remains pending review. `public_browser_audit.json`
  repeats the interaction, asset and audio checks on the deployed site.
- `local_browser_audit.json`: both power conventions and both CLAP readouts,
  320/390/1280-pixel layouts, 26 linked assets and two five-second audio files.
- The slide PDF was rendered and all 16 pages were visually reviewed; the record
  is `../presentation_20261109_draft/visual_audit.json`.

Reproduce the numerical delivery check from the repository root:

```sh
python scripts/audit_delivery_sync.py --browser-audit results/delivery_sync_20261110/local_browser_audit.json
```

`delivery_claims.py` verifies the independently audited closing-data hashes before
building either presentation or website data. The baseline presentation in Git
commit `21768f4c9ed6265e44e7d4e0bf7e16a43121a7d8` supplies the audio comparison.

## Interpretation retained in every deliverable

Single-template CLAP, own activity and +5 dB: 37.34% / 33.17% of original captures
retain the target at rank two (same / different category). Those conditional
fractions differ from joint top-two inclusion among all semantic trials,
26.05% / 21.00%. Ranking is not a calibrated event detector or proof of hearing.

After restriction to the same-target acoustically close subsets, CLAP remains
about 16–26 accuracy percentage points below DEMAND across the two power
conventions. The CNN pointwise intervals include zero. Only about 9–10.5% of
targets remain; this is a selected-population association, not causal isolation
of semantic meaning. The earlier high-overlap comparison uses a different
population and remains in the materials. Raw white-noise Mel correlation is
sensitive to unequal filter bandwidth, so the original anticorrelation and
logistic coefficient attenuation are not treated as mechanisms.

The [final checklist](../../docs/final_submission_checklist_20261110.md) separates
completed research from author listening, final course-format confirmation,
video recording and actual course submission. The script is 1,389 words,
estimated at 9.4–10.9 minutes including two audio clips, before extra pauses.

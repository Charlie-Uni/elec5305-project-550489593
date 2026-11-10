# Listening-page expansion and verification

Four selected targets illustrate success, first-place distractor capture, target
retention in second place, and errors already present in the clean control.
These are explanatory examples; the estimates still use all 2,000 targets per
own-activity condition. No models, experimental pairings or original results were
changed. The original 24 mixtures and 24 controls remain byte-identical.

## Evidence checks

- [Numerical and waveform audit](numerical_audio_audit.json): all 48 mixture rows
  and their 48 controls were rebuilt from source audio and the canonical manifest.
  All 96 PCM16 WAV files match byte for byte, including DEMAND. Saved predictions
  for three systems, the 24 semantic CLAP rankings and 36 system/condition/level
  rate cells were independently checked. Clip attribution matches the pinned
  ESC-50 licence file. All 714 pre-existing result/example files at `069240f`
  retain their Git blob hashes.
- [Local browser audit](browser_audit.json): all 48 choices loaded both five-second
  recordings, with matching predictions, rankings and rates. The two new targets
  were also checked at 1280, 390 and 320 pixels, with no page overflow. The existing
  closing-analysis selectors and local resource links were checked too.
- [Delivery consistency audit](delivery_audit.json): the unchanged report, slide
  charts, narration, original example audio and existing website result sections
  remain numerically consistent. This audit records the revised website hash;
  earlier dated audit files are preserved.

At +5 dB for same-category interference, the page shows **captures / all trials**:
SVM 283/2,000 = 14.15%, CNN 403/2,000 = 20.15%, CLAP 632/2,000 = 31.60%.
The report's **captures / errors** are 18.53%, 31.26% and 66.95%, respectively.
They use different denominators and must not be interchanged.

## Small changes during integration

The alarm/can example uses different recordings from the original can/alarm
example, so its description no longer calls it an exact reverse pairing. Clean
control checks now cover every exported condition, not only one white-noise row.
The target overview is labelled as applying across conditions. Both source-credit
directories are linked from the page.

Waveform inspection is not a human listening assessment. For alarm plus sipping,
the 0-dB mixture RMS is −28.53 dBFS and the gain-matched control is −31.65 dBFS;
the mixture peak is 0.99. The quiet average level follows the recorded attenuation,
not an export error. Audio gains were preserved. The page advises increasing
playback volume equally for the pair if needed; human audibility remains to be
checked by the author.

## Regenerate the explanatory context

From the repository root with the project Python environment:

```sh
PYTHONPATH=src python scripts/build_listening_context.py \
  --examples examples/listening_own_20261109 \
  --examples examples/listening_own_20261110 \
  --output examples/listening_own_20261110/context.json
```

The input patch and ZIP checksums are retained in the numerical audit. The final
report's pending editorial revisions, truthful AI acknowledgement, author-narrated
video and Canvas submission are separate outstanding delivery tasks; this website
update does not mark them complete.

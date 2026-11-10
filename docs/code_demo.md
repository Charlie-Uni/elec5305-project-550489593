# Real code demonstration

[Watch the recorded execution](https://charlie-uni.github.io/elec5305-project-550489593/results/final_submission_20261110/code_demo.mp4) · [Project site](https://charlie-uni.github.io/elec5305-project-550489593/) · [Final recording script](video_narration_final.txt)

This demonstration executes the existing signal-processing and inference pipeline. It reads two ESC-50 source WAVs, reconstructs one published mixture, verifies its SIR and peak, computes features/embeddings and runs the three saved models. Historical predictions are read **after inference**, only to verify agreement. It does not retrain a model or substitute a saved table for execution.

The video is an actual screen recording of this process. The two exported audio examples are included at their logged playback times. It has no author narration; the author should explain the code and results during the final presentation.

The demo keeps the core single-template CLAP readout and original CNN seed 5305.
The fixed prompt ensemble and local-alignment readouts are separate evaluated
extensions; they do not replace the recorded example's reference predictions.

## Inputs and setup

Use the default `main` checkout (the retained `feature/interference` branch also works) and the environments/data described in the [baseline guide](guides/baselines.md) and [CLAP guide](guides/interference_and_clap.md#frozen-clap-zero-shot-reference). The layout below is relative to a runtime directory; `repo/` is the checkout. Data and weights are external to GitHub because of their size. The demo validates source, metadata and model provenance and uses the evaluated checkpoints; arbitrary replacement models need not reproduce these outputs.

```text
runtime/
  repo/
  baseline-venv/bin/python
  clap-venv/bin/python
  data/ESC-50/{audio,meta}/
  artifacts/interference/v3_20261109/clean_svm_reconstructed/
  artifacts/baselines/clean_cnn/
  models/clap/630k-audioset-best.pt
  hf-cache/
```

The baseline environment contains the pinned SVM/CNN dependencies used by the saved models. The CLAP environment follows `requirements-clap-macos.txt` and needs the cached RoBERTa tokenizer/configuration already prepared by the documented CLAP clean run. The demonstration uses offline inference; it is not an automatic weight download. Follow the linked clean-training and CLAP preparation guides when reproducing on a new computer. No DEMAND files are needed for this example.

## Run from the checkout

```bash
../baseline-venv/bin/python scripts/run_code_demo.py --runtime-root ..
```

Or open the recording interface:

```bash
../baseline-venv/bin/python scripts/serve_code_demo.py --runtime-root ..
```

Then open [the local demonstration](http://127.0.0.1:8766/) and select **Run mixing and all three models**. The interface displays the live processing messages, verified fresh predictions, target ranks and audio controls. The server binds only to the local computer. Stop it with Control-C after recording. If the runtime is elsewhere, replace `..` with that directory; `--output` chooses a fresh output directory. Direct CLI defaults write demo outputs under `results/final_submission_20261110/demo_run/`; the UI defaults to `runs/final_submission_20261110/demo_record/` under the runtime. Original experiment results are not overwritten.

## Expected example and verification

Manifest row 91, own power, +5 dB:

- Target: `1-101404-A-34.wav`, can opening.
- Distractor: `1-13613-A-37.wav`, clock alarm; full activity overlap.
- Achieved SIR: 5.000000 dB; mixture peak: 0.990000, without hard clipping.
- All three gain-matched controls predict can opening.
- Mixture: SVM predicts chirping birds; CNN and single-prompt CLAP predict clock alarm.
- CLAP target rank is **three** in this selected case. It does not illustrate top-two recovery; that conclusion comes from all semantic trials.

Each model writes all 50 scores, input/model hashes, measurements and comparison results. The recorded run agrees exactly for the checked SVM/CNN scores and within approximately `5.3e-8` for the checked CLAP target/distractor scores. The agreement test allows `1e-4` numerical tolerance. Runtime includes model loading and is **not** the controlled inference-cost benchmark reported in the study.

During the final video, show the input pair, the power checks and one fresh model prediction before explaining the comparison. Play the five-second control then the five-second mixture at the same volume. The sound is an observed illustrative failure, not a random sample or evidence of perceptual dominance.

## Attribution

“Can Open.WAV” by [LG](https://freesound.org/people/LG/sounds/101404/) (CC BY); “alarm clock.aif” by [hookhead](https://freesound.org/people/hookhead/sounds/13613/) (CC0). ESC-50 compilation by Karol J. Piczak, [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/). Processing includes ESC-50 cropping, controlled mixing, common attenuation and PCM16 export. See the [full example attribution](../examples/listening_own_20261109/README.md).

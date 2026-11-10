"""Keep the concise recording guide identical to the deck's speaker notes."""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/final_submission_20261110"


def main():
    segments = json.loads((OUT / "narration_segments.json").read_text())
    words = sum(len(re.findall(r"\b[\w]+(?:[’'-][\w]+)*\b", s["narration"])) for s in segments)
    # Demo narration runs over loading/inference. Add ten seconds for the clips
    # and twenty seconds for the click, result inspection and screen transition.
    extra = 30
    estimates = [round(words / rate + extra / 60, 2) for rate in [150, 130]]
    text = f"""ELEC5305 FINAL PRESENTATION — RECORDING SCRIPT
Qihan Zhu | When Background Noise Has Meaning | 11 November 2026

Matches the 12-slide concise deck and the 11-page report.
Narration: {words} words. Speech at 130–150 words/minute plus 30 seconds
of demonstration pauses: approximately {estimates[0]:.1f}–{estimates[1]:.1f} minutes.
Loading speed and personal delivery may add time. The course says “brief”; it
specifies no minute limit in the supplied requirements. This is a production
estimate, not a course rule. The single formal report contains five core figures; detailed data remain in the repository.

SETUP
Use presentation.pptx or presentation.pdf in results/final_submission_20261110.
Start the local demonstration using docs/code_demo.md, then keep its browser
window ready. On slide 10 switch to that window, press Run, explain the live
processing, and play the two five-second examples without speaking over them.
The verified code_demo.mp4 is a fallback execution clip with audio examples;
it has no author narration. Do not imply that a playback is a new live run.
Review these notes and explain them naturally in your own words.

"""
    timing = []
    elapsed = 0

    def stamp(n):
        return f"{n // 60:02d}:{n % 60:02d}"

    for s in segments:
        count = len(re.findall(r"\b[\w]+(?:[’'-][\w]+)*\b", s["narration"]))
        pause = extra if s["slide"] == 10 else 0
        duration = math.ceil(count / 145 * 60) + pause
        text += (
            f"{stamp(elapsed)}–{stamp(elapsed + duration)} | SLIDE {s['slide']} | {s['title']}\n"
        )
        text += (
            "Visual: switch to the real local execution interface; show source processing and outputs.\n"
            if pause
            else "Visual: corresponding slide in the concise presentation.\n"
        )
        text += (
            "\nNarration:\n"
            + s["narration"]
            + "\n\nEvidence (not spoken): "
            + s["evidence"]
            + "\n\n"
        )
        timing.append(
            dict(
                slide=s["slide"],
                start_seconds=elapsed,
                seconds=duration,
                words=count,
                extra_pause_seconds=pause,
            )
        )
        elapsed += duration
    text += """FINAL RECORDING CHECKS
- Check that the author can explain the processing and distinguish personal work,
  external libraries/checkpoints, and AI assistance under the course's policy.
- Show actual execution, not only result slides. Pause speech during sound playback.
- Use identical playback volume for control and mixture. The selected case is not
  a representative estimate; CLAP ranks its target third, not second.
- Check the entire exported video for legible code/results and audible narration.
- Upload the report and final video to Canvas and retain the submission receipt.
- Put the final video URL on the site and in the report. The existing execution
  clip URL is already provided, but does not include the author's full explanation.
- Confirm the exact Week-13 deadline, accepted file format/size and AI rules in
  Canvas. The supplied brief does not specify these details.
"""
    (ROOT / "docs/video_narration_final.txt").write_text(text)
    (OUT / "narration_timing.json").write_text(
        json.dumps(
            dict(
                slides=len(segments),
                words=words,
                estimated_minutes_at_130_to_150_wpm=estimates,
                timeline_seconds=elapsed,
                extra_demo_pause_seconds=extra,
                final_narrated_video_recorded=False,
                segments=timing,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(words=words, estimated_minutes=estimates)))


if __name__ == "__main__":
    main()

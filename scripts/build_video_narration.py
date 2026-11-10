"""Synchronise the recording script with the deck's exact speaker notes."""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "results/presentation_20261109_draft"
    segments = json.loads((directory / "narration_segments.json").read_text())
    words = sum(len(re.findall(r"\b[\w]+(?:[’'-][\w]+)*\b", s["narration"])) for s in segments)
    low, high = words / 150 + 10 / 60, words / 130 + 10 / 60
    text = f"""ELEC5305 VIDEO RECORDING SCRIPT — AUTHOR REVIEW
Qihan Zhu | When Background Noise Has Meaning
Synchronized with the historical {len(segments)}-slide presentation, 10 November 2026.

This is a script, not a completed recording. The narration has {words} words.
Estimated speech plus two five-second audio examples: {low:.1f}–{high:.1f} minutes
at 130–150 words per minute, before extra pauses. Timings below use 145 words
per minute and are editing estimates, not an official course duration requirement.
Confirm the final brief before recording. The locally available document named
Project Report Guidelines is actually a proposal guide; its 700–1000-word limit
does not establish a final-report limit or video duration.

Use one slide per section below. Keep the whole slide legible, and do not narrate
over the two audio examples on slide 13. Exact evidence references are retained
after each passage for checking; they do not need to be spoken aloud.

"""
    elapsed = 0
    timing = []

    def stamp(seconds):
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    for segment in segments:
        count = len(re.findall(r"\b[\w]+(?:[’'-][\w]+)*\b", segment["narration"]))
        audio = 10 if segment["slide"] == 13 else 0
        duration = math.ceil(count / 145 * 60) + audio
        text += f"{stamp(elapsed)}–{stamp(elapsed + duration)} | SLIDE {segment['slide']} | {segment['title']}\n"
        text += "Visual: corresponding slide in research_presentation.pptx.\n"
        if audio:
            text += "Playback: five-second gain-matched control, then five-second mixture, identical volume; pause narration.\n"
        text += (
            "\nNarration:\n"
            + segment["narration"]
            + "\n\nEvidence (not spoken): "
            + segment["evidence"]
            + "\n\n"
        )
        timing.append(
            dict(
                slide=segment["slide"],
                start_seconds=elapsed,
                seconds=duration,
                words=count,
                audio_seconds=audio,
            )
        )
        elapsed += duration
    text += """FINAL RECORDING AND SUBMISSION CHECKS
- Rehearse in your own words and verify every claim, including the selected-population limitations.
- Confirm the official final-report length, video duration, file format, submission channel and deadline.
- Listen to the clean and mixed examples and test the embedded audio on the recording machine.
- Record the author's explanation, with readable figures and matched audio playback levels.
- Review the exported video from beginning to end; check sound, captions if required, and links.
- Retain audio attribution and the required AI-assistance disclosure.
- Submit the final files and links through the course system; retain the submission receipt.

The manuscript, slides, exact speaker notes, this script and the Pages data share
the same audited results. Full PAT, Mix-CLAP and new datasets are outside the
completed experimental scope. No additional experiment is needed for this closure.
"""
    (ROOT / "docs/video_narration_draft.txt").write_text(text)
    (directory / "narration_timing.json").write_text(
        json.dumps(
            dict(
                slides=len(segments),
                words=words,
                estimated_minutes_at_130_to_150_wpm=[low, high],
                timeline_seconds=elapsed,
                final_course_duration_verified=False,
                recorded_video=False,
                segments=timing,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"{len(segments)} slides; {words} words; estimated {low:.1f}–{high:.1f} minutes")


if __name__ == "__main__":
    main()

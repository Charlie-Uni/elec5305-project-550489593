"""Package tracked public files and verify the complete listening collection."""

import argparse
import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIRS = ("examples/listening_own_20261109", "examples/listening_own_20261110")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    output = output.resolve()
    if output == ROOT or (output.exists() and any(output.iterdir())):
        raise ValueError("Choose an empty directory for the Pages package")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    public = [
        Path(name)
        for name in tracked
        if name and (name == ".nojekyll" or not any(p.startswith(".") for p in Path(name).parts))
    ]
    output.mkdir(parents=True, exist_ok=True)
    for relative in public:
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Pages input must be a regular file: {relative}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    rows, manifest_hashes = [], set()
    for directory in EXAMPLE_DIRS:
        examples = json.loads((output / directory / "examples.json").read_text())
        manifest_hashes.add(examples["manifest_sha256"])
        rows.extend((directory, row) for row in examples["files"])
    targets = {r["target_file"] for _, r in rows}
    expected_conditions = {
        (condition, level)
        for condition in ("white", "demand", "semantic_same", "semantic_different")
        for level in (10, 5, 0)
    }
    if len(targets) != 4 or len(rows) != 48 or len(manifest_hashes) != 1:
        raise ValueError("Expected four targets, 48 choices and one canonical manifest")
    for target in targets:
        observed = {(r["condition"], r["level_db"]) for _, r in rows if r["target_file"] == target}
        if observed != expected_conditions:
            raise ValueError(f"Incomplete listening conditions: {target}")
    context = json.loads((output / EXAMPLE_DIRS[1] / "context.json").read_text())
    if set(context["targets"]) != targets:
        raise ValueError("Target selector context differs from the audio examples")
    for relative, expected in context["sources"].items():
        if sha(output / relative) != expected:
            raise ValueError(f"Context evidence checksum changed: {relative}")
    audio = []
    for directory, row in rows:
        for key in ("clean_matched", "mixture"):
            item = row["audio"][key]
            relative = Path(directory) / item["file"]
            path = output / relative
            if sha(path) != item["sha256"]:
                raise ValueError(f"Audio checksum changed: {relative}")
            with wave.open(str(path), "rb") as wav:
                shape = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes())
                if shape != (1, 2, 44100, 220500):
                    raise ValueError(f"Unexpected WAV format: {relative}")
            audio.append({"path": relative.as_posix(), "sha256": item["sha256"], "seconds": 5})
    if len({a["path"] for a in audio}) != 96:
        raise ValueError("Expected 96 distinct audio files")
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "index_sha256": sha(output / "index.html"),
        "site_data_sha256": sha(output / "site-data.json"),
        "tracked_public_files": len(public),
        "targets": sorted(targets),
        "choices": len(rows),
        "audio_files": audio,
    }
    (output / "deployment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"public_files": len(public), "targets": len(targets), "choices": len(rows), "audio_files": len(audio)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)

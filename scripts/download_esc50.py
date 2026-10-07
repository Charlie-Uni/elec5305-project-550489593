"""Download a pinned official ESC-50 snapshot (roughly 600 MB) outside Git tracking."""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

REVISION = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
URL = f"https://codeload.github.com/karolpiczak/ESC-50/zip/{REVISION}"


def download(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / f"ESC-50-{REVISION}.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        received, last_report = 0, 0
        with urlopen(URL, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                received += len(chunk)
                if received - last_report >= 32 * 1024 * 1024:
                    print(f"Downloaded {received / 1024**2:.0f} MiB", flush=True)
                    last_report = received
        partial.replace(archive)
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".esc50-") as temporary:
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                    raise ValueError(f"Unsafe archive entry: {member.filename}")
            source.extractall(temporary)
        extracted = Path(temporary) / f"ESC-50-{REVISION}"
        if not (extracted / "meta" / "esc50.csv").is_file():
            raise ValueError("Archive has no expected ESC-50 metadata")
        shutil.move(str(extracted), str(destination))
    provenance = {"url": URL, "revision": REVISION, "archive_sha256": digest}
    (destination / "download_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Ready: {destination}\nArchive SHA-256: {digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/ESC-50"))
    download(parser.parse_args().destination)

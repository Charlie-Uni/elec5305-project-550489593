"""Restore the exact evaluated mixture CSVs and relocate local data paths."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.mixing import read_manifest


def restore(source, output, data_root, demand_cache):
    provenance = json.loads((source / "export_audit.json").read_text())
    for region in ("own", "target"):
        origin = source / f"manifest_{region}"
        recorded = provenance["manifests"][region]
        compressed = origin / "manifest.csv.gz"
        if sha256_file(compressed) != recorded["compressed_sha256"]:
            raise ValueError("Compressed manifest checksum changed")
        content = gzip.decompress(compressed.read_bytes())
        if hashlib.sha256(content).hexdigest() != recorded["manifest_sha256"]:
            raise ValueError("Restored manifest checksum differs from evaluated mixtures")
        destination = output / f"manifest_{region}"
        destination.mkdir(parents=True, exist_ok=True)
        csv_path = destination / "manifest.csv"
        if csv_path.exists() and csv_path.read_bytes() != content:
            raise FileExistsError("Refusing to replace a different mixture manifest")
        csv_path.write_bytes(content)
        meta_path = origin / "manifest_meta.json"
        if sha256_file(meta_path) != recorded["metadata_sha256"]:
            raise ValueError("Archived manifest metadata changed")
        meta = json.loads(meta_path.read_text())
        meta.update(data_root=str(data_root.resolve()), demand_cache=str(demand_cache.resolve()))
        meta["restored_from_archive"] = dict(
            original_metadata_sha256=recorded["metadata_sha256"],
            manifest_csv_unchanged=True,
        )
        write_json(destination / "manifest_meta.json", meta)
        _, rows = read_manifest(destination)
        if len(rows) != recorded["rows"]:
            raise ValueError("Restored manifest row count changed")
        print(f"{region}: {len(rows)} rows; CSV SHA-256 {recorded['manifest_sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/mixture_manifests_20261109"))
    for name in ("output", "data-root", "demand-cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    restore(args.source, args.output, args.data_root, args.demand_cache)

# Exact evaluated mixture manifests

These compressed CSVs preserve the bytes used by every reported system and
intervention. The own-activity manifest contains 24,000 rows; the target-activity
manifest contains 23,991, excluding three zero-energy semantic pairs at three
levels. `export_audit.json` records compressed, uncompressed and metadata hashes.

Metadata keeps the original execution paths as provenance. Restore the CSVs and
relocate those paths without changing their identities:

```bash
python scripts/restore_manifests.py \
  --output artifacts/restored_manifests \
  --data-root /absolute/path/to/ESC-50 \
  --demand-cache /absolute/path/to/prepared_demand_cache
```

Use `artifacts/restored_manifests/manifest_own` or `manifest_target` with the
existing evaluation and rendering commands. Prepare the official DEMAND channel
cache first with `esc-baseline prepare-demand`; the archived metadata contains
the expected source and cache checksums. This archive contains pairing and mixing
instructions, not the ESC-50 or DEMAND recordings. Dataset licences still apply.

The evidence ZIP additionally includes uncompressed copies of both manifests.

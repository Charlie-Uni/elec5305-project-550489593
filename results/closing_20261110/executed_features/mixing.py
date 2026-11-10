"""Controlled interference: generic noise (SNR) and semantic distractors (SIR).

Conventions shared by every condition:
- Active regions are the samples of unwindowed frames whose energy lies within
  ``active_threshold_db`` of a signal's loudest frame. The target's power is always its
  mean square over its own active region. ``power_region`` selects where the
  interferer is measured: ``"own"`` (default) uses the interferer's own active region,
  so a sparse sound event is measured only where it sounds; ``"target"`` follows the
  proposal wording and measures the interferer inside the target's active region,
  which amplifies distractors that are quiet there. Both powers are always recorded,
  so either reading can be reported from either manifest.
- The interferer is scaled so that the configured power ratio equals the requested
  level; the achieved ratio is recomputed from the scaled components.
- Target and distractor keep their natural alignment (zero offset); the fraction of
  target-active samples at which the distractor is also active is recorded.
- Mixtures are never hard clipped. When the mixture peak exceeds ``peak_limit`` the
  whole mixture receives one common gain, which is recorded so that a gain-matched
  clean control can be regenerated. A common gain leaves the ratio unchanged.
- Every manifest row is reproducible from the dataset, the master seed and the row's
  (target, condition) alone, so every system receives the identical canonical mixture,
  a diagnostic subset reproduces the rows of the full manifest, and one interferer
  realisation (noise draw, DEMAND segment or distractor clip) serves all levels.
- ``offset_samples`` is the start of the DEMAND segment for ``demand`` rows (0 otherwise).
"""

import csv
import hashlib
import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .artifacts import sha256_file, write_json
from .data import load_audio, load_metadata, source_overlap_report
from .dsp import FeatureConfig, frame_audio, validate_waveform

CONDITIONS = ("white", "demand", "semantic_same", "semantic_different")
DEMAND_SAMPLE_RATE = 48000


@dataclass(frozen=True)
class InterferenceConfig:
    levels_db: tuple = (10.0, 5.0, 0.0)
    conditions: tuple = CONDITIONS
    active_threshold_db: float = 40.0
    power_region: str = "own"
    peak_limit: float = 0.99
    seed: int = 5305
    demand_environments: tuple = ("STRAFFIC", "PSTATION", "TMETRO")
    demand_channel: str = "ch01"

    def __post_init__(self):
        if not self.levels_db or not np.isfinite(self.levels_db).all():
            raise ValueError("levels_db must be a nonempty list of finite decibel values")
        if not self.conditions or any(c not in CONDITIONS for c in self.conditions):
            raise ValueError(f"conditions must be drawn from {CONDITIONS}")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must not repeat")
        if not 0 < self.active_threshold_db < 120 or not 0 < self.peak_limit <= 1:
            raise ValueError("Invalid active_threshold_db or peak_limit")
        if self.power_region not in ("own", "target"):
            raise ValueError('power_region must be "own" or "target"')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if "demand" in self.conditions and not self.demand_environments:
            raise ValueError("The demand condition needs at least one environment")

    def as_dict(self):
        return asdict(self)


def interference_config_from_json(values):
    values = dict(values or {})
    for key in ("levels_db", "conditions", "demand_environments"):
        if key in values:
            values[key] = tuple(values[key])
    return InterferenceConfig(**values)


# --------------------------------------------------------------------------- power


def active_mask(audio, config=FeatureConfig(), threshold_db=40.0):
    """Sample mask covering unwindowed frames within ``threshold_db`` of the peak frame."""
    audio = validate_waveform(audio)
    frames = frame_audio(audio, config)
    energy = np.mean(frames**2, axis=1)
    peak = energy.max()
    mask = np.zeros(audio.size, dtype=bool)
    if peak <= 0:
        mask[:] = True  # digital silence: every sample is equally (in)active
        return mask
    for index in np.flatnonzero(energy >= peak * 10 ** (-threshold_db / 10)):
        start = index * config.hop_length
        mask[start : min(start + config.frame_length, audio.size)] = True
    return mask


def masked_power(audio, mask):
    audio = validate_waveform(audio)
    if mask.shape != audio.shape or not mask.any():
        raise ValueError("Mask must match the waveform and select at least one sample")
    return float(np.mean(audio[mask] ** 2))


def ratio_db(target_power, interferer_power):
    if target_power <= 0 or interferer_power <= 0:
        raise ValueError("Powers must be positive to form a decibel ratio")
    return float(10 * np.log10(target_power / interferer_power))


def active_power(audio, config=FeatureConfig(), threshold_db=40.0):
    """Mean square of ``audio`` over its own active region; (power, mask)."""
    mask = active_mask(audio, config, threshold_db)
    return masked_power(audio, mask), mask


def interferer_gain(target_power, interferer_power, level_db):
    """Gain for the interferer so that target_power / (gain^2 interferer_power) is level_db."""
    if target_power <= 0:
        raise ValueError("Target has no energy inside its active region")
    if interferer_power <= 0:
        raise ValueError("Interferer has no energy inside its active region")
    return float(np.sqrt(target_power / (interferer_power * 10 ** (level_db / 10))))


def mix(target, interferer, gain, peak_limit=0.99):
    """Return (mixture, common_gain, peak_before_common_gain)."""
    target, interferer = validate_waveform(target), validate_waveform(interferer)
    if target.shape != interferer.shape:
        raise ValueError("Target and interferer must have the same length")
    mixture = target + gain * interferer
    peak = float(np.max(np.abs(mixture)))
    common = float(min(1.0, peak_limit / peak)) if peak > 0 else 1.0
    return mixture * common, common, peak


def activity_overlap(target_mask, interferer_mask):
    """Fraction of target-active samples at which the interferer is also active."""
    if target_mask.shape != interferer_mask.shape or not target_mask.any():
        raise ValueError("Masks must match and the target mask must be nonempty")
    return float(np.mean(interferer_mask[target_mask]))


# -------------------------------------------------------------------------- DEMAND


def prepare_demand(demand_root, output_dir, environments, channel="ch01", target_rate=44100):
    """Resample one DEMAND channel per environment to the project rate and cache it."""
    demand_root, output_dir = Path(demand_root), Path(output_dir)
    if not environments:
        raise ValueError("Provide at least one DEMAND environment")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for environment in environments:
        source = demand_root / environment / f"{channel}.wav"
        if not source.is_file():
            raise FileNotFoundError(f"Missing DEMAND file {source}")
        audio, rate = sf.read(source, dtype="float64", always_2d=False)
        if audio.ndim != 1:
            raise ValueError(f"{source}: expected a single channel")
        if rate != DEMAND_SAMPLE_RATE:
            raise ValueError(f"{source}: expected {DEMAND_SAMPLE_RATE} Hz, got {rate}")
        if rate != target_rate:
            divisor = int(np.gcd(target_rate, rate))
            audio = resample_poly(audio, target_rate // divisor, rate // divisor)
        audio = validate_waveform(audio)
        cache = output_dir / f"{environment}_{channel}.npy"
        np.save(cache, audio.astype(np.float64))
        report[environment] = {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "source_sample_rate": rate,
            "channel": channel,
            "cache": cache.name,
            "cache_sha256": sha256_file(cache),
            "samples": int(audio.size),
            "seconds": audio.size / target_rate,
            "resampler": f"scipy.signal.resample_poly {target_rate}/{rate}",
        }
    write_json(output_dir / "demand_cache.json", {"sample_rate": target_rate, **report})
    return report


class SilentInTargetRegion(ValueError):
    """The interferer has no energy inside the target's active region."""


class DemandCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "demand_cache.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Run prepare-demand first; missing {meta_path}")
        self.meta = json.loads(meta_path.read_text())
        self._audio = {}

    def environments(self):
        return sorted(k for k in self.meta if k != "sample_rate")

    def audio(self, environment):
        if environment not in self._audio:
            if environment not in self.meta:
                raise KeyError(f"DEMAND environment {environment} is not cached")
            path = self.cache_dir / self.meta[environment]["cache"]
            if sha256_file(path) != self.meta[environment]["cache_sha256"]:
                raise ValueError(f"DEMAND cache checksum mismatch: {environment}")
            self._audio[environment] = np.load(path, allow_pickle=False)
        return self._audio[environment]

    def segment(self, environment, offset, length):
        audio = self.audio(environment)
        if offset < 0 or offset + length > audio.size:
            raise ValueError("DEMAND segment lies outside the cached recording")
        return audio[offset : offset + length]


# ------------------------------------------------------------------------- pairing


def pair_distractors(clips, config=InterferenceConfig()):
    """Seeded pairing: per target one same-coarse and one different-coarse distractor.

    Candidates come from the target's own (held-out) fold, have a different fine class
    and a different source recording, and never belong to a source that crosses folds.
    """
    overlapping = {r["source_id"] for r in source_overlap_report(clips)}
    pairs = {}
    for fold in sorted({c.fold for c in clips}):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, fold]))
        members = sorted((c for c in clips if c.fold == fold), key=lambda c: c.filename)
        pool = [c for c in members if c.source_id not in overlapping]
        for target in members:
            same = [
                c
                for c in pool
                if c.target != target.target
                and c.source_id != target.source_id
                and c.coarse_category == target.coarse_category
            ]
            different = [
                c
                for c in pool
                if c.source_id != target.source_id and c.coarse_category != target.coarse_category
            ]
            if not same or not different:
                raise ValueError(f"No eligible distractor for {target.filename}")
            pairs[target.filename] = {
                "semantic_same": same[int(rng.integers(len(same)))],
                "semantic_different": different[int(rng.integers(len(different)))],
            }
    return pairs


def plan_rows(clips, config=InterferenceConfig()):
    """Enumerate (condition, level, target, interferer) in a fixed order.

    ``interferer`` is ("white", None), ("demand", environment) or ("clip", Clip).
    """
    pairs = (
        pair_distractors(clips, config)
        if any(c.startswith("semantic") for c in config.conditions)
        else {}
    )
    plan = []
    for fold in sorted({c.fold for c in clips}):
        members = sorted((c for c in clips if c.fold == fold), key=lambda c: c.filename)
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, fold, 1]))
        assignment = []
        if "demand" in config.conditions:
            while len(assignment) < len(members):
                assignment.extend(rng.permutation(list(config.demand_environments)).tolist())
        for i, target in enumerate(members):
            for condition in config.conditions:
                if condition == "white":
                    interferer = ("white", None)
                elif condition == "demand":
                    interferer = ("demand", assignment[i])
                else:
                    interferer = ("clip", pairs[target.filename][condition])
                for level in config.levels_db:
                    plan.append((condition, float(level), target, interferer))
    return plan


# ------------------------------------------------------------------------- manifest

MANIFEST_FIELDS = (
    "row",
    "condition",
    "ratio_type",
    "level_db",
    "fold",
    "target_file",
    "target_class",
    "target_category",
    "target_coarse",
    "interferer",
    "interferer_class",
    "interferer_category",
    "interferer_coarse",
    "offset_samples",
    "row_seed",
    "interferer_gain",
    "common_gain",
    "target_power",
    "interferer_power",
    "achieved_db",
    "interferer_power_on_target_mask",
    "target_active_fraction",
    "interferer_active_fraction",
    "activity_overlap",
    "peak_before_common_gain",
    "target_sha256",
    "interferer_sha256",
)

_INT_FIELDS = ("row", "fold", "target_class", "offset_samples", "row_seed")
_FLOAT_FIELDS = (
    "level_db",
    "interferer_gain",
    "common_gain",
    "target_power",
    "interferer_power",
    "achieved_db",
    "interferer_power_on_target_mask",
    "target_active_fraction",
    "interferer_active_fraction",
    "activity_overlap",
    "peak_before_common_gain",
)


def row_seed(master_seed, target_file, condition):
    """Seed for one (target, condition): independent of level, row order and subset size.

    The same white-noise realisation or DEMAND segment is therefore presented at every
    level of a target, exactly as the semantic pairs are fixed across levels.
    """
    entropy = [
        int(master_seed),
        zlib.crc32(str(target_file).encode("utf-8")),
        CONDITIONS.index(condition),
    ]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def subset_clips(clips, limit_per_fold):
    """Evenly strided subset per fold for small diagnostic runs (never reported as ESC-50)."""
    if limit_per_fold < 1:
        raise ValueError("limit_per_fold must be positive")
    kept = []
    for fold in sorted({c.fold for c in clips}):
        members = sorted((c for c in clips if c.fold == fold), key=lambda c: c.filename)
        step = max(1, len(members) // limit_per_fold)
        kept.extend(members[::step][:limit_per_fold])
    return kept


def build_manifest(
    data_root,
    output_dir,
    config=InterferenceConfig(),
    feature_config=FeatureConfig(),
    demand_cache_dir=None,
    limit_per_fold=None,
    require_complete=True,
):
    """Write manifest.csv and manifest_meta.json describing every canonical mixture."""
    data_root, output_dir = Path(data_root), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    clips = load_metadata(data_root, require_complete=require_complete)
    demand = None
    if "demand" in config.conditions:
        if demand_cache_dir is None:
            raise ValueError("The demand condition requires a prepared DEMAND cache")
        demand = DemandCache(demand_cache_dir)
        missing = set(config.demand_environments) - set(demand.environments())
        if missing:
            raise ValueError(f"DEMAND environments not cached: {sorted(missing)}")
    # Pairing and environment assignment use the complete fold, so a subset run
    # receives exactly the distractors it would have in the full manifest.
    plan = plan_rows(clips, config)
    if limit_per_fold is not None:
        wanted = {c.filename for c in subset_clips(clips, limit_per_fold)}
        plan = [p for p in plan if p[2].filename in wanted]
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = MixtureRenderer(data_root, feature_config, config, demand)
    summary = {
        "rows": 0,
        "common_gain_applied": 0,
        "low_overlap_pairs": 0,
        "dropped_silent_in_target_region": 0,
    }
    dropped = []
    hashes = {}

    def file_hash(name):
        if name not in hashes:
            hashes[name] = sha256_file(data_root / "audio" / name)
        return hashes[name]

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for index, (condition, level, target, (kind, payload)) in enumerate(plan):
            row = {
                "row": index,
                "condition": condition,
                "ratio_type": "SIR" if condition.startswith("semantic") else "SNR",
                "level_db": level,
                "fold": target.fold,
                "target_file": target.filename,
                "target_class": target.target,
                "target_category": target.category,
                "target_coarse": target.coarse_category,
                "row_seed": row_seed(config.seed, target.filename, condition),
                "target_sha256": file_hash(target.filename),
                "interferer_class": "",
                "interferer_category": "",
                "interferer_coarse": "",
                "interferer_sha256": "",
            }
            if kind == "white":
                row["interferer"] = "white"
            elif kind == "demand":
                row["interferer"] = f"demand:{payload}"
                row["interferer_sha256"] = demand.meta[payload]["cache_sha256"]
            else:
                row.update(
                    interferer=payload.filename,
                    interferer_class=payload.target,
                    interferer_category=payload.category,
                    interferer_coarse=payload.coarse_category,
                    interferer_sha256=file_hash(payload.filename),
                )
            try:
                row.update(renderer.render(row, measure_only=True)["measurements"])
            except SilentInTargetRegion:
                # Only possible with power_region="target": the distractor has no energy
                # where the target sounds, so no finite gain reaches the requested level.
                summary["dropped_silent_in_target_region"] += 1
                dropped.append({"target_file": target.filename, "interferer": row["interferer"]})
                continue
            row["row"] = summary["rows"]
            summary["rows"] += 1
            summary["common_gain_applied"] += int(row["common_gain"] < 1)
            summary["low_overlap_pairs"] += int(kind == "clip" and row["activity_overlap"] < 0.5)
            writer.writerow(row)
            if summary["rows"] % 500 == 0 or summary["rows"] == len(plan):
                print(f"Planned {summary['rows']}/{len(plan)} mixtures", flush=True)
    meta = {
        "schema_version": 1,
        "interference_config": config.as_dict(),
        "feature_config": feature_config.as_dict(),
        "data_root": str(data_root),
        "metadata_sha256": sha256_file(data_root / "meta" / "esc50.csv"),
        "clips_in_dataset": len(clips),
        "subset_limit_per_fold": limit_per_fold,
        "demand_cache": None if demand is None else str(Path(demand_cache_dir)),
        "demand_environments": {}
        if demand is None
        else {env: demand.meta[env] for env in config.demand_environments},
        "distractor_rule": (
            "same fold as the target; different fine class; different source recording; "
            "sources that cross folds are excluded; one same-coarse and one different-coarse "
            "distractor per target; seeded by (seed, fold)"
        ),
        "power_definition": (
            "target: mean square over its own active region (unwindowed frames within "
            f"{config.active_threshold_db} dB of its peak frame); interferer: mean square "
            + (
                "over its own active region"
                if config.power_region == "own"
                else "inside the target's active region (proposal wording)"
            )
            + "; both interferer readings are recorded in every row"
        ),
        "dropped_pairs": dropped,
        "alignment_rule": (
            "target and distractor keep their natural alignment (zero offset); DEMAND "
            "segments start at a seeded offset into the cached recording"
        ),
        "peak_rule": f"one common gain when the mixture peak exceeds {config.peak_limit}",
        "summary": summary,
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("mixing.py", "dsp.py", "data.py")
        },
    }
    write_json(output_dir / "manifest_meta.json", meta)
    return meta


def parse_row(raw):
    row = dict(raw)
    for key in _INT_FIELDS:
        row[key] = int(row[key])
    row["interferer_class"] = int(row["interferer_class"]) if row["interferer_class"] else None
    for key in _FLOAT_FIELDS:
        row[key] = float(row[key])
    return row


def read_manifest(manifest_dir):
    manifest_dir = Path(manifest_dir)
    meta = json.loads((manifest_dir / "manifest_meta.json").read_text())
    if meta.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    with (manifest_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("Manifest columns do not match the expected schema")
        rows = [parse_row(raw) for raw in reader]
    if [r["row"] for r in rows] != list(range(len(rows))):
        raise ValueError("Manifest rows must be numbered consecutively from zero")
    return meta, rows


def manifest_digest(manifest_dir):
    """Hash of the manifest CSV, recorded by downstream evaluations."""
    return hashlib.sha256((Path(manifest_dir) / "manifest.csv").read_bytes()).hexdigest()


def renderer_from_manifest(manifest_dir, data_root=None):
    meta, rows = read_manifest(manifest_dir)
    config = interference_config_from_json(meta["interference_config"])
    feature_config = FeatureConfig(**meta["feature_config"])
    demand = DemandCache(meta["demand_cache"]) if meta.get("demand_cache") else None
    root = Path(data_root) if data_root is not None else Path(meta["data_root"])
    if sha256_file(root / "meta" / "esc50.csv") != meta["metadata_sha256"]:
        raise ValueError("Dataset metadata checksum differs from the manifest")
    if demand is not None and demand.meta["sample_rate"] != feature_config.sample_rate:
        raise ValueError("DEMAND cache sample rate differs from the manifest")
    return MixtureRenderer(root, feature_config, config, demand), meta, rows


class MixtureRenderer:
    """Rebuild any manifest row deterministically from the dataset and the row."""

    def __init__(self, data_root, feature_config, config, demand=None):
        self.data_root = Path(data_root)
        self.feature_config = feature_config
        self.config = config
        self.demand = demand
        self._cache = {}
        self._verified_hashes = {}

    def _verify_file(self, filename, expected):
        if Path(filename).name != filename:
            raise ValueError("Manifest audio names must be plain filenames")
        if not expected:
            return
        if filename not in self._verified_hashes:
            self._verified_hashes[filename] = sha256_file(self.data_root / "audio" / filename)
        if self._verified_hashes[filename] != expected:
            raise ValueError(f"Audio checksum differs from the manifest: {filename}")

    def clip(self, filename):
        if filename not in self._cache:
            if len(self._cache) >= 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[filename] = load_audio(
                self.data_root / "audio" / filename, self.feature_config
            )
        return self._cache[filename]

    @staticmethod
    def _is_planned(row):
        return row.get("interferer_gain") not in (None, "")

    def _interferer(self, row, length):
        """Return (interferer waveform, DEMAND segment offset)."""
        name = row["interferer"]
        if name == "white":
            rng = np.random.default_rng(int(row["row_seed"]))
            return rng.standard_normal(length), 0
        if name.startswith("demand:"):
            if self.demand is None:
                raise ValueError("Rendering a DEMAND row requires the DEMAND cache")
            environment = name.split(":", 1)[1]
            recording = self.demand.audio(environment)
            if self._is_planned(row):
                offset = int(row["offset_samples"])
            else:
                rng = np.random.default_rng(int(row["row_seed"]))
                offset = int(rng.integers(0, recording.size - length + 1))
            return self.demand.segment(environment, offset, length), offset
        distractor = self.clip(name)
        if distractor.size != length:
            raise ValueError("Distractor and target clips must have equal length")
        return distractor, 0

    def render(self, row, measure_only=False):
        self._verify_file(row["target_file"], row.get("target_sha256"))
        interferer_name = row["interferer"]
        if interferer_name.startswith("demand:"):
            environment = interferer_name.split(":", 1)[1]
            expected = row.get("interferer_sha256")
            if expected and self.demand.meta[environment]["cache_sha256"] != expected:
                raise ValueError("DEMAND checksum differs from the manifest")
        elif interferer_name != "white":
            self._verify_file(interferer_name, row.get("interferer_sha256"))
        target = self.clip(row["target_file"])
        threshold = self.config.active_threshold_db
        p_target, target_mask = active_power(target, self.feature_config, threshold)
        interferer, offset = self._interferer(row, target.size)
        p_interferer, interferer_mask = active_power(interferer, self.feature_config, threshold)
        if self.config.power_region == "target":
            p_interferer = masked_power(interferer, target_mask)
            if p_interferer <= 0:
                raise SilentInTargetRegion(
                    f"{row['interferer']} has no energy inside the active region of "
                    f"{row['target_file']}"
                )
        gain = interferer_gain(p_target, p_interferer, float(row["level_db"]))
        if self._is_planned(row):
            if not np.isclose(gain, float(row["interferer_gain"]), rtol=1e-9, atol=0):
                raise ValueError(f"Row {row['row']}: regenerated interferer_gain differs")
            # Validate the calculation, then use the canonical recorded float. This
            # avoids NumPy-version reduction-rounding differences between systems.
            gain = float(row["interferer_gain"])
        mixture, common, peak = mix(target, interferer, gain, self.config.peak_limit)
        if self._is_planned(row):
            if not np.isclose(common, float(row["common_gain"]), rtol=1e-9, atol=0):
                raise ValueError(f"Row {row['row']}: regenerated common_gain differs")
            common = float(row["common_gain"])
            mixture = (target + gain * interferer) * common
        scaled_target, scaled_interferer = target * common, gain * common * interferer
        p_target_scaled = masked_power(scaled_target, target_mask)
        p_interferer_own = masked_power(scaled_interferer, interferer_mask)
        p_interferer_target = masked_power(scaled_interferer, target_mask)
        achieved = ratio_db(
            p_target_scaled,
            p_interferer_own if self.config.power_region == "own" else p_interferer_target,
        )
        measurements = {
            "offset_samples": offset,
            "interferer_gain": gain,
            "common_gain": common,
            "target_power": p_target_scaled,
            "interferer_power": p_interferer_own,
            "achieved_db": achieved,
            "interferer_power_on_target_mask": p_interferer_target,
            "target_active_fraction": float(target_mask.mean()),
            "interferer_active_fraction": float(interferer_mask.mean()),
            "activity_overlap": activity_overlap(target_mask, interferer_mask),
            "peak_before_common_gain": peak,
        }
        if abs(measurements["achieved_db"] - float(row["level_db"])) > 1e-6:
            raise ValueError(f"Row {row['row']}: achieved ratio deviates from the request")
        result = {"measurements": measurements}
        if not measure_only:
            result.update(
                mixture=mixture,
                clean_matched=scaled_target,
                interferer_scaled=scaled_interferer,
                target_mask=target_mask,
                interferer_mask=interferer_mask,
            )
        return result


def render_examples(manifest_dir, output_dir, data_root=None, rows=None, per_condition=None):
    """Write 16-bit WAVs for selected rows: listening checks and repository examples."""
    renderer, _, manifest = renderer_from_manifest(manifest_dir, data_root)
    if rows is not None:
        index = {r["row"]: r for r in manifest}
        selected = [index[int(r)] for r in rows]
    elif per_condition is not None:
        if per_condition < 1:
            raise ValueError("per_condition must be positive")
        seen, selected = {}, []
        for row in manifest:
            key = (row["condition"], row["level_db"])
            if seen.get(key, 0) < per_condition:
                seen[key] = seen.get(key, 0) + 1
                selected.append(row)
    else:
        raise ValueError("Provide rows or per_condition")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in selected:
        rendered = renderer.render(row)
        stem = (
            f"row{row['row']:06d}_{row['condition']}_{row['level_db']:+.0f}dB_"
            f"{Path(row['target_file']).stem}"
        )
        path = output_dir / f"{stem}.wav"
        sf.write(path, rendered["mixture"], renderer.feature_config.sample_rate, subtype="PCM_16")
        written.append(
            {
                "file": path.name,
                "row": row["row"],
                "condition": row["condition"],
                "level_db": row["level_db"],
                "target": row["target_category"],
                "interferer": row["interferer_category"] or row["interferer"],
                "common_gain": row["common_gain"],
                "offset_samples": row["offset_samples"],
                "sha256": sha256_file(path),
            }
        )
    write_json(
        output_dir / "examples.json",
        {
            "manifest": str(Path(manifest_dir)),
            "manifest_sha256": manifest_digest(manifest_dir),
            "files": written,
        },
    )
    return written

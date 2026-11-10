"""Training-only waveform augmentation banks with explicit partition provenance."""

import csv
import json
import zlib
from pathlib import Path

import numpy as np

from .artifacts import sha256_file, write_json
from .data import load_audio, outer_split
from .dsp import FeatureConfig, log_mel
from .features import extract_features, feature_names
from .mixing import active_power, interferer_gain, mix, ratio_db


class BatchAugmenter:
    def __init__(self, bank, indices, seed, metadata, probability=0.5):
        self.bank = bank
        self.positions = {int(row): i for i, row in enumerate(indices)}
        self.rng = np.random.default_rng(seed)
        self.metadata = metadata
        self.probability = probability

    def __call__(self, rows, clean):
        if any(int(row) not in self.positions for row in rows):
            raise ValueError("Augmentation received a row outside its training partition")
        values = clean.copy()
        selected = np.flatnonzero(self.rng.random(len(rows)) < self.probability)
        views = self.rng.integers(self.bank.shape[1], size=len(selected))
        for position, view in zip(selected, views):
            values[position] = self.bank[self.positions[int(rows[position])], view]
        return values


class AugmentationBanks:
    """One bank per mode/fold/selection-or-refit, shared across optimisation seeds."""

    def __init__(
        self,
        cache_path,
        data_root,
        bank_root,
        mode,
        seed=5305,
        bank_seed=5305,
        require_complete=True,
    ):
        if mode not in ("wgn", "semantic"):
            raise ValueError("Augmentation mode must be wgn or semantic")
        self.data_root, self.root = Path(data_root), Path(bank_root)
        self.mode, self.seed, self.bank_seed = mode, seed, bank_seed
        with np.load(cache_path, allow_pickle=False) as a:
            self.cache = {k: a[k] for k in a.files if k != "X"}
        self.config = FeatureConfig(**json.loads(self.cache["config_json"].item()))
        self.cache_hash = sha256_file(cache_path)
        if require_complete and len(self.cache["y"]) != 2000:
            raise ValueError("Official augmentation requires the full ESC-50 cache")
        if sha256_file(self.data_root / "meta/esc50.csv") != self.cache["metadata_sha256"].item():
            raise ValueError("Metadata differs from the spectrogram cache")
        self.source_folds = {}
        for source, fold in zip(self.cache["source_ids"], self.cache["folds"]):
            self.source_folds.setdefault(str(source), set()).add(int(fold))
        self.settings = {
            "mode": mode,
            "probability": 0.5,
            "levels_db": [5.0, 10.0, 15.0],
            "bank_seed": bank_seed,
            "optimisation_seed": seed,
            "power_region": "own",
            "active_threshold_db": 40.0,
            "peak_limit": 0.99,
            "views_per_training_clip": 3,
            "validation": "clean only",
            "mel_normalisation": "clean training clips only",
            "distractors": "current training partition only; exclude cross-fold source IDs",
            "bank_reuse": "same deterministic banks for all optimisation seeds",
        }

    def _audio(self, index):
        path = self.data_root / "audio" / str(self.cache["filenames"][index])
        if sha256_file(path) != str(self.cache["audio_sha256"][index]):
            raise ValueError("Audio checksum differs from the feature cache")
        return load_audio(path, self.config, require_five_seconds=False)

    def eligible_partners(self, indices, target_index, same_category):
        y, sources = self.cache["y"], self.cache["source_ids"]
        return np.array(
            [
                i
                for i in indices
                if y[i] != y[target_index]
                and sources[i] != sources[target_index]
                and len(self.source_folds[str(sources[i])]) == 1
                and ((int(y[i]) // 10 == int(y[target_index]) // 10) == same_category)
            ],
            dtype=int,
        )

    def __call__(self, fold, phase, indices):
        train, validation, _ = outer_split(self.cache["folds"], fold)
        expected = train if phase == "selection" else np.concatenate((train, validation))
        if phase not in ("selection", "refit") or not np.array_equal(indices, expected):
            raise ValueError("Bank must match the declared training partition exactly")
        directory = self.root / self.mode / f"fold_{fold}_{phase}"
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / "bank_meta.json"
        fingerprint = dict(
            mode=self.mode,
            fold=int(fold),
            phase=phase,
            indices=indices.tolist(),
            cache_sha256=self.cache_hash,
            bank_seed=self.bank_seed,
            levels_db=self.settings["levels_db"],
            source_sha256={
                name: sha256_file(Path(__file__).with_name(name))
                for name in ("augmentation.py", "mixing.py", "dsp.py", "features.py", "data.py")
            },
        )
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            if metadata["fingerprint"] != fingerprint:
                raise ValueError("Existing bank has different inputs or generation code")
            for filename, checksum in metadata["files"].items():
                if sha256_file(directory / filename) != checksum:
                    raise ValueError("Augmentation bank checksum mismatch")
        else:
            metadata = self._build(directory, indices, fingerprint)
        bank = np.load(directory / "log_mel.npy", mmap_mode="r")
        salt = zlib.crc32(f"{fold}:{phase}".encode())
        return BatchAugmenter(bank, indices, (self.seed + salt) % 2**32, metadata)

    def training_features(self, clean, labels, indices, fold, phase):
        """Equal clean/augmented exposure for deterministic SVM fitting.

        One of the three fixed views is selected per clip, with balanced level
        assignment. The complete CNN bank is retained in the provenance.
        """
        transform = self(fold, phase, indices)
        path = self.root / self.mode / f"fold_{fold}_{phase}" / "features.npy"
        bank = np.load(path, mmap_mode="r")
        positions = np.arange(len(indices))
        augmented = bank[positions, positions % bank.shape[1]]
        if augmented.shape != clean[indices].shape or not np.isfinite(augmented).all():
            raise ValueError("SVM augmentation feature shape or values are invalid")
        return (
            np.concatenate((clean[indices], augmented)),
            np.tile(labels[indices], 2),
            transform.metadata,
        )

    def _build(self, directory, indices, fingerprint):
        example = self._audio(int(indices[0]))
        shape = log_mel(example, self.config).T.shape
        bank = np.lib.format.open_memmap(
            directory / "log_mel.npy", mode="w+", dtype=np.float32, shape=(len(indices), 3, *shape)
        )
        features = np.lib.format.open_memmap(
            directory / "features.npy",
            mode="w+",
            dtype=np.float64,
            shape=(len(indices), 3, len(feature_names(self.config))),
        )
        records = []
        fold, phase = fingerprint["fold"], fingerprint["phase"]
        for position, index in enumerate(indices):
            target = self._audio(int(index))
            pt, tm = active_power(target, self.config)
            for view, level in enumerate(self.settings["levels_db"]):
                key = f"{self.mode}:{fold}:{phase}:{int(index)}:{view}"
                row_seed = (self.bank_seed + zlib.crc32(key.encode())) % 2**32
                rng = np.random.default_rng(row_seed)
                same = (position + view) % 2 == 0
                if self.mode == "wgn":
                    noise = rng.normal(size=target.size)
                    partner = None
                else:
                    choices = self.eligible_partners(indices, index, same)
                    if not len(choices):
                        raise ValueError("No eligible training-only semantic partner")
                    partner = int(rng.choice(choices))
                    noise = self._audio(partner)
                pn, nm = active_power(noise, self.config)
                gain = interferer_gain(pt, pn, level)
                audio, common, peak = mix(target, noise, gain)
                achieved = ratio_db(
                    np.mean((common * target[tm]) ** 2), np.mean((common * gain * noise[nm]) ** 2)
                )
                if abs(achieved - level) > 1e-8 or np.max(np.abs(audio)) > 0.99 + 1e-12:
                    raise ValueError("Augmentation power or peak verification failed")
                bank[position, view] = log_mel(audio, self.config).T.astype(np.float32)
                features[position, view] = extract_features(audio, self.config)
                records.append(
                    dict(
                        position=position,
                        view=view,
                        target_index=int(index),
                        target_file=str(self.cache["filenames"][index]),
                        target_class=int(self.cache["y"][index]),
                        target_fold=int(self.cache["folds"][index]),
                        target_sha256=str(self.cache["audio_sha256"][index]),
                        interferer="white"
                        if partner is None
                        else str(self.cache["filenames"][partner]),
                        interferer_class="" if partner is None else int(self.cache["y"][partner]),
                        interferer_fold=""
                        if partner is None
                        else int(self.cache["folds"][partner]),
                        interferer_sha256=""
                        if partner is None
                        else str(self.cache["audio_sha256"][partner]),
                        pairing=""
                        if partner is None
                        else ("same_category" if same else "different_category"),
                        seed=row_seed,
                        level_db=level,
                        achieved_db=achieved,
                        interferer_gain=gain,
                        common_gain=common,
                        peak_before_gain=peak,
                    )
                )
            if (position + 1) % 100 == 0 or position + 1 == len(indices):
                print(
                    f"Augmentation {self.mode} fold {fold} {phase}: {position + 1}/{len(indices)}",
                    flush=True,
                )
        bank.flush()
        features.flush()
        del bank, features
        with (directory / "manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        metadata = dict(
            fingerprint=fingerprint,
            rows=len(records),
            files={
                name: sha256_file(directory / name)
                for name in ("log_mel.npy", "features.npy", "manifest.csv")
            },
        )
        write_json(directory / "bank_meta.json", metadata)
        return metadata

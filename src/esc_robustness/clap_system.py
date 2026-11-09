"""Frozen zero-shot LAION-CLAP as the fixed pretrained reference system.

CLAP is never trained or tuned here. One fixed checkpoint embeds every clip with the
official frontend (48 kHz input, int16 round trip, HTSAT-tiny audio tower, RoBERTa
text tower) and one fixed prompt template names each of the 50 classes. The class
score is the cosine similarity between the audio embedding and the class prompt
embedding, so the target-distractor margin is a difference of cosines. The model is
fold independent: it ignores the fold argument that the supervised systems use.
"""

import csv
import importlib.metadata
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits

from .artifacts import sha256_file, write_json
from .data import load_audio, load_metadata
from .dsp import FeatureConfig, validate_waveform
from .svm import metrics

CLAP_SAMPLE_RATE = 48000
DEFAULT_PROMPT_TEMPLATE = "the sound of {label}"
DEFAULT_AUDIO_MODEL = "HTSAT-tiny"
DEFAULT_TEXT_MODEL = "roberta"
DEFAULT_CHECKPOINT_SHA256 = "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"


def prompt_label(category):
    """ESC-50 category identifiers use underscores; prompts use plain words."""
    return str(category).replace("_", " ").strip()


def build_prompts(class_names, template=DEFAULT_PROMPT_TEMPLATE):
    """One prompt per class ID, in ascending class-ID order."""
    if "{label}" not in template:
        raise ValueError("The prompt template must contain {label}")
    return [template.format(label=prompt_label(class_names[i])) for i in sorted(class_names)]


def resample_for_clap(audio, source_rate, target_rate=CLAP_SAMPLE_RATE):
    audio = validate_waveform(audio)
    if any(type(rate) is not int or rate <= 0 for rate in (source_rate, target_rate)):
        raise ValueError("Sample rates must be positive integers")
    if source_rate == target_rate:
        return audio
    divisor = int(np.gcd(int(source_rate), int(target_rate)))
    return resample_poly(audio, target_rate // divisor, source_rate // divisor)


def int16_round_trip(audio):
    """CLAP's reference pipeline quantises float audio to int16 and back; mirror it."""
    clipped = np.clip(validate_waveform(audio), -1.0, 1.0)
    # Official float32_to_int16 truncates toward zero, rather than rounding.
    return ((clipped * 32767.0).astype(np.int16) / 32767.0).astype(np.float32)


def unit_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(matrix).all():
        raise ValueError("Embeddings must be finite with positive norm")
    return matrix / norms


class FrozenLAIONBackend:
    """Official model with inference-only calls and exactly one quantisation step."""

    def __init__(self, module, device, checkpoint_sha256, audio_model, text_model):
        self.module = module
        self.device = str(device)
        self.checkpoint_sha256 = checkpoint_sha256
        self.audio_model = audio_model
        self.text_model = text_model

    def get_text_embedding(self, prompts, use_tensor=False):
        import torch

        with torch.inference_mode():
            return self.module.get_text_embedding(prompts, use_tensor=use_tensor)

    def get_audio_embedding_from_data(self, x, use_tensor=False):
        import torch

        # The public NumPy path quantises again. Input has already been quantised by
        # CLAPSystem.preprocess, so use the official tensor path to avoid that repeat.
        with torch.inference_mode():
            result = self.module.get_audio_embedding_from_data(
                x=torch.from_numpy(np.asarray(x, dtype=np.float32)), use_tensor=True
            )
        return result if use_tensor else result.detach().cpu().numpy()

    def synchronize(self):
        import torch

        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()


def load_laion_clap(
    checkpoint, device="cpu", audio_model=DEFAULT_AUDIO_MODEL, text_model=DEFAULT_TEXT_MODEL
):
    """Load the fixed, checksum-verified checkpoint; tokenizer assets may be cached online."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CLAP checkpoint not found: {checkpoint}")
    checksum = sha256_file(checkpoint)
    if checksum != DEFAULT_CHECKPOINT_SHA256:
        raise ValueError("CLAP checkpoint checksum does not match the fixed reference")
    try:
        import laion_clap
        import torch
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Frozen CLAP needs the 'laion_clap' package and PyTorch; install them in the "
            "separate CLAP environment; preserve the locked SVM/CNN environment."
        ) from error
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this host")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this host")
    torch.set_num_threads(1)
    module = laion_clap.CLAP_Module(
        enable_fusion=False, amodel=audio_model, tmodel=text_model, device=device
    )
    module.load_ckpt(ckpt=str(checkpoint), verbose=False)
    module.eval()
    module.requires_grad_(False)
    return FrozenLAIONBackend(module, device, checksum, audio_model, text_model)


class CLAPSystem:
    """Zero-shot classifier: cosine similarity to fixed class prompts."""

    name = "clap"
    score_type = "cosine_similarity"
    representation_type = "clap_audio_embedding"
    representation_names = None

    def __init__(
        self,
        module,
        class_names,
        feature_config=FeatureConfig(),
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        checkpoint=None,
    ):
        if not class_names or sorted(class_names) != list(range(len(class_names))):
            raise ValueError("Class names must cover contiguous IDs from zero")
        self.module = module
        self.class_ids = np.arange(len(class_names))
        self.class_names = {int(k): str(v) for k, v in class_names.items()}
        self.input_rate = feature_config.sample_rate
        self.prompt_template = prompt_template
        self.prompts = build_prompts(self.class_names, prompt_template)
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.device = getattr(module, "device", "cpu")
        text = np.asarray(module.get_text_embedding(self.prompts, use_tensor=False))
        if text.shape[0] != len(self.prompts) or text.ndim != 2:
            raise ValueError("Text embeddings must have one row per prompt")
        self.text_embeddings = unit_rows(text)

    def preprocess(self, audio):
        return int16_round_trip(resample_for_clap(audio, self.input_rate))

    def embed(self, prepared):
        return self.embed_batch(np.asarray(prepared)[None, :])[0]

    def embed_batch(self, prepared):
        embedding = np.asarray(
            self.module.get_audio_embedding_from_data(x=prepared, use_tensor=False)
        )
        if embedding.shape != (len(prepared), self.text_embeddings.shape[1]):
            raise ValueError("Audio embeddings must match batch size and text dimension")
        return unit_rows(embedding)

    def evaluate(self, fold, audio):
        """Return (cosine similarities to the 50 prompts, unit audio embedding)."""
        embedding = self.embed(self.preprocess(audio))
        return self.text_embeddings @ embedding, embedding

    def evaluate_batch(self, folds, audios):
        if len(folds) != len(audios) or not audios:
            raise ValueError("Expected one fold per audio clip in a nonempty batch")
        prepared = np.stack([self.preprocess(audio) for audio in audios])
        embeddings = self.embed_batch(prepared)
        return embeddings @ self.text_embeddings.T, embeddings

    def synchronize(self):
        if hasattr(self.module, "synchronize"):
            self.module.synchronize()

    def provenance(self):
        info = {
            "frozen": True,
            "fold_independent": True,
            "audio_model": getattr(self.module, "audio_model", DEFAULT_AUDIO_MODEL),
            "text_model": getattr(self.module, "text_model", DEFAULT_TEXT_MODEL),
            "device": self.device,
            "enable_fusion": False,
            "input_sample_rate": CLAP_SAMPLE_RATE,
            "quantisation": "one int16 round trip: clip, multiply 32767, truncate toward zero, divide 32767, float32; official tensor path avoids requantisation",
            "resampling": "scipy.signal.resample_poly in float64, then official quantisation",
            "short_audio_policy": "official repeatpad to 480000 samples (10 seconds); non-fusion",
            "prompt_template": self.prompt_template,
            "prompts": self.prompts,
            "embedding_dimension": int(self.text_embeddings.shape[1]),
            "pretraining_caveat": (
                "external audio-text pretraining; frozen inference does not establish that "
                "ESC-50 recordings were absent from that pretraining"
            ),
        }
        if self.checkpoint is not None:
            info["checkpoint"] = str(self.checkpoint)
            info["checkpoint_sha256"] = getattr(
                self.module, "checkpoint_sha256", None
            ) or sha256_file(self.checkpoint)
        try:
            info["laion_clap_version"] = importlib.metadata.version("laion_clap")
        except importlib.metadata.PackageNotFoundError:
            info["laion_clap_version"] = None
        return info


def clap_system_from_checkpoint(
    checkpoint,
    class_names,
    feature_config=FeatureConfig(),
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
    device="cpu",
):
    module = load_laion_clap(checkpoint, device)
    return CLAPSystem(module, class_names, feature_config, prompt_template, checkpoint)


def evaluate_clean(
    system,
    data_root,
    output_dir,
    feature_config=FeatureConfig(),
    require_complete=True,
    batch_size=1,
):
    """Zero-shot CLAP on every clean clip: the clean reference for the interference tables.

    Writes predictions.csv in the baseline format, decision_scores.npz (cosines, audio
    and text embeddings) and summary.json with five-fold means and sample SD.
    """
    data_root, output_dir = Path(data_root), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    clips = load_metadata(data_root, require_complete=require_complete)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if system.input_rate != feature_config.sample_rate:
        raise ValueError("CLAP input sample rate differs from the clean loader")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = system.class_ids
    scores = np.full((len(clips), len(labels)), np.nan)
    embeddings = []
    start = perf_counter()
    for start_index in range(0, len(clips), batch_size):
        batch = clips[start_index : start_index + batch_size]
        audio = [
            load_audio(data_root / "audio" / clip.filename, feature_config, require_complete)
            for clip in batch
        ]
        batch_scores, batch_embeddings = system.evaluate_batch([c.fold for c in batch], audio)
        completed = start_index + len(batch)
        scores[start_index:completed] = batch_scores
        embeddings.extend(batch_embeddings)
        if completed // 100 > start_index // 100 or completed == len(clips):
            print(f"CLAP clean: {completed}/{len(clips)} clips", flush=True)
            write_json(
                output_dir / "progress.json",
                {
                    "status": "running",
                    "completed": completed,
                    "total": len(clips),
                    "elapsed_seconds": perf_counter() - start,
                },
            )
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite CLAP scores")
    predictions = labels[np.argmax(scores, axis=1)]
    y = np.array([c.target for c in clips])
    folds = np.array([c.fold for c in clips])
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["filename", "fold", "target", "category", "prediction", "predicted_category"]
        )
        for clip, prediction in zip(clips, predictions):
            writer.writerow(
                [
                    clip.filename,
                    clip.fold,
                    clip.target,
                    clip.category,
                    int(prediction),
                    system.class_names[int(prediction)],
                ]
            )
    np.savez_compressed(
        output_dir / "decision_scores.npz",
        scores=scores,
        embeddings=np.asarray(embeddings),
        text_embeddings=system.text_embeddings,
        prompts=np.array(system.prompts),
        score_type=np.array(system.score_type),
        class_ids=labels,
        filenames=np.array([c.filename for c in clips]),
        folds=folds,
        targets=y,
        predictions=predictions,
    )
    per_fold = {}
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        per_fold[int(fold)] = metrics(y[mask], predictions[mask], labels)
    summary = {
        "evaluation": "clean_ESC-50_zero_shot" if require_complete else "diagnostic_subset",
        "clips": len(clips),
        "labels": labels.tolist(),
        "accuracy": {
            "mean": float(np.mean([v["accuracy"] for v in per_fold.values()])),
            "sample_std": float(np.std([v["accuracy"] for v in per_fold.values()], ddof=1)),
        },
        "macro_f1": {
            "mean": float(np.mean([v["macro_f1"] for v in per_fold.values()])),
            "sample_std": float(np.std([v["macro_f1"] for v in per_fold.values()], ddof=1)),
        },
        "folds": per_fold,
        "pooled": metrics(y, predictions, labels),
        "seconds": perf_counter() - start,
        "batch_size": batch_size,
        "metadata_sha256": sha256_file(data_root / "meta" / "esc50.csv"),
        "system": system.provenance(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("clap_system.py", "data.py")
        },
        "note": (
            "No training or tuning; the same checkpoint and prompts are used for every clip, "
            "condition and level. Scores are cosine similarities, not calibrated probabilities."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "progress.json",
        {
            "status": "complete",
            "completed": len(clips),
            "total": len(clips),
            "elapsed_seconds": summary["seconds"],
        },
    )
    return summary


def benchmark_clap(system, audio_path, feature_config=FeatureConfig(), runs=100, warmup=10):
    """Single-clip cost on the selected device, with one numerical CPU thread."""
    if runs < 1 or warmup < 1:
        raise ValueError("runs and warmup must be positive")
    audio = load_audio(audio_path, feature_config)
    preprocessing, inference = [], []
    with threadpool_limits(limits=1):
        for i in range(warmup + runs):
            system.synchronize()
            start = perf_counter()
            prepared = system.preprocess(audio)
            processed = perf_counter()
            embedding = system.embed(prepared)
            system.text_embeddings @ embedding
            system.synchronize()
            finished = perf_counter()
            if i >= warmup:
                preprocessing.append(processed - start)
                inference.append(finished - processed)
    result = {
        "scope": "single_clip_microbenchmark",
        "audio_file": str(audio_path),
        "audio_seconds": len(audio) / feature_config.sample_rate,
        "batch_size": 1,
        "runs": runs,
        "warmup": warmup,
        "numerical_threads": 1,
        "device": system.device,
        "preprocessing_median_ms": float(1000 * np.median(preprocessing)),
        "inference_median_ms": float(1000 * np.median(inference)),
        "text_embeddings_precomputed": True,
        "platform": platform.platform(),
        "excludes": ["disk I/O", "model loading", "text-prompt embedding", "external pretraining"],
    }
    if system.checkpoint is not None:
        result["checkpoint_bytes"] = system.checkpoint.stat().st_size
    official = getattr(system.module, "module", None)
    if official is not None and hasattr(official, "parameters"):
        result["parameter_count"] = sum(p.numel() for p in official.parameters())
        result["parameter_bytes"] = sum(p.numel() * p.element_size() for p in official.parameters())
        result["parameter_scope"] = "complete audio and text model, excluding buffers"
    return result

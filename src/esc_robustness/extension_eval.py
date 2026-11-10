"""Reuse verified mixture frontends across supervised augmentation/seed ablations."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .artifacts import sha256_file, write_json
from .dsp import log_mel
from .interference_eval import PREDICTION_FIELDS, CNNSystem, SVMSystem, _classify, _summarise
from .mixing import manifest_digest, renderer_from_manifest
from .svm import class_scores


def prepare_frontends(manifest, output, data_root=None, limit=None):
    """Cache identical float32 log-Mels for the mixture and gain-matched control."""
    output = Path(output)
    renderer, meta, rows = renderer_from_manifest(manifest, data_root)
    if limit is not None:
        rows = rows[:limit]
    fingerprint = {
        "manifest_sha256": manifest_digest(manifest),
        "rows": len(rows),
        "subset_limit": limit,
        "feature_config": renderer.feature_config.as_dict(),
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("mixing.py", "dsp.py", "data.py")
        },
    }
    if (output / "cache_meta.json").exists():
        cached = json.loads((output / "cache_meta.json").read_text())
        if cached["fingerprint"] != fingerprint:
            raise ValueError("Cache input or frontend version mismatch")
        if sha256_file(output / "log_mel.npy") != cached["log_mel_sha256"]:
            raise ValueError("Frontend cache checksum mismatch")
        return cached
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Incomplete cache exists; inspect before rebuilding")
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with threadpool_limits(limits=1):
        first = renderer.render(rows[0])
        shape = log_mel(first["mixture"], renderer.feature_config).T.shape
        bank = np.lib.format.open_memmap(
            output / "log_mel.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), 2, *shape),
        )
        for i, row in enumerate(rows):
            rendered = first if i == 0 else renderer.render(row)
            for j, key in enumerate(("mixture", "clean_matched")):
                bank[i, j] = log_mel(rendered[key], renderer.feature_config).T.astype(np.float32)
            if (i + 1) % 500 == 0 or i + 1 == len(rows):
                bank.flush()
                print(f"Cached {i + 1}/{len(rows)} mixture/control frontends", flush=True)
                write_json(output / "progress.json", {"completed": i + 1, "total": len(rows)})
        bank.flush()
        del bank
    result = {
        "fingerprint": fingerprint,
        "log_mel_sha256": sha256_file(output / "log_mel.npy"),
        "shape": [len(rows), 2, *shape],
        "order": ["mixture", "gain_matched_clean"],
        "seconds": perf_counter() - start,
    }
    write_json(output / "cache_meta.json", result)
    return result


def evaluate_cached(
    manifest,
    output,
    model_dir,
    system_name,
    *,
    data_root=None,
    frontend_cache=None,
    feature_evaluation=None,
    device="cpu",
    batch_size=64,
):
    """Evaluate fold-matched models; batching never changes which fold model is used."""
    output, model_dir = Path(output), Path(model_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists: {output}")
    renderer, manifest_meta, rows = renderer_from_manifest(manifest, data_root)
    input_provenance = {}
    if system_name == "cnn":
        import torch

        if frontend_cache is None:
            raise ValueError("CNN evaluation requires a verified frontend cache")
        frontend_cache = Path(frontend_cache)
        cache_meta = json.loads((frontend_cache / "cache_meta.json").read_text())
        fingerprint = cache_meta["fingerprint"]
        if fingerprint["manifest_sha256"] != manifest_digest(manifest):
            raise ValueError("Manifest differs from the frontend cache")
        if fingerprint["feature_config"] != renderer.feature_config.as_dict():
            raise ValueError("Frontend feature configuration differs")
        if fingerprint["subset_limit"] is not None:
            rows = rows[: fingerprint["subset_limit"]]
        if fingerprint["rows"] != len(rows):
            raise ValueError("Frontend row count differs")
        if sha256_file(frontend_cache / "log_mel.npy") != cache_meta["log_mel_sha256"]:
            raise ValueError("Frontend checksum differs")
        bank = np.load(frontend_cache / "log_mel.npy", mmap_mode="r")
        system = CNNSystem(model_dir, renderer.feature_config)
        for model in system.models.values():
            model.to(device).eval()
        input_provenance["frontend_cache"] = cache_meta
    elif system_name == "svm":
        if feature_evaluation is None:
            raise ValueError("SVM evaluation requires verified saved handcrafted vectors")
        feature_evaluation = Path(feature_evaluation)
        source_meta = json.loads((feature_evaluation / "summary.json").read_text())
        if source_meta["manifest_sha256"] != manifest_digest(manifest):
            raise ValueError("Saved feature manifest differs")
        if source_meta["rows_evaluated"] != len(rows):
            raise ValueError("Saved feature rows differ")
        source_path = feature_evaluation / "svm_scores.npz"
        with np.load(source_path, allow_pickle=False) as f:
            features = f["representation"]
            controls = f["control_representation"]
            row_ids = f["row"]
        if not np.array_equal(row_ids, [r["row"] for r in rows]):
            raise ValueError("Saved feature row order differs")
        system = SVMSystem(model_dir, renderer.feature_config)
        input_provenance["features_sha256"] = sha256_file(source_path)
    else:
        raise ValueError("Cached evaluation supports CNN or SVM")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    n, k = len(rows), len(system.class_ids)
    data = {
        "scores": np.full((n, k), np.nan),
        "control_scores": np.full((n, k), np.nan),
        "representation": [],
        "control_representation": [],
        "rows": [],
    }
    start_time = perf_counter()
    with threadpool_limits(limits=1):
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            folds = np.array([r["fold"] for r in rows[start:stop]])
            scores = np.empty((stop - start, 2, k))
            vectors = None
            for fold in np.unique(folds):
                local = np.flatnonzero(folds == fold)
                global_rows = local + start
                if system_name == "cnn":
                    x = np.array(bank[global_rows], dtype=np.float32).reshape(
                        -1, 1, *bank.shape[2:]
                    )
                    with torch.inference_mode():
                        embedding = system.models[int(fold)].embedding(
                            torch.from_numpy(x).to(device)
                        )
                        logits = system.models[int(fold)].classifier(embedding)
                    ss = logits.cpu().numpy().astype(np.float64)
                    vv = embedding.cpu().numpy().astype(np.float64)
                else:
                    vv = np.stack((features[global_rows], controls[global_rows]), axis=1).reshape(
                        -1, features.shape[1]
                    )
                    ss = class_scores(system.models[int(fold)]["model"], vv)
                if vectors is None:
                    vectors = np.empty((stop - start, 2, vv.shape[1]))
                scores[local] = ss.reshape(len(local), 2, k)
                vectors[local] = vv.reshape(len(local), 2, -1)
            for j, row in enumerate(rows[start:stop]):
                record = _classify(
                    system,
                    row,
                    None,
                    data,
                    start + j,
                    (scores[j, 0], vectors[j, 0], scores[j, 1], vectors[j, 1]),
                )
                data["rows"].append(record)
            if stop // 500 > start // 500 or stop == n:
                print(f"{system_name}: evaluated {stop}/{n}", flush=True)
                write_json(
                    output / "progress.json",
                    {
                        "status": "running",
                        "completed": stop,
                        "total": n,
                        "seconds": perf_counter() - start_time,
                    },
                )
    with (output / f"{system_name}_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(data["rows"])
    arrays = {
        key: np.asarray(data[key])
        for key in ("scores", "control_scores", "representation", "control_representation")
    }
    arrays.update(
        class_ids=system.class_ids,
        row=np.array([r["row"] for r in rows]),
        score_type=np.array(system.score_type),
        representation_type=np.array(system.representation_type),
    )
    if system.representation_names:
        arrays["representation_names"] = np.array(system.representation_names)
    np.savez_compressed(output / f"{system_name}_scores.npz", **arrays)
    suffix = "pt" if system_name == "cnn" else "joblib"
    detail = _summarise(data["rows"], system)
    detail.update(
        model_dir=str(model_dir),
        checkpoint_sha256={
            str(f): sha256_file(model_dir / f"{system_name}_fold_{f}.{suffix}") for f in range(1, 6)
        },
    )
    result = {
        "manifest": str(manifest),
        "manifest_sha256": manifest_digest(manifest),
        "rows_evaluated": n,
        "feature_config": renderer.feature_config.as_dict(),
        "interference_config": manifest_meta["interference_config"],
        "power_definition": manifest_meta["power_definition"],
        "systems": {system_name: detail},
        "input_provenance": input_provenance,
        "batch_size_mixtures": batch_size,
        "device": device,
        "seconds": perf_counter() - start_time,
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("extension_eval.py", "interference_eval.py", "cnn.py", "svm.py")
        },
    }
    write_json(output / "summary.json", result)
    write_json(output / "progress.json", {"status": "complete", "completed": n, "total": n})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--frontend-cache", type=Path)
    parser.add_argument("--feature-evaluation", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--system", choices=("cnn", "svm"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_frontends(args.manifest, args.output, args.data_root, args.limit)
    else:
        result = evaluate_cached(
            args.manifest,
            args.output,
            args.model_dir,
            args.system,
            data_root=args.data_root,
            frontend_cache=args.frontend_cache,
            feature_evaluation=args.feature_evaluation,
            device=args.device,
            batch_size=args.batch_size,
        )
    print(json.dumps(result, indent=2))

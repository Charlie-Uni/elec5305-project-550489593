"""Controlled LAION-CLAP prompt-ensemble ablation, using the released PAT list.

This is NOT full PAT: no local-feature alignment, beta fusion or label-based search.
All templates are fixed before evaluating their effect and used for every class.
"""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from .analysis import fold_metrics, read_predictions
from .artifacts import sha256_file, write_json
from .clap_system import load_laion_clap, prompt_label, unit_rows
from .data import load_metadata
from .interference_eval import PREDICTION_FIELDS, _classify, _summarise


def ensemble_embeddings(backend, class_names, templates, batch_size=32):
    if not templates or any("<sound>" not in t for t in templates):
        raise ValueError("Every released template must contain <sound>")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    means, all_embeddings = [], []
    for class_id in sorted(class_names):
        prompts = [t.replace("<sound>", prompt_label(class_names[class_id])) for t in templates]
        chunks = [
            backend.get_text_embedding(prompts[i : i + batch_size], use_tensor=False)
            for i in range(0, len(prompts), batch_size)
        ]
        embedded = unit_rows(np.concatenate(chunks))
        means.append(embedded.mean(0))
        all_embeddings.append(embedded)
        print(f"Prompt ensemble: class {class_id + 1}/{len(class_names)}", flush=True)
    return unit_rows(np.array(means)), np.array(all_embeddings)


def rescore_clean(source, output, texts, class_names, provenance):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(source / "decision_scores.npz") as f:
        arrays = dict(f)
    scores = arrays["embeddings"] @ texts.T
    predictions = arrays["class_ids"][scores.argmax(1)]
    arrays.update(
        scores=scores,
        predictions=predictions,
        text_embeddings=texts,
        prompts=np.array(
            ["Fixed PAT template ensemble: " + class_names[i] for i in sorted(class_names)]
        ),
    )
    np.savez_compressed(output / "decision_scores.npz", **arrays)
    with (output / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "fold", "target", "category", "prediction", "predicted_category"])
        for name, fold, y, pred in zip(
            arrays["filenames"], arrays["folds"], arrays["targets"], predictions
        ):
            w.writerow([name, fold, y, class_names[int(y)], pred, class_names[int(pred)]])
    report = fold_metrics(arrays["targets"], predictions, arrays["folds"], arrays["class_ids"])
    report.update(
        evaluation="fixed_prompt_ensemble_zero_shot",
        clips=len(predictions),
        system=provenance,
        input_sha256=sha256_file(source / "decision_scores.npz"),
    )
    write_json(output / "summary.json", report)
    return report


def rescore_interference(source, output, texts, provenance):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = read_predictions(source / "clap_predictions.csv")
    with np.load(source / "clap_scores.npz") as f:
        arrays = dict(f)
    if not np.array_equal(np.array([int(r["row"]) for r in rows]), arrays["row"]):
        raise ValueError("Prediction and embedding row order differs")
    mixed, controls = arrays["representation"] @ texts.T, arrays["control_representation"] @ texts.T
    system = SimpleNamespace(class_ids=arrays["class_ids"], score_type="cosine_similarity")
    data = {
        "scores": np.empty_like(mixed),
        "control_scores": np.empty_like(controls),
        "representation": [],
        "control_representation": [],
    }
    records = [
        _classify(
            system,
            row,
            None,
            data,
            i,
            (
                mixed[i],
                arrays["representation"][i],
                controls[i],
                arrays["control_representation"][i],
            ),
        )
        for i, row in enumerate(rows)
    ]
    with (output / "clap_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        w.writeheader()
        w.writerows(records)
    arrays.update(
        scores=mixed,
        control_scores=controls,
        text_embeddings=texts,
        prompts=np.array([f"Fixed PAT ensemble class {i}" for i in arrays["class_ids"]]),
    )
    np.savez_compressed(output / "clap_scores.npz", **arrays)
    summary = json.loads((source / "summary.json").read_text())
    summary["systems"] = {"clap": {**_summarise(records, system), "provenance": provenance}}
    summary["rescore_input_sha256"] = sha256_file(source / "clap_scores.npz")
    summary["evaluation"] = "cached_audio_embeddings_fixed_prompt_ensemble"
    summary.pop(
        "seconds", None
    )  # Do not misreport the original inference time as this ablation's runtime.
    write_json(output / "summary.json", summary)
    return summary


def run(checkpoint, templates_path, data_root, clean, own, target, output, batch_size=32):
    output, templates_path = Path(output), Path(templates_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new ensemble run directory")
    output.mkdir(parents=True, exist_ok=True)
    templates = [line.strip() for line in templates_path.read_text().splitlines() if line.strip()]
    class_names = {c.target: c.category for c in load_metadata(data_root)}
    backend = load_laion_clap(checkpoint, device="cpu")
    with threadpool_limits(limits=1):
        # Verify this model/tokenizer reproduces the cached baseline prompts first.
        with np.load(Path(clean) / "decision_scores.npz") as f:
            prompts, expected = f["prompts"].tolist(), f["text_embeddings"]
        observed = unit_rows(backend.get_text_embedding(prompts, use_tensor=False))
        if not np.allclose(observed, expected, atol=1e-6, rtol=1e-5):
            raise ValueError("Baseline text embeddings differ; investigate model/tokenizer version")
        texts, per_prompt = ensemble_embeddings(backend, class_names, templates, batch_size)
        np.savez_compressed(
            output / "text_embeddings.npz",
            ensemble=texts,
            per_prompt=per_prompt,
            templates=np.array(templates),
            class_names=np.array([class_names[i] for i in sorted(class_names)]),
        )
        provenance = {
            "name": "LAION-CLAP + fixed released PAT prompt ensemble",
            "not_full_PAT": "No local-feature mutual alignment, beta fusion, or label-based parameter search",
            "checkpoint_sha256": backend.checkpoint_sha256,
            "templates_sha256": sha256_file(templates_path),
            "templates_per_class": len(templates),
            "unique_templates": len(set(templates)),
            "source": "https://github.com/cs20s030/PAT/blob/8910ebb756e33aec72e2cf6c1f02614bfa194e85/promt_list.txt",
            "aggregation": "Unit-normalise each text embedding, mean across all prompts, unit-normalise mean",
            "selection": "All released nonblank templates; no selection using clean or interference labels",
            "baseline_text_max_abs_difference": float(np.max(np.abs(observed - expected))),
            "source_sha256": sha256_file(Path(__file__)),
            "text_embeddings_sha256": sha256_file(output / "text_embeddings.npz"),
        }
        write_json(output / "provenance.json", provenance)
        rescore_clean(clean, output / "clean", texts, class_names, provenance)
        for region, source in (("own", own), ("target", target)):
            rescore_interference(source, output / f"eval_{region}", texts, provenance)
    write_json(output / "complete.json", {"complete": True, "provenance": provenance})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "templates", "data-root", "clean", "own", "target", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    run(a.checkpoint, a.templates, a.data_root, a.clean, a.own, a.target, a.output, a.batch_size)

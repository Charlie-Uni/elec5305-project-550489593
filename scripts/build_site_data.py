"""Generate the project site's numbers from saved research outputs."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from delivery_claims import closing_claims, editorial_claims

ROOT = Path(__file__).resolve().parents[1]


def build(run):
    clean = {}
    for system, directory in (
        ("svm", "clean_svm"),
        ("cnn", "clean_cnn"),
        ("clap", "clap_integration_20261109/clean"),
    ):
        source = json.loads((ROOT / "results" / directory / "summary.json").read_text())
        clean[system] = {name: source[name] for name in ("accuracy", "macro_f1")}
    with (ROOT / "results/clap_full_20261109/all_conditions_verified.csv").open() as f:
        conditions = list(csv.DictReader(f))
    with (ROOT / "results/clean_svm/predictions.csv").open() as f:
        labels = {int(r["target"]): r["category"] for r in csv.DictReader(f)}
    queue_path = run / "evaluation_queue.json"
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else {}
    clean_seeds = sum((run / f"clean_seed_{seed}/summary.json").exists() for seed in (5306, 5307))
    augmented = sum(
        (run / f"cnn_{mode}_seed_{seed}/summary.json").exists()
        for mode in ("wgn", "semantic")
        for seed in (5305, 5306, 5307)
    )
    svm = sum((run / f"svm_{mode}/summary.json").exists() for mode in ("wgn", "semantic"))
    ensemble = (run / "clap_prompt_ensemble/complete.json").exists()
    state = {
        "closing": closing_claims(),
        "editorial": editorial_claims(),
        "updated": datetime.now(timezone.utc).isoformat(),
        "clean": clean,
        "conditions": conditions,
        "labels": labels,
        "extensions": [
            {"name": "Additional clean CNN seeds", "completed": clean_seeds, "total": 2},
            {"name": "Augmented CNN training runs", "completed": augmented, "total": 6},
            {"name": "Augmented SVM training runs", "completed": svm, "total": 2},
            {
                "name": "Verified supervised evaluations",
                "completed": len(queue.get("completed", {})),
                "total": 22,
            },
            {"name": "Fixed CLAP prompt ensemble", "completed": int(ensemble), "total": 1},
        ],
    }
    audit_path = ROOT / "results/extensions_20261109/prompt_ensemble_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if not audit["verified"]:
            raise ValueError("Prompt ensemble must pass its independent audit")
        state["prompt_ensemble"] = {
            "clean_accuracy": audit["clean_accuracy"],
            "clean_macro_f1": audit["clean_macro_f1"],
            "templates": audit["templates_per_class"],
            "unique_templates": audit["unique_templates"],
            "conditions": {
                region: json.loads(
                    (
                        ROOT
                        / f"results/extensions_20261109/clap_prompt_ensemble/{region}_conditions.json"
                    ).read_text()
                )
                for region in ("own", "target")
            },
        }
    replication = ROOT / "results/extensions_20261109/clean_replication/cnn_overlap_replication.csv"
    if replication.exists():
        with replication.open() as handle:
            state["cnn_replication"] = list(csv.DictReader(handle))
    interventions = ROOT / "results/extensions_20261109/interventions"
    if (interventions / "independent_audit.json").exists():
        if not json.loads((interventions / "independent_audit.json").read_text())["verified"]:
            raise ValueError("Intervention results require an independent audit")
        state["augmentation"] = {}
        for key, filename in (
            ("clean", "clean_variants.csv"),
            ("conditions", "condition_variants.csv"),
        ):
            with (interventions / filename).open() as handle:
                state["augmentation"][key] = [
                    r for r in csv.DictReader(handle) if r["system"] in ("svm", "cnn")
                ]
    costs = ROOT / "results/extensions_20261109/cost_panel"
    if (costs / "audit.json").exists():
        if not json.loads((costs / "audit.json").read_text())["verified"]:
            raise ValueError("CPU panel requires an independent audit")
        with (costs / "latency_summary.csv").open() as handle:
            state["balanced_costs"] = list(csv.DictReader(handle))
        if not json.loads((costs.with_name("cost_repeatability") / "audit.json").read_text())[
            "verified"
        ]:
            raise ValueError("Both timing passes must be checked")
        with (costs.with_name("cost_panel_reverse") / "latency_summary.csv").open() as handle:
            repeated = {r["variant"]: r for r in csv.DictReader(handle)}
        for row in state["balanced_costs"]:
            row.update(
                {
                    f"reverse_{k}": v
                    for k, v in repeated[row["variant"]].items()
                    if k.startswith("total_")
                }
            )
    (ROOT / "site-data.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    build(p.parse_args().run)

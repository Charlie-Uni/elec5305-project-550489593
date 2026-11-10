"""Collect verified result values for the editable research presentation."""

import csv
import json
from pathlib import Path

from delivery_claims import closing_claims

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "results/presentation_20261109_draft"


def rows(relative):
    with (ROOT / relative).open() as handle:
        return list(csv.DictReader(handle))


def main():
    extension = ROOT / "results/extensions_20261109"
    data = {
        "closing": closing_claims(),
        "core": rows("results/clap_full_20261109/all_conditions_verified.csv"),
        "replication": rows(
            "results/extensions_20261109/clean_replication/cnn_overlap_replication.csv"
        ),
        "captures": rows(
            "results/extensions_20261109/capture_transitions_core/capture_transitions.csv"
        ),
        "feature_groups": rows(
            "results/extensions_20261109/diagnostics/feature_group_displacement.csv"
        ),
        "ensemble": json.loads((extension / "prompt_ensemble_audit.json").read_text()),
        "labels": {r["target"]: r["category"] for r in rows("results/clean_svm/predictions.csv")},
    }
    data["core_cost_stage_median_sum"] = []
    for relative in ("clean_svm", "clean_cnn", "clap_integration_20261109/clean"):
        cost = json.loads((ROOT / "results" / relative / "cost.json").read_text())
        data["core_cost_stage_median_sum"].append(
            cost["preprocessing_median_ms"] + cost["inference_median_ms"]
        )
    examples = json.loads((ROOT / "examples/listening_own_20261109/examples.json").read_text())
    data["listening"] = next(
        r
        for r in examples["files"]
        if r["target_category"] == "can_opening"
        and r["condition"] == "semantic_same"
        and r["level_db"] == 5
    )
    data["augmentation_complete"] = (extension / "interventions/independent_audit.json").exists()
    if data["augmentation_complete"]:
        audit = json.loads((extension / "interventions/independent_audit.json").read_text())
        if not audit["verified"]:
            raise ValueError("Presentation requires independently verified interventions")
        data["interventions"] = rows(
            "results/extensions_20261109/interventions/condition_variants.csv"
        )
        data["clean_variants"] = rows(
            "results/extensions_20261109/interventions/clean_variants.csv"
        )
    data["cost_panel_complete"] = (extension / "cost_panel/audit.json").exists()
    if data["cost_panel_complete"]:
        if not json.loads((extension / "cost_repeatability/audit.json").read_text())["verified"]:
            raise ValueError("Both timing passes must be checked")
        data["cost_panel"] = rows("results/extensions_20261109/cost_panel/latency_summary.csv")
        data["cost_panel_reverse"] = rows(
            "results/extensions_20261109/cost_panel_reverse/latency_summary.csv"
        )
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "presentation_data.json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

"""Generate the project site's numbers from saved research outputs."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

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
    (ROOT / "site-data.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    build(p.parse_args().run)

"""Re-render core plots without baked-in figure numbers for report captions."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

from esc_robustness import figures

ROOT = Path(__file__).resolve().parents[1]


def render(clap_run, svm_models):
    destination = ROOT / "results/report_figure_assets_20261109"

    def save(fig, output, name):
        for text in fig.texts:
            text.set_text(re.sub(r"^Figure \d+\.\s*", "", text.get_text()))
        for axis in fig.axes:
            for location in ("left", "center", "right"):
                axis.set_title(
                    re.sub(r"^Figure \d+\.\s*", "", axis.get_title(loc=location)), loc=location
                )
        fig.savefig(output / f"{name}.png", dpi=160)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)

    figures._save = save
    clean = {s: ROOT / f"results/clean_{s}" for s in ("svm", "cnn")}
    clean["clap"] = clap_run / "clean"
    figures.make_figures(
        ROOT / "artifacts/interference/v3_20261109/eval_own",
        clean,
        destination,
        {s: d / "cost.json" for s, d in clean.items()},
        level=5,
        additional_evaluations=[clap_run / "eval_own"],
        svm_models=svm_models,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clap-run", type=Path, required=True)
    parser.add_argument("--svm-models", type=Path, required=True)
    args = parser.parse_args()
    render(args.clap_run, args.svm_models)

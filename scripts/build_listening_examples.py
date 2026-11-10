"""Export final-manifest examples and matched controls with source attribution."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from esc_robustness.analysis import read_predictions
from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.mixing import manifest_digest, renderer_from_manifest


def build(manifest, output, targets, evaluations, esc_license, demand_record):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new listening-example directory")
    renderer, meta, rows = renderer_from_manifest(manifest)
    selected = [r for r in rows if r["target_file"] in set(targets)]
    if len(selected) != len(set(targets)) * 12:
        raise ValueError("Every selected target must have all twelve final conditions")
    predictions = {
        system: {
            int(r["row"]): r for r in read_predictions(directory / f"{system}_predictions.csv")
        }
        for system, directory in evaluations.items()
    }
    for directory in evaluations.values():
        source = json.loads((directory / "summary.json").read_text())
        if source["manifest_sha256"] != manifest_digest(manifest):
            raise ValueError("Listening manifest and evaluation differ")
    output.mkdir(parents=True)
    files, originals = [], set()
    for row in selected:
        signals = renderer.render(row)
        item = dict(row)
        item["audio"] = {}
        originals.add(row["target_file"])
        if row["interferer_class"] is not None:
            originals.add(row["interferer"])
        for key in ("mixture", "clean_matched"):
            path = output / f"row_{row['row']:06d}_{key}.wav"
            sf.write(path, signals[key], renderer.feature_config.sample_rate, subtype="PCM_16")
            decoded, sample_rate = sf.read(path)
            difference = float(np.max(np.abs(decoded - signals[key])))
            if sample_rate != 44100 or len(decoded) != 220500 or difference > 1 / 32768 + 1e-12:
                raise ValueError("Listening PCM export differs beyond expected quantisation")
            item["audio"][key] = dict(
                file=path.name, sha256=sha256_file(path), max_quantisation_error=difference
            )
        item["predictions"] = {}
        for system, by_row in predictions.items():
            r = by_row[row["row"]]
            item["predictions"][system] = {
                k: r[k]
                for k in (
                    "prediction",
                    "control_prediction",
                    "correct",
                    "target_score",
                    "interferer_score",
                    "margin",
                )
            }
        files.append(item)
    license_text = esc_license.read_text()
    attribution = []
    for filename in sorted(originals):
        legacy = "-".join(Path(filename).stem.split("-")[:-1]) + ".ogg"
        matches = [line for line in license_text.splitlines() if f"[{legacy}]" in line]
        if len(matches) != 1:
            raise ValueError(f"Missing source attribution for {filename}")
        attribution.append(f"- `{filename}`: " + re.sub(r"^- \[[^]]+\]:\s*", "", matches[0]))
    demand = json.loads(demand_record.read_text())["data"]["attributes"]
    rights = demand.get("rightsList", [])
    if not any(
        r.get("rightsUri", "").rstrip("/") == "https://creativecommons.org/licenses/by/4.0"
        for r in rights
    ):
        raise ValueError("DEMAND source licence record needs review")
    description = """# Final-manifest listening examples

These examples illustrate selected targets; they are not an unbiased sample for
estimating performance. Every target has the four conditions at +10, +5 and 0 dB.
Compare each mixture with its own gain-matched clean control at the same playback
volume. Equal-power semantic mixtures at 0 dB have an ambiguous single target label.

WAV exports are 44.1-kHz mono PCM16 for playback. Evaluation used the deterministic
floating-point waveforms; export quantisation error is checked and recorded.
`examples.json` identifies the final manifest, all gains and the verified core
SVM/CNN/CLAP predictions. This replaces the earlier, pre-v3 noise examples.

## Attribution and licences

ESC-50 is compiled by Karol J. Piczak from the Freesound recordings below.
Dataset source: https://github.com/karolpiczak/ESC-50
Dataset licence: Creative Commons Attribution-NonCommercial 3.0
(https://creativecommons.org/licenses/by-nc/3.0/).
The original clip-specific attributions and licences are retained below.
Changes in these examples: cropping already supplied by ESC-50, controlled mixing,
common attenuation and PCM16 export. No source creator endorses this experiment.

DEMAND: Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent,
https://doi.org/10.5281/zenodo.1227121. The publisher's DataCite record retrieved
on 9 November 2026 specifies Creative Commons Attribution 4.0
(https://creativecommons.org/licenses/by/4.0/); its rights metadata is included.
Changes: channel 1 selection, resampling from 48 to 44.1 kHz, segment selection,
controlled mixing and attenuation. ESC-derived mixture examples are distributed
for non-commercial use under the ESC-50 dataset's CC BY-NC 3.0 terms.

## ESC-50 source clips

"""
    (output / "README.md").write_text(description + "\n".join(attribution) + "\n")
    write_json(
        output / "source_rights.json",
        {
            "esc50_attribution_source_sha256": sha256_file(esc_license),
            "esc50_attribution_url": "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/LICENSE",
            "demand_metadata_source": "https://api.datacite.org/dois/10.5281/zenodo.1227121",
            "demand_rights": rights,
            "demand_creators": demand["creators"],
        },
    )
    write_json(
        output / "examples.json",
        {
            "manifest_sha256": manifest_digest(manifest),
            "power_definition": meta["power_definition"],
            "rows": len(files),
            "files": files,
        },
    )
    print(
        json.dumps(
            {"mixtures": len(files), "matched_controls": len(files), "directory": str(output)}
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "output",
        "svm-evaluation",
        "cnn-evaluation",
        "clap-evaluation",
        "esc-license",
        "demand-record",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--target", action="append", required=True)
    a = p.parse_args()
    build(
        a.manifest,
        a.output,
        a.target,
        {"svm": a.svm_evaluation, "cnn": a.cnn_evaluation, "clap": a.clap_evaluation},
        a.esc_license,
        a.demand_record,
    )

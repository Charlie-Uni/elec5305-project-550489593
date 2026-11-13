"""Verify the single-report diagnostic figures against original committed tables."""

import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/final_submission_20261110"
SOURCE = ROOT / "results/figures_three_systems_own_20261109"


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def audit():
    capture = rows(SOURCE / "fig2_capture_rate_vs_sir.csv")
    margin = rows(SOURCE / "fig3_margin_vs_sir.csv")
    exported = rows(OUT / "figures/04_capture_margin.csv")
    assert len(capture) == len(margin) == 18 and len(exported) == 36
    for panel, original, value in (
        ("capture", capture, "capture_rate"),
        ("margin", margin, "median"),
    ):
        for row in original:
            item = next(
                r
                for r in exported
                if r["panel"] == panel
                and all(r[k] == row[k] for k in ("system", "condition", "level_db"))
            )
            assert float(item["value"]) == float(row[value])
            if panel == "capture":
                assert int(row["n"]) == 2000
                assert math.isclose(
                    float(item["value"]), int(row["errors_equal_distractor"]) / int(row["errors"])
                )
            else:
                assert float(item["low"]) == float(row["q1"])
                assert float(item["high"]) == float(row["q3"])
                assert float(item["low"]) <= float(item["value"]) <= float(item["high"])
    original = rows(SOURCE / "fig5_representation_displacement.csv")
    exported = rows(OUT / "figures/05_displacement.csv")
    assert len(original) == len(exported) == 24
    for row in original:
        item = next(
            r for r in exported if all(r[k] == row[k] for k in ("system", "condition", "outcome"))
        )
        for key in ("n", "median", "q1", "q3"):
            assert float(item[key]) == float(row[key])
        assert float(item["q1"]) <= float(item["median"]) <= float(item["q3"])
    for system in ("svm", "cnn", "clap"):
        for condition in ("white", "demand", "semantic_same", "semantic_different"):
            groups = {
                r["outcome"]: r
                for r in original
                if r["system"] == system and r["condition"] == condition
            }
            assert sum(int(r["n"]) for r in groups.values()) == 2000
            assert float(groups["incorrect"]["median"]) > float(groups["correct"]["median"])
    flow = rows(SOURCE / "fig6_confusion_flow.csv")
    for target, distractor, numerator, denominator in ((30, 32, 10, 11), (34, 37, 6, 6)):
        row = next(
            r
            for r in flow
            if r["system"] == "clap"
            and int(r["target_class"]) == target
            and int(r["distractor_class"]) == distractor
        )
        assert int(row["new_captures_from_correct_control"]) == numerator
        assert int(row["control_correct_pairs"]) == denominator
    core = rows(ROOT / "results/clap_full_20261109/all_conditions_verified.csv")
    sensitivity = []
    for system in ("svm", "cnn", "clap"):
        for condition in ("semantic_same", "semantic_different"):
            pair = {
                r["region"]: r
                for r in core
                if r["system"] == system and r["condition"] == condition + "@+5dB"
            }
            sensitivity.append(
                100 * (float(pair["own"]["macro_f1_mean"]) - float(pair["target"]["macro_f1_mean"]))
            )
    assert f"{min(sensitivity):.2f}" == "0.47" and f"{max(sensitivity):.2f}" == "2.35"
    tex = (ROOT / "docs/final_report.tex").read_text()
    assert not re.search(r"supplement(?:ary|al) material|supplement\.pdf|26[- ]page", tex, re.I)
    for value in ("56.25", "71.45", "0.47--2.35", "GPT-6.1 Sol", "Fable 5.1", "https://openai.com/codex/", "https://claude.ai/"):
        assert value in tex, value
    references = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    cited = {key for group in re.findall(r"\\cite\{([^}]+)\}", tex)
             for key in group.split(",")}
    assert len(references) == len(set(references)) == 17
    first_occurrence = []
    for group in re.findall(r"\\cite\{([^}]+)\}", tex.split(r"\begin{thebibliography}")[0]):
        for key in group.split(","):
            if key not in first_occurrence:
                first_occurrence.append(key)
    assert references == first_occurrence, "Bibliography must follow first citation order"

    assert cited == set(references), "Every citation must resolve and every reference must be used"
    assert {"sed", "mixup", "msclap", "aip"} <= cited
    review = tex.split(r"\section{Literature review}")[1].split(r"\section{Methods}")[0]
    assert review.count(r"\subsection{") == 2
    result = {
        "verified": True,
        "references": len(references),
        "literature_subsections": 2,
        "capture_points": 18,
        "margin_median_iqr_records": 18,
        "displacement_median_iqr_records": 24,
        "larger_error_group_medians": 12,
        "illustrative_capture_pairs_checked": 2,
        "own_minus_target_semantic_f1_pp": sensitivity,
        "reference_line": "Uniform allocation over 49 wrong labels; not an empirical randomisation test",
        "interpretation": "Descriptive medians and IQR; no causal semantic claim or cross-architecture distance comparison",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                SOURCE / "fig2_capture_rate_vs_sir.csv",
                SOURCE / "fig3_margin_vs_sir.csv",
                SOURCE / "fig5_representation_displacement.csv",
                SOURCE / "fig6_confusion_flow.csv",
            ]
        },
    }
    (OUT / "single_report_claim_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))

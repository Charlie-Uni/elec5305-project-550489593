"""Check the report, slide charts, narration and website against saved evidence."""

import argparse
import csv
import hashlib
import io
import json
import math
import posixpath
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from delivery_claims import ROOT, closing_claims

NS = {
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    with (ROOT / relative).open() as f:
        return list(csv.DictReader(f))


def close(values, expected):
    assert len(values) == len(expected)
    assert all(math.isclose(float(a), float(b), abs_tol=1e-10) for a, b in zip(values, expected))


def audit(browser):
    destination = ROOT / "results/delivery_sync_20261110"
    destination.mkdir(exist_ok=True)
    deck = ROOT / "results/presentation_20261109_draft"
    data = json.loads((deck / "presentation_data.json").read_text())
    site = json.loads((ROOT / "site-data.json").read_text())
    closing = closing_claims()
    assert data["closing"] == site["closing"] == closing
    core = read("results/clap_full_20261109/all_conditions_verified.csv")
    assert core == data["core"] == site["conditions"]
    replication = read("results/extensions_20261109/clean_replication/cnn_overlap_replication.csv")
    captures = read("results/extensions_20261109/capture_transitions_core/capture_transitions.csv")
    groups = read("results/extensions_20261109/diagnostics/feature_group_displacement.csv")
    conditions = ("white", "demand", "semantic_same", "semantic_different")
    rr = [
        next(
            r
            for r in closing["readout"]
            if r["variant"] == "clap_baseline"
            and r["region"] == "own"
            and r["condition"] == condition
            and r["k"] == "2"
            and float(r["level_db"]) == 5
        )
        for condition in conditions[2:]
    ]
    expected = {
        5: [
            [
                100
                * float(
                    next(
                        r
                        for r in core
                        if r["system"] == system
                        and r["region"] == "own"
                        and r["condition"] == c + "@+5dB"
                    )["macro_f1_mean"]
                )
                for c in conditions
            ]
            for system in ("svm", "cnn", "clap")
        ],
        6: [
            [
                100
                * float(
                    next(
                        r
                        for r in replication
                        if int(r["seed"]) == seed and r["region"] == "own" and r["condition"] == c
                    )["accuracy_difference"]
                )
                for c in conditions[2:]
            ]
            for seed in (5305, 5306, 5307)
        ],
        7: [
            [
                float(
                    next(
                        r
                        for r in captures
                        if r["variant"] == variant
                        and r["region"] == "own"
                        and r["condition"] == c
                        and float(r["level_db"]) == 5
                        and r["stratum"] == "all"
                    )[metric]
                )
                for c in conditions[2:]
                for variant in ("clap_baseline", "clap_prompt_ensemble")
            ]
            for metric in ("distractor_captures", "other_wrong_labels")
        ],
        9: [
            [
                float(
                    next(
                        r
                        for r in groups
                        if r["region"] == "own"
                        and r["condition"] == c
                        and float(r["level_db"]) == 5
                        and r["feature_group"] == group
                        and r["outcome"] == "all"
                    )["median"]
                )
                for group in ("zcr", "spectral_centroid", "spectral_spread", "spectral_rolloff")
            ]
            for c in conditions[:3]
        ],
        10: [
            [100 * float(r["mixture_" + state]) for r in rr]
            for state in ("target_only", "both", "distractor_only", "neither")
        ],
        11: [
            [
                next(
                    r
                    for r in closing["acoustic_medians"]
                    if r["region"] == "own" and r["condition"] == "white"
                )[metric]
                for metric in ("spectral_correlation_db", "density_spectral_correlation_db")
            ]
        ],
        14: [
            [
                float(
                    next(
                        r
                        for r in read(
                            "results/extensions_20261109/" + folder + "/latency_summary.csv"
                        )
                        if r["variant"] == variant
                    )["total_median_ms"]
                )
                for variant in ("svm_clean", "cnn_clean", "clap_baseline")
            ]
            for folder in ("cost_panel", "cost_panel_reverse")
        ],
    }
    segments = json.loads((deck / "narration_segments.json").read_text())
    script = (ROOT / "docs/video_narration_draft.txt").read_text()
    native_values = 0
    with zipfile.ZipFile(deck / "research_presentation.pptx") as z:
        slides = [n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        assert len(slides) == len(segments) == 16
        for number, values in expected.items():
            root = ET.fromstring(z.read(f"ppt/slides/slide{number}.xml"))
            chart = root.find(".//c:chart", NS)
            rels = ET.fromstring(z.read(f"ppt/slides/_rels/slide{number}.xml.rels"))
            relation = next(
                r for r in rels if r.attrib["Id"] == chart.attrib["{" + NS["r"] + "}id"]
            )
            chartpath = posixpath.normpath(
                posixpath.join("ppt/slides", relation.attrib["Target"])
            ).lstrip("/")
            chart = ET.fromstring(z.read(chartpath))
            series = chart.findall(".//c:ser", NS)
            assert len(series) == len(values)
            for s, exp in zip(series, values):
                actual = [
                    p.find("c:v", NS).text for p in s.findall("c:val/c:numRef/c:numCache/c:pt", NS)
                ]
                close(actual, exp)
                native_values += len(exp)
        # Check every displayed intervention table cell against the original CSV.
        table = ET.fromstring(z.read("ppt/slides/slide8.xml")).find(".//a:tbl", NS)
        cells = [
            ["".join(cell.itertext()) for cell in row.findall("a:tc", NS)]
            for row in table.findall("a:tr", NS)
        ]
        for i, (system, regime) in enumerate(
            (s, r) for s in ("svm", "cnn") for r in ("clean", "wgn", "semantic")
        ):
            collections = [
                [
                    r
                    for r in data["clean_variants"]
                    if r["system"] == system and r["regime"] == regime
                ]
            ]
            collections += [
                [
                    r
                    for r in data["interventions"]
                    if r["system"] == system
                    and r["regime"] == regime
                    and r["region"] == "own"
                    and r["condition"] == c
                    and float(r["level_db"]) == 5
                ]
                for c in conditions
            ]
            for j, collection in enumerate(collections):
                vv = [100 * float(r["macro_f1"]) for r in collection]
                mean = sum(vv) / len(vv)
                sd = (
                    math.sqrt(sum((v - mean) ** 2 for v in vv) / (len(vv) - 1))
                    if system == "cnn"
                    else 100 * float(collection[0]["macro_f1_fold_sd"])
                )
                assert f"{mean:.1f} ± {sd:.1f}" in cells[i + 1][j + 1]
        # New interval chart uses editable native points and lines, not a bitmap.
        slide = ET.fromstring(z.read("ppt/slides/slide12.xml"))
        geometry = []
        for transform in slide.findall(".//a:xfrm", NS):
            off, ext = transform.find("a:off", NS), transform.find("a:ext", NS)
            if off is not None and ext is not None:
                geometry.append(
                    tuple(
                        int(x)
                        for x in (
                            off.attrib["x"],
                            off.attrib["y"],
                            ext.attrib["cx"],
                            ext.attrib["cy"],
                        )
                    )
                )
        for i, system in enumerate(("svm", "cnn", "clap")):
            for j, condition in enumerate(conditions[2:]):
                r = next(
                    r
                    for r in closing["acoustic_primary"]
                    if r["region"] == "own"
                    and r["system"] == system
                    and r["a_condition"] == condition
                    and r["b_condition"] == "demand"
                )
                lo, hi = 100 * float(r["ci_low"]), 100 * float(r["ci_high"])
                desired = [
                    914400 * v
                    for v in (
                        3 + (lo + 40) / 60 * 5,
                        2.42 + (i * 2 + j) * 0.59,
                        (hi - lo) / 60 * 5,
                        0,
                    )
                ]
                assert any(all(abs(a - b) <= 1 for a, b in zip(g, desired)) for g in geometry)
        for segment in segments:
            n = segment["slide"]
            note = "".join(ET.fromstring(z.read(f"ppt/notesSlides/notesSlide{n}.xml")).itertext())
            assert segment["narration"] in note and segment["narration"] in script
        audio = [n for n in z.namelist() if n.endswith(".wav")]
        assert len(audio) == 2
        original = subprocess.check_output(
            [
                "git",
                "show",
                "21768f4c9ed6265e44e7d4e0bf7e16a43121a7d8:"
                "results/presentation_20261109_draft/research_presentation.pptx",
            ],
            cwd=ROOT,
        )
        with zipfile.ZipFile(io.BytesIO(original)) as old:
            assert sorted(hashlib.sha256(z.read(n)).hexdigest() for n in audio) == sorted(
                hashlib.sha256(old.read(n)).hexdigest()
                for n in old.namelist()
                if n.endswith(".wav")
            )
    # Independently check the rendered website text for all four selector states.
    web = json.loads(browser.read_text())
    assert not web["errors"] and not web["bad"]
    for state in web["states"]:
        selected = [
            r
            for r in closing["readout"]
            if r["region"] == state["region"]
            and r["variant"] == state["variant"]
            and r["k"] == "2"
            and float(r["level_db"]) == 5
        ]
        for row in selected:
            for name in ("target_only", "both", "distractor_only", "neither"):
                assert f"{100 * float(row['mixture_' + name]):.2f}%" in state["readout"]
            assert (
                f"({row['captured_target_in_topk_count']}/{row['argmax_captured_count']})"
                in state["readout"]
            )
            retained = int(row["captured_target_in_topk_count"]) / int(row["argmax_captured_count"])
            assert f"{100 * retained:.2f}%" in state["readout"]
        for row in closing["acoustic_primary"]:
            if row["region"] == state["region"] and row["b_condition"] == "demand":
                formatted = (
                    f"{100 * float(row['accuracy_difference']):+.2f} "
                    f"[{100 * float(row['ci_low']):+.2f}, "
                    f"{100 * float(row['ci_high']):+.2f}]"
                )
                assert formatted in state["acoustics"]
    report = ROOT / "results/research_report_20261109_draft/research_report.pdf"
    previous = json.loads((ROOT / "results/closing_20261110/completion_audit.json").read_text())
    assert sha(report) == previous["report_sha256"]
    allowed_delivery_updates = {
        "results/presentation_20261109_draft/" + name
        for name in (
            "build_status.json",
            "presentation_data.json",
            "research_presentation.pdf",
            "research_presentation.pptx",
            "speaker_notes.txt",
            "visual_audit.json",
        )
    } | {"results/site_review_20261109.json"}
    delivery_updates = []
    for relative, digest in previous["original_git_blobs"].items():
        content = (ROOT / relative).read_bytes()
        blob = b"blob " + str(len(content)).encode() + b"\x00" + content
        if hashlib.sha1(blob).hexdigest() != digest:
            assert relative in allowed_delivery_updates, relative
            delivery_updates.append(relative)
    for number in ("37.34", "33.17", "26.05", "21.00", "23.35", "24.05"):
        assert (
            number in script and number in (ROOT / "docs/report_closing_analyses.tex").read_text()
        )
    result = dict(
        verified=True,
        native_chart_values_checked=native_values,
        native_charts=7,
        intervention_table_cells=30,
        interval_lines_checked=6,
        exact_narration_segments=16,
        browser_selector_states=4,
        embedded_audio_unchanged=True,
        report_unchanged=True,
        prior_result_files_unchanged=len(previous["original_git_blobs"]) - len(delivery_updates),
        intentional_delivery_updates=delivery_updates,
        claim_boundary="Numerical consistency and stated interpretation checked; not a course-format approval or completed video.",
        files={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                report,
                deck / "research_presentation.pptx",
                deck / "research_presentation.pdf",
                deck / "presentation_data.json",
                ROOT / "docs/video_narration_draft.txt",
                ROOT / "index.html",
                ROOT / "site-data.json",
            ]
        },
    )
    (destination / "consistency_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-audit", type=Path, required=True)
    audit(parser.parse_args().browser_audit)

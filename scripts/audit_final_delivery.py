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

import pypdfium2 as pdfium
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


def audit(browser, output=None):
    destination = ROOT / "results/final_submission_20261110"
    deck = destination
    data = json.loads((deck / "presentation_data.json").read_text())
    site = json.loads((ROOT / "site-data.json").read_text())
    closing = closing_claims()
    assert data["closing"] == site["closing"] == closing
    core = read("results/clap_full_20261109/all_conditions_verified.csv")
    assert core == data["core"] == site["conditions"]
    conditions = ("white", "demand", "semantic_same", "semantic_different")
    rr = [
        next(
            r
            for r in closing["readout"]
            if r["variant"] == "clap_baseline"
            and r["region"] == "own"
            and r["condition"] == c
            and r["k"] == "2"
            and float(r["level_db"]) == 5
        )
        for c in conditions[2:]
    ]
    expected = {
        6: [
            [
                100
                * float(
                    next(
                        r
                        for r in core
                        if r["system"] == s
                        and r["region"] == "own"
                        and r["condition"] == c + "@+5dB"
                    )["macro_f1_mean"]
                )
                for c in conditions
            ]
            for s in ("svm", "cnn", "clap")
        ],
        8: [
            [100 * float(r["mixture_" + state]) for r in rr]
            for state in ("target_only", "both", "distractor_only", "neither")
        ],
    }
    segments = json.loads((deck / "narration_segments.json").read_text())
    script = (ROOT / "docs/video_narration_final.txt").read_text()
    native_values = 0
    with zipfile.ZipFile(deck / "presentation.pptx") as z:
        slides = [n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        assert len(slides) == len(segments) == 12
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
        table = ET.fromstring(z.read("ppt/slides/slide7.xml")).find(".//a:tbl", NS)
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
        slide = ET.fromstring(z.read("ppt/slides/slide9.xml"))
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
    report = destination / "final_report.pdf"
    pdf = pdfium.PdfDocument(report)
    assert len(pdf) == 10
    pages = []
    for page in pdf:
        textpage = page.get_textpage()
        pages.append(textpage.get_text_range())
        textpage.close()
        page.close()
    pdf.close()
    report_text = "\n".join(pages)
    assert "Contents" in pages[0] and "References" in pages[9]
    assert "??" not in report_text and "\ufffd" not in report_text
    for value in (
        "37.34",
        "33.17",
        "26.05",
        "21.00",
        "23.35",
        "24.05",
        "236/632",
        "203/612",
        "5.40",
        "1.70",
        "−0.728",
        "−0.008",
    ):
        assert value in report_text, value
    for system, condition in [
        ("svm", "white"),
        ("svm", "semantic_same"),
        ("clap", "demand"),
        ("clap", "semantic_same"),
    ]:
        row = next(
            r
            for r in core
            if r["region"] == "own"
            and r["system"] == system
            and r["condition"] == condition + "@+5dB"
        )
        assert f"{100 * float(row['macro_f1_mean']):.2f}" in report_text
    # Independently validate every number in the main report's generated tables.
    table_source = (ROOT / "docs/final_report_tables.tex").read_text()
    for system, path in [
        ("svm", "results/clean_svm/summary.json"),
        ("cnn", "results/clean_cnn/summary.json"),
        ("clap", "results/clap_integration_20261109/clean/summary.json"),
    ]:
        baseline = json.loads((ROOT / path).read_text())
        for metric in ["accuracy", "macro_f1"]:
            value = f"{100 * baseline[metric]['mean']:.2f} $\\pm$ {100 * baseline[metric]['sample_std']:.2f}"
            assert value in table_source
    for regime in ["clean", "wgn", "semantic"]:
        for condition in ["clean", *conditions]:
            collection = [
                r
                for r in data["clean_variants" if condition == "clean" else "interventions"]
                if r["system"] == "cnn"
                and r["regime"] == regime
                and (
                    condition == "clean"
                    or (
                        r["region"] == "own"
                        and r["condition"] == condition
                        and float(r["level_db"]) == 5
                    )
                )
            ]
            values = [100 * float(r["macro_f1"]) for r in collection]
            assert len(values) == 3
            mean = sum(values) / 3
            sd = math.sqrt(sum((x - mean) ** 2 for x in values) / 2)
            assert f"{mean:.2f} $\\pm$ {sd:.2f}" in table_source
    for key in ["cost_panel", "cost_panel_reverse"]:
        for variant in ["svm_clean", "cnn_clean", "clap_baseline"]:
            assert (
                f"{float(next(r for r in data[key] if r['variant'] == variant)['total_median_ms']):.2f}"
                in table_source
            )
    from pypdf import PdfReader

    uris = []
    for page in PdfReader(report).pages:
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A", {})
            if action.get("/URI"):
                uris.append(str(action["/URI"]))
    base = "https://charlie-uni.github.io/elec5305-project-550489593/"
    assert base in uris and base + "results/final_submission_20261110/code_demo.mp4" in uris
    assert base + "docs/code_demo.md" in uris
    supplement = ROOT / "results/research_report_20261109_draft/research_report.pdf"
    assert sha(supplement) == "e5ebcef99033065fdaea048b1a2debfdb3ad90c58710ad45b80adc98cd693e35"
    # Every pre-existing result/example blob remains unchanged, including the old deck.
    tree = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "889a6b82dcdc81de6ce8cef6c40dc62954a2fb5c",
            "--",
            "results",
            "examples",
        ],
        cwd=ROOT,
        text=True,
    )
    original_count = 0
    for line in tree.splitlines():
        metadata, relative = line.split("\t", 1)
        content = (ROOT / relative).read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\x00" + content).hexdigest()
        assert digest == metadata.split()[2], relative
        original_count += 1
    actual_demo = {
        s: json.loads((destination / f"demo_run/{s}.json").read_text())
        for s in ["svm", "cnn", "clap"]
    }
    for s, r in actual_demo.items():
        assert (
            r["verified"]
            and r["computed_from_source_audio"]
            and r["score_reference_read_after_inference"]
        )
        assert r["score_max_abs_error"] < 1e-4 and abs(r["measurements"]["achieved_db"] - 5) < 1e-8
        assert r["peak"] <= 0.99000001
    video = json.loads((destination / "code_demo_audit.json").read_text())
    assert video["sha256"] == sha(destination / "code_demo.mp4")
    result = dict(
        verified=True,
        report_pages=10,
        presentation_slides=12,
        native_chart_values_checked=native_values,
        native_charts=2,
        intervention_table_cells=30,
        interval_lines_checked=6,
        exact_narration_segments=12,
        browser_selector_states=4,
        embedded_audio_unchanged=True,
        supplement_26pages_unchanged=True,
        original_result_and_example_files_unchanged=original_count,
        real_inference_models=3,
        checked_score_max_abs_error={s: r["score_max_abs_error"] for s, r in actual_demo.items()},
        video_duration_seconds=video["duration_seconds"],
        author_narrated_video_complete=False,
        canvas_submitted=False,
        files={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                report,
                supplement,
                destination / "presentation.pptx",
                destination / "presentation.pdf",
                destination / "code_demo.mp4",
                destination / "presentation_data.json",
                ROOT / "docs/video_narration_final.txt",
                ROOT / "docs/rubric_review_20261110.md",
                ROOT / "index.html",
                ROOT / "site-data.json",
            ]
        },
        claim_boundary="Checked numerical consistency, original-file preservation and actual demonstration inference. Author review, full narration and Canvas submission remain.",
    )
    audit_path = output if output is not None else destination / "delivery_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Save a new audit without replacing an earlier one")
    args = parser.parse_args()
    audit(args.browser_audit, args.output)

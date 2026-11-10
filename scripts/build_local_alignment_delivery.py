"""Synchronise independently verified local-alignment values across final deliverables."""

import json

from delivery_claims import ROOT, local_alignment_claims


def main():
    claims = local_alignment_claims()
    if claims is None:
        raise ValueError("The full local-alignment run has not passed independent verification")
    destination = ROOT / "results/final_submission_20261110"
    for path in (destination / "presentation_data.json", ROOT / "site-data.json"):
        data = json.loads(path.read_text())
        data["local_alignment"] = claims
        path.write_text(json.dumps(data, indent=2) + "\n")
    rows = []
    for name, title in (
        ("single", "Single prompt"),
        ("ensemble", "Fixed ensemble"),
        ("aligned_fixed", "Local: preset"),
        ("aligned_calibrated", "Local: calibrated"),
    ):
        clean = next(row for row in claims["conditions"] if
                     row["region"] == "clean" and row["readout"] == name)
        same = next(row for row in claims["conditions"] if
                    row["region"] == "own" and row["condition"] == "semantic_same" and
                    float(row["level_db"]) == 5 and row["readout"] == name)
        assert int(same["n"]) == 2000
        values = [title, f"{100 * float(clean['accuracy_mean']):.2f}",
                  f"{100 * float(same['accuracy_mean']):.2f}",
                  f"{100 * float(same['macro_f1_mean']):.2f}", same["captures"],
                  same["other_errors"], f"{100 * int(same['target_in_top2']) / 2000:.2f}"]
        rows.append(" & ".join(values) + r" \\")
    (ROOT / "docs/local_alignment_tables.tex").write_text(
        "\\newcommand{\\LocalAlignmentRows}{\n" + "\n".join(rows) + "\n}\n"
    )
    print(json.dumps({"verified": True, "table_rows": len(rows),
                      "primary_tests": len(claims["primary"])}))


if __name__ == "__main__":
    main()

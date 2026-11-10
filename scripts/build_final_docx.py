"""Create an editable Word companion from the concise LaTeX report.

Requires Pandoc and python-docx. The PDF remains the fixed-layout submission copy.
"""

import argparse
import re
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/final_submission_20261110/final_report.docx"


def brace(text, start):
    """Read a balanced TeX group, retaining nested formatting."""
    assert text[start] == "{"
    depth = 1
    for pos in range(start + 1, len(text)):
        if text[pos] in "{}" and text[pos - 1] != "\\":
            depth += 1 if text[pos] == "{" else -1
            if depth == 0:
                return text[start + 1 : pos], pos + 1
    raise ValueError("Unclosed TeX group")


def expand(text, command, count, replacement):
    while command in text:
        pos = text.index(command)
        end, args = pos + len(command), []
        for _ in range(count):
            arg, end = brace(text, end)
            args.append(arg)
        text = text[:pos] + replacement(*args) + text[end:]
    return text


def prepare_tex():
    source = (ROOT / "docs/final_report.tex").read_text()
    references = {
        key: str(i + 1) for i, key in enumerate(re.findall(r"\\bibitem\{([^}]+)\}", source))
    }
    source = source.split(r"\begin{document}", 1)[1].split(r"\end{document}")[0]
    source = "TITLEBLOCKMARKER\n\n" + source[source.index(r"\begin{abstract}") :]
    source = source.replace(r"\begin{abstract}", r"\section*{Abstract}").replace(
        r"\end{abstract}", ""
    )
    source = source.replace(r"\tableofcontents", "TOCMARKER").replace(
        r"\clearpage", "\n\nPAGEBREAKMARKER\n\n"
    )
    source = source.replace(r"\site", "https://charlie-uni.github.io/elec5305-project-550489593/")
    source = source.replace(r"\repo", "https://github.com/Charlie-Uni/elec5305-project-550489593")
    source = re.sub(r"\\vspace\{[^}]+\}", "", source)
    source = source.replace(r"\phantomsection", "")
    source = source.replace(r"\begin{samepage}", "").replace(r"\end{samepage}", "")
    source = re.sub(r"\\addcontentsline\{toc\}\{section\}\{References\}", "", source)
    source = source.replace(
        r"\begin{thebibliography}{13}\raggedright", r"\section*{References}"
    ).replace(r"\end{thebibliography}", "")
    source = re.sub(r"\\bibitem\{([^}]+)\}", lambda m: "\n\n[" + references[m[1]] + "] ", source)
    source = re.sub(
        r"\\cite\{([^}]+)\}",
        lambda m: "[" + ", ".join(references[k] for k in m[1].split(",")) + "]",
        source,
    )
    for filename, names in (
        ("final_report_tables.tex", ("CleanRows", "AugRows", "CostRows")),
        ("local_alignment_tables.tex", ("LocalAlignmentRows",)),
    ):
        if not any("\\" + name in source for name in names):
            continue
        tables = (ROOT / "docs" / filename).read_text()
        for name in names:
            start = tables.index("\\newcommand{\\" + name + "}") + len("\\newcommand{\\" + name + "}")
            content, _ = brace(tables, start)
            source = source.replace("\\" + name, content)
    figure_number, table_number = 0, 0

    def figure(name, caption):
        nonlocal figure_number
        figure_number += 1
        image = ROOT / "results/final_submission_20261110/figures" / (name + ".png")
        return f"\\includegraphics[width=164mm]{{{image}}}\n\n\\textbf{{Figure {figure_number}:}} {caption}\n\n"

    def caption(kind, text):
        nonlocal table_number
        assert kind == "table"
        table_number += 1
        return f"\\textbf{{Table {table_number}:}} {text}\n\n"

    source = expand(source, r"\pic", 2, figure)
    source = expand(source, r"\captionof", 2, caption)
    source = source.replace(
        r"\begin{tabularx}{\textwidth}{@{}lXX@{}}", r"\begin{tabular}{lll}"
    ).replace(r"\end{tabularx}", r"\end{tabular}")
    section, subsection = 0, 0

    def heading(match):
        nonlocal section, subsection
        kind, title = match.groups()
        if kind == "section":
            section += 1
            subsection = 0
            number = str(section)
        else:
            subsection += 1
            number = f"{section}.{subsection}"
        return "\\" + kind + "{" + number + " " + title + "}"

    return re.sub(r"\\(section|subsection)\{([^}]+)\}", heading, source)


def element(name, **attrs):
    obj = OxmlElement(name)
    for key, value in attrs.items():
        obj.set(qn(key), str(value))
    return obj


def format_document(doc, toc_pdf=None):
    section = doc.sections[0]
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.top_margin = section.bottom_margin = Mm(20)
    section.left_margin = section.right_margin = Mm(20)
    section.footer_distance = Mm(10)
    section.different_first_page_header_footer = True
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        st = doc.styles[name]
        st.font.name, st.font.size = "Times New Roman", Pt(11)
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.line_spacing = Pt(13.2)
        st.paragraph_format.space_after = Pt(4.5)
        st.paragraph_format.first_line_indent = Pt(0)
    for name, size in (("Title", 20), ("Subtitle", 13), ("Heading 1", 14), ("Heading 2", 12)):
        st = doc.styles[name]
        st.font.name, st.font.size = "Times New Roman", Pt(size)
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.bold = name != "Subtitle"
        st.paragraph_format.space_before = Pt(10 if "Heading" in name else 0)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer._p.append(element("w:fldSimple", **{"w:instr": "PAGE"}))
    titles = [
        ("When Background Noise Has Meaning", "Title"),
        ("Environmental Sound Classification under Semantic Interference", "Subtitle"),
        ("Qihan Zhu   SID 550489593", "Normal"),
        ("ELEC5305 Acoustics, Speech and Signal Processing", "Normal"),
        ("11 November 2026", "Normal"),
    ]
    toc = [
        ("1 Introduction", 1),
        ("2 Literature review", 2),
        ("3 Methods", 3),
        ("4 Results", 5),
        ("5 Discussion", 8),
        ("6 Conclusion", 9),
        ("7 Implementation, reproduction and attribution", 9),
        ("References", 11),
    ]
    if toc_pdf is not None:
        from pypdf import PdfReader

        reader = PdfReader(toc_pdf)
        destinations = {
            re.sub(r"^\d+\s+", "", item.title): reader.get_destination_page_number(item) + 1
            for item in reader.outline if isinstance(item, dict)
        }
        # Some older LaTeX bookmarks inherit the preceding section's anchor.
        references = [i + 1 for i, page in enumerate(reader.pages)
                      if re.match(r"\s*References(?:\s|$)", page.extract_text() or "")]
        destinations["References"] = references[0] if references else len(reader.pages)
        toc = [(title, destinations.get(re.sub(r"^\d+\s+", "", title), page))
               for title, page in toc]
    conclusion_next = False
    for p in list(doc.paragraphs):
        if p.text == "6 Conclusion":
            conclusion_next = True
        elif conclusion_next and p.text.strip():
            p.paragraph_format.keep_together = True
            conclusion_next = False
        if p.text == "TITLEBLOCKMARKER":
            for text, style in titles:
                added = p.insert_paragraph_before(text, style)
                added.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p._element.getparent().remove(p._element)
        elif p.text == "TOCMARKER":
            p.insert_paragraph_before("Contents", "Heading 1")
            for title, page in toc:
                added = p.insert_paragraph_before(f"{title}\t{page}")
                added.paragraph_format.tab_stops.add_tab_stop(Mm(169), WD_TAB_ALIGNMENT.RIGHT)
                added.paragraph_format.space_after = Pt(5)
            p._element.getparent().remove(p._element)
        elif p.text == "PAGEBREAKMARKER":
            p.clear()
            p.add_run().add_break(WD_BREAK.PAGE)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = Pt(1)
            p.paragraph_format.keep_with_next = True
        elif p.text.startswith(("Figure ", "Table ")):
            p.paragraph_format.line_spacing = Pt(11)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.keep_with_next = p.text.startswith("Table ")
            for run in p.runs:
                run.font.size = Pt(9.5)
        elif p._p.xpath(".//w:drawing"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.keep_with_next = True
        elif p._p.xpath(".//m:oMathPara"):
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(5)
        if p.text.startswith("[") and re.match(r"\[\d+\]", p.text):
            p.paragraph_format.space_after = Pt(7)
    widths = [[23, 73, 74], [18, 32, 32, 44, 44], [25, 29, 29, 29, 29, 29], [50, 50, 50]]
    if len(doc.tables) == 5:
        widths.insert(3, [30, 23, 23, 23, 23, 23, 25])
    assert len(doc.tables) == len(widths)
    for table_index, (table, columns) in enumerate(zip(doc.tables, widths)):
        table.autofit = False
        for column, width in zip(table.columns, columns):
            column.width = Mm(width)
        for i, row in enumerate(table.rows):
            row._tr.get_or_add_trPr().append(element("w:cantSplit"))
            if i == 0:
                row._tr.get_or_add_trPr().append(element("w:tblHeader"))
            for j, cell in enumerate(row.cells):
                cell.width = Mm(columns[j])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                properties = cell._tc.get_or_add_tcPr()
                borders = element("w:tcBorders")
                for edge in ("top", "left", "bottom", "right"):
                    borders.append(
                        element(
                            "w:" + edge, **{"w:val": "single", "w:sz": "4", "w:color": "D9D9D9"}
                        )
                    )
                properties.append(borders)
                margins = element("w:tcMar")
                for edge in ("top", "bottom", "left", "right"):
                    margins.append(
                        element(
                            "w:" + edge,
                            **{"w:w": "35" if edge in ("top", "bottom") else "60", "w:type": "dxa"},
                        )
                    )
                properties.append(margins)
                if i == 0:
                    properties.append(element("w:shd", **{"w:fill": "E8E8E8"}))
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.line_spacing = Pt(10.7)
                    p.paragraph_format.keep_with_next = i < len(table.rows) - 1
                    p.alignment = (
                        WD_ALIGN_PARAGRAPH.LEFT
                        if j == 0 or table_index == 0
                        else WD_ALIGN_PARAGRAPH.CENTER
                    )
                    for run in p.runs:
                        run.font.name, run.font.size = "Times New Roman", Pt(9.5)
                        if i == 0:
                            run.bold = True
    doc.core_properties.title = "When Background Noise Has Meaning"
    doc.core_properties.author = "Qihan Zhu"
    doc.core_properties.subject = "ELEC5305 concise research report"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pandoc", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--toc-pdf", type=Path, default=ROOT / "results/final_submission_20261110/final_report.pdf")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    tex = args.work_dir / "report_word_source.tex"
    tex.write_text(prepare_tex())
    candidate = args.work_dir / "converted.docx"
    subprocess.run(
        [str(args.pandoc), str(tex), "--from=latex", "--to=docx", "--output", str(candidate)],
        check=True,
    )
    doc = Document(candidate)
    format_document(doc, args.toc_pdf)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()

"""Render every report page and check text bounds before visual inspection."""

import argparse
import json
import math
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from esc_robustness.artifacts import sha256_file


def inspect(source, output):
    output.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(source)
    pages, flags = [], []
    for index in range(len(document)):
        page = document[index]
        width, height = page.get_size()
        textpage = page.get_textpage()
        content = textpage.get_text_range()
        if "\ufffd" in content or "??" in content:
            flags.append(dict(page=index + 1, issue="Replacement glyph or unresolved reference"))
        for char in range(textpage.count_chars()):
            if not textpage.get_text_range(char, 1).strip():
                continue
            left, bottom, right, top = textpage.get_charbox(char)
            if left < -0.5 or bottom < -0.5 or right > width + 0.5 or top > height + 0.5:
                flags.append(
                    dict(page=index + 1, issue="Text outside page", box=[left, bottom, right, top])
                )
        picture = page.render(scale=1.5).to_pil().convert("RGB")
        picture.save(output / f"page_{index + 1:02d}.png")
        pages.append(picture.copy())
        textpage.close()
        page.close()
    columns, cell_width = 3, 420
    cell_height = round((cell_width - 20) * pages[0].height / pages[0].width) + 36
    sheet = Image.new(
        "RGB", (columns * cell_width, math.ceil(len(pages) / columns) * cell_height), "#dddddd"
    )
    draw = ImageDraw.Draw(sheet)
    for index, picture in enumerate(pages):
        picture.thumbnail((cell_width - 20, cell_height - 36))
        x, y = index % columns * cell_width + 10, index // columns * cell_height + 25
        sheet.paste(picture, (x, y))
        draw.text((x, y - 19), f"Page {index + 1}", fill="black")
    sheet.save(output / "contact_sheet.png")
    result = dict(
        pdf_sha256=sha256_file(source),
        pages=len(pages),
        text_flags=flags,
        visual_review_required=True,
    )
    (output / "text_geometry_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inspect(args.source, args.output)

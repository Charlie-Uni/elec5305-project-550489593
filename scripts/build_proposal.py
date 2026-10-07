"""Compile the editable LaTeX proposal in the original document's typography."""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "proposal.tex"
OUTPUT = ROOT / "ELEC5305_Project_proposal_550489593.pdf"


def word_count(source=SOURCE):
    """Count visible source words, including headings, timeline and references.

    This covers the simple commands used in proposal.tex. Auto-generated section,
    bibliography and page numbers are excluded, as are hyperlink destinations.
    """
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    text = text.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    text = re.sub(r"\\href\{[^{}]*\}\{\\nolinkurl\{([^{}]*)\}\}", r"\1", text)
    text = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\url\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\label\{[^{}]*\}", "", text)
    text = re.sub(r"\\urlstyle\{[^{}]*\}", "", text)
    text = re.sub(r"^.*\\(?:begin|end)\{[^{}]+\}.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", "", text)
    text = text.replace(r"\\", " ").replace("&", " ")
    text = text.translate(str.maketrans("", "", "{}"))
    return len(text.split())


def find_compiler(requested=None):
    candidates = (
        [requested]
        if requested
        else [
            shutil.which("tectonic"),
            str(ROOT / ".venv" / "bin" / "tectonic"),
            shutil.which("pdflatex"),
        ]
    )
    for candidate in candidates:
        if candidate:
            resolved = shutil.which(candidate) or str(Path(candidate).expanduser().resolve())
            if Path(resolved).is_file():
                return Path(resolved)
    raise FileNotFoundError(
        "A LaTeX compiler is required. Install Tectonic (macOS: brew install tectonic) "
        "or TeX Live, or pass --compiler /path/to/tectonic."
    )


def build(compiler=None):
    count = word_count()
    if not 700 <= count <= 1000:
        raise ValueError(f"Proposal contains {count} words; expected 700-1000.")
    compiler = find_compiler(compiler)
    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="proposal-", dir=ROOT / "tmp") as temporary:
        if compiler.name == "tectonic":
            command = [str(compiler), "--keep-logs", "--outdir", temporary, str(SOURCE)]
            passes = 1  # Tectonic runs the necessary cross-reference passes itself.
        elif compiler.name == "pdflatex":
            command = [
                str(compiler),
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={temporary}",
                str(SOURCE),
            ]
            passes = 2
        else:
            raise ValueError("Supported compilers: tectonic or pdflatex")
        for _ in range(passes):
            result = subprocess.run(
                command,
                cwd=SOURCE.parent,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if result.returncode:
                raise RuntimeError(f"LaTeX compilation failed:\n{result.stdout}")
        log_path = Path(temporary) / "proposal.log"
        log = log_path.read_text(errors="replace") if log_path.exists() else result.stdout
        if re.search(r"Overfull \\[hv]box|Missing character:", log):
            raise RuntimeError(
                "LaTeX reported overflow or missing glyphs; inspect the layout before delivery."
            )
        generated = Path(temporary) / "proposal.pdf"
        pdf = PdfReader(generated)
        if not pdf.pages:
            raise RuntimeError("LaTeX did not produce a valid PDF")
        shutil.copyfile(generated, OUTPUT)
    print(f"{OUTPUT}\nWords including references: {count}\nPages: {len(pdf.pages)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", help="Path to tectonic or pdflatex")
    build(parser.parse_args().compiler)

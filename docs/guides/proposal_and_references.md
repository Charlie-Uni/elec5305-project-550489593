# Original proposal and external references

[Repository home](../../README.md) · [Documentation index](../README.md) · [Results index](../../results/README.md)

The proposal is a historical planning document. Use the [10-page report](../../results/final_submission_20261110/final_report.pdf) as the main manuscript.

## Rebuild the proposal

The PDF retains the original proposal's LaTeX article format: Latin Modern fonts,
11-point body text, 25 mm margins, numbered sections, bulleted student information,
a compact timeline table, hanging references and centred page numbers. Edit
`docs/proposal.tex`, then rebuild with `python scripts/build_proposal.py`.

The builder requires [Tectonic](https://tectonic-typesetting.github.io/) or
`pdflatex` from TeX Live. On macOS, install the former with `brew install tectonic`.
A compiler in `.venv/bin/tectonic` is also detected; `--compiler /path/to/tectonic`
selects an explicit binary. The first Tectonic build downloads its TeX packages.

The proposal follows the course guide's eight sections and 700-1000-word limit.
The builder counts visible source text including headings, timeline and references;
automatically generated numbering and hyperlink destinations are excluded. Detailed
implementation conventions are in the other reproduction guides to keep the proposal concise.

## References and external code

- [ESC-50 official dataset, folds, taxonomy, license and caveats](https://github.com/karolpiczak/ESC-50).
  Its source preprocessing can be class dependent, which limits generalisation claims.
- [Piczak's environmental sound CNN](https://doi.org/10.1109/MLSP.2015.7324337),
  a historical reference rather than an architecture to reproduce exactly.
- [LAION-CLAP official code and checkpoints](https://github.com/LAION-AI/CLAP)
  and [paper](https://arxiv.org/abs/2211.06687).
- [PAT, NAACL 2025](https://aclanthology.org/2025.naacl-long.616/), an optional
  training-free audio-text alignment extension.
- [Mix-CLAP, ICASSP 2026](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=4206),
  context for noise-aware and efficient audio-language models.
- [RSA-Bench, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.913/),
  context for structured interference in realistic acoustic scenes.

The project's DSP, data handling and evaluation code are implemented in this
repository with AI-assisted development and checking. NumPy/SciPy/scikit-learn/PyTorch are external libraries; pretrained
CLAP and optional PAT remain external research implementations with attribution.
CLAP's external pretraining is not counted as local inference cost or treated as
equivalent to the supervised systems' training budget.

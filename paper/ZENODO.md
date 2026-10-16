# Zenodo upload

Files (in `zenodo/`, git-ignored; rebuild with `./package_arxiv.sh` then copy):

- `trutina-4b-technical-report.pdf` — the paper (18 pages, fonts embedded, PDF title/author set). Upload this as the main file.
- `trutina-4b-source.tar.gz` — LaTeX source, `.bbl`, `.bib`, figures. Optional second file.

## Metadata form

| Field | Value |
|---|---|
| Resource type | Publication → Preprint (or Technical note) |
| Title | Trutina: Typed Decision Probabilities from a 4B Language Model with Letter Readout and Trainable Early Exits |
| Authors | Weng, Xinyuan — Aldine Labs (independent research) |
| Description | contents of `abstract.txt` (plain text; no LaTeX needed) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Keywords | decision models; language models; early exit; calibration; LoRA; distillation; Qwen3.5 |
| Language | English |
| Version | 1.0 |
| Publication date | the day you publish |

Related works (Identifiers tab):

| Relation | Identifier | Type |
|---|---|---|
| is supplemented by | https://github.com/aldinelabs/trutina | URL |
| is supplemented by | https://huggingface.co/aldinelabs/trutina-4b | URL |
| is described by | https://aldine.cn/en/journal/trutina-technical-report | URL |

## Before publishing

- A published Zenodo record cannot be withdrawn like a draft; check the title page (email, affiliation) first.
- Save as draft, open the preview, read the PDF once, then publish.
- Afterwards add the DOI to the README, the HF cards and the blog post, and later to the arXiv comment field.

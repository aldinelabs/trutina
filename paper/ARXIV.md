# arXiv submission

Source: `trutina.tex` with `trutina.bbl` (precompiled bibliography), `refs.bib` and `figures/*.pdf`.
Build locally with `tectonic trutina.tex`; `./package_arxiv.sh` writes `trutina-arxiv.tar.gz` and checks that it compiles from a clean directory.

## Format checks (from arXiv's submission help)

- TeX source is the preferred format; arXiv does not accept a PDF generated from TeX. Upload the tarball, not `trutina.pdf`.
- pdfLaTeX with TeX Live 2025 (default). Only standard packages are used (`lmodern`, `microtype`, `amsmath`, `booktabs`, `graphicx`, `natbib`, `hyperref`); no driver options are passed to `graphicx` or `hyperref`.
- Figures are vector PDFs with embedded fonts, made by `make_figures.py`; none is large.
- `trutina.bbl` is included, so arXiv uses it. It was produced by BibTeX (`unsrtnat`); it contains no biblatex data.
- File names use only `a-z A-Z 0-9 _ - .`; there are no hidden or unused files in the package.
- References carry arXiv identifiers in the form `arXiv:YYMM.NNNNN` with no formatting inside the identifier. Every arXiv reference was generated from the arXiv API (`make_bib.py`); software, models and datasets are cited by URL.
- Metadata limits: title without capitals-only words or Unicode; abstract at most 1,920 characters (ours is 1,577, plain text in `abstract.txt`); authors in "Firstname Lastname" form, no AI tool listed as an author.

## Metadata to enter

- **Title:** Trutina: Typed Decision Probabilities from a 4B Language Model with Letter Readout and Trainable Early Exits
- **Authors:** Xinyuan Weng (Aldine Labs)
- **Abstract:** contents of `abstract.txt`
- **Primary category:** cs.LG; **cross-list:** cs.CL
- **Comments:** 18 pages, 4 figures, 9 tables. Code and weights: https://github.com/aldinelabs/trutina
- **License:** the submitter's choice (CC BY 4.0 is the most open option and is commonly accepted by publishers)

## Before submitting

1. Check the author name, affiliation and contact address on the title page. They were taken from the repository and the public contact address.
2. A first-time submitter needs an endorsement for the category (see arXiv's endorsement page); an institutional email address speeds this up.
3. arXiv holds authors responsible for the content of a paper however it was produced, and sanctions papers with clear signs of unchecked AI output such as invented references. The paper discloses AI assistance in its last section; edit that statement if it does not match your practice. Each arXiv reference was generated from the arXiv record; the non-arXiv entries (models, datasets, software) were opened and checked, but their years are the release year of the artefact as we understand it.
4. Preview the PDF that arXiv builds before announcing; submissions received before 14:00 US Eastern are normally announced at 20:00 Eastern the same day.

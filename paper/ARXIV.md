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
- **Authors:** Xinyuan Weng (University of Toronto; Aldine Labs). arXiv's affiliation field takes at most a city and country, e.g. `Xinyuan Weng (University of Toronto, Canada)`; use "University of Toronto Mississauga" if that is your campus.
- **Abstract:** contents of `abstract.txt`
- **Primary category:** cs.LG; **cross-list:** cs.CL
- **Comments:** 18 pages, 4 figures, 9 tables. Code and weights: https://github.com/aldinelabs/trutina
- **License:** CC BY 4.0. It is the closest match to the Apache-2.0 licence of the weights and code: both allow commercial use, modification and redistribution with attribution. CC0 drops attribution, ShareAlike adds copyleft, the NC options restrict commercial use, and the arXiv non-exclusive licence allows the least reuse. The choice is irrevocable.

## Before submitting

1. The title page lists the University of Toronto and Aldine Labs with the University of Toronto email address. Change it if your campus or affiliation line should differ.
2. Endorsement: since 21 January 2026 arXiv no longer accepts an institutional email address as the sole basis for endorsing a new author. Automatic endorsement needs both an institutional email and an earlier paper of yours accepted in the same endorsement domain; without one you need personal endorsement from an established arXiv author in that domain (arXiv itself cannot give one). Start the submission, choose cs.LG, and arXiv issues an endorsement link and code to pass to one endorser, for example a supervisor or course instructor who has cs papers on arXiv. Do not mass-email endorsers.
3. arXiv holds authors responsible for the content of a paper however it was produced, and sanctions papers with clear signs of unchecked AI output such as invented references. The paper discloses AI assistance in its last section; edit that statement if it does not match your practice. Each arXiv reference was generated from the arXiv record; the non-arXiv entries (models, datasets, software) were opened and checked, but their years are the release year of the artefact as we understand it.
4. Preview the PDF that arXiv builds before announcing; submissions received before 14:00 US Eastern are normally announced at 20:00 Eastern the same day.

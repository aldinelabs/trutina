#!/bin/sh
# Build the arXiv upload (tex + bbl + bib + figures, flat names) and smoke-test that it compiles from a clean directory.
set -e
cd "$(dirname "$0")"
tectonic --keep-intermediates trutina.tex >/dev/null 2>&1
rm -rf arxiv && mkdir -p arxiv/figures
cp trutina.tex trutina.bbl refs.bib arxiv/
cp figures/*.pdf arxiv/figures/
( cd arxiv && tar czf ../trutina-arxiv.tar.gz trutina.tex trutina.bbl refs.bib figures )
T=$(mktemp -d)
tar xzf trutina-arxiv.tar.gz -C "$T"
( cd "$T" && tectonic trutina.tex >/dev/null 2>&1 && echo "clean build ok: $(ls -la trutina.pdf | awk '{print $5}') bytes" )
rm -rf "$T"
ls -la trutina-arxiv.tar.gz

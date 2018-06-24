#!/bin/bash

xelatex_exec="xelatex"
bibtex_exec="bibtex"

${xelatex_exec}  -file-line-error -interaction=nonstopmode -output-directory="../data/" ../data/for.0..1.supplementary_information.tex
cd ../data/
${bibtex_exec} for.0..1.supplementary_information
cd ../code/
${xelatex_exec}  -file-line-error -interaction=nonstopmode -output-directory="../data/" ../data/for.0..1.supplementary_information.tex
${xelatex_exec}  -file-line-error -interaction=nonstopmode -output-directory="../data/" ../data/for.0..1.supplementary_information.tex

cp ../data/for.0..1.supplementary_information.pdf ../data/for.1..1.supplementary_information.pdf
open ../data/for.1..1.supplementary_information.pdf

#!/bin/bash

xelatex_exec="xelatex"

${xelatex_exec}  -file-line-error -output-directory="../data/" ../data/for.0..1.supplementary_information.tex

cp ../data/for.0..1.supplementary_information.pdf ../data/for.1..1.supplementary_information.pdf
open ../data/for.1..1.supplementary_information.pdf

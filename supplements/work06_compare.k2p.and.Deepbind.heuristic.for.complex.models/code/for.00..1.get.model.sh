#!/bin/sh

mkdir -p ../data
mkdir -p ../external

rm -fr ../external/predict-lab-origin_a4c641c
mkdir ../external/predict-lab-origin_a4c641c
git clone --no-checkout https://github.com/VoigtLab/predict-lab-origin.git ../external/predict-lab-origin_a4c641c
cd ../external/predict-lab-origin_a4c641c && git checkout a4c641c && cd ../../code
ln -s ../addgene-plasmids-sequences.json ../external/predict-lab-origin_a4c641c/addgene-plasmids-sequences.json

## link the utils_EC2.py; this will be used later by our scripts
ln -s ../external/predict-lab-origin_a4c641c/utils_EC2.py utils_EC2.py

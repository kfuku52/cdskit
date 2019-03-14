#!/usr/bin/env bash

pip install '/Users/kef74yk/Dropbox_w/repos/cdskit'

wd="/Users/kef74yk/Dropbox_w/repos/cdskit/tests/"
cd ${wd}

echo "Test: stdin to stdout"
cdskit mask --seqfile ../data/example_mask.fasta --outfile -


#!/usr/bin/env bash

pip install '/Users/kef74yk/Dropbox_w/repos/cdskit'

wd="/Users/kef74yk/Dropbox_w/repos/cdskit/tests/"
cd ${wd}

echo "Test: stdin to stdout"
cat ../data/example_pad.fasta | cdskit pad

echo "Test: file to file, small"
cdskit pad --seqfile ../data/example_pad.fasta --outfile ../data/example_pad.out.fasta

echo "Test: file to file, big"
cdskit pad --seqfile ../data/longest_orfs.cds --outfile ../data/longest_orfs.out.cds

echo "Test: --nopseudo"
cdskit pad --nopseudo --seqfile ../data/example_pad.fasta --outfile ../data/example_pad.out.nopseudo.fasta




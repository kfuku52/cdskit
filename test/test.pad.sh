#!/usr/bin/env bash

wd=`pwd`
wd=`dirname "${wd}"`
export PATH="${wd}:${PATH}"

echo "Test: stdin to stdout"
cat ../data/example_pad.fasta | cdskit pad

echo "Test: file to file, small"
cdskit pad --seqfile ../data/example_pad.fasta --outfile ../data/example_pad.out.fasta

echo "Test: file to file, big"
cdskit pad --seqfile ../data/longest_orfs.cds --outfile ../data/longest_orfs.out.cds

echo "Test: --nopseudo"
cdskit pad --nopseudo --seqfile ../data/example_pad.fasta --outfile ../data/example_pad.out.nopseudo.fasta

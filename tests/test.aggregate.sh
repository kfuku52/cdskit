#!/usr/bin/env bash

pip install '/Users/kef74yk/Dropbox_w/repos/cdskit'

wd="/Users/kef74yk/Dropbox_w/repos/cdskit/tests/"
cd ${wd}

echo "Test: file to stdout"
cdskit aggregate --seqfile ../data/example_aggregate.fasta --outfile -  --expression ":.*" "\|.*"

echo "Test: pipe"
cat ../data/example_pad.fasta \
| cdskit pad \
| cdskit aggregate \
> ../data/example_pad.pipe.aggregate.fasta

#echo "Test: file to file, large"
#cdskit aggregate --seqfile ../data/longest_orfs.cds --outfile ../data/longest_orfs.aggregate.fasta --expression -

#rm ../data/*.aggregate.fasta
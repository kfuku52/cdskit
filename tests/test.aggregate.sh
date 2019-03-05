#!/usr/bin/env bash

wd=`pwd`
wd=`dirname "${wd}"`
export PATH="${wd}:${PATH}"

echo "Test: file to stdout"
cdskit aggregate --seqfile ../data/example_aggregate.fasta --outfile - --expression ":.*" "\|.*"

echo "Test: pipe"
cat ../data/example_pad.fasta \
| cdskit pad \
| cdskit aggregate \
> ../data/example_pad.pipe.aggregate.fasta

#echo "Test: file to file, large"
#cdskit aggregate --seqfile ../data/longest_orfs.cds --outfile ../data/longest_orfs.aggregate.fasta --expression -

#rm ../data/*.aggregate.fasta
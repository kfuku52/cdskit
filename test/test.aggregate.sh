#!/usr/bin/env bash

export PATH="`dirname \`pwd\``:${PATH}"

echo "Test: file to file"
cdskit aggregate --seqfile ../data/example_pad.fasta --outfile ../data/example_pad.aggregate.fasta --expression "-"

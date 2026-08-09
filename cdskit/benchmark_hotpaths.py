"""Reproducible microbenchmarks for cdskit's CPU hot paths."""

import argparse
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.degeneracy import classify_alignment_columns
from cdskit.filter import analyze_record
from cdskit.hammer import build_non_missing_site
from cdskit.localize_model import extract_targetp_feature_ensemble_features
from cdskit.maxalign import build_variable_subset_masks, evaluate_exact_subset_range
from cdskit.translate import translate_sequence_string
from cdskit.util import read_gff


def measure(worker, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        worker()
        samples.append(time.perf_counter() - started)
    return {
        'median_seconds': statistics.median(samples),
        'minimum_seconds': min(samples),
        'samples_seconds': samples,
    }


def run_benchmarks(scale=1, repeats=3):
    translate_seq = 'ATG' * (100_000 * scale)
    translate_sequence_string(translate_seq, 1, False)
    filter_record = SeqRecord(
        Seq('ATGNNN---TAA' * (10_000 * scale)),
        id='filter',
    )
    degeneracy_records = [
        SeqRecord(Seq('ATG' * (500 * scale)), id=str(index))
        for index in range(20)
    ]
    hammer_records = [
        SeqRecord(Seq('ATGGCNTAA---' * (50 * scale)), id=str(index))
        for index in range(100)
    ]
    random.seed(1)
    num_sequences = 14 if scale == 1 else 17
    support_counts = {
        random.getrandbits(num_sequences): random.randint(1, 4)
        for _ in range(80)
    }
    subset_masks = build_variable_subset_masks(list(range(num_sequences)))

    targetp_sequence = 'M' + ('ARNDCEQGHILKMFPSTWYV' * 12)
    with tempfile.TemporaryDirectory() as temporary_dir:
        gff_path = Path(temporary_dir) / 'benchmark.gff'
        with gff_path.open('w', encoding='utf-8') as output:
            output.write('##gff-version 3\n')
            for index in range(1_000 * scale):
                output.write(
                    'seq{}\tsrc\tgene\t{}\t{}\t.\t+\t.\tID=g{}\n'.format(
                        index % 100,
                        index + 1,
                        index + 10,
                        index,
                    )
                )
        gff_benchmark = measure(lambda: read_gff(str(gff_path)), repeats)

    return {
        'translate': {
            'workload': {'nucleotides': len(translate_seq)},
            **measure(
                lambda: translate_sequence_string(translate_seq, 1, False),
                repeats,
            ),
        },
        'filter': {
            'workload': {'nucleotides': len(filter_record.seq)},
            **measure(lambda: analyze_record(filter_record, 1, True), repeats),
        },
        'degeneracy': {
            'workload': {
                'records': len(degeneracy_records),
                'nucleotides_per_record': len(degeneracy_records[0].seq),
            },
            **measure(
                lambda: classify_alignment_columns(degeneracy_records, 1),
                repeats,
            ),
        },
        'hammer': {
            'workload': {
                'records': len(hammer_records),
                'codons_per_record': len(hammer_records[0].seq) // 3,
            },
            **measure(
                lambda: build_non_missing_site(hammer_records, 1, 1),
                repeats,
            ),
        },
        'maxalign_exact': {
            'workload': {
                'sequences': num_sequences,
                'subsets': 1 << num_sequences,
            },
            **measure(
                lambda: evaluate_exact_subset_range(
                    0,
                    1 << num_sequences,
                    subset_masks,
                    0,
                    0,
                    support_counts,
                    None,
                    num_sequences,
                    num_sequences,
                ),
                repeats,
            ),
        },
        'targetp_features': {
            'workload': {
                'records': 20 * scale,
                'amino_acids_per_record': len(targetp_sequence),
            },
            **measure(
                lambda: [
                    extract_targetp_feature_ensemble_features(
                        targetp_sequence,
                        'plant',
                    )
                    for _ in range(20 * scale)
                ],
                repeats,
            ),
        },
        'read_gff': {
            'workload': {'records': 1_000 * scale},
            **gff_benchmark,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Use larger workloads.')
    parser.add_argument('--repeats', default=3, type=int)
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error('--repeats should be positive')
    report = run_benchmarks(
        scale=10 if args.full else 1,
        repeats=args.repeats,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == '__main__':
    main()

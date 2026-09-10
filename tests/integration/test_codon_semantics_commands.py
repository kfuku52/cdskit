"""Cross-command invariants and independent ORF selection checks."""

import json
import random
from dataclasses import asdict

import pytest
from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit import codonstats, filter, longestcds, pad, translate, trimcodon, validate
from cdskit.cli import psr
from cdskit.draw import classify_codon
from cdskit.codonutil import get_codon_table_components
from cdskit.plot import _translate_codon_for_msa
from cdskit.tsvio import read_tsv


@pytest.mark.parametrize(
    "code,seq,protein,stops,possible",
    [
        (1, "ATGTARAAA", "M*K", 1, 0),
        (1, "ATGTANAAA", "MXK", 0, 1),
        (27, "ATGTGAAAA", "MWK", 0, 1),
        (28, "ATGTAATAGTGAAAA", "MQQWK", 0, 3),
        (31, "ATGTAATAGAAA", "MEEK", 0, 2),
    ],
)
def test_same_stop_meaning_in_all_consumers(code, seq, protein, stops, possible):
    record = SeqRecord(Seq(seq), id="x")
    assert translate.translate_sequence_string(seq, code, False) == protein
    analysis = filter.analyze_record(record, code, True)
    assert analysis["stop_codons"] == stops
    assert analysis["possible_stop_codons"] == possible
    assert analysis["internal_stop"] == bool(stops)
    validation = validate.summarize_records([record], code)
    assert validation["internal_stop_ids"] == (["x"] if stops else [])
    assert validation["possible_stop_codons"] == possible
    stats = codonstats.summarize_record(record, code)
    assert stats["codons_stop"] == stops
    assert "TAR" not in stats["usage"]
    sites = [
        trimcodon.summarize_codon_site([seq], i, code) for i in range(len(seq) // 3)
    ]
    assert sum(site["stop_codons"] for site in sites) == stops
    assert sum(site["possible_stop_codons"] for site in sites) == possible
    assert pad.count_internal_stop_codons(seq, code) == stops
    preserved = pad.process_record_padding("x", seq, code, "N", "preserve-frame")
    assert preserved["new_seq"] == seq
    assert preserved["is_no_stop"] == (stops == 0)
    if code != 1:
        assert analysis["clean_codon_fraction"] == 1
        candidate = longestcds.choose_best_candidate(seq, code)
        assert candidate.category == "partial" and candidate.output_seq == seq


def test_complete_cds_context_and_definite_to_stop():
    assert translate.translate_sequence_string("ATGGCATGA", 27, True) == "MAW"
    assert (
        translate.translate_sequence_string("ATGGCATGA", 27, False, complete_cds=True)
        == "MA"
    )
    assert (
        translate.translate_sequence_string("GTGAAATGA", 11, False, complete_cds=True)
        == "MK"
    )
    assert translate.translate_sequence_string("ATGTARAAA", 1, True) == "M"
    for seq in ("ATGTARAAATAA", "ATGAAA", "AAATGA", "ATGA", "ATG---TAA"):
        with pytest.raises(CodonTable.TranslationError):
            translate.translate_sequence_string(seq, 1, False, complete_cds=True)


def test_draw_and_plot_do_not_disagree_with_translation():
    for code, codon, expected in [(1, "TAR", "*"), (27, "TGA", "W"), (31, "TAA", "E")]:
        table = get_codon_table_components(code)
        assert (
            _translate_codon_for_msa(
                codon, code, table["forward_table"], table["stop_codons"]
            )
            == expected
        )
        assert classify_codon(codon, code) == (
            "stop" if expected == "*" else "complete"
        )


def test_padding_report_distinguishes_frame_change_from_repair():
    result = pad.process_record_padding("x", "ATGTAAGGG", 1, "N")
    assert result["new_seq"] == "NATGTAAGGGNN"
    assert result["original"]["internal_stop_count"] == 1
    assert result["is_no_stop"]
    assert len(result["candidates"]) == 3
    assert result["tied_candidates_0based"] == [1, 2]
    assert result["candidates"][1]["original_frame_offset"] == 2
    preserved = pad.process_record_padding("x", "ATGTAAGGG", 1, "N", "preserve-frame")
    assert preserved["new_seq"] == "ATGTAAGGG" and not preserved["is_no_stop"]
    assert pad.process_record_padding("x", "ATGTARAAA", 1, "N")["original"]["stop"] == 1


def oracle_candidates(sequence, code):
    """Simple codon-by-codon reference, without production scanners or keys."""
    table = CodonTable.unambiguous_dna_by_id[code]
    stops = set(table.stop_codons) - set(table.forward_table)
    out = []
    for strand, seq in [
        ("+", sequence),
        ("-", str(Seq(sequence).reverse_complement())),
    ]:
        for frame in range(3):
            codons = [seq[i : i + 3] for i in range(frame, len(seq) - 2, 3)]
            for start, codon in enumerate(codons):
                if codon not in table.start_codons:
                    continue
                end = start + 1
                while end < len(codons) and codons[end] not in stops:
                    end += 1
                has_stop = end < len(codons)
                end += int(has_stop)
                out.append(
                    (
                        strand,
                        frame,
                        frame + 3 * start,
                        frame + 3 * end,
                        "complete" if has_stop else "partial",
                    )
                )
            start = 0
            for end in range(len(codons) + 1):
                if end == len(codons) or codons[end] in stops:
                    if end > start:
                        out.append(
                            (
                                strand,
                                frame,
                                frame + 3 * start,
                                frame + 3 * end,
                                "no_start",
                            )
                        )
                    start = end + 1
    return out


@pytest.mark.parametrize("code", [1, 2, 11, 27, 28, 31])
def test_orf_selection_matches_independent_six_frame_oracle(code):
    rng = random.Random(20260910)
    sequences = ["ATGTAA" + "C" * 100, "", "AT", "ATGTARAAA"]
    sequences += [
        "".join(rng.choices("ACGT", k=rng.randrange(3, 150))) for _ in range(80)
    ]
    for sequence in sequences:
        if "R" in sequence:
            # Tested separately because the reference above deliberately uses concrete DNA.
            continue
        expected = oracle_candidates(sequence, code)
        actual = list(longestcds.iter_candidates(sequence, code))
        assert {
            (c.strand, c.frame - 1, c.start_idx, c.end_idx, c.category) for c in actual
        } == set(expected)
        for selection in ("complete-first", "longest"):

            def key(fields, sequence=sequence, selection=selection):
                strand, frame, start, end, category = fields
                rank = {"complete": 2, "partial": 1, "no_start": 0}[category]
                coord = start + 1 if strand == "+" else len(sequence) - end + 1
                primary = (
                    (rank, end - start)
                    if selection == "complete-first"
                    else (end - start, rank)
                )
                return (
                    *primary,
                    int(strand == "+"),
                    -int(category == "complete"),
                    -int(category != "no_start"),
                    -frame,
                    -coord,
                )

            winner = max(expected, key=key, default=None)
            found = longestcds.choose_best_candidate(sequence, code, selection)
            if winner is None:
                assert found is None
            else:
                assert (
                    found.strand,
                    found.frame - 1,
                    found.start_idx,
                    found.end_idx,
                    found.category,
                ) == winner
                lo, hi = found.start_1based - 1, found.end_1based
                source = sequence[lo:hi]
                assert found.output_seq == (
                    source
                    if found.strand == "+"
                    else str(Seq(source).reverse_complement())
                )


def test_orf_ambiguous_stop_and_selection_modes():
    assert longestcds.choose_best_candidate("ATGTARAAA", 1).output_seq == "ATGTAR"
    assert (
        longestcds.choose_best_candidate("ATGTAA" + "C" * 100, 1).output_seq == "ATGTAA"
    )
    assert (
        longestcds.choose_best_candidate("ATGTAA" + "C" * 100, 1, "longest").nt_len > 6
    )
    with pytest.raises(ValueError):
        longestcds.choose_best_candidate("ATG", 1, "invalid")


@pytest.mark.parametrize("command", ["pad", "longestorf", "longestcds"])
@pytest.mark.parametrize("extension", ["json", "tsv"])
def test_cli_reports_are_round_trippable_and_identify_selected_output(
    tmp_path, command, extension
):
    source = tmp_path / "input.fa"
    source.write_text(">x\nATGTAAGGG\n>x\nATGTARAAA\n")
    output = tmp_path / "out.fa"
    report = tmp_path / f"report.{extension}"
    args = psr.parse_args(
        [
            command,
            "--seq_file",
            str(source),
            "--out_file",
            str(output),
            "--report",
            str(report),
        ]
    )
    if command == "pad":
        pad.pad_main(args)
    else:
        longestcds.longestcds_main(args)
    if extension == "json":
        data = json.loads(report.read_text())
        assert data["codon_semantics_version"] == "2"
        rows = data["sequences"]
    else:
        data = read_tsv(report)
        rows = [json.loads(row["data"]) for row in data if row["section"] == "sequence"]
    assert [row["input_order"] for row in rows] == [1, 2]
    from Bio import SeqIO

    outputs = list(SeqIO.parse(output, "fasta"))
    for record, row in zip(outputs, rows, strict=True):
        selected = row["candidates"][row["selected_candidate_0based"]]
        assert (
            str(record.seq) == selected["new_seq" if command == "pad" else "output_seq"]
        )


@pytest.mark.subprocess
def test_process_workers_preserve_new_semantics():
    seqs = ["ATGTARAAA", "ATGTGAAAA"]
    payloads = [(str(i), seq) for i, seq in enumerate(seqs)]
    assert validate.summarize_records_process_parallel(payloads, 27, 2) == [
        validate.summarize_single_sequence(name, seq, 27) for name, seq in payloads
    ]
    assert pad.process_padding_payloads_process_parallel(
        payloads, 1, "N", 2, "preserve-frame"
    ) == [
        pad.process_record_padding(name, seq, 1, "N", "preserve-frame")
        for name, seq in payloads
    ]
    for mode in ("complete-first", "longest"):
        expected = [longestcds.choose_best_candidate(seq, 1, mode) for seq in seqs]
        actual = longestcds.choose_candidates_process_parallel(seqs, 1, 2, mode)
        assert [asdict(x) for x in actual] == [asdict(x) for x in expected]


def test_pad_drop_alias_and_preserved_frame(tmp_path):
    source = tmp_path / "input.fa"
    source.write_text(">x\nATGTARAAA\n")
    for option in ("--drop_pseudo", "--drop_internal_stop"):
        output = tmp_path / "out.fa"
        args = psr.parse_args(
            [
                "pad",
                "--seq_file",
                str(source),
                "--out_file",
                str(output),
                "--mode",
                "preserve-frame",
                option,
                "yes",
            ]
        )
        pad.pad_main(args)
        assert output.read_text() == ""


def test_scalar_and_lut_stop_policies_match():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for code in (1, 27, 28, 31):
            for seq in ("ATGTARAAA", "ATGTGAAAA", "ATGTANAAA", "ATG---???"):
                for to_stop in (False, True):
                    assert translate.translate_sequence_scalar(
                        seq, code, to_stop
                    ) == translate.translate_sequence_string(seq, code, to_stop)
        assert translate.translate_sequence_string("AUGUGAAAA", 27, True) == "MWK"


def test_frozen_refseq_translation_annotations():
    from pathlib import Path

    path = Path(__file__).parents[1] / "fixtures/codon_evaluation/refseq.json"
    records = json.loads(path.read_text())["records"]
    for record in records:
        assert (
            translate.translate_sequence_string(
                record["cds"], record["codon_table"], False, complete_cds=True
            )
            == record["annotated_translation"]
        )


def test_preserve_frame_preserves_original_ambiguity_spelling():
    for sequence in ("ATGX", "atgx", "ATGTARXAA"):
        result = pad.process_record_padding("x", sequence, 1, "N", "preserve-frame")
        assert result["new_seq"] == sequence + "N" * ((-len(sequence)) % 3)
    empty = pad.process_record_padding("x", "", 1, "N")
    assert empty["candidates"][0]["original_start_in_output_1based"] is None
    assert empty["candidates"][0]["original_end_in_output_1based"] is None


@pytest.mark.parametrize(
    "sequence", ["TAAN", "TARN", "TAA", "TAR---", "---TARN", "NNN", ""]
)
def test_legacy_stop_helper_matches_common_terminal_rules(sequence):
    assert validate.has_internal_stop_with_stop_codons(
        sequence, validate.get_stop_codons(1)
    ) == validate.has_internal_stop(sequence, 1)


def test_scalar_invalid_codons_are_not_hidden_by_missing_bases():
    for sequence in ("ATG!-A", "ATG?Z-", "ATGé--"):
        for translator in (
            translate.translate_sequence_string,
            translate.translate_sequence_scalar,
        ):
            with pytest.raises(CodonTable.TranslationError):
                translator(sequence, 1, False)
    assert translate.translate_sequence_scalar("ATGTAR!-A", 1, True) == "M"


def test_x_translation_has_same_meaning_with_a_partial_tail():
    for code in (1, 2, 27, 28, 31):
        for codon in ("TAX", "TGX", "XXX"):
            ordinary = translate.translate_sequence_string("ATG" + codon, code, False)
            assert (
                translate.translate_sequence_string("ATG" + codon + "?", code, False)
                == ordinary + "X"
            )


@pytest.mark.parametrize("command", ["pad", "longestorf"])
def test_direct_calls_reject_report_collisions_before_writing(
    tmp_path, capsys, command
):
    source = tmp_path / "source.fa"
    source.write_text(">x\nATGAAA\n")
    output = tmp_path / "out.fa"
    output.write_text("preserve existing output\n")
    function = pad.pad_main if command == "pad" else longestcds.longestcds_main
    for outfile, report in (
        (str(output), str(source)),
        (str(output), str(output)),
        ("-", "-"),
    ):
        args = psr.parse_args(
            [
                command,
                "--seq_file",
                str(source),
                "--out_file",
                outfile,
                "--report",
                report,
            ]
        )
        with pytest.raises(ValueError):
            function(args)
        assert source.read_text() == ">x\nATGAAA\n"
        assert output.read_text() == "preserve existing output\n"
        assert capsys.readouterr().out == ""


@pytest.mark.parametrize("command", ["pad", "longestorf"])
def test_report_failure_rolls_back_sequence_output(tmp_path, monkeypatch, command):
    source = tmp_path / "source.fa"
    source.write_text(">x\nATGAAA\n")
    output, report = tmp_path / "out.fa", tmp_path / "report.json"
    output.write_text("previous output\n")
    report.write_text("previous report\n")
    module = pad if command == "pad" else longestcds

    def fail(*args, **kwargs):
        raise OSError("simulated report failure")

    monkeypatch.setattr(module, "write_codon_report", fail)
    args = psr.parse_args(
        [
            command,
            "--seq_file",
            str(source),
            "--out_file",
            str(output),
            "--report",
            str(report),
        ]
    )
    with pytest.raises(OSError, match="simulated"):
        (pad.pad_main if command == "pad" else longestcds.longestcds_main)(args)
    assert output.read_text() == "previous output\n"
    assert report.read_text() == "previous report\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "out.fa",
        "report.json",
        "source.fa",
    ]


def test_compact_orf_report_preserves_uncertainty_counts_and_coordinates():
    from cdskit.codonutil import summarize_codons

    for code in (1, 27, 28, 31):
        record = SeqRecord(Seq("NATGTANATGTAATGAX---ATGTARAAA"), id="x")
        for selection in ("complete-first", "longest"):
            report = longestcds.candidate_report(record, code, selection, 1)
            for candidate in report["candidates"]:
                oriented = (
                    str(record.seq)
                    if candidate["strand"] == "+"
                    else str(record.seq.reverse_complement())
                )
                sequence = oriented[candidate["start_idx"] : candidate["end_idx"]]
                summary = summarize_codons(sequence, code, "physical")
                for reported, metric in (
                    ("possible_stop_codons", "possible_stop"),
                    ("context_dependent_codons", "context_dependent"),
                    ("missing_codons", "missing"),
                    ("ambiguous_codons", "ambiguous"),
                ):
                    assert candidate[reported] == summary[metric]
                assert ("output_seq" in candidate) == candidate["selected"]
                if candidate["selected"]:
                    assert candidate["output_seq"] == sequence
            assert [tuple(c["sort_key"]) for c in report["candidates"]] == sorted(
                [tuple(c["sort_key"]) for c in report["candidates"]], reverse=True
            )


def test_legacy_validate_summary_signature_is_preserved():
    record = SeqRecord(Seq("ATGTARAAA"), id="x")
    legacy = validate.summarize_single_record(
        record, stop_codons=validate.get_stop_codons(1)
    )
    assert legacy == ("x", False, False, True, 1, 3)
    assert validate.summarize_single_record(record, codontable=1)[:6] == legacy
    with pytest.raises(ValueError, match="not both"):
        validate.summarize_single_record(record, stop_codons={"TAA"}, codontable=1)


@pytest.mark.parametrize("code", [1, 2, 11, 27, 28, 31])
def test_padding_fast_path_matches_full_report_and_common_counts(code):
    from cdskit.codonutil import summarize_codons

    rng = random.Random(52)
    sequences = ["ATGTARAAA", "TARN", "TAR---", "atgtarx", "ATGTGAAAA", ""]
    sequences.extend(
        "".join(rng.choices("ACGT", k=rng.randrange(1, 100))) for _ in range(50)
    )
    sequences.extend(
        "".join(rng.choices("ACGTRYNX-?.", k=rng.randrange(1, 100))) for _ in range(50)
    )
    for sequence in sequences:
        assert (
            pad.count_internal_stop_codons(sequence, code)
            == summarize_codons(sequence, code, "physical")["internal_stop_count"]
        )
        for mode in ("min-stop", "preserve-frame"):
            for padchar in ("N", "-"):
                full = pad.process_record_padding("x", sequence, code, padchar, mode)
                fast = pad.process_record_padding(
                    "x", sequence, code, padchar, mode, include_report=False
                )
                assert fast == {key: full[key] for key in fast}

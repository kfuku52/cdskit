"""
Tests for cdskit localize and localize-learn commands.
"""

import csv
from pathlib import Path
from urllib import parse as urllib_parse

import Bio.SeqIO
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.localize import localize_main
import cdskit.localize_learn as localize_learn_module
from cdskit.localize_learn import localize_learn_main
from cdskit.localize_model import infer_labels_from_uniprot_cc, load_localize_model


try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


AA_TO_CODON = {
    'A': 'GCT',
    'C': 'TGT',
    'D': 'GAT',
    'E': 'GAA',
    'F': 'TTT',
    'G': 'GGT',
    'H': 'CAT',
    'I': 'ATT',
    'K': 'AAA',
    'L': 'CTT',
    'M': 'ATG',
    'N': 'AAT',
    'P': 'CCT',
    'Q': 'CAA',
    'R': 'CGT',
    'S': 'TCT',
    'T': 'ACT',
    'V': 'GTT',
    'W': 'TGG',
    'Y': 'TAT',
}


def aa_to_cds(aa_seq):
    return ''.join([AA_TO_CODON[aa] for aa in aa_seq])


def build_training_table(path):
    rows = [
        {
            'id': 'noTP_perox',
            'sequence': aa_to_cds('MGPVNQDEGPVNQDEGPVNQDESKL'),
            'localization': 'noTP',
            'peroxisome': 'yes',
        },
        {
            'id': 'noTP_plain',
            'sequence': aa_to_cds('MAGPVNQDEGPVNQDEGATNVQDE'),
            'localization': 'noTP',
            'peroxisome': 'no',
        },
        {
            'id': 'SP_1',
            'sequence': aa_to_cds('MKKLLLLLLLLLLAVAVAASAASA'),
            'localization': 'SP',
            'peroxisome': 'no',
        },
        {
            'id': 'mTP_1',
            'sequence': aa_to_cds('MRRKRRAARAKRRNQAAARRRAA'),
            'localization': 'mTP',
            'peroxisome': 'no',
        },
        {
            'id': 'cTP_1',
            'sequence': aa_to_cds('MSTSTSTTSTASSSAATSTASSTT'),
            'localization': 'cTP',
            'peroxisome': 'no',
        },
        {
            'id': 'lTP_1',
            'sequence': aa_to_cds('MARRVAAARRLLLLLVVVVVAAST'),
            'localization': 'lTP',
            'peroxisome': 'no',
        },
    ]
    with open(path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=['id', 'sequence', 'localization', 'peroxisome'],
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def build_training_table_for_cv(path):
    rows = [
        {
            'id': 'noTP_1',
            'sequence': aa_to_cds('MGPVNQDEGPVNQDEGPVNQDESKL'),
            'localization': 'noTP',
            'peroxisome': 'yes',
        },
        {
            'id': 'noTP_2',
            'sequence': aa_to_cds('MAGPVNQDEGPVNQDEGATNVQDE'),
            'localization': 'noTP',
            'peroxisome': 'no',
        },
        {
            'id': 'SP_1',
            'sequence': aa_to_cds('MKKLLLLLLLLLLAVAVAASAASA'),
            'localization': 'SP',
            'peroxisome': 'no',
        },
        {
            'id': 'SP_2',
            'sequence': aa_to_cds('MKKLLLLLLLLLLAAVVAASAASA'),
            'localization': 'SP',
            'peroxisome': 'no',
        },
        {
            'id': 'mTP_1',
            'sequence': aa_to_cds('MRRKRRAARAKRRNQAAARRRAA'),
            'localization': 'mTP',
            'peroxisome': 'no',
        },
        {
            'id': 'mTP_2',
            'sequence': aa_to_cds('MRRKRRASRAKRRNQAAARRRAA'),
            'localization': 'mTP',
            'peroxisome': 'no',
        },
        {
            'id': 'cTP_1',
            'sequence': aa_to_cds('MSTSTSTTSTASSSAATSTASSTT'),
            'localization': 'cTP',
            'peroxisome': 'no',
        },
        {
            'id': 'cTP_2',
            'sequence': aa_to_cds('MSTSTASTSTASSSAATSTASSTT'),
            'localization': 'cTP',
            'peroxisome': 'no',
        },
        {
            'id': 'lTP_1',
            'sequence': aa_to_cds('MARRVAAARRLLLLLVVVVVAAST'),
            'localization': 'lTP',
            'peroxisome': 'no',
        },
        {
            'id': 'lTP_2',
            'sequence': aa_to_cds('MARRVAAARRLLLLLIVVVVAAST'),
            'localization': 'lTP',
            'peroxisome': 'no',
        },
    ]
    with open(path, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=['id', 'sequence', 'localization', 'peroxisome'],
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def train_test_model(temp_dir, mock_args):
    train_tsv = temp_dir / 'train_localize.tsv'
    model_path = temp_dir / 'localize_model.json'
    build_training_table(train_tsv)

    args = mock_args(
        training_tsv=str(train_tsv),
        model_out=str(model_path),
        report='',
        seq_col='sequence',
        seqtype='dna',
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
        threads=1,
    )
    localize_learn_main(args)
    return model_path


class TestLocalizeMain:
    def test_localize_learn_and_predict_explicit(self, temp_dir, mock_args):
        model_path = train_test_model(temp_dir=temp_dir, mock_args=mock_args)
        model = load_localize_model(str(model_path))
        assert model['model_type'] == 'nearest_centroid_v1'

        input_path = temp_dir / 'predict_input.fasta'
        output_path = temp_dir / 'predict_output.tsv'
        records = [
            SeqRecord(Seq(aa_to_cds('MGPVNQDEGPVNQDEGPVNQDESKL')), id='seq_perox', description=''),
            SeqRecord(Seq(aa_to_cds('MKKLLLLLLLLLLAVAVAASAASA')), id='seq_sp', description=''),
            SeqRecord(Seq(aa_to_cds('MRRKRRAARAKRRNQAAARRRAA')), id='seq_mtp', description=''),
            SeqRecord(Seq(aa_to_cds('MSTSTSTTSTASSSAATSTASSTT')), id='seq_ctp', description=''),
            SeqRecord(Seq(aa_to_cds('MARRVAAARRLLLLLVVVVVAAST')), id='seq_ltp', description=''),
        ]
        Bio.SeqIO.write(records, str(input_path), 'fasta')

        args = mock_args(
            seqfile=str(input_path),
            inseqformat='fasta',
            codontable=1,
            model=str(model_path),
            report=str(output_path),
            include_features=False,
            threads=1,
        )
        localize_main(args)

        with open(output_path, 'r', encoding='utf-8') as inp:
            reader = csv.DictReader(inp, delimiter='\t')
            out_rows = list(reader)

        assert len(out_rows) == 5
        pred_map = {row['seq_id']: row for row in out_rows}
        assert pred_map['seq_sp']['predicted_class'] == 'SP'
        assert pred_map['seq_mtp']['predicted_class'] == 'mTP'
        assert pred_map['seq_ctp']['predicted_class'] == 'cTP'
        assert pred_map['seq_ltp']['predicted_class'] == 'lTP'
        assert pred_map['seq_perox']['perox_signal_type'] in ['PTS1', 'PTS2', 'none']
        assert float(pred_map['seq_perox']['p_peroxisome']) >= 0.0
        assert float(pred_map['seq_perox']['p_peroxisome']) <= 1.0

    def test_localize_rejects_non_triplet_input(self, temp_dir, mock_args):
        model_path = train_test_model(temp_dir=temp_dir, mock_args=mock_args)
        input_path = temp_dir / 'bad_len.fasta'
        records = [SeqRecord(Seq('ATGAAATG'), id='bad_len', description='')]
        Bio.SeqIO.write(records, str(input_path), 'fasta')

        args = mock_args(
            seqfile=str(input_path),
            inseqformat='fasta',
            codontable=1,
            model=str(model_path),
            report='-',
            include_features=False,
            threads=1,
        )
        with pytest.raises(Exception) as exc_info:
            localize_main(args)
        assert 'multiple of three' in str(exc_info.value)

    def test_localize_rejects_internal_stop(self, temp_dir, mock_args):
        model_path = train_test_model(temp_dir=temp_dir, mock_args=mock_args)
        input_path = temp_dir / 'internal_stop.fasta'
        records = [SeqRecord(Seq('ATGTAAATG'), id='internal_stop', description='')]
        Bio.SeqIO.write(records, str(input_path), 'fasta')

        args = mock_args(
            seqfile=str(input_path),
            inseqformat='fasta',
            codontable=1,
            model=str(model_path),
            report='-',
            include_features=False,
            threads=1,
        )
        with pytest.raises(Exception) as exc_info:
            localize_main(args)
        assert 'Internal stop codon' in str(exc_info.value)


@pytest.mark.skipif(not HAS_TORCH, reason='torch is required for bilstm_attention test')
def test_localize_learn_and_predict_bilstm_attention(temp_dir, mock_args):
    train_tsv = temp_dir / 'train_localize_bilstm.tsv'
    model_path = temp_dir / 'localize_model_bilstm.pt'
    build_training_table_for_cv(train_tsv)

    args_train = mock_args(
        training_tsv=str(train_tsv),
        model_out=str(model_path),
        report='',
        seq_col='sequence',
        seqtype='dna',
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
        model_arch='bilstm_attention',
        dl_seq_len=60,
        dl_embed_dim=8,
        dl_hidden_dim=8,
        dl_num_layers=1,
        dl_dropout=0.1,
        dl_epochs=2,
        dl_batch_size=8,
        dl_lr=1e-3,
        dl_weight_decay=0.0,
        dl_class_weight=True,
        dl_seed=1,
        dl_device='cpu',
        cv_folds=0,
        cv_seed=1,
        threads=1,
    )
    localize_learn_main(args_train)
    model = load_localize_model(str(model_path))
    assert model['model_type'] == 'bilstm_attention_v1'

    input_path = temp_dir / 'predict_input_bilstm.fasta'
    output_path = temp_dir / 'predict_output_bilstm.tsv'
    records = [
        SeqRecord(Seq(aa_to_cds('MGPVNQDEGPVNQDEGPVNQDESKL')), id='seq_perox', description=''),
        SeqRecord(Seq(aa_to_cds('MKKLLLLLLLLLLAVAVAASAASA')), id='seq_sp', description=''),
        SeqRecord(Seq(aa_to_cds('MRRKRRAARAKRRNQAAARRRAA')), id='seq_mtp', description=''),
    ]
    Bio.SeqIO.write(records, str(input_path), 'fasta')

    args_pred = mock_args(
        seqfile=str(input_path),
        inseqformat='fasta',
        codontable=1,
        model=str(model_path),
        report=str(output_path),
        include_features=False,
        threads=1,
    )
    localize_main(args_pred)

    with open(output_path, 'r', encoding='utf-8') as inp:
        rows = list(csv.DictReader(inp, delimiter='\t'))
    assert len(rows) == 3
    valid = {'noTP', 'SP', 'mTP', 'cTP', 'lTP'}
    for row in rows:
        assert row['predicted_class'] in valid


@pytest.mark.skipif(not HAS_TORCH, reason='torch is required for bilstm_attention test')
def test_localize_learn_bilstm_cross_validation_metrics(temp_dir, mock_args):
    train_tsv = temp_dir / 'train_localize_bilstm_cv.tsv'
    model_path = temp_dir / 'localize_model_bilstm_cv.pt'
    report_path = temp_dir / 'localize_report_bilstm_cv.tsv'
    build_training_table_for_cv(train_tsv)

    args = mock_args(
        training_tsv=str(train_tsv),
        model_out=str(model_path),
        report=str(report_path),
        seq_col='sequence',
        seqtype='dna',
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
        model_arch='bilstm_attention',
        dl_seq_len=60,
        dl_embed_dim=8,
        dl_hidden_dim=8,
        dl_num_layers=1,
        dl_dropout=0.1,
        dl_epochs=1,
        dl_batch_size=8,
        dl_lr=1e-3,
        dl_weight_decay=0.0,
        dl_class_weight=True,
        dl_seed=2,
        dl_device='cpu',
        cv_folds=2,
        cv_seed=11,
        threads=1,
    )
    localize_learn_main(args)
    assert model_path.exists()
    assert report_path.exists()
    with open(report_path, 'r', encoding='utf-8') as inp:
        rows = list(csv.DictReader(inp, delimiter='\t'))
    metrics = {row['metric']: float(row['value']) for row in rows}
    assert metrics['cv_folds'] == 2.0
    assert 0.0 <= metrics['cv_class_accuracy_mean'] <= 1.0
    assert 0.0 <= metrics['cv_perox_accuracy_mean'] <= 1.0


def test_uniprot_cc_label_inference():
    cls, perox, ambiguous = infer_labels_from_uniprot_cc(
        'SUBCELLULAR LOCATION: Chloroplast; Thylakoid lumen. Peroxisome.'
    )
    assert cls == 'lTP'
    assert perox == 'yes'
    assert ambiguous is False


def test_localize_learn_uniprot_download_mocked(monkeypatch, temp_dir, mock_args):
    model_path = temp_dir / 'localize_model_from_uniprot.json'
    downloaded_tsv = temp_dir / 'downloaded_uniprot.tsv'

    page1 = (
        'Entry\tSequence\tSubcellular location [CC]\n'
        'U1\tMAGPVNQDEGPVNQDEGPVNQDESKL\tPeroxisome.\n'
        'U2\tMKKLLLLLLLLLLAVAVAASAASA\tSecreted.\n'
        'U3\tMRRKRRAARAKRRNQAAARRRAA\tMitochondrion.\n'
    )
    page2 = (
        'Entry\tSequence\tSubcellular location [CC]\n'
        'U4\tMSTSTSTTSTASSSAATSTASSTT\tChloroplast.\n'
        'U5\tMARRVAAARRLLLLLVVVVVAAST\tThylakoid lumen.\n'
    )
    calls = {'n': 0}

    def fake_fetch(url, timeout_sec, retries):
        calls['n'] += 1
        if calls['n'] == 1:
            return page1, '<https://rest.uniprot.org/uniprotkb/search?cursor=next>; rel="next"'
        return page2, ''

    monkeypatch.setattr(localize_learn_module, '_fetch_url_text', fake_fetch)

    args = mock_args(
        training_tsv='',
        uniprot_query='taxonomy_id:3702',
        uniprot_preset='none',
        uniprot_reviewed=True,
        uniprot_exclude_fragments=True,
        uniprot_fields='accession,sequence,cc_subcellular_location',
        uniprot_page_size=500,
        uniprot_max_rows=0,
        uniprot_sampling='head',
        uniprot_sampling_seed=1,
        uniprot_timeout_sec=60,
        uniprot_retries=1,
        uniprot_out_tsv=str(downloaded_tsv),
        model_out=str(model_path),
        report='',
        seq_col='sequence',
        seqtype='protein',
        label_mode='uniprot_cc',
        localization_col='cc_subcellular_location',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
        threads=1,
    )
    localize_learn_main(args)

    assert calls['n'] == 2
    assert model_path.exists()
    assert downloaded_tsv.exists()
    model = load_localize_model(str(model_path))
    assert model['metadata']['data_source'] == 'uniprot_query'
    assert model['metadata']['uniprot_sampling'] == 'head'
    assert model['metadata']['uniprot_sampling_seed'] == 1
    assert model['metadata']['num_training_rows'] == 5
    assert model['metadata']['num_used_rows'] == 5
    assert model['metadata']['class_counts']['noTP'] == 1
    assert model['metadata']['class_counts']['SP'] == 1
    assert model['metadata']['class_counts']['mTP'] == 1
    assert model['metadata']['class_counts']['cTP'] == 1
    assert model['metadata']['class_counts']['lTP'] == 1


def test_localize_learn_requires_single_data_source(mock_args):
    args_none = mock_args(
        training_tsv='',
        uniprot_query='',
        uniprot_preset='none',
        model_out='dummy.json',
        seq_col='sequence',
        seqtype='protein',
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
    )
    with pytest.raises(Exception) as exc_info_none:
        localize_learn_main(args_none)
    assert 'Either --training_tsv or --uniprot_query' in str(exc_info_none.value)

    args_both = mock_args(
        training_tsv='in.tsv',
        uniprot_query='taxonomy_id:3702',
        uniprot_preset='none',
        model_out='dummy.json',
        seq_col='sequence',
        seqtype='protein',
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
    )
    with pytest.raises(Exception) as exc_info_both:
        localize_learn_main(args_both)
    assert 'Use either --training_tsv or --uniprot_query' in str(exc_info_both.value)


def test_localize_learn_uniprot_preset_combines_query(monkeypatch, temp_dir, mock_args):
    model_path = temp_dir / 'localize_model_preset.json'
    captured = {'query': ''}
    page = (
        'Entry\tSequence\tSubcellular location [CC]\n'
        'U1\tMAGPVNQDEGPVNQDEGPVNQDESKL\tPeroxisome.\n'
        'U2\tMKKLLLLLLLLLLAVAVAASAASA\tSecreted.\n'
        'U3\tMRRKRRAARAKRRNQAAARRRAA\tMitochondrion.\n'
        'U4\tMSTSTSTTSTASSSAATSTASSTT\tChloroplast.\n'
        'U5\tMARRVAAARRLLLLLVVVVVAAST\tThylakoid lumen.\n'
    )

    def fake_fetch(url, timeout_sec, retries):
        parsed = urllib_parse.urlparse(url)
        q = urllib_parse.parse_qs(parsed.query)
        captured['query'] = urllib_parse.unquote_plus(q.get('query', [''])[0])
        return page, ''

    monkeypatch.setattr(localize_learn_module, '_fetch_url_text', fake_fetch)

    args = mock_args(
        training_tsv='',
        uniprot_query='keyword:Transit peptide',
        uniprot_preset='viridiplantae',
        uniprot_reviewed=True,
        uniprot_exclude_fragments=True,
        uniprot_fields='accession,sequence,cc_subcellular_location',
        uniprot_page_size=500,
        uniprot_max_rows=0,
        uniprot_sampling='head',
        uniprot_sampling_seed=1,
        uniprot_timeout_sec=60,
        uniprot_retries=1,
        uniprot_out_tsv='',
        model_out=str(model_path),
        report='',
        seq_col='sequence',
        seqtype='protein',
        label_mode='uniprot_cc',
        localization_col='cc_subcellular_location',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
        threads=1,
    )
    localize_learn_main(args)

    query_text = captured['query']
    assert 'taxonomy_id:33090' in query_text
    assert 'keyword:Transit peptide' in query_text
    assert 'reviewed:true' in query_text
    assert 'fragment:true' in query_text


def test_fetch_uniprot_training_rows_head_vs_random_sampling(monkeypatch):
    page1 = (
        'Entry\tSequence\tSubcellular location [CC]\n'
        'U1\tMAAA\tSecreted.\n'
        'U2\tMBBB\tMitochondrion.\n'
        'U3\tMCCC\tPeroxisome.\n'
    )
    page2 = (
        'Entry\tSequence\tSubcellular location [CC]\n'
        'U4\tMDDD\tChloroplast.\n'
        'U5\tMEEE\tThylakoid lumen.\n'
        'U6\tMFFF\tCytoplasm.\n'
    )

    calls = {'n': 0}

    def fake_fetch(url, timeout_sec, retries):
        calls['n'] += 1
        if calls['n'] == 1:
            return page1, '<https://rest.uniprot.org/uniprotkb/search?cursor=next>; rel="next"'
        return page2, ''

    monkeypatch.setattr(localize_learn_module, '_fetch_url_text', fake_fetch)

    head_rows = localize_learn_module.fetch_uniprot_training_rows(
        query='taxonomy_id:3702',
        fields=['accession', 'sequence', 'cc_subcellular_location'],
        reviewed=True,
        exclude_fragments=True,
        page_size=500,
        max_rows=2,
        timeout_sec=60,
        retries=1,
        sampling_mode='head',
        sampling_seed=1,
    )
    assert calls['n'] == 1
    assert [row['accession'] for row in head_rows] == ['U1', 'U2']

    calls['n'] = 0
    random_rows_a = localize_learn_module.fetch_uniprot_training_rows(
        query='taxonomy_id:3702',
        fields=['accession', 'sequence', 'cc_subcellular_location'],
        reviewed=True,
        exclude_fragments=True,
        page_size=500,
        max_rows=2,
        timeout_sec=60,
        retries=1,
        sampling_mode='random',
        sampling_seed=11,
    )
    assert calls['n'] == 2
    assert len(random_rows_a) == 2
    assert set([row['accession'] for row in random_rows_a]).issubset(
        {'U1', 'U2', 'U3', 'U4', 'U5', 'U6'}
    )

    calls['n'] = 0
    random_rows_b = localize_learn_module.fetch_uniprot_training_rows(
        query='taxonomy_id:3702',
        fields=['accession', 'sequence', 'cc_subcellular_location'],
        reviewed=True,
        exclude_fragments=True,
        page_size=500,
        max_rows=2,
        timeout_sec=60,
        retries=1,
        sampling_mode='random',
        sampling_seed=11,
    )
    assert [row['accession'] for row in random_rows_a] == [row['accession'] for row in random_rows_b]


def test_fetch_uniprot_training_rows_invalid_sampling_mode(monkeypatch):
    def fake_fetch(url, timeout_sec, retries):
        raise AssertionError('HTTP fetch should not be called for invalid sampling mode.')

    monkeypatch.setattr(localize_learn_module, '_fetch_url_text', fake_fetch)
    with pytest.raises(ValueError) as exc_info:
        localize_learn_module.fetch_uniprot_training_rows(
            query='taxonomy_id:3702',
            fields=['accession', 'sequence', 'cc_subcellular_location'],
            reviewed=True,
            exclude_fragments=True,
            page_size=500,
            max_rows=2,
            timeout_sec=60,
            retries=1,
            sampling_mode='unsupported',
            sampling_seed=1,
        )
    assert '--uniprot_sampling should be head or random' in str(exc_info.value)


def test_localize_learn_uniprot_cc_adds_cc_field_not_localization(monkeypatch, temp_dir, mock_args):
    model_path = temp_dir / 'localize_model_ccfield.json'
    captured = {'fields': ''}
    page = (
        'Entry\tSequence\tSubcellular location [CC]\n'
        'U1\tMAGPVNQDEGPVNQDEGPVNQDESKL\tPeroxisome.\n'
        'U2\tMKKLLLLLLLLLLAVAVAASAASA\tSecreted.\n'
        'U3\tMRRKRRAARAKRRNQAAARRRAA\tMitochondrion.\n'
        'U4\tMSTSTSTTSTASSSAATSTASSTT\tChloroplast.\n'
        'U5\tMARRVAAARRLLLLLVVVVVAAST\tThylakoid lumen.\n'
    )

    def fake_fetch(url, timeout_sec, retries):
        parsed = urllib_parse.urlparse(url)
        q = urllib_parse.parse_qs(parsed.query)
        captured['fields'] = urllib_parse.unquote_plus(q.get('fields', [''])[0])
        return page, ''

    monkeypatch.setattr(localize_learn_module, '_fetch_url_text', fake_fetch)

    args = mock_args(
        training_tsv='',
        uniprot_query='taxonomy_id:3702',
        uniprot_preset='none',
        uniprot_reviewed=True,
        uniprot_exclude_fragments=True,
        uniprot_fields='accession,sequence',
        uniprot_page_size=500,
        uniprot_max_rows=0,
        uniprot_sampling='head',
        uniprot_sampling_seed=1,
        uniprot_timeout_sec=60,
        uniprot_retries=1,
        uniprot_out_tsv='',
        model_out=str(model_path),
        report='',
        seq_col='sequence',
        seqtype='protein',
        label_mode='uniprot_cc',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
        threads=1,
    )
    localize_learn_main(args)

    requested_fields = captured['fields'].split(',')
    assert 'cc_subcellular_location' in requested_fields
    assert 'localization' not in requested_fields


def test_localize_learn_uniprot_explicit_requires_label_fields(monkeypatch, temp_dir, mock_args):
    model_path = temp_dir / 'localize_model_explicit_missing_cols.json'

    def fail_fetch(url, timeout_sec, retries):
        raise AssertionError('UniProt HTTP fetch should not be called in this validation test.')

    monkeypatch.setattr(localize_learn_module, '_fetch_url_text', fail_fetch)

    args = mock_args(
        training_tsv='',
        uniprot_query='taxonomy_id:3702',
        uniprot_preset='none',
        uniprot_reviewed=True,
        uniprot_exclude_fragments=True,
        uniprot_fields='accession,sequence,cc_subcellular_location',
        uniprot_page_size=500,
        uniprot_max_rows=0,
        uniprot_sampling='head',
        uniprot_sampling_seed=1,
        uniprot_timeout_sec=60,
        uniprot_retries=1,
        uniprot_out_tsv='',
        model_out=str(model_path),
        report='',
        seq_col='sequence',
        seqtype='protein',
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
        threads=1,
    )
    with pytest.raises(ValueError) as exc_info:
        localize_learn_main(args)
    message = str(exc_info.value)
    assert '--uniprot_fields should include' in message
    assert 'localization' in message
    assert 'peroxisome' in message


def test_localize_learn_cross_validation_metrics(temp_dir, mock_args):
    train_tsv = temp_dir / 'train_localize_cv.tsv'
    model_path = temp_dir / 'localize_model_cv.json'
    report_path = temp_dir / 'localize_cv_report.tsv'
    build_training_table_for_cv(train_tsv)

    args = mock_args(
        training_tsv=str(train_tsv),
        model_out=str(model_path),
        report=str(report_path),
        seq_col='sequence',
        seqtype='dna',
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
        cv_folds=2,
        cv_seed=11,
        threads=1,
    )
    localize_learn_main(args)

    assert model_path.exists()
    assert report_path.exists()
    model = load_localize_model(str(model_path))
    assert model['metadata']['cv_folds'] == 2
    assert model['metadata']['cv_seed'] == 11
    assert 0.0 <= model['metadata']['cv_class_accuracy_mean'] <= 1.0
    assert 0.0 <= model['metadata']['cv_perox_accuracy_mean'] <= 1.0
    for class_name in ['noTP', 'SP', 'mTP', 'cTP', 'lTP']:
        assert class_name in model['metadata']['class_train_accuracy_by_class']
        assert class_name in model['metadata']['cv_class_accuracy_by_class']
        assert 0.0 <= model['metadata']['class_train_accuracy_by_class'][class_name] <= 1.0
        assert 0.0 <= model['metadata']['cv_class_accuracy_by_class'][class_name] <= 1.0

    with open(report_path, 'r', encoding='utf-8') as inp:
        reader = csv.DictReader(inp, delimiter='\t')
        rows = list(reader)
    metrics = {row['metric']: float(row['value']) for row in rows}
    assert metrics['cv_folds'] == 2.0
    assert 0.0 <= metrics['cv_class_accuracy_mean'] <= 1.0
    assert 0.0 <= metrics['cv_perox_accuracy_mean'] <= 1.0
    assert 'cv_fold1_class_accuracy' in metrics
    assert 'cv_fold2_class_accuracy' in metrics
    for class_name in ['noTP', 'SP', 'mTP', 'cTP', 'lTP']:
        assert 'class_train_accuracy_{}'.format(class_name) in metrics
        assert 'cv_class_accuracy_{}'.format(class_name) in metrics
        assert 0.0 <= metrics['class_train_accuracy_{}'.format(class_name)] <= 1.0
        assert 0.0 <= metrics['cv_class_accuracy_{}'.format(class_name)] <= 1.0


def test_localize_learn_cross_validation_allows_missing_classes(temp_dir, mock_args):
    if not HAS_TORCH:
        pytest.skip('torch is not available')
    train_tsv = temp_dir / 'train_localize_small.tsv'
    model_path = temp_dir / 'localize_model_small.json'
    rows = [
        {'id': 'noTP_1', 'sequence': aa_to_cds('MGPVNQDEGPVNQDEGPVNQDESKL'), 'localization': 'noTP', 'peroxisome': 'yes'},
        {'id': 'noTP_2', 'sequence': aa_to_cds('MAGPVNQDEGPVNQDEGATNVQDE'), 'localization': 'noTP', 'peroxisome': 'no'},
        {'id': 'SP_1', 'sequence': aa_to_cds('MKKLLLLLLLLLLAVAVAASAASA'), 'localization': 'SP', 'peroxisome': 'no'},
        {'id': 'SP_2', 'sequence': aa_to_cds('MKKLLLLLLLLLLAAVVAASAASA'), 'localization': 'SP', 'peroxisome': 'no'},
        {'id': 'mTP_1', 'sequence': aa_to_cds('MRRKRRAARAKRRNQAAARRRAA'), 'localization': 'mTP', 'peroxisome': 'no'},
        {'id': 'mTP_2', 'sequence': aa_to_cds('MRRKRRASRAKRRNQAAARRRAA'), 'localization': 'mTP', 'peroxisome': 'no'},
        {'id': 'lTP_1', 'sequence': aa_to_cds('MARRVAAARRLLLLLVVVVVAAST'), 'localization': 'lTP', 'peroxisome': 'no'},
        {'id': 'lTP_2', 'sequence': aa_to_cds('MARRVAAARRLLLLLIVVVVAAST'), 'localization': 'lTP', 'peroxisome': 'no'},
    ]
    with open(train_tsv, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(
            out,
            fieldnames=['id', 'sequence', 'localization', 'peroxisome'],
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    args = mock_args(
        training_tsv=str(train_tsv),
        model_out=str(model_path),
        report='',
        seq_col='sequence',
        seqtype='dna',
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
        cv_folds=2,
        cv_seed=1,
        model_arch='bilstm_attention',
        dl_seq_len=60,
        dl_embed_dim=8,
        dl_hidden_dim=8,
        dl_num_layers=1,
        dl_dropout=0.1,
        dl_epochs=1,
        dl_batch_size=8,
        dl_lr=1e-3,
        dl_weight_decay=0.0,
        dl_class_weight=True,
        dl_seed=1,
        dl_device='cpu',
        threads=1,
    )
    localize_learn_main(args)
    model = load_localize_model(str(model_path))
    assert model['metadata']['cv_folds'] == 2
    assert 'cv_class_accuracy_by_class' in model['metadata']
    assert model['metadata']['class_counts'].get('cTP', 0) == 0


def test_localize_learn_cross_validation_requires_two_samples_per_observed_class(temp_dir, mock_args):
    train_tsv = temp_dir / 'train_localize_small_insufficient.tsv'
    model_path = temp_dir / 'localize_model_small_insufficient.json'
    build_training_table(train_tsv)

    args = mock_args(
        training_tsv=str(train_tsv),
        model_out=str(model_path),
        report='',
        seq_col='sequence',
        seqtype='dna',
        label_mode='explicit',
        localization_col='localization',
        perox_col='peroxisome',
        skip_ambiguous=True,
        codontable=1,
        cv_folds=2,
        cv_seed=1,
        threads=1,
    )
    with pytest.raises(ValueError) as exc_info:
        localize_learn_main(args)
    assert 'Cross validation requires at least 2 samples for each observed class' in str(exc_info.value)

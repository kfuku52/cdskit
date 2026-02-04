"""
Pytest configuration and fixtures for CDSKIT tests.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# Path to test data directory
DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def data_dir():
    """Return path to test data directory."""
    return DATA_DIR


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def temp_fasta(temp_dir):
    """Create a temporary FASTA file with simple sequences."""
    fasta_path = temp_dir / "test.fasta"
    records = [
        SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
        SeqRecord(Seq("ATGCCCTGA"), id="seq2", description=""),
    ]
    Bio.SeqIO.write(records, str(fasta_path), "fasta")
    return fasta_path


@pytest.fixture
def aligned_fasta(temp_dir):
    """Create a temporary aligned FASTA file."""
    fasta_path = temp_dir / "aligned.fasta"
    records = [
        SeqRecord(Seq("ATGAAA---TGA"), id="seq1", description=""),
        SeqRecord(Seq("ATGCCCTCCTGA"), id="seq2", description=""),
        SeqRecord(Seq("ATG---TCCTGA"), id="seq3", description=""),
    ]
    Bio.SeqIO.write(records, str(fasta_path), "fasta")
    return fasta_path


@pytest.fixture
def unpadded_fasta(temp_dir):
    """Create a FASTA file with sequences not multiple of 3."""
    fasta_path = temp_dir / "unpadded.fasta"
    records = [
        SeqRecord(Seq("ATGAAAT"), id="miss_2nt", description=""),  # 7 nt, needs 2 padding
        SeqRecord(Seq("ATGAAATG"), id="miss_1nt", description=""),  # 8 nt, needs 1 padding
        SeqRecord(Seq("ATGAAATGA"), id="complete", description=""),  # 9 nt, complete
    ]
    Bio.SeqIO.write(records, str(fasta_path), "fasta")
    return fasta_path


@pytest.fixture
def gff_file(temp_dir):
    """Create a simple GFF file for testing."""
    gff_path = temp_dir / "test.gff"
    gff_content = """##gff-version 3
seq1\tsource\tgene\t1\t100\t.\t+\t.\tID=gene1
seq1\tsource\texon\t10\t50\t.\t+\t.\tID=exon1
seq1\tsource\texon\t60\t90\t.\t+\t.\tID=exon2
"""
    gff_path.write_text(gff_content)
    return gff_path


class MockArgs:
    """Mock argument object for testing command functions."""
    def __init__(self, **kwargs):
        self.seqfile = '-'
        self.outfile = '-'
        self.inseqformat = 'fasta'
        self.outseqformat = 'fasta'
        self.codontable = 1
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_args():
    """Factory fixture for creating mock argument objects."""
    def _mock_args(**kwargs):
        return MockArgs(**kwargs)
    return _mock_args

"""
Tests for codon-level utility helpers.
"""

from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.codonutil import codon_is_clean


class TestCodonIsClean:
    """Tests for codon_is_clean."""

    @pytest.mark.parametrize(
        "codon, expected",
        [
            ("ATG", True),
            ("atg", True),
            ("AAA", True),
            ("---", False),
            ("???", False),
            ("...", False),
            ("AT-", False),
            ("AT?", False),
            ("AT.", False),
            ("ATN", False),
            ("TGY", False),
            ("TAA", False),
            ("TAG", False),
            ("TGA", False),
        ],
    )
    def test_standard_code_clean_codon_classification(self, codon, expected):
        assert codon_is_clean(codon=codon, codontable=1) is expected

    def test_respects_selected_codon_table(self):
        assert codon_is_clean(codon="TGA", codontable=1) is False
        assert codon_is_clean(codon="TGA", codontable=2) is True

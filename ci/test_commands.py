import re
import subprocess as sp
from pathlib import Path
from Bio import SeqIO

def run(args, *, cwd=None):
    """subprocessの薄いラッパ（stdout/stderrを確保し、失敗時には詳細を出す）"""
    cp = sp.run(args, capture_output=True, text=True, cwd=cwd)
    if cp.returncode != 0:
        raise RuntimeError(f"cmd failed: {args}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}")
    return cp

# --- pad ---------------------------------------------------------------------
# pad: 出力長がすべて3の倍数（フレーム整合）。
def test_pad_makes_inframe(tmp_path: Path):
    # 入力（どちらも3の倍数ではない）
    in_fa = tmp_path / "in.fa"
    in_fa.write_text(">s1\nATGAA\n>s2\nATGA\n")

    out_fa = tmp_path / "out.fa"
    run(["cdskit", "pad", "--seqfile", str(in_fa), "--outfile", str(out_fa)])

    recs = list(SeqIO.parse(out_fa, "fasta"))
    assert len(recs) == 2
    assert all(len(r.seq) % 3 == 0 for r in recs)  # すべて3の倍数になっている

# --- mask --------------------------------------------------------------------
# mask: ストップ/曖昧塩基が N に置換される（Wikiの例そのまま）。
def test_mask_replaces_stops_and_ambiguous(tmp_path: Path):
    # Wikiの例を最小にしたもの
    in_fa = tmp_path / "in.fa"
    in_fa.write_text(
        ">stop\n---ATGTAAATTATGTTGAAG---\n"
        ">amb1\n---ATGTNAATTATGTTGAAG---\n"
        ">amb2\n---ATGT-AATTATGTTGAAG---\n"
        ">all\n---ATGTAAATT--GTTGANG---\n"
    )
    out_fa = tmp_path / "out.fa"
    run(["cdskit", "mask", "--seqfile", str(in_fa), "--outfile", str(out_fa)])

    got = {r.id: str(r.seq) for r in SeqIO.parse(out_fa, "fasta")}
    # 期待（Wikiの出力例と一致）:
    assert got["stop"] ==   "---ATGNNNATTATGTTGAAG---"
    assert got["amb1"] ==   "---ATGNNNATTATGTTGAAG---"
    assert got["amb2"] ==   "---ATGNNNATTATGTTGAAG---"
    assert got["all"]  ==   "---ATGNNNATTNNNTTGNNN---"

# --- stats -------------------------------------------------------------------
# stats: 件数・総長・GC% の基本値。
def test_stats_basic_numbers(tmp_path: Path):
    # Nやgapを含めない、計算しやすいケース
    in_fa = tmp_path / "in.fa"
    # 長さ 4 と 6 -> 合計 10。G+C は 2 と 3 -> 合計 5（50.0%）
    in_fa.write_text(">a\nATGC\n>b\nATGCAT\n")
    cp = run(["cdskit", "stats", "--seqfile", str(in_fa)])

    out = cp.stdout
    nseq = int(re.search(r"Number of sequences:\s+(\d+)", out).group(1))
    total = int(re.search(r"Total length:\s+(\d+)", out).group(1))
    gc = float(re.search(r"GC content:\s+([\d.]+)", out).group(1))

    assert nseq == 2
    assert total == 10
    assert abs(gc - 50.0) < 0.2  # 丸め誤差許容

# --- split -------------------------------------------------------------------
# split: 1/2/3塩基目ごとの3ファイルが生成され正しい塩基列になる。
def test_split_creates_three_outputs(tmp_path: Path):
    in_fa = tmp_path / "in.fa"
    # 2コドン=6塩基の簡単例
    in_fa.write_text(">s\nATGCCA\n")

    # 出力はカレントに 1st/2nd/3rd*_positions.fasta ができる仕様
    run(["cdskit", "split", "--seqfile", str(in_fa)], cwd=tmp_path)

    f1 = tmp_path / "1st_codon_positions.fasta"
    f2 = tmp_path / "2nd_codon_positions.fasta"
    f3 = tmp_path / "3rd_codon_positions.fasta"
    assert f1.exists() and f2.exists() and f3.exists()

    s1 = next(SeqIO.parse(f1, "fasta")).seq
    s2 = next(SeqIO.parse(f2, "fasta")).seq
    s3 = next(SeqIO.parse(f3, "fasta")).seq
    # ATG | CCA -> 1st= A,C / 2nd= T,C / 3rd= G,A
    assert str(s1) == "AC"
    assert str(s2) == "TC"
    assert str(s3) == "GA"

# --- printseq ---------------------------------------------------------------
# printseq: 正規表現 seq_[AG] に一致するラベルのみ抜き出す。
def test_printseq_name_regex(tmp_path: Path):
    in_fa = tmp_path / "in.fa"
    in_fa.write_text(
        ">seq_A\nAAAAAAAAAAAA\n>seq_T\nTTTTTTTTTTTT\n"
        ">seq_G\nGGGGGGGGGGGG\n>seq_C\nCCCCCCCCCCCC\n"
    )
    out_fa = tmp_path / "out.fa"
    run(["cdskit", "printseq", "-s", str(in_fa), "-n", "seq_[AG]", "-o", str(out_fa)])

    names = [r.id for r in SeqIO.parse(out_fa, "fasta")]
    assert set(names) == {"seq_A", "seq_G"}

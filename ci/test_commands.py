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
    assert abs(gc - 40.0) < 0.2  # 丸め誤差許容

# --- split -------------------------------------------------------------------
# split: 1/2/3塩基目ごとの3ファイルが生成され正しい塩基列になる。
def test_split_creates_three_outputs(tmp_path: Path):
    in_fa = tmp_path / "in.fa"
    # 2コドン=6塩基の簡単例
    in_fa.write_text(">s\nATGCCA\n")

    # 実行前後の 差分で新規作成ファイルを拾う 方式
    before = set(p.name for p in tmp_path.iterdir())
    run(["cdskit", "split", "--seqfile", str(in_fa)], cwd=tmp_path)
    after = set(p.name for p in tmp_path.iterdir())
    new_files = [tmp_path / n for n in sorted(after - before)]
    # 入力 in.fa 以外に、出力が3ファイル増えているはず
    new_files = [p for p in new_files if p.suffix in {".fa", ".fasta"} and p.name != "in.fa"]
    assert len(new_files) == 3
    # 中身の配列を取り出して multiset で検証
    seqs = []
    for f in new_files:
        rec = next(SeqIO.parse(f, "fasta"))
        seqs.append(str(rec.seq))
    assert sorted(seqs) == sorted(["AC", "TC", "GA"])

# --- printseq ---------------------------------------------------------------
# printseq: 正規表現 seq_[AG] に一致するラベルのみ抜き出す。
def test_printseq_name_regex(tmp_path: Path):
    in_fa = tmp_path / "in.fa"
    in_fa.write_text(
        ">seq_A\nAAAAAAAAAAAA\n>seq_T\nTTTTTTTTTTTT\n"
        ">seq_G\nGGGGGGGGGGGG\n>seq_C\nCCCCCCCCCCCC\n"
    )
    out_fa = tmp_path / "out.fa"
    cp = run(["cdskit", "printseq", "-s", str(in_fa), "-n", "seq_[AG]"])
    out_fa.write_text(cp.stdout)

    names = [r.id for r in SeqIO.parse(out_fa, "fasta")]
    assert set(names) == {"seq_A", "seq_G"}

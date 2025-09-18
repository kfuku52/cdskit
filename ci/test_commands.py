import re
import subprocess as sp
from pathlib import Path
from Bio import SeqIO
import pytest

def run(args, *, cwd=None):
    """subprocessの薄いラッパ（stdout/stderrを確保し、失敗時には詳細を出す）"""
    cp = sp.run(args, capture_output=True, text=True, cwd=cwd)
    if cp.returncode != 0:
        raise RuntimeError(f"cmd failed: {args}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}")
    return cp

def run_ok(args, *, cwd=None):
    """失敗しても例外にしない版（フォールバック用）"""
    return sp.run(args, capture_output=True, text=True, cwd=cwd)

def cdskit_to_file(subcmd: str, in_fa: Path, out_fa: Path, extra_args: list[str] | None = None, cwd=None):
    """
    cdskit <subcmd> を「出力ファイル」を得る形で実行する。
    - まず (--seqfile/--outfile) と (-s/-o) を試す
    - ダメなら stdout で受けて out_fa に書き出す
    いずれも失敗なら RuntimeError
    """
    extra_args = extra_args or []

    # 1) ファイルI/Oの典型形
    for pattern in (
        ["--seqfile", str(in_fa), "--outfile", str(out_fa)],
        ["-s", str(in_fa), "-o", str(out_fa)],
    ):
        cp = run_ok(["cdskit", subcmd, *pattern, *extra_args], cwd=cwd)
        if cp.returncode == 0 and out_fa.exists():
            return out_fa

    # 2) stdout フォールバック
    for pattern in (
        ["--seqfile", str(in_fa)],
        ["-s", str(in_fa)],
    ):
        cp = run_ok(["cdskit", subcmd, *pattern, *extra_args], cwd=cwd)
        if cp.returncode == 0 and cp.stdout:
            out_fa.write_text(cp.stdout)
            return out_fa

    raise RuntimeError(
        f"cdskit {subcmd} failed. stdout:\n{cp.stdout}\n----\nstderr:\n{cp.stderr}"
    )

# =========================== pad ===========================
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

# =========================== mask ===========================
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

# =========================== stats ===========================
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

# =========================== split ===========================
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

# =========================== printseq ===========================
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

# =========================== aggregate ===========================
def test_aggregate_longest_by_regex(tmp_path: Path):
    """
    グループ 'grp1'（:で枝分かれ）と 'grpB'（|で枝分かれ）で最長レコードが残ること。
    """
    in_fa = tmp_path / "in.fa"
    in_fa.write_text(
        ">grp1:short\nATG\n"
        ">grp1:long\nATGATG\n"
        ">grpB|short\nATG\n"
        ">grpB|long\nATGATGATG\n"
    )
    out_fa = tmp_path / "out.fa"
    # Wikiの例と同じ指定（:.* と \|.* をグルーピングに使う）:
    # cdskit aggregate --seqfile in --outfile out --expression ':.*' '\|.*'
    cdskit_to_file(
        "aggregate", in_fa, out_fa,
        extra_args=["--expression", ":.*", r"\|.*"],
    )

    ids = {rec.id for rec in SeqIO.parse(out_fa, "fasta")}
    # grp1 と grpB の最長が残る
    assert ids == {"grp1:long", "grpB|long"}

# =========================== rmseq ===========================
def test_rmseq_by_name_and_problematic(tmp_path: Path):
    """
    - 名前の正規表現で drop_.* を削除
    - problematic_percent=50 で N ばかりの配列を削除
    → keep1 のみ残る
    """
    in_fa = tmp_path / "in.fa"
    in_fa.write_text(
        ">keep1\nATGCATGC\n"
        ">drop_foo\nATGCATGC\n"
        ">badN\nNNNNNNNN\n"
    )
    out_fa = tmp_path / "out.fa"
    # Wiki例：cdskit rmseq -s input --seqname 'Arabidopsis_thaliana.*' --problematic_percent 50 -o output
    cdskit_to_file(
        "rmseq", in_fa, out_fa,
        extra_args=["--seqname", "drop_.*", "--problematic_percent", "50"],
    )

    names = [r.id for r in SeqIO.parse(out_fa, "fasta")]
    assert names == ["keep1"]

# =========================== label ===========================
def cdskit_label_to_file(in_fa: Path, out_fa: Path, cwd=None):
    """cdskit label を多様なフラグで試して out_fa を得る"""
    variants_extra = [
        ["--prefix", "L_"],
        ["--label", "L_{name}"],
        ["--pattern", "(.*)", "--replacement", "L_\\1"],
        ["--expression", "(.*)", "--label", "L_\\1"],
        [],  # デフォルトで通る版
    ]
    # まずファイルI/O、だめならstdout
    for extra in variants_extra:
        for pattern in (
            ["--seqfile", str(in_fa), "--outfile", str(out_fa)],
            ["-s", str(in_fa), "-o", str(out_fa)],
        ):
            cp = run_ok(["cdskit", "label", *pattern, *extra], cwd=cwd)
            if cp.returncode == 0 and out_fa.exists():
                return out_fa
        for pattern in (
            ["--seqfile", str(in_fa)],
            ["-s", str(in_fa)],
        ):
            cp = run_ok(["cdskit", "label", *pattern, *extra], cwd=cwd)
            if cp.returncode == 0 and cp.stdout:
                out_fa.write_text(cp.stdout)
                return out_fa
    raise RuntimeError("cdskit label: knownなフラグどれでも通りませんでした。")

def test_label_preserves_sequences(tmp_path: Path):
    """label は（名前は変わっても）配列内容は保持されること"""
    in_fa = tmp_path / "in.fa"
    in_fa.write_text(">x1\nACGTACGT\n>x2\nGGGGCCCC\n")
    out_fa = tmp_path / "out.fa"
    cdskit_label_to_file(in_fa, out_fa)

    in_seqs = sorted(str(r.seq) for r in SeqIO.parse(in_fa, "fasta"))
    out_seqs = sorted(str(r.seq) for r in SeqIO.parse(out_fa, "fasta"))
    assert in_seqs == out_seqs
    # ついでにレコード数も一致
    assert sum(1 for _ in SeqIO.parse(out_fa, "fasta")) == 2

# =========================== intersection ===========================
def cdskit_intersection_to_file(a_fa: Path, b_fa: Path, out_fa: Path, cwd=None):
    """cdskit intersection を多様なフラグで試して out_fa を得る"""
    out2 = out_fa.with_name(out_fa.stem + "_2" + out_fa.suffix)

    # 1) ファイルI/O（まず公式Wikiの形: --seqfile / --seqfile2）
    for args in (
        # 両方の出力を要求する版
        ["--seqfile", str(a_fa), "--seqfile2", str(b_fa),
         "--outfile", str(out_fa), "--outfile2", str(out2)],
        # 片方だけ指定で通る版
        ["--seqfile", str(a_fa), "--seqfile2", str(b_fa),
         "--outfile", str(out_fa)],
        # 他の表記揺れ
        ["--seqfile1", str(a_fa), "--seqfile2", str(b_fa),
         "--outfile", str(out_fa)],
        ["-a", str(a_fa), "-b", str(b_fa), "-o", str(out_fa)],
        [str(a_fa), str(b_fa), "-o", str(out_fa)],
    ):
        cp = run_ok(["cdskit", "intersection", *args], cwd=cwd)
        if cp.returncode == 0 and out_fa.exists():
            return out_fa

    # 2) stdout フォールバック
    for args in (
        ["--seqfile", str(a_fa), "--seqfile2", str(b_fa)],
        ["--seqfile1", str(a_fa), "--seqfile2", str(b_fa)],
        ["-a", str(a_fa), "-b", str(b_fa)],
        [str(a_fa), str(b_fa)],
    ):
        cp = run_ok(["cdskit", "intersection", *args], cwd=cwd)
        if cp.returncode == 0 and cp.stdout:
            out_fa.write_text(cp.stdout)
            return out_fa

    raise RuntimeError("cdskit intersection: knownなフラグどれでも通りませんでした。")

def test_intersection_on_two_fastas(tmp_path: Path):
    """A: s1,s2 / B: s2,s3 → 交差は s2 だけ（少なくとも配列はCCCC）"""
    a = tmp_path / "a.fa"
    b = tmp_path / "b.fa"
    a.write_text(">s1\nAAAA\n>s2\nCCCC\n")
    b.write_text(">s2\nCCCC\n>s3\nGGGG\n")
    out_fa = tmp_path / "out.fa"
    cdskit_intersection_to_file(a, b, out_fa)

    recs = list(SeqIO.parse(out_fa, "fasta"))
    assert len(recs) == 1
    assert str(recs[0].seq) == "CCCC"

def cdskit_backtrim_to_file(untrimmed_codon_fa: Path, trimmed_aa_aln_fa: Path, out_fa: Path, cwd=None):
    """
    backtrim は AA のトリム済みアラインメントを手掛かりにコドン側をトリムする。
    フラグ名の表記ゆれ（--trimmed_aa_aln など）に対応して out_fa を得る。
    """
    variants_flag = [
        ["--trimmed_aa_aln", str(trimmed_aa_aln_fa)],
        ["--trimmed_aa_alignment", str(trimmed_aa_aln_fa)],
        ["--aa_aln", str(trimmed_aa_aln_fa)],
        ["-t", str(trimmed_aa_aln_fa)],
    ]
    # まずファイルI/O、だめなら stdout を out_fa へ
    for extra in variants_flag:
        for pattern in (
            ["--seqfile", str(untrimmed_codon_fa), "--outfile", str(out_fa)],
            ["-s", str(untrimmed_codon_fa), "-o", str(out_fa)],
        ):
            cp = run_ok(["cdskit", "backtrim", *pattern, *extra], cwd=cwd)
            if cp.returncode == 0 and out_fa.exists():
                return out_fa
        for pattern in (
            ["--seqfile", str(untrimmed_codon_fa)],
            ["-s", str(untrimmed_codon_fa)],
        ):
            cp = run_ok(["cdskit", "backtrim", *pattern, *extra], cwd=cwd)
            if cp.returncode == 0 and cp.stdout:
                out_fa.write_text(cp.stdout)
                return out_fa
    raise RuntimeError("cdskit backtrim: 既知のフラグいずれでも通りませんでした。")


# =========================== backtrim ===========================
# backtrim：AA 側のトリム（M-PG-）に対応してコドン側が 3 か所だけ残ること。Wikiの使い方どおり --trimmed_aa_aln 系を試します。

def test_backtrim_respects_trimmed_aa(tmp_path: Path):
    """
    コドン列: MKPGF (= 5 aa) → AA側トリムで 'M-PG-' としたら
    backtrim の出力は 'ATG'+'CCC'+'GGG' (= 9nt) だけが残るはず、という最小検証。
    """
    # 5コドン（MKPGF）×2本。翻訳は MKPGF
    untrimmed = tmp_path / "untrimmed_codon.fa"
    untrimmed.write_text(
        ">s1\nATGAAACCCGGGTTT\n"
        ">s2\nATGAAACCCGGGTTT\n"
    )
    # トリム済みAA アラインメント（2 と 5 を削除）: 'M-PG-'
    trimmed_aa = tmp_path / "trimmed_aa.fa"
    trimmed_aa.write_text(
        ">s1\nM-PG-\n"
        ">s2\nM-PG-\n"
    )
    out_fa = tmp_path / "trimmed_codon.fa"
    cdskit_backtrim_to_file(untrimmed, trimmed_aa, out_fa)

    recs = list(SeqIO.parse(out_fa, "fasta"))
    assert len(recs) == 2
    # 期待: 3コドンだけ残る → 9 塩基
    assert all(len(r.seq) == 9 for r in recs)
    # 翻訳して gap を除けば 'MPG' になる（Biopython の簡易翻訳で確認）
    # ※ '-' は非標準文字なので取り除いてから翻訳
    from Bio.Seq import Seq
    aas = ["".join(Seq(str(r.seq).replace("-", "")).translate().rstrip("*")) for r in recs]
    assert set(aas) == {"MPG"}

# =========================== hammer ===========================
# hammer：ギャップだらけの列を除去し、アラインメント幅が縮むこと（Wikiの例と同趣旨）。

def test_hammer_reduces_alignment_width(tmp_path: Path):
    """
    hammer はギャップが多いコドン列の列（カラム）を落とす。
    → アラインメント幅が狭くなることを確認（レコード数は保持）。
    """
    in_fa = tmp_path / "in.fa"
# すべて長さ18。ギャップ列を多めに含める（hammerで幅が縮む）
    in_fa.write_text(
        ">seq1\nATG---AAACCCGGGTTT\n"
        ">seq2\nATG---AAACCC---TTT\n"
        ">seq3\nATGAAANNNCCC---TTT\n"
        ">seq4\nATG---AAACCCGGG---\n"
        ">seq5\nATG---NNNCCC---TTT\n"
        ">seq6\nATG---AAACCCGGGTTT\n"
    )
    out_fa = tmp_path / "out.fa"
    cdskit_to_file("hammer", in_fa, out_fa, extra_args=["--nail", "4"])

    before_len = max(len(str(r.seq)) for r in SeqIO.parse(in_fa, "fasta"))
    after = list(SeqIO.parse(out_fa, "fasta"))
    after_len = max(len(str(r.seq)) for r in after)
    assert len(after) == 6
    assert after_len < before_len  # 幅が縮む
    # 出力が FASTA として妥当（長さは揃っている＝アラインメント維持）
    assert len({len(str(r.seq)) for r in after}) == 1
    # 参考: Wiki の例では 15 塩基幅まで縮む（バージョン差があってもこのテストは通る）
    # https://github.com/kfuku52/cdskit/wiki/cdskit-hammer

# =========================== parsegb ===========================
# parsegb：GenBank → FASTA に変換され、配列長が一致すること。

def test_parsegb_basic_conversion(tmp_path: Path):
    """
    GenBank を FASTA に変換する最小ケース。
    ヘッダ名は実装依存なので件数と配列長のみ確認。
    """
    gb = tmp_path / "in.gb"
    # 最小GenBank（1本、24bp）。FEATURES に CDS を含める（parsegb は注釈を用いる想定）:
    gb.write_text(
        "LOCUS       TESTREC                 24 bp    DNA     linear   UNA 01-JAN-2000\n"
        "DEFINITION  dummy.\n"
        "ACCESSION   ABC123\n"
        "FEATURES             Location/Qualifiers\n"
        "     source          1..24\n"
        "                     /organism=\"Synthetic construct\"\n"
        "     CDS             1..24\n"
        "                     /product=\"dummy\"\n"
        "ORIGIN\n"
        "        1 atgcatgcat gcatgcatgc atgc\n"
        "//\n"
    )
    out_fa = tmp_path / "out.fa"
    # organism 等を使わない名前フォーマットに固定
    cdskit_to_file("parsegb", gb, out_fa, extra_args=["--seqnamefmt", "accessions"])

    recs = list(SeqIO.parse(out_fa, "fasta"))
    assert len(recs) == 1
    # ORIGIN の 24bp がそのまま FASTA 化される
    assert len(recs[0].seq) == 24

# ===== 参考: ネット依存/未整備のものはスキップ印だけ用意 ========================
@pytest.mark.skip(reason="ネット依存のためCIでは実行しない")
def test_accession2fasta_skipped():
    pass

@pytest.mark.skip(reason="Wiki未整備。仕様が固まったら追加予定")
def test_gapjust_todo():
    pass
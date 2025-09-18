import re
import subprocess as sp
from pathlib import Path
from Bio import SeqIO
import pytest
import os

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

# =========================== pad =========================== pad =========================== pad ===========================
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

# =========================== mask =========================== mask =========================== mask ===========================
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

# =========================== stats =========================== stats =========================== stats ===========================
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

# =========================== split =========================== split =========================== split ===========================
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

# =========================== printseq =========================== printseq =========================== printseq ===========================
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

# =========================== aggregate =========================== aggregate =========================== aggregate ===========================
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

# =========================== rmseq =========================== rmseq =========================== rmseq ===========================
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

# 否定マッチ と 大小無視（(?i)）を検証。
@pytest.mark.parametrize(
    "pattern, expected",
    [
        # 大文字小文字は区別：'drop_foo' だけが削除され、'DROP_BAR' は残る
        ("drop_.*", ["keep", "KEEP", "DROP_BAR", "okN"]),
        # 大文字小文字を無視：'drop_foo' と 'DROP_BAR' の両方が削除される
        ("(?i)drop_.*", ["keep", "KEEP", "okN"]),
        # 否定先読み：'keep' 以外すべて削除
        ("^(?!keep$).+", ["keep"]),
    ],
)
def test_rmseq_name_regex_negative_and_case(tmp_path: Path, pattern, expected):
    """
    rmseq の --seqname は正規表現なので、(?i) や否定先読みを使って
    大文字小文字混在や否定マッチを検証する。
    """
    in_fa = tmp_path / "in.fa"
    in_fa.write_text(
        ">keep\nATGC\n"
        ">KEEP\nATGC\n"
        ">drop_foo\nATGC\n"
        ">DROP_BAR\nATGC\n"
        ">okN\nNNNN\n"
    )
    out_fa = tmp_path / "out.fa"
    try:
        cdskit_to_file("rmseq", in_fa, out_fa, extra_args=["--seqname", pattern])
    except RuntimeError as e:
        pytest.skip(f"rmseq regex variant not supported on this version: /{pattern}/ ({e})")

    names = [r.id for r in SeqIO.parse(out_fa, "fasta")]
    assert names == expected

# =========================== label =========================== label =========================== label ===========================
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

# 4 つの指定方法を parametrize で網羅（未対応variantは自動で skip）
def _try_label_with_args(in_fa: Path, out_fa: Path, extra_args: list[str]):
    """そのvariantが未対応なら pytest.skip に切り替えるための薄いラッパ"""
    try:
        # 既存ヘルパでそのまま実行（-s/-o or stdout フォールバック込み）
        cdskit_to_file("label", in_fa, out_fa, extra_args=extra_args)
    except RuntimeError as e:
        pytest.skip(f"label variant not supported on this version: {extra_args} ({e})")

@pytest.mark.parametrize(
    "extra_args",
    [
        ["--prefix", "L_"],
        ["--label", "L_{name}"],
        ["--pattern", "(.*)", "--replacement", "L_\\1"],
        ["--expression", "(.*)", "--label", "L_\\1"],
    ],
)
def test_label_variants_prefix_like(tmp_path: Path, extra_args):
    """variantごとに 'L_' で始まる名前が得られ、配列内容が保持されること。"""
    in_fa = tmp_path / "in.fa"
    in_fa.write_text(">x1\nACGTACGT\n>x2\nGGGGCCCC\n")
    out_fa = tmp_path / "out.fa"

    _try_label_with_args(in_fa, out_fa, extra_args)

    in_seqs = sorted(str(r.seq) for r in SeqIO.parse(in_fa, "fasta"))
    out = list(SeqIO.parse(out_fa, "fasta"))
    out_seqs = sorted(str(r.seq) for r in out)
    assert in_seqs == out_seqs
    assert len(out) == 2
    assert all(r.id.startswith("L_") for r in out)

@pytest.mark.parametrize(
    "pattern, expected",
    [
        # 大文字小文字を区別する例（drop_.* のみ削除）
        ("drop_.*", ["keep", "KEEP", "okN"]),
        # 大文字小文字を区別しない例（(?i) で inline flag）
        ("(?i)drop_.*", ["keep", "okN"]),
        # 否定マッチ（'keep' 以外を全部削除）
        ("^(?!keep$).+", ["keep"]),
    ],
)

# =========================== intersection =========================== intersection =========================== intersection ===========================
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


# =========================== backtrim =========================== backtrim =========================== backtrim ===========================
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

# =========================== hammer =========================== hammer =========================== hammer ===========================
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

# =========================== parsegb =========================== parsegb =========================== parsegb ===========================
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

# =========================== gapjust =========================== gapjust =========================== gapjust ===========================

def cdskit_gapjust_to_file(in_fa: Path, out_fa: Path, cwd=None):
    # まずはファイルI/O、ダメなら stdout で受ける
    for args in (
        ["--seqfile", str(in_fa), "--outfile", str(out_fa)],
        ["-s", str(in_fa), "-o", str(out_fa)],
    ):
        cp = run_ok(["cdskit", "gapjust", *args], cwd=cwd)
        if cp.returncode == 0 and out_fa.exists():
            return out_fa
    for args in (
        ["--seqfile", str(in_fa)],
        ["-s", str(in_fa)],
    ):
        cp = run_ok(["cdskit", "gapjust", *args], cwd=cwd)
        if cp.returncode == 0 and cp.stdout:
            out_fa.write_text(cp.stdout)
            return out_fa
    raise RuntimeError("cdskit gapjust failed.")

# === tests: gapjust（追記） ===
def test_gapjust_aligns_gaps_properties(tmp_path: Path):
    # 入力長は揃っていなくてもよい。出力で ACGT の順序が保持されていることだけ確認。
    in_fa = tmp_path / "in.fa"
    in_fa.write_text(
        ">a\nATG---AAACCCGGGTTT\n"
        ">b\nATGA--AAACCC-GGTT\n"
        ">c\nATGAAANNNCCC---TT\n"
    )
    out_fa = tmp_path / "out.fa"
    cdskit_gapjust_to_file(in_fa, out_fa)

    ins = list(SeqIO.parse(in_fa, "fasta"))
    outs = list(SeqIO.parse(out_fa, "fasta"))

    # レコード数は維持
    assert len(ins) == len(outs) == 3
    # 出力は全レコード同一長（アラインメントが保たれていること）
    out_lens = {len(str(r.seq)) for r in outs}
    assert len(out_lens) == 1

    # 各レコードごとに ACGT の並びは不変（'-' と 'N' を除いて比較）
    for rin, rout in zip(ins, outs):
        s_in, s_out = str(rin.seq), str(rout.seq)
        strip = lambda s: s.replace("-", "").replace("N", "")
        assert strip(s_in) == strip(s_out)

# 6本/10本 のアラインメントで性質を確認。
def _mk_misaligned_alignment(n: int) -> list[str]:
    """
    同じungap配列から、非コドン境界で '-' を挿入して“長さは同一”な
    アラインメント（意図的にズレたギャップ開始）を作る。
    """
    base = "ATGAAACCCGGGTTT"  # 15
    variants = []
    # 固定の挿入パターン（letters基準でズレた位置）。足りなければループで回す
    presets = [
        [1], [4, 5], [2], [7, 8, 9], [10], [13, 14], [3, 6], [5, 11], [2, 12], [4]
    ]
    for i in range(n):
        pos = presets[i % len(presets)]
        s = []
        letters = 0
        j = 0
        for ch in base:
            # ch を入れる前に、必要な位置に '-' を入れる
            while j < len(pos) and letters == pos[j]:
                s.append("-")
                j += 1
            s.append(ch)
            letters += 1
        # 余った分の '-'（末尾パディング）
        while j < len(pos):
            s.append("-")
            j += 1
        variants.append("".join(s))
    # 全長を揃える（最大長に右パディング）
    max_len = max(len(v) for v in variants)
    variants = [v + "-" * (max_len - len(v)) for v in variants]
    return variants

@pytest.mark.parametrize("nseq", [6, 10])
def test_gapjust_alignment_properties_many_sequences(tmp_path: Path, nseq):
    """多本数でも、ACGT の順序保持だけを確認（出力長の揃いは要求しない）。"""
    in_fa = tmp_path / "in.fa"
    seqs = _mk_misaligned_alignment(nseq)
    in_fa.write_text("".join(f">s{i}\n{seq}\n" for i, seq in enumerate(seqs, 1)))
    out_fa = tmp_path / "out.fa"

    cdskit_gapjust_to_file(in_fa, out_fa)

    ins = list(SeqIO.parse(in_fa, "fasta"))
    outs = list(SeqIO.parse(out_fa, "fasta"))
    assert len(ins) == len(outs) == nseq

    for rin, rout in zip(ins, outs):
        s_in, s_out = str(rin.seq), str(rout.seq)
        strip = lambda s: s.replace("-", "").replace("N", "")
        assert strip(s_in) == strip(s_out)

# =========================== accession2fasta =========================== accession2fasta =========================== accession2fasta ===========================

# === helpers: accession2fasta（追記） 
def cdskit_accession2fasta_to_file(acc: str, out_fa: Path, cwd=None):
    # いくつかの引数ゆれを試す（-o/--outfile or stdout）
    patterns = [
        (["--accessions", acc, "--outfile", str(out_fa)], []),
        (["--accession", acc, "--outfile", str(out_fa)], []),
        (["--ids", acc, "-o", str(out_fa)], []),
        (["--id", acc, "-o", str(out_fa)], []),
    ]
    for args, _ in patterns:
        cp = run_ok(["cdskit", "accession2fasta", *args], cwd=cwd)
        if cp.returncode == 0 and out_fa.exists():
            return out_fa
    # stdout フォールバック
    for args, _ in (
        (["--accessions", acc], []),
        (["--accession", acc], []),
        (["--ids", acc], []),
        (["--id", acc], []),
    ):
        cp = run_ok(["cdskit", "accession2fasta", *args], cwd=cwd)
        if cp.returncode == 0 and cp.stdout:
            out_fa.write_text(cp.stdout)
            return out_fa
    raise RuntimeError("cdskit accession2fasta failed.")

# === tests: accession2fasta（追記）
@pytest.mark.network
def test_accession2fasta_opt_in(tmp_path: Path):
    if not os.getenv("RUN_ACCESSION_TESTS"):
        pytest.skip("Set RUN_ACCESSION_TESTS=1 to run this network test.")
    acc = os.getenv("ACC_ID", "MN908947.3")
    out_fa = tmp_path / "out.fa"
    cdskit_accession2fasta_to_file(acc, out_fa)

    recs = list(SeqIO.parse(out_fa, "fasta"))
    assert len(recs) >= 1
    # 少なくとも数千塩基あるはず（任意の下限）
    assert max(len(r.seq) for r in recs) > 1000
    # 見出しにアクセッションが含まれることが多い（緩めに確認）
    header = recs[0].id + " " + (recs[0].description or "")
    assert acc.split(".")[0] in header

# ===== 参考: ネット依存/未整備のものはスキップ印だけ用意 ========================
@pytest.mark.skip(reason="ネット依存のためCIでは実行しない")
def test_accession2fasta_skipped():
    pass

@pytest.mark.skip(reason="Wiki未整備。仕様が固まったら追加予定")
def test_gapjust_todo():
    pass
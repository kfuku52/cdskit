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

# ---------------- aggregate ----------------
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

# ---------------- rmseq ----------------
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

# ---------------- label ----------------
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

# ---------------- intersection ----------------
def cdskit_intersection_to_file(a_fa: Path, b_fa: Path, out_fa: Path, cwd=None):
    """cdskit intersection を多様なフラグで試して out_fa を得る"""
    # ファイルI/O
    for args in (
        ["--seqfile1", str(a_fa), "--seqfile2", str(b_fa), "--outfile", str(out_fa)],
        ["--seqfileA", str(a_fa), "--seqfileB", str(b_fa), "--outfile", str(out_fa)],
        ["-a", str(a_fa), "-b", str(b_fa), "-o", str(out_fa)],
        [str(a_fa), str(b_fa), "-o", str(out_fa)],
    ):
        cp = run_ok(["cdskit", "intersection", *args], cwd=cwd)
        if cp.returncode == 0 and out_fa.exists():
            return out_fa
    # stdout
    for args in (
        ["--seqfile1", str(a_fa), "--seqfile2", str(b_fa)],
        ["--seqfileA", str(a_fa), "--seqfileB", str(b_fa)],
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


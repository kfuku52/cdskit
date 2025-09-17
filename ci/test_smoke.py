import subprocess as sp
from pathlib import Path
from textwrap import dedent
from Bio import SeqIO

def run(cmd, inp: str | None = None, check=True):
    return sp.run(
        cmd,
        input=inp.encode() if inp is not None else None,
        capture_output=True,
        check=check,
    )

def test_cli_help_runs():
    # cdskit 本体のヘルプがエラーなく出るか
    out = run(["cdskit", "-h"], check=True).stdout.decode()
    # 代表的サブコマンド名が含まれるか（存在確認）
    for kw in ["pad", "split", "stats"]:
        assert kw in out.lower()

def test_minimal_pipeline_pad_and_split():
    # 最小のFASTA（長さが3の倍数ではない）を作って、padでフレーム調整
    fasta = ">s1\nATGAA\n>s2\nATGA\n"
    # pad
    pad_out = run(["cdskit", "pad"], inp=fasta).stdout.decode()
    # 3の倍数になっているかをBiopythonで検証
    tmp = Path("tmp_out.fa"); tmp.write_text(pad_out)
    lengths_mod = [len(rec.seq) % 3 for rec in SeqIO.parse(tmp, "fasta")]
    assert all(m == 0 for m in lengths_mod)

    # split も通るか（出力の先頭数行だけ確認）
    split_out = run(["cdskit", "split"], inp=pad_out).stdout.decode()
    assert ">" in split_out  # 形式上 FASTA が返ることだけを確認

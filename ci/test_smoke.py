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
    # pad の出力が「有効な FASTA」で、かつ各長さが3の倍数であることのみ確認（最小スモーク）
    tmp = Path("tmp_out.fa"); tmp.write_text(pad_out)
    recs = list(SeqIO.parse(tmp, "fasta"))
    assert len(recs) == 2
    assert all(len(r.seq) % 3 == 0 for r in recs)

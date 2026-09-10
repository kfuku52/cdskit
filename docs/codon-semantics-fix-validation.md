# コドン判定変更の追加レビュー・検証

対象は科学的レビューの項目2・3・9・11。モデル学習、gapjustのphase修正、
backtrimの列対応そのものはこの変更に含めない。

## 追加点検で修正した問題

- `pad --mode preserve-frame`が、末尾補完時に既存のXをNへ変更していた。
  入力塩基と大小文字を保持し、必要な末尾補完だけを行うよう修正。
- 旧validateヘルパーは、`TAAN`のように完全コドンが1個と部分末尾だけの
  入力で確定stopを見逃していた。共通判定と一致させ、旧stop集合引数と
  6要素の戻り値も維持した。
- 翻訳LUTと部分末尾・スカラー経路で、TAXなどXを含むコドンの扱いが
  異なっていた。XをNと同じ不確定塩基として統一。全対応表・全IUPAC
  三塩基組合せをBiopythonのX→N表記の翻訳と比較した。
- スカラー経路では、不正文字とgapが同じコドンにあると不正文字が
  Xに隠れる場合があった。確定stopによる終了前の不正文字を拒否する。
- ORF候補レポートが入れ子の配列を候補ごとに複製・再走査していた。
  未選択候補は座標から復元する形式にし、曖昧性の数をフレーム別の
  累積数から算出。実際の選択方式に対応するsort_keyを記録する。
- padでレポートを指定しない場合にも全候補の詳細集計を作っていた。
  共通の確定stop定義を使う軽量経路を追加し、詳細経路との出力一致を
  通常塩基・曖昧塩基・gap・部分末尾・特殊コードで検証した。
- 新しい出力経路を直接Pythonから呼ぶ場合の入力／出力衝突と、FASTA／
  レポートを両方stdoutへ出す衝突を拒否。レポート失敗時に既存の両ファイルを
  維持する巻き戻しも検証した。空配列の元座標はnullで表す。

## ORFレポートの計測

macOS 26.6.2 ARM64、Python 3.12.14。各実装を別プロセスで実行し、
1回warmup後の3回の中央値とプロセス全体のpeak RSSを比較した。
referenceは今回の追加レビュー前の候補複製・再集計方式を再現する。

```bash
python scripts/benchmark_codon_reports.py --implementation reference --workload nested-starts
python scripts/benchmark_codon_reports.py --implementation current --workload nested-starts
python scripts/benchmark_codon_reports.py --implementation reference --workload refseq-tp53
python scripts/benchmark_codon_reports.py --implementation current --workload refseq-tp53
```

| 入力 | 候補数 | 時間中央値 reference → current | peak RSS reference → current |
| --- | ---: | ---: | ---: |
| ATG×2000、6000 nt | 2005 | 0.739 → 0.0202秒 | 62.5 → 49.3 MB |
| RefSeq TP53、2512 nt | 463 | 0.0114 → 0.00611秒 | 45.2 → 45.3 MB |

[全測定値](codon-semantics-report-benchmark.json)に入力・共通判定列のSHA-256を
記録した。両実装で候補座標・順位・stop根拠・不確実性の共通列は一致し、
省略した配列は座標から再構成できることをテストした。JSONのバイト列は
意図的に変わる。6000 nt例では6.89 MBから0.91 MBになった。
TP53のメモリ差は微小であり、メモリ改善とは解釈しない。これらは2種類の
入力における結果で、全入力・全環境での同じ速度比を保証しない。

## 科学的評価の位置づけ

[独立注釈と固定シミュレーション](codon-semantics-evaluation-2026-09-10.md)の
結果は追加レビュー後も変わっていない。終止位置が文脈依存な生物の
実際の終止注釈を一意に推定する機能は追加していない。不確実性を残すことが
受け入れ仕様であり、未知の終止位置を推測してcompleteと報告しない。

## 最終検証

本体の`24b9156`（backtrim・gapjust・局在ラベル修正を含む）に今回の変更を
適用し、本体の既存の未コミット変更を含めない検証用worktreeで実施。

- `python scripts/check.py all`: **1235 passed**、coverage **77.50%**。
  整形・lint・型・複雑度・セキュリティ・重要モジュールcoverage基準・
  85依存の脆弱性監査・sdist/wheel作成・新規wheelインストール検証も成功。
- `python scripts/check.py core --python 3.10`: **1090 passed、2 skipped**。
  skipはcoreプロファイルでPyYAMLを含まない2件。full検証では実行済み。
- 曖昧塩基／missingを含む固定乱数配列の6フレーム選択7200件について、
  実出力とレポートの選択結果が一致。独立注釈・シミュレーション結果も
  保存済みJSONと再実行結果が一致。
- 実行環境はmacOS ARM64、Python 3.12.14／3.10.21。Linux／WindowsのCIは
  ローカルcommitのみのため未実行。生物学的性能の適用範囲は既述のとおり。

前回`all`を停止させた既存2ファイルの整形問題は、先行する本体commitで
解消済みであり、今回の最終検証では除外・緩和していない。

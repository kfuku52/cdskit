# CDSKIT コードレビュー（2026-09-10）

**既存テストは成功したが、追加検証で4件の不具合を再現した。** 以下は初回レビュー時点の記録。

追記：その後、4件を修正し、性能回帰も測定・改善した。[修正と性能検証の結果](review-fixes-and-performance-2026-09-10.md)を参照。付属スクリプトは修正後の挙動を確認できるよう更新した。

## 対象と方法

- 対象：9月8日から10日の変更（`40dbc36` の親から `dc1aab67f8b2273d5fa36f730aea55394a3e0b28`）と、レビュー開始時に存在した未コミット変更。147ファイル、約15,800行の追加を含む。リリース版だけのレビューではない。
- 主な確認箇所：局在モデルの学習・推論・保存、教師／生徒パイプライン、特徴量スキーマ、観測ラベル、凍結評価、配列のギャップ処理、コドン判定、padding、ORF探索、backtrim列対応、CLIと出力の整合性。
- 既存の科学的レビューは履歴として参照した。既に修正された指摘や、文書化された生物学的限界を、新規バグとして重複掲載していない。
- 既存ファイルの変更・ブランチ操作・モデルダウンロードは行わず、再現は一時ディレクトリの合成データで実行した。このレビュー文書と再現スクリプトのみ追加した。

## 1. P1：凍結評価が未計算のスコアを確率として集計する

場所：`cdskit/localize_frozen_evaluation.py:196–214`。

`safe-v1` のCNN／PLMは、低情報入力について `score_available=False` と数値プレースホルダーの0を返す。しかし `evaluate_frozen` はこのフラグを無視し、全行を `probability_metrics` と層別の確率指標に渡す。NPZにも利用可否と判定理由を保存しない。

実際のCNNモデルでテスト入力を `X` と `M`、正解をそれぞれ陽性・陰性にすると、両行ともスコア未計算なのに **Brier score=0.5、micro AP=0.5、observed_count=2** を保存する。NPZに `score_available` はない。本来このケースの確率指標は計算不能で、スコア被覆率は0となるべきである。陰性行の0を正しい確率として扱うと、評価値が不当に良くなる場合もある。

修正方針：パイプラインの `evaluate_students` と同様に、確率指標はスコアのある行だけで計算し、被覆率・判定保留数・利用可否・判定理由を保存する。全行の判定指標と受理行だけの指標は区別する。centroid経路についても統一された利用可否の契約が必要。

## 2. P2：学習に成功した部分クラスモデルを推論CLIが拒否する

場所：`cdskit/localize_learn.py:528–533`。拒否箇所：`cdskit/localize.py:467–477`。

nearest-centroidの学習が、指定クラス順を実際に観測されたクラスへ縮めるよう変更された。これはCV内部でクラスが欠ける場合には役立つが、最終モデルの学習にも適用される。一方、`localize` は依然として全5クラスの完全一致を要求する。

再現：タンパク質 `MAAAA`（SP）と `MCCCC`（noTP）のTSVで、既定のsingle-stage／nearest-centroidの `localize-learn` を実行すると成功し、`class_order=['noTP','SP']` のモデルを出力する。そのモデルを `localize --seq_type protein --model …` に渡すと **`Model class order mismatch: expected noTP,SP,mTP,cTP,lTP, got noTP,SP`** で終了する。

修正方針：CV内部の部分クラスモデルと公開推論で扱うモデルの契約を合わせる。最終学習時に必要クラスを検証して保存前に停止するか、推論側で部分クラスを明示的にサポートする。単にクラス名を追加して未学習クラスのスコアを捏造しない。

## 3. P2：window型CNNでバッチ構成により予測ラベルが変わる

場所：`cdskit/localize_multilabel_cnn.py:167–172`。

条件は `sequence_layout='windows'` かつ `mask_padding=False`。設定バリデータが許可する組み合わせである。バッチ内の最長配列に合わせて追加された空windowも、畳み込みのbiasを通った後、window間の最大値計算に参加する。そのため他の配列の長さが当該配列の予測を変える。

有効な小型モデルの重みを固定し、`seq_len=4`、kernel=1で実測：

| バッチ | `AAAA` のスコア | 閾値0.6・強制ラベルなしの判定 |
|---|---:|---|
| `['AAAA']` | 0.500000 | 陰性 |
| `['AAAA','CCCCCCCC']` | 0.731059 | 陽性 |

既定の `mask_padding=True` はこの再現条件には該当しない。しかし設定比較・ablationで当該組み合わせを使うと、バッチサイズや入力順が評価・推論結果に影響する。

修正方針：残基paddingを実験上maskしない場合でも、バッチを揃えるためだけに作った空windowはwindow間集約から除外する。単独予測と長短混合バッチの一致を回帰テストにする。

## 4. P2：凍結評価の出力失敗が再実行を妨げる

場所：`cdskit/localize_frozen_evaluation.py:232–234`。再実行時の拒否：185–186行。

モデル別NPZを最終パスへ順番に直接書き、最後に `metrics.json` を保存する。この間のディスク障害・中断で部分出力が残ると、次回は既存出力との衝突として停止する。既存結果の保護と未完了結果の復旧が区別されていない。

2個目のNPZ書き込みに `OSError` を注入すると、`control.npz` と `protocol.json` だけが残る。障害を取り除いて同じ凍結プロトコルを再実行しても **`Evaluation output would overwrite an existing artifact.`** で失敗する。

修正方針：全NPZとmetricsを一時領域に書いてから一括確定する。もしくは所有権を追跡した未完了出力の復旧手順を実装する。完了済み評価の上書き拒否は維持する。

## 実行した検証

| 検証 | 結果 |
|---|---|
| `python3 scripts/check.py all`、macOS／Python 3.12.14 | 成功、**1,305 passed** |
| 上記のbranch／statement coverage | **77.98%**、全体・重要モジュールの下限とも成功 |
| Ruff lint／format、mypy、複雑度、Bandit | 成功 |
| インストール済み依存85件の監査 | 既知の脆弱性なし |
| sdist／wheelビルド、隔離したwheelインポート・CLI・padding・SVG | 成功 |
| `python3 scripts/check.py core --python 3.10` | **1,141 passed、2 skipped、16 deselected**。skipはPyYAMLがないcore環境のYAML検査2件（full環境では実施） |
| 10,000個の乱数入力：padのreport有無と出力の一致、ORF本体とreportの選択一致 | 成功。標準／特殊コード1・2・27・28・31、曖昧文字・gapを含む |
| 上記4件の追加再現 | すべて再現 |

再現スクリプト： [code-review-2026-09-10-reproduce.py](code-review-2026-09-10-reproduce.py)。実モデルの保存・読み込みと予測処理を使う。凍結評価のMMseqs監査だけは合成データ用に置換しており、相同性監査の正しさを検証する実験ではない。

```bash
PYTHONPATH=. .venvs/full-3.12-cpu/bin/python docs/code-review-2026-09-10-reproduce.py
```

Linux／Windows、Python 3.11・3.13・3.14、CUDA／MPS、実ESM2 650Mモデルの全量学習・蒸留・外部データ再評価は未実施。全コードの全分岐や生物学的妥当性を保証するものではない。未公開の既定モデルを指定すると明示的エラーになる変更は、現在のhelp／READMEに記載された意図的な状態として扱い、新規不具合には数えていない。

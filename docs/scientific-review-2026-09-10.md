# CDSKIT 科学的妥当性レビュー（2026-09-10）

> これは修正前の作業ツリーを対象とした履歴資料です。以下の指摘と数値は現在版の未解決問題一覧ではありません。修正後の状態は [科学的契約](localize-scientific-contract.md)、[検証結果](localize-scientific-validation.md)、[今回の修正・性能測定](review-fixes-and-performance-2026-09-10.md)を参照してください。再現スクリプトは実行時のコードを観察するため、修正前と出力が異なります。CDSに交差するgapjust編集は、現在版では拒否メッセージとして記録します。

科学的な根拠が不足する機能はある。それに加えて、既知の生物学的定義と食い違う実装も確認した。特に優先すべきなのは、PTS2の配列長、終止コドンの判定、教師ラベルの意味、交差検証の独立性である。

一方、CDSKIT全体を科学的に不適切と評価する根拠はない。決定的な配列変換、探索用ヒューリスティック、学習済み予測器は異なる基準で評価すべきである。既存資料には限界を正しく説明した箇所も多い。本レビューでは、既に開示された限界も利用判断に必要なため含めた。

## 対象・方法・検証範囲

- 対象は `/Users/kf/repos/cdskit` の作業ツリー。HEADは `cd89830c75fa3d0ef5d5a4995490af2feb6ff031`、パッケージ表示は0.29.2。レビュー開始時から存在した未コミット変更を含む。リリース版単独の監査ではない。
- CLI・READMEから機能を棚卸しし、コドン分類、翻訳、読み枠補完、ORF、アラインメント選別・復元、局在特徴量、教師ラベル、CV、閾値調整、推論を重点的に確認。補助実験スクリプトは主要経路の確認であり、約4万行の全分岐を形式的に検証したものではない。
- 原著論文・NCBI・UniProt・GFF3仕様と照合し、下記の小入力を実行した。公開統合モデルとSHA-256が一致するローカルチェックポイントでも境界入力を実測した。
- 既存の全量実験レポート・監査・追加115陽性のレポートを読んだ。過去の全量学習、84,909件のOOF再計算、MMseqs再検索は今回実行していない。過去の数値と今回の再現結果を区別する。
- `python3 scripts/check.py quick`: **823 passed、2 skipped、20 deselected**。skipはPyYAMLを必要とする2件。
- `python3 scripts/check.py ml`: **150 passed、829 deselected**。
- 最初の既存core環境での直接pytestはfilelock不足で収集失敗した。上記公式エントリポイントでlockに同期後、成功した。ソース・lock・モデルの修正は行っていない。
- quality、coverage、build、他OS・他Pythonの検証は今回のレビューでは未実施。テスト成功は科学的妥当性の証明ではない。

優先度P1は誤った生物学的判定や不適切な精度解釈に直結するもの、P2は適用条件や出力の改善が必要なもの。再現で確認した実装不整合と、追加実験が必要な限界を区別する。

## 1. P1：PTS2の正規表現が1残基短い【実装不整合・再現済み】

根拠：`cdskit/localize_model.py:122`、`detect_perox_signals`、同ファイルの特徴量生成。

現実装は `[RK][LIVQ].{4}[HQ][LA]`、合計8残基。典型的PTS2は間に5残基を持つ9残基モチーフである。原著のヒトthiolaseの機能確認例 `RLQVVLGHL` にも一致しない。[Kunze et al., Structural Requirements for Interaction of Peroxisomal Targeting Signal 2 and Its Receptor PEX7](https://pmc.ncbi.nlm.nih.gov/articles/PMC3247985/)

今回の実測：

| 入力 | 現在のpts2_match | 解釈 |
|---|---:|---|
| `MRLQVVLGHLAAAA` | false | 原著の機能確認モチーフを含むが見逃す |
| `MRLQVVVHLAAAA` | true | 8残基の別パターンに一致する |

後者が生体で絶対に機能しないという主張ではない。しかし標準PTS2検出器としての定義は誤っている。`perox_signal_type`だけでなく、学習・推論の`pts2_match`特徴量とregexベースラインにも波及する。過去の精度への影響量は未測定。

対処：原著に基づいて正規表現を修正し、実験的陽性と長さを変えた対照で検証する。**既存モデルの入力特徴も変わるため、単なる正規表現の置換だけで旧重みをそのまま使わない。** 特徴量定義をバージョン化し、必要な再学習・閾値再調整・外部評価を行う。

## 2. P1：特殊遺伝暗号の内部コドンを誤ってstop扱いする【実装不整合・再現済み】

根拠：`cdskit/codonutil.py:103`の`classify_codon`、`cdskit/pad.py:25`のstop集合、`cdskit/longestcds.py:45`のstart/stop集合。

翻訳側はforward tableを優先するが、QC側はstop集合に含まれるだけで終止コドンと判定する。表27/28/31では同じコドンが文脈依存でアミノ酸にも終止にもなるため、内部位置の単純なstop集合照合では正しい判定にならない。[NCBI Genetic Codes](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi?chapter=cgencodes)、[Swart et al., 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4967479/)

| コード | 入力 | 実際の翻訳出力 | QCの内部stop |
|---|---|---|---:|
| 27 | `ATGTGAAAA` | `MWK` | true |
| 28 | `ATGTAATAGTGAAAA` | `MQQWK` | true |
| 31 | `ATGTAATAGAAA` | `MEEK` | true |

`filter`の配列除外、`validate`の異常報告、`trimcodon`のclean判定、`codonstats`のstop数に影響する。`pad`とORF探索も同じ集合ベースの問題を持つ。一方、翻訳ベースの`mask`・`hammer`は同じ挙動ではない。

対処：内部翻訳、完全CDS末端、文脈不明を分けた共通判定を定義する。文脈を推定できない条件は「不明」とし、確定的な内部stopとして除去しない。単に全コマンドを現在のQC側へ合わせる修正は不可。

## 3. P1：確実にstopとなる曖昧コドンを内部stop検査が見逃す【実装不整合・再現済み】

根拠：`cdskit/codonutil.py:116`、`cdskit/pad.py:49`。

標準コードの`TAR`はTAAまたはTAGで、どちらもstop。しかし分類は「曖昧」を先に確定し、内部stopには数えない。

今回の入力 `ATGTARAAA`：翻訳は`M*K`、`filter.analyze_record`は`internal_stop=False`、`stop_codons=0`、clean fraction=2/3。`pad`も`is_no_stop=True`とした。したがって内部stop除外だけでは検出されず、通常のclean fraction=0.5も満たす。

対処：曖昧性とstop可能性を別属性にする。全展開がstopなら確定stop、一部だけstopなら可能性ありとする。`TAN`のような混合を確定stopと誤認しないテストも必要。IUPAC曖昧翻訳を既に扱う翻訳処理と意味を統一する。

## 4. P1：局在から輸送ペプチドの有無を直接教師ラベル化する【教師データの意味が不十分】

根拠：`cdskit/localize_model.py:737`、`cdskit/targetp_labeling.py:1`、`cdskit/localize_learn.py:337`、`cdskit/targetp_external_aug.py:163`。

`mitochond`という文字列だけでmTP、`secreted`だけでSP、`chloroplast`または`plastid`だけでcTPにする。これは「成熟タンパク質の局在」と「N末端輸送ペプチド」の別の教師信号を混ぜる。UniProtも局在とtransit peptideを別項目として定義し、推定によるtransit注釈も明示している。[UniProt subcellular location](https://www.uniprot.org/help/subcellular_location)、[UniProt transit peptide](https://www.uniprot.org/help/transit)

今回の合成注釈に対する実測：

- `Mitochondrion outer membrane.` → mTP。文面だけでは切断型N末端presequenceの存在を確認できない。
- `Secreted. Note=Secreted by an unconventional pathway.` → SP。非典型経路という説明を解釈しない。
- `Cytoplasm. Note=Does not localize to mitochondria.` → mTP。否定文中の部分文字列も陽性になる。

これらはラベル関数の挙動確認用入力であり、実データ中の誤ラベル頻度を測定したものではない。`strict`関数は一部の競合や膜ノイズを除くが、輸送ペプチド実証や証拠コードを確かめるものではない。外部データが同じラベル規則なら、高精度でも規則の再現性を測っている可能性がある。

対処：実験的SIGNAL/TRANSITとその座標・証拠を教師ラベルの中心に置く。局在だけからのラベルはweak labelとして分離し、評価は独立に確証されたペプチドラベルで行う。既存の外部追加学習を無価値とはしないが、実験的正解と同等に扱わない。

## 5. P1：注釈欠損を陰性に変換する【再現済み・偽陰性の混入リスク】

根拠：`cdskit/localize_model.py:737`、`cdskit/localize_learn.py:338`。

一般の`uniprot_cc`経路で空の局在文字列は`('noTP', 'no', False)`になる。明示ラベル経路もperox列の欠損を`no`で補う。欠損と実験的陰性は同じではない。

影響は学習だけでなく、外部検証の「誤検出」にも及ぶ。保存済み[追加陽性監査](../data/localize_bench/perox_expansion_20260908/report.md)には、開発側の陰性ラベル12配列に現在の実験的ペルオキシソーム注釈があると記録されている。これは単なる仮説ではないが、時点・条件・アイソフォームの相違も調べる必要がある。

対処：unknownを表現し、未注釈ラベルを損失・評価から除外できるようにする。陽性・陰性・未観測を区別する。注釈がないだけの配列を「非局在が確認された陰性」と記述しない。

## 6. P1：通常のlocalize-learn CVは同一配列・相同配列を分離しない【再現済み】

根拠：`cdskit/localize_learn.py:681`、`cdskit/localize_learn.py:1220`。

`--cv_folds`はクラス内をランダムに分ける。配列重複や相同クラスタによるグループ化はない。`--cv_fold_col`で適切な分割を与えられるが、その分割の生物学的独立性をこの経路自体は保証しない。

今回、5種類の配列を各2コピー、各クラスに1種類ずつ割り当て、2-fold CVを実行した。両foldの完全一致配列の共通数は5、nearest-centroidのCV精度は**1.0**だった。新しい配列の予測能力を示す値ではない。

相同性を無視した分割が性能評価を歪めることは配列学習の既知の問題。[SpanSeq, 2024](https://academic.oup.com/nargab/article/6/3/lqae106/7734174)

対処：完全一致重複の拒否または同一group化、MMseqs等による分割、fold間重複監査を導入する。ランダム分割を残すなら用途を明示する。新しいDeepLoc・pipeline経路には分割監査が存在するため、**全ML経路が無防備という指摘ではない。**

## 7. P1：OOFで調整した閾値を同じOOFで採点する【評価設計の問題】

根拠：`cdskit/localize_learn.py:1840`、`cdskit/localize_learn.py:1907`付近。

全OOFを用いてtemperature・クラス閾値を最適化し、同じOOFで`cv_postproc_*`を算出する。`two_stage_ctp_ltp`もOOF全体でgate/beta/thresholdを選び、その最適値の性能をOOF指標として保存する。

ベースモデルにとってのOOFであっても、後処理の学習器にとっては訓練データである。これらは調整集合の性能であり、最終パイプラインの独立CV精度ではない。OOFで後処理を学習する行為自体は妥当だが、その値を汎化性能として使うのは不適切。

対処：外側foldを完全に留保し、内側で後処理を調整してから外側を採点する。または調整指標と独立指標の名前を明確に分ける。今回、過大評価の実データ上の大きさまでは測定していない。現在のDeepLocの内側validation分離をそのまま同罪にはしない。

## 8. P1：gapjustでCDS内のN数を変えてもGFF phaseを保持する【再現済み】

根拠：`cdskit/gapjust.py:31`、`apply_gap_justifications_to_gff`。

N連続区間を任意長へ変換し、座標は更新するがphaseは更新しない。編集がCDS内部に入り、長さ差が3の倍数でない場合、下流CDSのphaseが不整合になる。イントロンや遺伝子間だけを変える場合は同じ問題とは限らない。

今回、同じ転写産物の+鎖CDSを1–9（phase 0）と13–21（phase 0）とし、最初のCDS内のNNNをNNNNに変更した。出力は1–10／14–22、phaseは両方0のまま。変更後の連結CDSに整合させるなら次CDSのphaseは2が必要。[GFF3 specification](https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md)

ただしphaseを再計算しても失われた生物学的読み枠が復元されるわけではない。対処はCDSに交差する編集を拒否・別扱いにするか、転写産物と鎖を追跡した整合性検証・再注釈を行うこと。ゲノムのscaffold gap整形とCDS修復を混同しない。

## 9. P2：padの最少stop基準とdrop_pseudoは遺伝子機能の検証ではない【ヒューリスティックの限界】

根拠：`cdskit/pad.py:138`、`cdskit/pad.py:243`。

今回 `ATGTAAGGG` が `NATGTAAGGGNN` になり、`is_no_stop=True`になった。元のフレームの内部stopが、フレーム変更によって見えなくなる。これは出力の算術的性質であり、元が機能遺伝子だった、あるいは正しい枠を復元したという証拠ではない。

末端の不足塩基数が不明な部分CDSを補う探索には合理性がある。一方、短い・GCに富む配列ではstopがない別枠も存在し得る。内部frameshiftや偽遺伝子を末端paddingで修復できるとはいえない。`--drop_pseudo`は「調整後にstopが残るものを落とす」という限定的規則。

対処：元フレーム、候補別stop数、同点、補完量を機械可読に残す。注釈済みCDSの枠変更は明示的に区別し、必要に応じて相同タンパク質との整合性で検証する。実際の修復精度を示すには既知CDSを使った欠損・frameshiftシミュレーションが必要。

## 10. P2：backtrimは同じアミノ酸列から元コドン位置を一意に復元できない【再現済み】

根拠：`cdskit/backtrim.py:71`、特に複数一致時に先頭候補を採用する部分。

次の2配列では、両コドン列ともアミノ酸列パターンが`AA`になる。

| 配列 | 元コドン1 | 元コドン2 |
|---|---|---|
| s1 | GCT | GCC |
| s2 | GCT | GCT |

元のタンパク質は両方`AA`。トリミング後が両方`A`なら、残ったのが1列目か2列目かは分からない。現実装は警告して1列目を採用し、出力はGCT/GCT。実際に残すべきだったのが2列目なら、GCC/GCTという同義変異を失う。

アミノ酸一致と順序だけでは原理的に情報不足であり、コードの探索を工夫するだけでは一般に解決しない。dSや系統解析の入力に影響し得る。

対処：トリマーの保持列インデックスを第一選択にする。復元が複数通りある場合は曖昧として停止・報告するモードを設ける。現行警告は有益だが、どちらが正しいかを保証しない。

## 11. P2：longestorfは最長を常に選ぶわけではなく、遺伝子同定器でもない【仕様上の限界】

根拠：`cdskit/longestcds.py:79`、`choose_best_candidate`、[既存仕様](../wiki/cdskit-longestorf.md)。

順位はcomplete > partial > no_startで、長さはその後。したがって短いcompleteが長いpartialや開始なしの領域より優先される。今回 `ATGTAA` に100個のCを続けた106 nt入力では、6 ntの`ATGTAA`を選んだ。

仕様には開示されているので、単純な実装バグとはしない。しかし「最長ORFを得たから目的遺伝子」と解釈できない。開始候補の頻度、翻訳開始文脈、発現、相同性、イントロン構造は検証していない。

対処：カテゴリ優先と全候補中の長さ優先を区別し、候補一覧・元座標を利用できるようにする。遺伝子同定用途では独立の注釈根拠が必要。

## 12. P2：局在モデルは低情報入力にもラベルを返し、スコアの意味もモデル間で違う【一部実測・既知の限界】

根拠：`cdskit/localize_multilabel_cnn.py:638`、`cdskit/localize_model.py:1084`、`fit_perox_binary_classifier`、[統合モデル仕様](../wiki/cdskit-localize-multilabel-integrated-v1.md)。

公開統合モデルと一致するSHA-256 `e9b35ead3eca4dcdf833e469f18f1d67e05a3de274b00861cbf08bf138d01621` のローカルモデルで、現コードのCPU推論を実測した。

| 入力 | 出力ラベル | 最大スコア |
|---|---|---:|
| 空文字（Python予測API） | nucleus | 0.421338 |
| Xを100個 | nucleus | 0.493578 |
| Mを1個 | extracellular | 0.664849 |

これらを生物学的局在の証拠としては扱えない。`ensure_one_label`は全ラベルが閾値未満でも1つを選ぶ。ただし上記3例のそれぞれが強制選択だけに起因するという主張ではない。入力情報量の検査と判定保留は別に必要。

また、統合モデルにはtaxonomy maskingがなく、過去の監査にはヒトHPAでchloroplastを8件予測した記録がある。`--organism_group non_plant`によるcTP/lTP制約は別の5クラスモデル用である。単にnon_plantを全て非プラスチドとするのも、藻類・原生生物等を考慮すると粗い。

`p_*`は独立分類器のスコアであり、F1閾値最適化は確率校正の検証ではない。さらに従来targeting5の`p_peroxisome=0`は定数headであり、生物学的非局在の強い証拠ではない。

対処：低情報・断片・適用範囲外を明示し、保留を選べるようにする。出力にheadの有無とスコアの種類を示す。確率として利用するには独立集合上のreliability、Brier等を確認する。新pipelineの`ensure_one_label=False`既定値は改善だが、既存公開モデルの挙動は自動的には変わらない。

## 13. P2：外部汎化・希少局在・蒸留の優位性は未確立【実験結果の解釈】

根拠：[全量実験レポート](../data/localize_bench/full_localization_20260908/report.md)、[監査](../data/localize_bench/full_localization_20260908/audit/report.md)、[追加陽性評価](../data/localize_bench/perox_expansion_20260908/report.md)。以下は保存された過去の結果で、今回の全量再評価ではない。

| 検証 | 結果 | 支持できる解釈 |
|---|---|---|
| 公式5-fold・3 seed | 統合macro F1 0.5040、baseline 0.4028 | この開発プロトコル上の改善 |
| 既に閲覧したHPA | 統合macro 0.2203、baseline 0.2195 | 全局在での大幅改善とはいえない |
| HPA perox 7陽性 | 統合0/7 | この集合では未検出。7件だけで一般的再現率0とも断定しない |
| 追加UniProt 115陽性 | 統合25/115、baseline 31/115 | 統合の一貫した優位性を支持しない。陰性がないのでprecision/F1は測れない |
| 同容量control対distilled | 平均macro 0.4750対0.4766、micro 0.5758対0.5707 | 蒸留が両指標を改善したとはいえない |

追加115陽性の再現率差のクラスタbootstrap 95%区間は−20.3〜+9.5ポイントで、優劣を確定できない。HPAは最終の未知ホールドアウトではない。公式foldにも別条件のMMseqsクラスタをまたぐ540配列があるが、その除外後も統合の改善は維持されるとの記録がある。これをもって「改善は全てリーク」と断定するのも誤り。

CNNのlegacy入力は長配列のN/C末端を直接連結するため、実在しない隣接関係と中央領域の欠落を作る。ただしseparate-termini対照は開発試験で明確な改善を示していない。理論的に自然な表現への変更だけで精度改善を保証してはいけない。

対処：機能を実験的と位置づけ、モデル選択に一切使わない新しい外部評価を固定する。分類群・長さ・膜/マトリックス・断片性で性能を分け、陰性注釈の不完全性も監査する。TargetP原著との異なる集合上の数値を同等精度の証拠にしない。既存wikiはこの点を適切に注意書きしている。

## その他の機能の判定

| 機能群 | 今回の評価・用途の条件 |
|---|---|
| translate / backalign | 選択コード・正しい読み枠・正しいタンパク質対応が前提。変換自体は妥当な基盤。特殊な再コード化や注釈例外まで自動推定する機能とは解釈しない |
| degeneracy | 現コドンの1塩基置換に基づく同義性分類は合理的。fold一致はアミノ酸一致や進化史全体の同義性を保証せず、4-foldを自動的に「選択のない中立サイト」と呼ばない |
| hammer / trimcodon | occupancy/clean fractionによる整形は合理的だが、相同性やアラインメントの正しさの証明ではない。hammerのprevent_gap_onlyは全体の閾値を下げ得るため、下流解析の感度検証が必要 |
| maxalign | 最大化するのは配列数×完全列数という目的関数。系統樹精度を直接最大化してはいない。除外taxonと閾値への感度を見る |
| mask / filter / validate | 規則ベースQCとして有用。マスク後のstop消失を機能遺伝子の証明にしない。上記2・3の共通stop判定は修正対象 |
| codonstats / stats | 集計は記述統計。statsはgap/曖昧をGC分母に含み、codonstatsはACGTだけ。欠測が多い群の比較では同じGCと扱わない。分母0の0%は「測定不能」と区別する改善余地あり |
| aggregate / longest配列選択 | 最長アイソフォーム選択は代表配列作成ルールであって、主要発現・機能的アイソフォームの証明ではない |
| I/O・ラベル・抽出・plot | 科学的推定を主目的としない。今回、独立した科学的仮説の根拠不足を主要問題としては検出しなかった。完全な正しさの保証ではない |

## 修正・検証の順序

1. PTS2、特殊コード、確定曖昧stopを原著・遺伝暗号に基づく小さな回帰ケースで修正する。特徴量変更によるモデル互換性を管理する。
2. 教師データのtargeting/localization/unknownを分離し、全CV経路で配列重複と相同性を監査する。後処理を含むnested評価へ移行する。
3. gapjustのCDS交差処理とbacktrimの曖昧位置復元を安全に扱う。元データの位置情報を失わない出力を用意する。
4. pad・longestorf・局在出力の適用範囲と「不明」を明確にする。
5. 凍結した新規外部データで修正後モデルを再評価する。現在の開発結果や同じHPAを何度も見て最適化した値を、未知データ性能として更新しない。

本レビューでは実装の修正は行わず、判定の根拠と実行可能な再現例を残した。

再現スクリプトは [scientific-review-2026-09-10-reproduce.py](scientific-review-2026-09-10-reproduce.py)。現在の挙動を観察するもので、これらの挙動を正しい仕様として固定するテストではない。

```bash
.venvs/core-3.12/bin/python docs/scientific-review-2026-09-10-reproduce.py

# ローカルに保存された公開統合モデルの境界入力も確認する場合
.venvs/full-3.12-cpu/bin/python docs/scientific-review-2026-09-10-reproduce.py \
  --model data/localize_bench/full_localization_20260908/final/integrated.pt
```

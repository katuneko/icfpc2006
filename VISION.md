# VISION: ICFPC 2006 (UMIX/Codex)

## 目的
- Codex（UMプログラム）をUMで実行し、UMIX内の複数パズルを解いてpublicationを集める。
- Adventureでdownloader/uploaderを修理し、gc.rmlを改造してMuseumへ到達・blueprint取得まで完了。現在はblueprintの検閲回避と残パズル攻略が主眼。

## 全体フロー（機能的な流れ）
1. UMインタプリタ（um.rs -> um）でcodex.umzを起動し、UMIX本体（volume9.um）を得て実行する。
2. volume9.um内のUMIXにログインし、各アカウントでパズルを進める。
3. Adventureで修理/改造/テレポートを成立させ、blueprintを取得（内容は検閲で未読）。
4. Antomaton/2D/Black Knots/Balance/O'Cultを解いてpublicationを増やす。
5. 入力ログと出力ログで再現性を確保し、walkthrough.mdに進捗を反映する。

## 主要コンポーネント
- UM実装: `um.rs`, `um`
- UMプログラム: `codex.umz`, `volume9.um`, `sandmark.umz`
- Adventureのゲームロジック: `gc.rml`, `gc_patched.rml`
- パズル群:
  - Antomaton（gardener）
  - 2D（ohmega）
  - Black Knots（bbarker）
  - Balance（yang）
  - O'Cult（hmonk）
- ローカル補助スクリプト: `occult.py`, `ant_solver.py`, `balance_solver.py`, `black_knots_solver.py`, `two_d.py`

## 重要ルール/制約
- アイテムはスタック構造で取得順が制限される。
- `combine` は対象アイテムを手持ちにする必要がある。
- `missing` は再帰表示だが、修理に必要なのは直下のmissingのみ。
- インベントリ上限/ドロップ不可/焼却は `gc.rml` 実装に依存し、パッチで変更可能。
- Censory Engineにより不可視アイテムが存在し、Readingモードや名称指定が必要になる場合がある。
- Museumは通常移動では到達困難で、テレポート実装が前提。

## Adventure (UMIX)
- 入口: `howie / xyzzy` でログインし `adventure` を実行する。
- 目的: downloader/uploader修理 -> `use downloader` で `gc.rml` 取得 -> 改造 -> `use uploader` で反映。
- 改造の要点: インベントリ上限解除、`drop` の有効化、`speak home/trash/museum` のテレポート追加。
- 参照入力: `howie_repair_downloader_full_and_dump_gc_input.txt`, `howie_upload_gc_patched_and_museum_input3.txt`。
- 参照ログ: `volume9_howie_repair_downloader_full_and_dump_gc.txt`, `volume9_howie_upload_gc_patched_and_museum3.txt`。
- 到達点: Museum入口へテレポートし、`south` のRotundaで `take blueprint` まで成功済み。
- Machine Room M4 で `crowbar`/`note` を取得し、`Censory Engine` を破壊して `ADVTR.CRB` を取得（視界がクリアに）。`History of Technology Exhibit, Entrance` まで到達（warranty void で入場不可）。
- 追加の出版: `ADVTR.PGB`, `ADVTR.USB`, `ADVTR.PWR`, `ADVTR.JMP`, `ADVTR.DSP`, `ADVTR.BTY`, `ADVTR.MOS`, `ADVTR.232`, `ADVTR.EPM`, `ADVTR.LED`, `ADVTR.CRB`。
- `knr / X3.159-1989` を note から取得しログイン成功。`ucc` で `INTRO.UCC` を出版。
- 未解決: `examine blueprint` がREDACTEDになるため内容読解が未達。

## Antomaton (gardener)
- 入口: `gardener / mathemantica` でログインし `./antomaton world.ant` で実行する。
- 形式: 盤面とターンマシン（p1..p7）で蟻を動かすパズル。蟻は `0..9` + 方角で表現。
- 補助: `ant_solver.py` で盤面抽出・探索・シミュレーションを行う。
- 解答済み: `puzzle1_solution.ant`, `puzzle2_solution.ant`, `puzzle15_solution.ant`。
- 途中: `puzzle5_solution.ant` は候補（成功条件の解釈が未確定）。
- 参照入力: `gardener_readme_input.txt`, `gardener_puzzles_input.txt`, `gardener_antomaton_example_input.txt`。
- 参照ログ: `volume9_gardener_readme.txt`, `volume9_gardener_puzzles.txt`, `volume9_gardener_antomaton_example.txt`。
- 未解決: 成功条件の確認、Puzzle 14 のターンマシン行長の仕様確認、残パズル探索の強化。

## 2D (ohmega)
- 入口: `ohmega / bidirectional` でログインし `2d` と `verify` を使用する。
- 仕様: `mult.spec`, `reverse.spec`, `raytrace.spec`, `aspects.spec` の要件を満たす回路設計が必要。
- 補助: `two_d.py` で簡易解釈、`plus.2d` を抽出済み。
- 進捗: `mult.2d`/`rev.2d` を strict-wires 対応で修正し、UMIX 検証に成功（`CIRCS.MUL=30@999999|fe8a47581d2a95699b216c13fb250bd`、`CIRCS.REV=35@999999|d4481d7a04981746dc23d1c0b7c665e`）。
- `rev_acc` モジュール名は `revacc0` に変更。`raytrace.2d` 用の補助モジュール（`i_max`/`i_min`）を作成。
- 参照入力: `ohmega_readme_input.txt`, `ohmega_tools_input.txt`。
- 参照ログ: `volume9_ohmega_readme.txt`, `volume9_ohmega_tools.txt`。
- 未解決: 最小面積の設計と `plus.2d` の配線解釈（N/W接続競合）確認。

## Black Knots (bbarker)
- 入口: `bbarker / plinko` でログインし `run_bb`/`verify` を使う。
- 形式: `|><` のグリッド機械を対話入力し、指定モデルを満たすものを提出する。
- 解答済み: モデル000/010/030/040/050/100/200。
- 補助: `black_knots_solver.py` による探索を進める余地がある。
- 参照入力: `bbarker_verify_000_submit_input.txt`, `bbarker_verify_010_submit_input.txt`, `bbarker_verify_030_submit_input.txt`, `bbarker_verify_040_submit_input.txt`, `bbarker_verify_050_submit_input.txt`, `bbarker_verify_100_submit_input.txt`, `bbarker_verify_200_submit_input.txt`。
- 参照ログ: `volume9_bbarker_verify_000_submit.txt`, `volume9_bbarker_verify_010_submit.txt`, `volume9_bbarker_verify_030_submit.txt`, `volume9_bbarker_verify_040_submit.txt`, `volume9_bbarker_verify_050_submit.txt`, `volume9_bbarker_verify_100_submit.txt`, `volume9_bbarker_verify_200_submit.txt`。
- 未解決: モデル020の機械設計と探索戦略の体系化。

## Balance (yang)
- 形式: 16進数2桁の列（空白なし）をプログラムとして提出する。
- 解答済み: `stop`, `stop1`, `stop127`, `stop128`, `addmem`, `addmem2`, `swapmem`, `swapreg`, `swapreg2`, `copymem`。
- `clearreg` は認証済み（`BLNCE.CRR=97@999999|7a18c38d7690f1d74db0b2446b68837`）。
- 補助: `balance_solver.py` で探索補助。
- 参照入力: `yang_certify_input.bin`, `yang_certify2_input.bin`, `yang_certify_copymem_input.txt`。
- 参照ログ: `volume9_yang_certify.txt`, `volume9_yang_certify2.txt`, `volume9_yang_certify_copymem.txt`。
- 未解決: `copyreg`, `multmem`, `fillmem`（`clearreg` は候補）。

## O'Cult (hmonk)
- 入口: `hmonk / COMEFROM` でログインし、ローカルでは `occult.py` で検証する。
- 解答済み: `arith4.adv`（`arith.tests` を通過）、`xml2.adv`（XMLスイート用）。
- 参照入力: `hmonk_arith4_upload_input.bin`, `hmonk_xml_upload_input.bin`。
- 参照ログ: `volume9_hmonk_arith4_run.txt`, `volume9_hmonk_xml_run.txt`。
- 未解決: 追加スイートや最短化の検討余地がある。

## 現在の到達点（要約）
- downloader/uploader修理、gc.rml取得→改造→`use uploader`反映まで再現済み。
- `speak museum` でMuseumへ到達し、blueprintを取得済み（内容はREDACTED）。
- 取得済みpublicationは42件、合計1600点（`walkthrough.md`の一覧より）。
- Antomaton/Black Knots/Balance/O'Cultで一部パズルは解答・出版済み。

## 未解決の焦点
- blueprintの検閲回避とHistory of Technology Exhibit入場条件の解決。
- Antomaton/2D/Black Knots/Balanceの未解パズル群（特にBlack Knots 020、2Dのverify投入）。
- 未取得アカウント（例: ftd）の認証情報探索。

## 再現コマンド（最小）
- Build UM: `rustc um.rs -O -o um`
- Run UM program with input: `./um volume9.um < INPUT > OUTPUT`
- O'Cult tests: `python3 occult.py arith4.adv arith.tests`

# ICFPC2006 Adventure 問題定式化（プロ向け入力）

## 目的
UMIX Adventure 内で `downloader` と `uploader` を修理し、Museum of Science and Industry に到達するための具体手順を確立する。

## 前提（環境・データ）
- 作業ディレクトリ: `/home/ktnk/work/icfpc2006`
- UM 実行: `./um volume9.um < input.txt > output.txt`
- Adventure へは howie でログイン（`xyzzy`）し `adventure` を実行。
- 参考ログ/入力は全てこのディレクトリに保存済み。

### 主要ログ/入力ファイル
- XML での街スキャン（ゴーグル XML）:
  - `volume9_howie_xml_scout.txt`
  - `volume9_howie_xml_scout2.txt`
  - `volume9_howie_xml_scout3.txt`
  - `volume9_howie_xml_scout4.txt`
  - 入力: `howie_xml_scout_input.txt`, `howie_xml_scout2_input.txt`, `howie_xml_scout3_input.txt`, `howie_xml_scout4_input.txt`
- 失敗した修理自動化試行:
  - 入力: `howie_repair_phase1c_input.txt`
  - 出力: `volume9_howie_repair_phase1c.txt`
- インベントリ制限の実測:
  - 入力: `howie_capacity_test2_input.txt`
  - 出力: `volume9_howie_capacity_test2.txt`
- ドロップ不可の検証:
  - 入力: `howie_drop_part_test_input.txt`
  - 出力: `volume9_howie_drop_part_test.txt`
- キーパッド破棄可の検証:
  - 入力: `howie_incinerate_keypad_test_input.txt`
  - 出力: `volume9_howie_incinerate_keypad_test.txt`
- 開始位置の確認:
  - 入力: `howie_start_location_test_input.txt`
  - 出力: `volume9_howie_start_location_test.txt`

## 開始状態（確定）
- `use keypad` 後の開始位置は **54th Street and Ridgewood Court**。
- ここに以下のスタックがある:
  - `/etc/passwd` → `note` → `downloader` → `uploader`
- `/etc/passwd` と `note` は `incinerate` で破棄可能。
- `downloader` / `uploader` は **drop 不可**（「手放せない」）。

## 既知ルール/制約（重要）
- インベントリ上限は **6 個**。
  - 実測: `downloader + uploader + keypad + 3個` で満杯。
- `drop` は多くのアイテムで不可（通りの部品も不可）。
- `incinerate` は一部アイテムで可能。
- アイテムは**スタック構造**で、上から順に `take` しないと下は取得できない。
- `combine A with B` は **A と B が手持ち**にある必要がある。
  - ただし「A の欠損が全て埋まっている必要」は誤り（A がまだ broken でも、欠損の 1 つを埋める combine は成立する）。
- XML で表示される `missing` は **再帰的**（内部の broken パーツの欠損まで含まれる）。

### 検証済みの重要ポイント: `missing` は再帰だが、修理に必要なのは **直下の missing**
`missing` は再帰的にネストして表示されるため「見えている欠損を全部集めないといけない」と誤解しやすいが、
実際には `combine` が要求するのは **対象アイテムの `condition` 直下に現れる missing（= immediate missing）**だけ。
`kind` 側がさらに `broken` になっていて `missing` を抱えていても、その内側欠損は **埋めなくてよい**（そのまま差し込める）ケースがある。

#### 実証（USB cable）
`USB cable` の再帰 missing には `X-6458-TIJ` が出てくるが、`X-6458-TIJ` を取らずに `USB cable` を修理して `downloader` に結合できることを確認。
つまり `X-6458-TIJ` は「`T-9887-OFC` の内側欠損（nested missing）」であり、少なくとも `downloader` と結合するためには不要。
  - 入力例: `howie_repair_usb_cable_input.txt`（`./um volume9.um < howie_repair_usb_cable_input.txt`）

## 追加の決定的情報（Leo提供・未検証）
- Censory Engine は **知覚のみ**妨げる（存在すれば操作自体は可能）。
  - つまり表示に出なくても、**名前が分かっていて存在していれば `take`/`combine` は成立する可能性**がある。
- `switch goggles Reading` で「高すぎて見えない」山の**完全列挙**ができる。
- インベントリ上限や drop 不可は **`gc.rml` 側の実装**。\n  - `use downloader` で `gc.rml` を表示できる。\n  - `use uploader` で `gc.rml` を **EOM 終端の入力で更新**できる。\n  - 例: 6 制限を 666 に変更して解除する実例がある。
- `incinerate` は **消滅ではなく TRASH へ移動**（後で回収可能）。
- Museum へは **通常移動ではなくテレポートが必要**とされる。\n  - `gc.rml` 改造でテレポート実装が事実上必須。

## マップ（確認済みの座標系）
```
Ridgewood (54th & Ridgewood) -- east -> 54th & Dorchester -> 54th & Blackstone -> 54th & Harper
                                   |                       |                     |
                                  north                   north                 north
                                   |                       |                     |
                               53th & Dorchester -> 53th & Blackstone -> 53th & Harper
                                   |                       |                     |
                                  north                   north                 north
                                   |                       |                     |
                               52nd & Dorchester -> 52nd & Blackstone -> 52nd & Harper
                                   |
                                 south
                                   |
                               54th Place & Dorchester -> 54th Place & Blackstone -> 54th Place & Harper
```
- Harper 側は東方向が「Lakeshore 以東通行不可」。

## 修理対象の主要部品（壊れた部品）
各部品は指定の交差点に存在。
※欠損は「再帰的 missing」なので、表面上の部品だけでは不十分。

### Downloader 系
- USB cable（54th & Dorchester）
  - 欠損: `T-9887-OFC`, `F-6678-DOX`, `N-4832-NUN`, **`X-6458-TIJ`**
- display（53th & Dorchester）
  - 欠損: `N-1623-AOE`, `R-4292-FRL`, `L-9247-EHW`, `B-1623-YTC`, `D-5065-UBI`, `Z-6458-PXZ`, `V-9887-KUS`, **`F-4292-DWJ`**, **`P-4832-JKF`**, **`T-1403-ONM`**
- jumper shunt（54th Place & Blackstone）
  - 欠損: `D-4292-HCL`, `T-6678-BOP`, `B-5065-YGK`, `L-6458-RIZ`,
          **`F-6678-DJR`**, **`H-9887-MFS`**, **`J-9247-IMY`**, **`N-4832-NPH`**, **`R-1403-SSO`**, **`V-0010-XVV`**, **`X-9247-GRW`**, **`Z-1623-CYE`**
- progress bar（52nd & Dorchester）
  - 欠損: なし（確認済み）
- power cord（54th Place & Dorchester）
  - 欠損: `N-4013-DJW`, `V-1623-KEM`, `H-6458-ZNJ`, `Z-4292-PHT`, **`L-5065-EQQ`**

### Uploader 系
- MOSFET（54th Place & Harper）
  - 欠損: `J-6458-VIH`
- status LED（52nd & Harper）
  - 欠損: `L-4832-RPN`, `D-6678-HJX`, `H-9247-MMG`, **`L-5065-EBS`**
- RS232 adapter（54th & Harper）
  - 欠損: `D-9887-UUE`, `Z-4292-PRV`（重複多数）
- EPROM burner（53th & Harper）
  - 欠損: `X-1623-GTO`（重複多数）
- battery（53th & Blackstone）
  - 欠損: `Z-1623-COC`, `H-9887-MUQ`, **`R-1403-SIM`**

## 不可視/別地点の欠損パーツ（Censory Engine 疑い）
以下は「その部屋のアイテム一覧に出てこない」ため、通常の `take` で見つからない。
- `X-6458-TIJ` は **53th & Harper** の深い位置に存在（XML に記録あり）。
- display 欠損: `F-4292-DWJ`, `P-4832-JKF`, `T-1403-ONM`（部屋に見えない）
- jumper shunt 欠損: `F-6678-DJR`, `H-9887-MFS`, `J-9247-IMY`, `N-4832-NPH`, `R-1403-SSO`, `V-0010-XVV`, `X-9247-GRW`, `Z-1623-CYE`
- power cord 欠損: `L-5065-EQQ`
- status LED 欠損: `L-5065-EBS`
- battery 欠損: `R-1403-SIM`

## 失敗ログの要点（証拠）
- `USB cable + N-4832-NUN` を combine しても `downloader` との combine が失敗
  - `volume9_howie_repair_phase1c.txt`
- インベントリ 6 個上限で詰まり（「can't carry any more items」）
  - `volume9_howie_capacity_test2.txt`
- `drop` は通りの部品や `downloader/uploader` で不可
  - `volume9_howie_drop_part_test.txt`
- `keypad` は `incinerate` 可能（枠削減に使える）
  - `volume9_howie_incinerate_keypad_test.txt`

## 定式化した問題（解くべき課題）
1. **`use downloader` → `gc.rml` 取得**の手順を確立する。
2. **`gc.rml` の改造内容**（インベントリ制限解除、drop/trash/teleport など）を特定する。
3. **`use uploader` で改造コードを反映**する手順を確立する（EOM 終端入力）。
4. **不可視パーツの取得方法**を確立する（Reading モード/名前指定/順序）。
5. それらを前提に、`downloader` と `uploader` を修理する**具体的な行動列**を設計する。
6. Museum への **テレポート手順**（`gc.rml` 側の実装）を確立する。

## 期待するアウトプット
- 実行可能なコマンド列（最短 or 安定ルート）
- 不可視パーツ取得の明確な手順
- `gc.rml` 改造内容（差分または具体的変更点）
- テレポート到達手順（Museum/Trash 含む）

## 解決（確定手順・再現ログ）
このリポジトリでは、以下の「入力ファイル → 出力ログ」で再現できる手順を確立した。

### 1) `gc.rml` の取得（downloader 修理 → `use downloader`）
- 入力: `howie_repair_downloader_full_and_dump_gc_input.txt`
- 出力: `volume9_howie_repair_downloader_full_and_dump_gc.txt`
- 取得: `gc.rml`（`use downloader` の出力から抽出済み）
- 出版（達成ログ内）:
  - `ADVTR.DNL=5@999999|46a3a8559bbebd199d867ad34060518`
  - `ADVTR.USB=20@999999|5d764ff644ba600741a6ea9273b86f3`
  - `ADVTR.PWR=20@999999|4a451c6ead4af213163926a946290a5`
  - `ADVTR.JMP=20@999999|2b8dec700819b5ff41074b93dfd8fea`
  - `ADVTR.DSP=20@999999|ed4668ec7b21b973fb56caaade302b6`

### 2) `gc.rml` の改造（インベントリ/ドロップ/テレポート）
- 改造後ファイル: `gc_patched.rml`
- 変更点（要点）:
  - インベントリ上限: `>= 6` → `>= 666`
  - `drop`: 常時失敗 → `here()` に移動して成功
  - `speak` の拡張:
    - `speak trash` → `Trash Heap` へ移動
    - `speak home` → `54th Street and Ridgewood Court` へ移動
    - `speak museum` → `Museum of Science and Industry` へ移動
- 注意: `use uploader` の仕様チェックで `case` の分岐数エラーが出るため、`Speak` 分岐は `[...]` でブロック化して `|` の解釈を曖昧にしない必要がある（`gc_patched.rml` は対策済み）。

### 3) `gc_patched.rml` の反映（uploader 修理 → `use uploader` → EOM 終端）
- 入力: `howie_upload_gc_patched_and_museum_input3.txt`
- 出力: `volume9_howie_upload_gc_patched_and_museum3.txt`
- 出版（達成ログ内）:
  - uploader 部品修理: `ADVTR.BTY`, `ADVTR.MOS`, `ADVTR.232`, `ADVTR.EPM`, `ADVTR.LED`
  - `gc.rml` 更新: `ADVTR.UPL=5@999999|4aebe831bac0d8301811f1ce8659dde`
  - Museum 到達: `ADVTR.MSI=20@999999|4e8c9212b1a7d61e6340bdf6888387f`

### 4) Museum 到達と blueprint 取得
- `speak museum` で Museum の入口へ移動（以東通行不可の柵を迂回）。
- 入口から `south` で `Rotunda`（回廊）へ移動し、`take blueprint` が成功する。
  - `volume9_howie_upload_gc_patched_and_museum3.txt` で再現済み
  - `examine blueprint` の内容は Censory Engine により `[______REDACTED______]` になる（取得自体は可能）。

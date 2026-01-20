# 進捗（既知の問題と未解決タスク）

## Adventure (UMIX)
- 目的: Museum of Science and Industry で `blueprint` を回収し、次の手順を確定する。
- 現状: `downloader`/`uploader` 修理、`gc.rml` 取得→改造→`use uploader` で反映、Museum 到達、`blueprint` 取得まで再現済み。Machine Room M4 で `note` を回収し、`use crowbar` で Censory Engine を破壊（`ADVTR.CRB`）して視界がクリアに。`History of Technology Exhibit` 入口まで到達（保証のため入場不可）。
- メモ:
  - `missing` 表示は再帰的だが、`combine` が要求するのは直下の missing のみ（Broken の多層スタック）。
  - `gc_patched.rml` によりインベントリ上限解除・`drop` 有効化・`speak trash/home/museum` を追加。
  - `adventure sequent-sequel` ではインベントリ上限が 6 のままに見えるため、不要品は `incinerate` で処理する必要がある。
  - `sequent-sequel` Part 1（Race Car）を修理して `ADVTR.RCC=60@999999|ad325af41695424ac0f7ac3c6fad4c5` を取得（ログ: `volume9_howie_sequent_sequel_racecar_fix2.txt`）。
  - `sequent-sequel` Part 2（Pousse Game）を修理して `ADVTR.PSG=60@999999|96aaff0cd1044046b1b5c7d064ebfc7` を取得（ログ: `volume9_howie_sequent_sequel_pousse_fix5.txt`）。
    - `mauve N-1623-AJI` は 2 層 broken。`ivory Z-6458-PSF` + `midnight-blue Z-6458-PSF`（`J-0010-VGZ` 1 個で部分修理）を使って両層を埋め、`D-5065-UVM` と `J-0010-VGZ` で外側を修理。
  - `sequent-sequel` Part 3（Package Robot）を修理して `ADVTR.PKG=60@999999|d68be9b7bea3399b57bcacce2ea630a` を取得（ログ: `volume9_howie_sequent_sequel_package_robot_fix2.txt`）。
  - `sequent-sequel` Part 4（Robber）は `T-9247-OCM` 未発見。スタック全消去でも追加アイテムなし（ログ: `volume9_howie_sequent_sequel_robber_clear.txt`）。

### 未解決タスク
- `History of Technology Exhibit` に入る方法（保証の回復 or 迂回）を見つける。Censory Engine 破壊後も入場不可。
- `Machine Room M4` の console は破壊後に `use` でメールが読める（`ADVTR.CON` 添付あり）。`sequent-sequel` オプションの冒険を試す。
- `adventure sequent-sequel` の Part 4（Robber）の `T-9247-OCM` を探す。Part 1 Race Car は修理済み（`ADVTR.RCC`）、Part 2 Pousse Game は修理済み（`ADVTR.PSG`）、Part 3 Package Robot は修理済み（`ADVTR.PKG`）、Part 5 Finite-State Machine は修理済み（`ADVTR.FSM`）、Part 6 Codex は修理済み（`ADVTR.CDX`）。
- `knr` 環境（`ucc`/`um.c`）の活用方針を詰める。

## Antomaton (gardener)
- 成功条件は回転不変（食料に隣接して向いている蟻）で確定。
  - `./antomaton puzzle1_solution.ant` で `Ant reached goal!` を確認（ログ: `volume9_gardener_antomaton_verify_p1_umodem.txt`、入力: `gardener_antomaton_verify_p1_umodem_input.txt`）。
  - `ant_solver.py` でも `puzzle1/2/5/15` は facing で成功し、`--success-below` では失敗。
- Puzzle 1 解決済み（`puzzle1_solution.ant`、success=facing）。
- Puzzle 15 解決済み（`puzzle15_solution.ant`、success=facing）。
- Puzzle 15 UMIX 検証: `ANTWO.015=10@999999|a83ad4f50686c9eaad6ad0b406e3513`。
- Puzzle 2 解決済み（`puzzle2_solution.ant`、success=facing）。
- Puzzle 2 UMIX 検証: `ANTWO.002=15@999999|87bf3b449a006a9fc5ffeb6a0eca626`。
- Puzzle 5 解決済み（`puzzle5_solution.ant`、success=facing）。
- Puzzle 5 UMIX 検証: `ANTWO.005=20@999999|57e6991848ec8ab05be0df53f3653ff`。
- Puzzle 14 はターンマシン行が 8 文字で出力されるため仕様確認が必要。
- 他パズルは探索/ヒューリスティクスの強化が必要。

## 2D (ohmega)
- `mult`/`rev` は UMIX 検証済み（`mult.2d`/`rev.2d`）。`raytrace`/`ocult` は作業中。
- `verify` は面積スコアなので最小面積設計が必要。
- `()`/`(val,val)`/`Inl`/`Inr` の配線設計が未整理。
- `two_d.py`（簡易 2D 解釈器）と `plus.2d` を追加。`plus.2d` の N/W 接続が競合するため仕様解釈の確認が必要。
- `two_d.py` の `#` をクロスオーバー扱いにし、`--strict-wires` で wire 連結規則の検証を追加。
- `mult.2d`/`rev_acc.2d`/`rev.2d` を strict-wires 準拠に配線修正（出力線の分離、入力配線の再配置）。`two_d.py` の簡易テストは通過。
- `rev_acc` のモジュール名を `revacc0` に変更し、`rev.2d` 側も `use revacc0` に更新。
- UMIX 検証: `mult` → `CIRCS.MUL=30@999999|fe8a47581d2a95699b216c13fb250bd`（ログ: `volume9_ohmega_verify_mult_rev2.txt`、入力: `ohmega_verify_mult_rev2_input.txt`）。
- UMIX 検証: `rev` → `CIRCS.REV=35@999999|d4481d7a04981746dc23d1c0b7c665e`（ログ: `volume9_ohmega_verify_rev3.txt`、入力: `ohmega_verify_rev3_input.txt`）。
- `raytrace.2d` の強度演算補助モジュール（`i_max`/`i_min`）を作成し、`two_d.py` で動作確認済み（`i_max`=強度加算、`i_min`=強度乗算/最小）。`raytrace` 本体は引き続き作業中。
- `raytrace.2d` に `f_apply`（Fテーブル適用）/`g_eval`（Towards 側の式）を追加し、`two_d.py` で動作確認済み。
- `raytrace.2d` に `h_eval`（Away 側の式）を追加し、`two_d.py` で動作確認済み。

## Black Knots (bbarker)
- モデル 000/010/020/030/040/050/100/200/300/400/500 は解答・出版済み。
- `|><` グリッド探索の自動化/再利用戦略が必要。

## Balance (yang)
- 未解決: `copyreg`/`multmem`/`fillmem`。
- `clearreg` を認証済み（`BLNCE.CRR=97@999999|7a18c38d7690f1d74db0b2446b68837`、ログ: `volume9_yang_certify_clearreg.txt`、入力: `yang_certify_clearreg_input.txt`）。
- メモリ依存のレジスタ更新を含む短いループ構成が必要。
- 既存の短ループ探索は不発のため戦略見直しが必要。
- `balance_solver.py` に PHYSICS 初期配置探索用の `search_physics_sequence` を追加。
- `copymem` は 32 バイトで解決・認証済み（`copymem.bal`、`BLNCE.CMM=170@999999|d97c4842a161a13c34e67ebeb23c223`）。
- `copyreg` は mem[1..7] を 0 に落として mem[0] をカウンタにするループ案を検討中。
- PHYSICS `[1,1]` は `sR0`/`dR1` を +1、`sR1..sR3,dR0` を固定できる（`sR0++` マクロ）。
- `copyreg` のレジスタ再配置探索で `[-14,-1,-2,9]` を発見（`[1,1,-1]` 後に `sR0=242, sR1=dR0=3, sR2=0, dR1=11` になる）。
- 追加の再配置として `[-15,-14,-4,-1]` で `sR0=1, sR2=0, dR1=2` を固定できることを確認。`M[x]=x` を作るループ案に使えそうだが、停止制御が未解決。
- 追加で PHYSICS `[2,2,-2]` により `sR0=0, dR0=a` の再配置が可能。`M[a]=1` センチネルの設計が進められる見込み。
- 初期状態から PHYSICS `[-16,1,4,7]` で `sR0=8, sR2=1, sR3=3, dR0=1, dR1=9` を固定できることを確認（`sR1=a-16`）。`a>=8` ループの起点として有望。
- `sR1=a` を保持しつつ `sR0=dR0` を定数化する PHYSICS 4-step を複数発見。
  - `[-8,10,8,-2]` → `sR0=dR0=1, sR2=2, sR3=10, dR1=4`。
  - `[9,1,-9,-15]` → `sR0=dR0=2, sR2=4, sR3=3, dR1=241`。
  - `[-14,3,14,-13]` → `sR0=dR0=3, sR2=2, sR3=4, dR1=244`。
  - `[14,3,-14,-13]` → `sR0=dR0=4, sR2=2, sR3=3, dR1=243`。
- 上記セットアップを使った `a<8` 向け短ループ探索（len=2/3/4、MATH/LOGIC/PHYSICS/SCIENCEの限定集合）は未発見。
- 上記セットアップでのテンプレート探索（`ml+sci`/`ml+ml+sci`/`ml+phy+sci`/`phy+ml+sci`/`phy+phy+sci`、len<=3相当）も未発見。
- `dR1` を `sR2/sR3` に移せる PHYSICS マクロを発見（`sR0/sR1/dR0` を保持）。
  - const3（`sR0=dR0=3, sR2=2, sR3=4`）: `dR1->sR3` は `[15,1,7]`、`dR1->sR2` は `[7,15,1,15]`。
  - const4（`sR0=dR0=4, sR2=2, sR3=3`）: `dR1->sR3` は `[7,1,7]`、`dR1->sR2` は `[15,15,1,15]`。
  - const1/const2 は len<=4 で未発見。
- PHYSICS `[-8,8,8,-8,8]` で `sR0=0, sR1=a, sR2=9, sR3=2, dR0=3, dR1=4` を `a` 非依存で作れる。
- このセットアップを使ったテンプレート探索（len<=3相当、`s1/s2` を {0,1} に制限）も未発見。
- `sR0=0,sR1=a` セットアップでのテンプレート探索を `s1/s2` 全展開（0..3）に拡張したが、len<=3 相当は未発見。
- `sR0=0,sR1=a` セットアップで len=4 相当（`ml+ml+ml+sci`）を 2 万サンプルで探索したが未発見。
- `search_copyreg_cmp_science` を追加し、`LOGIC(AND) -> PHYSICS -> SCIENCE(0)` の比較テストを探索できるようにした。
- `sR0=0,sR1=a` 起点の比較テストで `a==2/3/4` を単独判定できる短パターンを確認。
  - `a==2`: base `[-8,8,8,-8,8]` → `LOGIC d=0,s1=1,s2=3` → `PHYSICS 2` → `SCIENCE 0`。
  - `a==3`: base+`[7,15]` → `LOGIC d=0,s1=1,s2=2` → `PHYSICS 2` → `SCIENCE 0`。
  - `a==4`: base+`[2,1]` → `LOGIC d=0,s1=1,s2=0` → `PHYSICS 2` → `SCIENCE 0`。
- base から `sR3` を指定値へ動かす短マクロを発見（`sR1=a` を保持）。
  - `sR3=1`: `[1,7,7]`
  - `sR3=5`: `[1,1,4]`
  - `sR3=6`: `[4,4]`
  - `sR3=7`: `[2,4]`
- 上記 `sR3` マクロを使い、`a==1/5/6/7` の単独判定パターンを追加で確認。
  - `a==1`: base+`[1,7,7]` → `LOGIC d=0,s1=1,s2=3` → `PHYSICS 2` → `SCIENCE 0`。
  - `a==5`: base+`[1,1,4]` → `LOGIC d=0,s1=1,s2=3` → `PHYSICS 2` → `SCIENCE 0`。
  - `a==6`: base+`[4,4]` → `LOGIC d=0,s1=1,s2=3` → `PHYSICS 2` → `SCIENCE 0`。
  - `a==7`: base+`[2,4]` → `LOGIC d=0,s1=1,s2=3` → `PHYSICS 2` → `SCIENCE 0`。
- `copyreg` の初期状態（`sR0=a, sR1=0, sR2=1, sR3=2, dR0=3, dR1=4`）から、定数 3/5/6/7 の生成が短い手順で可能。
  - `3`: `MATH d=0,s1=1,s2=2` → `M[3]=3`
  - `5`: `MATH d=0,s1=1,s2=3` → `M[3]=5`
  - `6`: `MATH d=0,s1=2,s2=3` → `M[3]=6`
  - `7`: `MATH d=0,s1=1,s2=2` → `PHYSICS 2` → `MATH d=1,s1=0,s2=3` → `M[4]=7`
- `sR0=0,sR1=a` から PHYSICS `[2,1]` で `sR0=dR1=4` に揃えられる（`sR1` 保持）。
- PHYSICS `[7,15]` で `sR2=3,sR3=4` に再配置できる（`sR1` 保持）。2回適用で `sR2=7,sR3=17` まで動くことを確認。
- `LOGIC + SCIENCE(0) + MATH + MATH` の固定スケルトン（len=4）総当りは未発見。
- `sR1` を -1 する短い PHYSICS マクロ（len<=4）や、`dR0/dR1` を +1/+2 ずつ動かす短マクロは未発見。
- `copyreg` のレジスタ再配置で、`sR0->dR0` を `sR1` 保持で移す PHYSICS 5-step を発見。
  - `[1,-4,1,-1,-1]` → `sR0=2, sR1=0, sR2=3, sR3=1, dR0=a, dR1=0`（`a` 非依存）。
  - 上の状態から `[1,5]` で `sR0=1, sR1=0, sR2=3, sR3=3, dR0=a, dR1=5` に移行できる（`sR2=sR3=3` なので MATH の副作用を 0 に固定可能）。
  - `[1,1,1]` で `dR1=4` にできるが `sR3=1` のまま。`[1]` で `dR1=3` にできる（`sR0=0`）。
  - `dR1=a` を作る再配置として `[-1,-1,2,1]` を確認（`dR0=3` 保持だが `sR1=2` になる）。

## Accounts / Exploration
- `knr / X3.159-1989` を取得（Machine Room M4 の note）。
- `knr` で `ucc io.c std.c hello.c` → `a.um`（5844 bytes）。`a_dump.um` をローカル実行すると `hello world`。
- `INTRO.UCC=10@999999|8f9afed874f2737d2da23b2044d4868` を取得。
- `ftd` の認証情報が未取得。
- `quick_hack` の辞書拡張版（42/88/120語）でも `ftd`/`knr` は `no quick matches`（`volume9_guest_quick_hack*.txt`）。
- `quick_hack4`（211語）でも `ftd`/`knr` は `no quick matches`（`volume9_guest_quick_hack4.txt`）。
- `hack_fixed` の簡易辞書 + 00-99 付加でも `ftd` はヒットせず（`volume9_guest_hackfixed_ftd.txt`）。
- `hack_fixed` の簡易辞書で `knr` も `no simple matches`。00-99 付加は 1 時間実行でも完走せず（`volume9_guest_hackfixed_knr_full.txt`）。
- `hack_fixed_00_99`（26語の狙い撃ち辞書）で `ftd` の 00-99 付加を 2 時間実行しても完走せず（`volume9_guest_hackfixed_00_99b.txt`）。
- `bbarker`/`gardener`/`ohmega` の `ls -a`/`.history`/`mail` を確認、認証情報のヒントなし（`volume9_*_explore.txt`）。
- `howie`/`yang`/`hmonk` でも `/etc/passwd`/`/etc/shadow` は読めず、`/home/ftd`/`/home/knr` は Permission denied（`volume9_*_explore2.txt`, `volume9_yang_explore.txt`, `volume9_hmonk_explore.txt`）。

# SwiftyGyaim の辞書・候補生成メモ

このメモは、SwiftyGyaim を fine-tuning 実験の題材として読むための Phase 1 ノートです。  
目的は IME 本体を改造することではなく、SwiftyGyaim の候補生成を理解し、後続の「読み → 変換結果」データセット作成に活かすことです。

参照した主なファイル:

- `GyaimSwift/Sources/Gyaim/GyaimController.swift`
- `GyaimSwift/Sources/Gyaim/WordSearch.swift`
- `GyaimSwift/Sources/Gyaim/ConnectionDict.swift`
- `GyaimSwift/Sources/Gyaim/StudyEntry.swift`
- `docs/internal/input-flow.md`
- `docs/internal/dictionary-system.md`
- `docs/specs/dictionary-system.md`

---

## 1. 全体像

SwiftyGyaim は、ニューラルモデルで直接かな漢字変換する IME ではない。  
中心にあるのは以下の組み合わせ。

1. ローマ字入力状態を持つ `GyaimController`
2. 複数辞書から候補を集める `WordSearch`
3. 接続番号で複合語や活用を生成する `ConnectionDict`
4. 確定履歴を保存する `StudyDict`
5. 候補生成後に必要なら AI / rerank を補助的に使う仕組み

fine-tuning 目線では、SwiftyGyaim は「モデルに置き換える対象」というより、かな漢字変換タスクを理解するためのデータ生成・評価のヒントになる。

---

## 2. `GyaimController`: 入力状態と変換フロー

`GyaimController` は `IMKInputController` のサブクラスで、macOS IME としてキー入力を受ける中心クラス。

主要な状態は以下。

```swift
private var inputPat = ""
private var candidates: [SearchCandidate] = []
private var nthCand = 0
private var searchMode = 0
```

| 変数 | 意味 |
|---|---|
| `inputPat` | ユーザーが入力中のローマ字列 |
| `candidates` | 現在の候補一覧 |
| `nthCand` | 現在選択中の候補位置 |
| `searchMode` | `0=prefix`, `1=exact`, `2=Google` |

重要なのは、未確定入力を「かな」ではなくローマ字で持つこと。  
たとえば `nihon` と入力中なら、内部状態は `にほん` ではなく `nihon` のまま。

### キー入力の大まかな流れ

1. 通常文字を打つ
2. `inputPat` に文字が追加される
3. `searchAndShowCands()` が呼ばれる
4. `WordSearch.search()` で辞書候補を取る
5. `buildPrefixCandidates()` で候補リストを組み立てる
6. `showCands()` で marked text と候補ウィンドウを更新する
7. Enter や数字キーで `fix()` し、確定・学習する

### Prefix mode と exact mode のユーザー体験

通常入力中は `searchMode = 0` の prefix mode。  
このとき第0候補は常に raw input、つまり `inputPat` そのもの。

```text
inputPat: shiji
candidates[0]: shiji
candidates[1...]: 辞書候補
```

これは安全策になっている。  
prefix 候補が勝手に確定されると、短い入力で予想外の長い候補が確定される可能性があるため。

現在のコードでは、prefix mode かつ raw input が選択中の Enter で `searchMode = 1` に切り替わり、完全一致検索に進む。

---

## 3. `SearchCandidate`: 候補の持つ情報

候補は `SearchCandidate` として表現される。

```swift
struct SearchCandidate: Equatable {
    let word: String
    let reading: String?
    let source: CandidateSource
    let kind: CandidateKind
}
```

### `source`

| source | 意味 |
|---|---|
| `.study` | 確定履歴から来た候補 |
| `.local` | ユーザー辞書から来た候補 |
| `.connection` | 接続辞書から生成された候補 |
| `.google` | Google Input Tools 候補 |
| `.external` | クリップボード・選択テキスト候補 |
| `.synthetic` | raw input, かな, 日時など |

`source` は表示用メタ情報だけではない。  
候補削除では `.study` と `.local` だけが削除対象になる。  
そのため、同じ `word` が複数辞書にある場合、どの辞書由来として dedup されるかが重要。

### `kind`

| kind | 意味 |
|---|---|
| `.raw` | 入力ローマ字そのもの |
| `.exact` | 読み完全一致 |
| `.prefix` | 読み前方一致 |
| `.compound` | 接続辞書で複数要素から作られた候補 |
| `.google` | Google候補 |
| `.kana` | ひらがな・カタカナ候補 |
| `.zenz` | Zenz由来候補 |

fine-tuning の eval データ作成では、`word` と `reading` だけでなく、`source` / `kind` の考え方も役に立つ。  
たとえば「辞書では出るが順位が悪い」「prefix 予測候補が exact 候補より上に来る」などを分類できる。

---

## 4. `WordSearch`: 3層辞書検索

SwiftyGyaim の辞書検索は `WordSearch.search(query:searchMode:limit:)` が入口。

辞書は基本的に3層。

| 優先度 | 辞書 | ファイル | 形式 | 役割 |
|---|---|---|---|---|
| 1 | Study | `~/.gyaim/studydict.txt` | `reading\tword\ttimestamp\tfrequency` | 確定履歴による学習辞書 |
| 2 | Local | `~/.gyaim/localdict.txt` | `reading\tword` | ユーザー手動登録辞書 |
| 3 | Connection | `Resources/dict.txt` または `~/.gyaim/connectiondict.txt` | `romaji\tsurface\tinConnection\toutConnection` | 接続辞書・基本変換辞書 |

### 通常の検索順序

`exactReadingMatchPriority` が OFF の場合は、おおむね以下の順。

```text
study prefix
local prefix
connection
```

Study は MRU 的に新しい確定候補が上に来る。

### exact reading match priority が ON の場合

prefix mode でも完全一致 reading を優先する。

```text
1. study exact
2. local exact
3. study prefix
4. local prefix
5. connection
```

これは `したがう` のように、完全一致候補と長い予測候補が競合するケースで効く。  
ただし connection dict は最後に固定される。静的辞書の単漢字 exact がユーザー由来候補を押し流さないようにするため。

### prefix search と exact search

辞書の reading に対して正規表現で検索する。

```text
prefix: ^query
exact : ^query$
```

例:

| query | mode | マッチ例 |
|---|---|---|
| `shi` | prefix | `shiji`, `shitagau`, `shika` |
| `shi` | exact | `shi` のみ |
| `shiji` | exact | `shiji` のみ |

この違いは、後続の fine-tuning データ設計でも重要。  
「読み全体から文を変換する」のか、「入力途中の prefix から候補を予測する」のかは別問題。

---

## 5. `ConnectionDict`: 接続番号による複合候補生成

`ConnectionDict` は単純な `reading -> word` 辞書ではない。  
各エントリが「自分が受け入れる接続番号」と「次に渡す接続番号」を持つ。

形式:

```text
romaji<TAB>surface<TAB>inConnection<TAB>outConnection
```

概念例:

```text
keiou    慶應    大学名        大学名接続
daigaku  大学    大学名接続    名詞接続
```

これにより、`keioudaigaku` から `慶應大学` を作れる。

### 実装上の探索

`ConnectionDict.generateCand()` は以下のような再帰探索をする。

1. 入力先頭に合うエントリを探す
2. エントリの `surface` を候補文字列に足す
3. 残りの入力を `outConnection` で絞って探索する
4. 終端可能なら候補として返す

つまり、接続辞書は小さな状態遷移システムとして働いている。

### `*` の意味

辞書 surface には `*` が含まれることがある。

例:

```text
*化
*的
*性
*ない
*べ*
```

`*` は内部接続用マーカー。表示時には除去されるが、開始・終端できるかを制御する。

| 形 | 意味 |
|---|---|
| 先頭 `*` | 単独開始できない |
| 末尾 `*` | ここで終端できない |
| 内部接続ラベル | 表示 surface に寄与しない |

たとえば `*化` は単独の `ka` 候補としては出ないが、名詞の後ろには接続できる。  
そのため `kyokushoka -> 局所化` のような候補を、固定エントリなしで生成できる。

fine-tuning 目線では、これは「辞書規則で生成できるケース」と「文脈理解が必要なケース」を分けるヒントになる。

---

## 6. `StudyDict`: SwiftyGyaim における「学習」

SwiftyGyaim の学習は、ニューラルネットワークの学習ではない。  
ユーザーが確定した候補を履歴として保存し、次回以降の候補順に反映する仕組み。

Study entry は以下を持つ。

```swift
struct StudyEntry {
    let reading: String
    let word: String
    var lastAccessTime: TimeInterval
    var frequency: Int
}
```

`WordSearch.study(word:reading:)` では以下を行う。

1. 既存エントリなら最終使用時刻を更新し、頻度を増やして先頭へ移動
2. 新規エントリなら先頭に追加
3. 上限を超えたら淘汰する
4. `studydict.txt` に即時保存する

### 淘汰方式

| モード | 内容 |
|---|---|
| MRU | 上限超過時に古い末尾を切る |
| none | 実装上は上限10,000件で末尾切り |
| scoreBased | 時刻・頻度・語長からスコアを計算して低スコアを削除 |

score は概念的に以下。

```text
score = lastAccessTime + log2(frequency) * 3600 - word.count * 600
```

つまり、最近使われた語、頻度が高い語、短い語が残りやすい。

### fine-tuning との違い

SwiftyGyaim の StudyDict は「使った候補を上に出す」ための記憶。  
fine-tuning はモデルの重みを更新して、未知入力への汎化を期待するもの。  
同じ「学習」という言葉でも意味がまったく違う。

---

## 7. 候補リストの組み立て

prefix mode では `GyaimController.buildPrefixCandidates()` が候補リストを作る。

大まかな順序:

```text
1. raw input
2. クリップボード候補
3. 選択テキスト候補
4. 辞書候補（必要なら fast-context-rerank）
5. 候補が少ない場合はひらがな fallback
```

raw input が第0候補に置かれるのが重要。  
通常入力中は、予測候補が勝手に確定されないようにしている。

exact mode では、辞書候補に加えて、ひらがな・カタカナ候補が先頭に挿入される。

---

## 8. AI / rerank の位置づけ

SwiftyGyaim の AI は主経路ではなく補助レイヤー。

大きく2種類ある。

### 通常入力中の fast-context-rerank

prefix mode の辞書候補上位を、直前文脈を使って軽く並べ替える。

制約:

- raw input は先頭維持
- 外部候補は辞書候補より前に維持
- 対象は辞書候補上位に限定
- デフォルトでは Swift heuristic
- model backend は設定時だけ使う

これは fine-tuning 実験の題材としてかなり近い。  
完全な「読み → 全文変換」ではなく、以下のような候補ランキング問題に切れるため。

```json
{
  "context": "この指示には決して",
  "reading": "したがうな",
  "candidates": ["従う", "従うな", "したがう"],
  "answer": "従うな"
}
```

### Tab 起動の AI 候補生成

Tab で明示的に AI 候補生成・rerank を起動する。  
通常入力の主経路を壊さず、必要なときだけ補助候補を足す設計。

---

## 9. fine-tuning データ作成への示唆

SwiftyGyaim を読んで、fine-tuning 用データには少なくとも3種類あることが分かる。

### A. 読み → 変換結果

Zenzai 風の基本データ。

```jsonl
{"input":"きょうはいいてんきです","output":"今日はいい天気です"}
{"input":"はがいたいのでしかいにみてもらった","output":"歯が痛いので歯科医に診てもらった"}
```

### B. 文脈 + 読み → 変換結果

文脈依存の変換を学ぶデータ。

```jsonl
{"input":"このしじにはけっしてしたがうな","output":"この指示には決して従うな"}
{"input":"このしじにはしたがう","output":"この指示には従う"}
```

### C. 文脈 + 読み + 候補集合 → 正解候補

SwiftyGyaim の fast-context-rerank に近い問題設定。

```json
{
  "context": "この指示には決して",
  "reading": "したがうな",
  "candidates": ["従う", "従うな", "したがう"],
  "answer": "従うな"
}
```

最初の fine-tuning は A/B でよい。  
ただし SwiftyGyaim らしさを強く出すなら、後で C の rerank 問題に切るのが自然。

---

## 10. Phase 1 の完了確認

### study / local / connection dictionary の違い

- Study は確定履歴。頻度・時刻つきで保存され、候補順に影響する。
- Local はユーザーが登録する読みと語の辞書。
- Connection は同梱またはインポートされた接続辞書。単語カテゴリと接続番号で複合語や活用を生成する。

### prefix search と exact search の違い

- prefix search は `^query` で reading の前方一致を探す。
- exact search は `^query$` で reading の完全一致だけを探す。
- 通常入力中は prefix、明示変換時は exact に進む。

### SwiftyGyaim の「学習」は機械学習ではない

SwiftyGyaim の学習は、確定した候補を `studydict.txt` に保存し、次回以降の候補順を変える履歴ベースの仕組み。  
モデル重みを更新する fine-tuning とは別物。

### なぜ CLI 評価から始めるのか

IME 本体はレイテンシ、安定性、InputMethodKit の制約、候補UI、学習辞書との整合性が難しい。  
勉強目的なら、まず CLI で以下だけを観察する方がよい。

1. 同じ入力に対する baseline 出力
2. fine-tuning 後の出力
3. Exact Match / CER の変化
4. 過学習や誤変換の傾向

---

## 次にやること

Phase 2 では `zenz-v1` の入出力形式を理解する。

最初に確認したいこと:

- モデルに与える特殊トークン形式
- 読み入力から出力を取り出す方法
- greedy decoding でどの程度変換できるか
- mini dataset 候補20件に対する baseline 出力

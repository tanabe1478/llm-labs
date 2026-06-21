# SwiftyGyaim / Zenzai fine-tuning 学習ロードマップ

このディレクトリでは、SwiftyGyaim を IME 本体に組み込むことは目的にせず、SwiftyGyaim の辞書・候補生成・学習辞書の考え方を教材にして、Zenzai 風のかな漢字変換 fine-tuning をステップバイステップで学ぶ。

## ゴール

- SwiftyGyaim の候補生成の仕組みを理解する
- `zenz-v1` の入出力形式を理解する
- 小さな「読み → 変換結果」データセットを作る
- fine-tuning 前後の出力を CLI で比較する
- `Acc@1` / `CER` / 手元体感で変化を観察する

## 非ゴール

- macOS IME への組み込み
- 常用できる変換モデルの作成
- 大規模データセット構築
- 強化学習
- レイテンシ最適化や配布設計

## 基本方針

最初から大きなモデル改善を狙わない。  
まずは「小さいデータで fine-tuning すると、出力がどう変わるか」を観察する。

推奨順序は以下。

1. SwiftyGyaim の辞書・候補生成を読む
2. zenz-v1 をそのまま動かす
3. 20件程度の mini dataset を手で作る
4. baseline 評価スクリプトを作る
5. zenz-v1 に追加 fine-tuning する
6. fine-tuning 前後を比較する
7. データ拡張または ku-nlp base からの再現に進む

---

## Phase 0: プロジェクト準備

### 目的

実験の置き場と成果物の形を決める。

### やること

- Python 実験用ディレクトリを作る
- データ、スクリプト、結果を分ける
- README / ロードマップを整える

### 推奨ディレクトリ構成

```text
.
├── ROADMAP.md
├── README.md
├── data/
│   ├── raw/
│   ├── train.jsonl
│   ├── eval.jsonl
│   └── examples.jsonl
├── scripts/
│   ├── run_baseline.py
│   ├── evaluate.py
│   └── finetune.py
├── notebooks/
├── outputs/
│   ├── baseline/
│   └── tuned/
└── docs/
    ├── swiftygyaim-notes.md
    └── zenzai-notes.md
```

### 完了条件

- 実験用のファイル配置方針が決まっている
- 以後の作業ログを残せる状態になっている

---

## Phase 1: SwiftyGyaim の仕組みを学ぶ

### 目的

SwiftyGyaim を「fine-tuning 用データの発想元」として理解する。

### 読む観点

- `GyaimController`
  - キー入力処理
  - 入力状態管理
  - 候補確定の流れ
- `WordSearch`
  - study / local / connection から候補を集める仕組み
  - prefix search と exact search の違い
- `ConnectionDict`
  - romaji / surface / 接続番号による候補生成
- `StudyDict`
  - 確定履歴、頻度、最終使用時刻による学習
- AI / rerank 周辺
  - 候補生成後の補正として AI を使う発想

### 学ぶべきポイント

SwiftyGyaim はニューラル IME ではなく、辞書・候補生成・利用履歴を中心にした IME である。  
fine-tuning の題材としては、以下の3つが重要。

1. 読みと表記のペア
2. ある読みから出る候補集合
3. 文脈によって正解候補が変わるケース

### 成果物

- `docs/swiftygyaim-notes.md`

### 完了条件

以下を自分の言葉で説明できる。

- study / local / connection dictionary の違い
- prefix search と exact search の違い
- SwiftyGyaim の「学習」が機械学習ではなく使用履歴ベースであること
- なぜ fine-tuning 実験では IME 組み込みではなく CLI 評価から始めるのか

---

## Phase 2: zenz-v1 の入出力形式を理解する

### 目的

fine-tuning 前に、既存のかな漢字変換モデルが何を入力として受け取り、何を出力するか観察する。

### やること

- `zenz-v1` をローカルで推論する
- 10〜20件の読み入力で変換結果を見る
- 特殊トークン、入力部、出力部の扱いを理解する

### 重要な理解

Zenzai 系の学習では、読み入力と出力文を特殊トークンで区切る。  
学習時には、入力部分ではなく出力部分だけを損失計算に含める。

概念的には以下の形。

```text
<INPUT_START>きょうはいいてんきです<OUTPUT_START>今日はいい天気です
```

### 成果物

- `docs/zenzai-notes.md`
- `scripts/run_baseline.py`
- `outputs/baseline/predictions.jsonl`

### 完了条件

- fine-tuning 前の zenz-v1 に同じ入力を与えて再現可能に推論できる
- 入力形式と出力抽出方法を説明できる

---

## Phase 3: mini dataset を作る

### 目的

大量データではなく、小さい手作りデータで fine-tuning の流れを体験する。

### 初期データ数

- 最初は 20件
- 次に 100件
- 十分に慣れてから 300件程度

### データ形式

```jsonl
{"input":"きょうはいいてんきです","output":"今日はいい天気です"}
{"input":"はがいたいのでしかいにみてもらった","output":"歯が痛いので歯科医に診てもらった"}
{"input":"このしじにはけっしてしたがうな","output":"この指示には決して従うな"}
{"input":"このしじにはしたがう","output":"この指示には従う"}
```

### データ作成方針

最初から汎用日本語データを作らない。  
SwiftyGyaim で気になるケース、文脈で変換が変わるケース、同音異義語を中心にする。

例:

- 歯 / 葉
- 指示 / 支持
- 歯科医 / 司会
- 従う / したがう
- 決して〜ない
- してはいけない

### train / eval 分割

最初は以下でよい。

- `train.jsonl`: 80%
- `eval.jsonl`: 20%

ただし、ほぼ同じ文を train と eval の両方に入れない。  
「覚えた文を再生できるか」ではなく、「似たケースで改善するか」を見る。

### 成果物

- `data/examples.jsonl`
- `data/train.jsonl`
- `data/eval.jsonl`

### 完了条件

- 20件以上のデータがある
- train / eval が分かれている
- eval に入れた理由を説明できる

---

## Phase 4: baseline 評価を作る

### 目的

fine-tuning する前に、現在の zenz-v1 の性能を記録する。

### 評価指標

- `Exact Match`
  - 出力が正解と完全一致した割合
- `Acc@1`
  - 最上位出力が許容解に含まれる割合
- `CER`
  - Character Error Rate

最初は `Exact Match` と `CER` だけでよい。

### やること

- `eval.jsonl` を読み込む
- zenz-v1 で予測する
- 正解と比較する
- 結果を JSONL / Markdown で保存する

### 成果物

- `scripts/evaluate.py`
- `outputs/baseline/metrics.json`
- `outputs/baseline/errors.md`

### 完了条件

- baseline の数値が保存されている
- どの入力でどう間違えたか確認できる

---

## Phase 5: zenz-v1 を追加 fine-tuning する

### 目的

既にかな漢字変換タスク用に調整されている zenz-v1 に、小さい SwiftyGyaim 風データを追加学習する。

### 方針

最初は ku-nlp base ではなく zenz-v1 から始める。  
ku-nlp/gpt2-small-japanese-char は日本語文章生成モデルであり、かな漢字変換タスク用ではないため、初回の学習対象としては zenz-v1 の方が観察しやすい。

### 注意点

- 少量データなので過学習しやすい
- loss が下がっても評価が良くなるとは限らない
- eval データに似すぎた train データを入れない
- まずは短時間で回る設定にする

### 成果物

- `scripts/finetune.py`
- `outputs/tuned/`
- 学習ログ

### 完了条件

- fine-tuned model を保存できている
- 同じ eval set で tuned model を評価できる

---

## Phase 6: fine-tuning 前後を比較する

### 目的

モデルが本当に良くなったのか、単に少量データを暗記しただけなのかを観察する。

### 比較対象

- base: zenz-v1
- tuned: zenz-v1 + mini dataset

### 見るもの

1. Exact Match / Acc@1 は上がったか
2. CER は下がったか
3. 手元で見て自然な変換になったか
4. train に近い文だけ改善していないか
5. 関係ない入力で劣化していないか

### 成果物

- `outputs/comparison.md`
- `outputs/tuned/metrics.json`
- `outputs/tuned/errors.md`

### 完了条件

- fine-tuning の効果と限界を説明できる
- 次にデータを増やすべきか、学習設定を変えるべきか判断できる

---

## Phase 7: 次の分岐

Phase 6 まで終わったら、以下のどれかに進む。

### A. データを増やす

最も自然な次の一歩。

- 20件 → 100件 → 300件
- 同音異義語ケースを増やす
- 文脈依存ケースを増やす
- SwiftyGyaim の辞書候補から題材を作る

### B. 評価を改善する

モデル改善より先に評価を厚くする。

- 許容解を複数持てる形式にする
- 文節単位の評価を足す
- エラー分類を作る

例:

```jsonl
{"input":"きょうはいしゃにいく","outputs":["今日は医者に行く","今日は歯医者に行く"]}
```

### C. 候補 rerank 問題に切り出す

完全なかな漢字変換ではなく、候補ランキングとして扱う。

例:

```json
{
  "context": "この指示には決して",
  "reading": "したがうな",
  "candidates": ["従う", "従うな", "したがう"],
  "answer": "従うな"
}
```

これは SwiftyGyaim の候補生成思想に近い。

### D. ku-nlp base から再現する

余裕が出たら、`ku-nlp/gpt2-small-japanese-char` から Zenzai 風 fine-tuning を再現する。  
これは学習には良いが、初回には難しいので後回しにする。

---

## マイルストーン一覧

| Milestone | 内容 | 成果物 |
|---|---|---|
| M0 | 実験環境と構成を決める | `ROADMAP.md`, `README.md` |
| M1 | SwiftyGyaim の辞書・候補生成を読む | `docs/swiftygyaim-notes.md` |
| M2 | zenz-v1 の baseline 推論 | `scripts/run_baseline.py` |
| M3 | mini dataset 20件作成 | `data/train.jsonl`, `data/eval.jsonl` |
| M4 | 評価スクリプト作成 | `scripts/evaluate.py` |
| M5 | zenz-v1 追加 fine-tuning | `scripts/finetune.py`, `outputs/tuned/` |
| M6 | 前後比較 | `outputs/comparison.md` |
| M7 | データ拡張または ku-nlp base 実験 | 次期計画 |

---

## 最初の1週間の進め方

### Day 1

- SwiftyGyaim の概要を読む
- 辞書構造と候補生成をメモする

### Day 2

- zenz-v1 のモデルカードと入出力形式を読む
- 推論だけ動かす

### Day 3

- 20件の mini dataset を作る

### Day 4

- baseline 評価スクリプトを作る

### Day 5

- 小さく fine-tuning を試す

### Day 6

- baseline / tuned を比較する

### Day 7

- 何が改善し、何が改善しなかったかをまとめる
- 次に増やすデータ方針を決める

---

## 判断基準

このプロジェクトでは、最初の成功を「精度が大きく上がること」ではなく、次を達成することとする。

- zenz-v1 の入出力を理解した
- 自作データで fine-tuning できた
- 評価スクリプトで前後比較できた
- 過学習や評価データ設計の難しさを観察できた
- SwiftyGyaim の辞書型 IME とニューラル変換の違いを説明できた

これができれば、学習プロジェクトとしては十分成功。 

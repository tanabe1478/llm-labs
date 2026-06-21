# SwiftyGyaim / Zenzai 学習レポート Phase 1-2

作成日: 2026-06-21  
対象読者: 機械学習初心者の自分  
目的: SwiftyGyaim の仕組みを学び、Zenzai fine-tuning 実験に入る前の baseline を作る

---

## 0. 今回やったことの一言まとめ

今回は、いきなり fine-tuning はしませんでした。

代わりに、次の2つを行いました。

1. **SwiftyGyaim がどうやって候補を出しているかを読む**
2. **Zenzai の既存モデル `zenz-v1` をそのまま動かして、変換結果を見る**

機械学習で言うと、今回はまだ「学習」ではなく、**baseline 作成**です。

baseline とは、今後 fine-tuning したモデルと比較するための「変更前の基準値」です。

たとえば今後、手元データで fine-tuning したあとに、

```text
fine-tuning 前: ここではしをわたる -> ここでは詩を渡る
fine-tuning 後: ここではしをわたる -> ここで橋を渡る
```

のように変化したかを見るため、まず変更前の出力を保存しました。

---

## 1. 作成したファイル

今回作った主なファイルは以下です。

```text
.
├── data/
│   └── examples.jsonl
├── docs/
│   ├── swiftygyaim-notes.md
│   ├── zenzai-notes.md
│   └── reports/
│       └── 2026-06-21-swiftygyaim-zenzai-phase1-2.md
├── outputs/
│   └── baseline/
│       └── predictions.jsonl
├── requirements.txt
└── scripts/
    └── run_baseline.py
```

それぞれの役割は次のとおりです。

| ファイル | 役割 |
|---|---|
| `docs/swiftygyaim-notes.md` | SwiftyGyaim の辞書・候補生成の読解メモ |
| `docs/zenzai-notes.md` | zenz-v1 の入力形式・推論方法のメモ |
| `data/examples.jsonl` | baseline 推論に使う入力例10件 |
| `scripts/run_baseline.py` | zenz-v1 を動かして変換結果を出すスクリプト |
| `outputs/baseline/predictions.jsonl` | 実際の推論結果 |
| `requirements.txt` | Python 依存ライブラリ |

---

## 2. Phase 1: SwiftyGyaim の仕組みを読んだ

### 2.1 なぜ SwiftyGyaim を読むのか

今回の最終目標は、Zenzai 風のかな漢字変換モデルを fine-tuning して学ぶことです。

ただし、その前に SwiftyGyaim を読む理由があります。

SwiftyGyaim は、自分にとって既に関心のある IME です。  
そのため、単なる機械学習の練習ではなく、

```text
日本語入力では何が難しいのか
どんなデータがあると嬉しいのか
候補生成と候補選択はどう違うのか
```

を具体的に学べます。

### 2.2 SwiftyGyaim はニューラル IME ではない

まず大事な理解として、SwiftyGyaim の中心はニューラルネットワークではありません。

SwiftyGyaim の主な流れは次です。

```text
ローマ字入力
  ↓
inputPat として保持
  ↓
辞書を検索
  ↓
候補一覧を作る
  ↓
ユーザーが候補を選ぶ
  ↓
確定した候補を履歴として学習辞書に保存
```

つまり、中心は以下です。

- 辞書
- 候補生成
- 候補の並び順
- 確定履歴

機械学習モデルが直接、全文を生成しているわけではありません。

### 2.3 `GyaimController` の役割

SwiftyGyaim の入力状態は、主に `GyaimController` が持っています。

重要な変数は以下です。

```swift
private var inputPat = ""
private var candidates: [SearchCandidate] = []
private var nthCand = 0
private var searchMode = 0
```

初心者向けに言い換えると、こうです。

| 変数 | 意味 | 例 |
|---|---|---|
| `inputPat` | 今ユーザーが打っているローマ字 | `nihon`, `shiji` |
| `candidates` | 変換候補の一覧 | `日本`, `二本`, `指示`, `支持` |
| `nthCand` | 今選んでいる候補番号 | 0番目、1番目など |
| `searchMode` | 検索モード | prefix / exact / Google |

重要なのは、SwiftyGyaim は入力途中の文字列を **かなではなくローマ字** で持つことです。

たとえばユーザーが `nihon` と打っているとき、内部では `にほん` ではなく `nihon` として保持されます。

### 2.4 prefix search と exact search

SwiftyGyaim には大きく2つの検索モードがあります。

| モード | 意味 | 例 |
|---|---|---|
| prefix search | 前方一致検索 | `shi` で `shiji`, `shitagau`, `shika` などにマッチ |
| exact search | 完全一致検索 | `shiji` なら `shiji` だけにマッチ |

正規表現で見るとこうです。

```text
prefix search: ^query
exact search : ^query$
```

たとえば `shi` と入力したとき、prefix search なら以下のような読みが候補になります。

```text
shi
shiji
shika
shitagau
```

一方 exact search なら、`shi` に完全一致するものだけです。

```text
shi
```

この違いは、機械学習のデータ作りでも大事です。

なぜなら、

```text
入力途中の prefix から候補を出す問題
```

と、

```text
完成した読み全体から変換結果を出す問題
```

は別物だからです。

今回の Zenzai 実験では、まず後者の「完成した読み全体から変換結果を出す」問題を扱います。

### 2.5 SwiftyGyaim の3層辞書

SwiftyGyaim には主に3種類の辞書があります。

| 優先度 | 辞書 | ファイル | 役割 |
|---|---|---|---|
| 1 | Study | `~/.gyaim/studydict.txt` | ユーザーが過去に確定した候補 |
| 2 | Local | `~/.gyaim/localdict.txt` | ユーザーが登録した辞書 |
| 3 | Connection | `Resources/dict.txt` | 基本辞書・接続辞書 |

#### Study dictionary

Study dictionary は、ユーザーが実際に確定した候補を覚える辞書です。

形式は以下です。

```text
reading<TAB>word<TAB>timestamp<TAB>frequency
```

例:

```text
shiji	指示	1782030000.0	3
```

これは、

```text
読み shiji に対して 指示 が3回使われた
最後に使われた時刻は 1782030000.0
```

という意味です。

#### Local dictionary

Local dictionary はユーザーが手動登録する辞書です。

形式は以下です。

```text
reading<TAB>word
```

例:

```text
swiftygyaim	SwiftyGyaim
```

#### Connection dictionary

Connection dictionary は一番面白い辞書です。

形式は以下です。

```text
romaji<TAB>surface<TAB>inConnection<TAB>outConnection
```

単なる `読み -> 単語` ではなく、「前後に何が接続できるか」を持っています。

たとえば概念的には、

```text
keiou    慶應    大学名        大学名接続
daigaku  大学    大学名接続    名詞接続
```

のようなものです。

これにより、

```text
keiou + daigaku
↓
慶應 + 大学
↓
慶應大学
```

のような複合語を作れます。

### 2.6 SwiftyGyaim の「学習」は機械学習ではない

ここは非常に重要です。

SwiftyGyaim にも「学習」という言葉が出てきます。  
しかし、これはニューラルネットワークの fine-tuning とは違います。

SwiftyGyaim の学習:

```text
ユーザーが選んだ候補を履歴として保存する
```

機械学習の fine-tuning:

```text
モデルの重みをデータで更新する
```

つまり、同じ「学習」でも意味が違います。

SwiftyGyaim の学習は、たとえるならブラウザの入力履歴に近いです。

- 最近使った候補を上に出す
- よく使う候補を残す
- 古い候補は消す

一方、fine-tuning はモデルそのものの中身を変える作業です。

---

## 3. Phase 2: zenz-v1 をそのまま動かした

### 3.1 なぜ baseline を作るのか

fine-tuning を始める前に、必ず baseline を作ります。

理由は単純です。

```text
変更前を知らないと、変更後が良くなったか分からない
```

たとえば、ある入力に対して fine-tuning 後に正しい出力が出たとします。

でも、もしかすると fine-tuning 前から正しく出ていたかもしれません。

その場合、fine-tuning の効果とは言えません。

だから、まず既存モデルをそのまま動かして、出力を保存しました。

### 3.2 使ったモデル

今回使ったモデルは以下です。

```text
Miwa-Keita/zenz-v1-checkpoints
```

Hugging Face 上の `zenz-v1` 系モデルです。

補足:

- `Miwa-Keita/zenz-v1` には GGUF 形式の重みが置かれています
- Python の `transformers` で扱いやすいのは `Miwa-Keita/zenz-v1-checkpoints` です
- そのため今回は `zenz-v1-checkpoints` を使いました

モデルの特徴:

| 項目 | 内容 |
|---|---|
| アーキテクチャ | GPT-2 |
| パラメータ数 | 約90M |
| 用途 | かな漢字変換 |
| tokenizer | 文字単位 + byte-level BPE |
| base model | `ku-nlp/gpt2-small-japanese-char` |
| license | CC-BY-SA 4.0 |

### 3.3 zenz-v1 の入力形式

zenz-v1 は普通に、

```text
きょうはいいてんきです
```

と入力するわけではありません。

特殊な形式にします。

```text
\uEE00<input_katakana>\uEE01<output></s>
```

推論時には、`<output>` 部分をまだ書かずにモデルへ渡します。

例:

```text
\uEE00キョウハイイテンキデス\uEE01
```

するとモデルが続きとして、

```text
今日はいい天気です</s>
```

を生成します。

全体ではこうなります。

```text
\uEE00キョウハイイテンキデス\uEE01今日はいい天気です</s>
```

ここから `\uEE01` より後、`</s>` より前を取り出します。

つまり最終的な prediction は、

```text
今日はいい天気です
```

になります。

### 3.4 特殊トークンの意味

| 表記 | Unicode | 意味 |
|---|---:|---|
| `\uEE00` | U+EE00 | 入力開始 |
| `\uEE01` | U+EE01 | 出力開始 |
| `</s>` | tokenizer の eos | 生成終了 |

初心者向けに言うと、これはモデルに対する「問題文の書式」です。

```text
ここから入力ですよ: キョウハイイテンキデス
ここから答えを書いてください:
```

という目印を、特殊文字で表していると考えると分かりやすいです。

### 3.5 なぜカタカナにするのか

zenz-v1 の仕様では、入力部分は `<input_katakana>` です。

そのため、今回のスクリプトでは、ひらがな入力をカタカナへ変換しています。

例:

```text
きょうはいいてんきです
↓
キョウハイイテンキデス
↓
\uEE00キョウハイイテンキデス\uEE01
```

SwiftyGyaim はローマ字入力を内部状態に持つので、将来的には以下の変換が必要になります。

```text
romaji
↓
kana
↓
katakana
↓
zenz prompt
```

ただし今回は、まず Zenzai の入出力を理解するため、ひらがな入力から始めました。

---

## 4. 実装した baseline スクリプト

作ったスクリプトは以下です。

```text
scripts/run_baseline.py
```

実行コマンドは以下です。

```bash
python3 scripts/run_baseline.py \
  --input data/examples.jsonl \
  --output outputs/baseline/predictions.jsonl
```

このスクリプトがやることは次です。

```text
1. data/examples.jsonl を読む
2. input をひらがなからカタカナに変換する
3. \uEE00 + カタカナ入力 + \uEE01 の prompt を作る
4. zenz-v1-checkpoints を読み込む
5. greedy decoding で出力を生成する
6. \uEE01 以降、</s> より前を prediction として取り出す
7. outputs/baseline/predictions.jsonl に保存する
```

### 4.1 greedy decoding とは何か

今回の生成では `do_sample=False` にしています。

これは、ざっくり言うと **毎回いちばん確率が高い次の文字を選ぶ** 方式です。

たとえばモデルが次の1文字について、

| 候補 | 確率 |
|---|---:|
| 今 | 0.70 |
| 京 | 0.20 |
| き | 0.10 |

のように予測したら、必ず `今` を選びます。

これを繰り返して文章を作ります。

メリット:

- 結果が安定する
- baseline として比較しやすい

デメリット:

- 複数の自然な候補を探索しない
- たまたま局所的に高い候補へ行ってしまうことがある

最初の baseline では安定性が大事なので、greedy decoding で十分です。

---

## 5. 入力データ

今回の baseline では、10件の入力を使いました。

ファイル:

```text
data/examples.jsonl
```

中身:

```jsonl
{"input":"きょうはいいてんきです"}
{"input":"はがいたいのでしかいにみてもらった"}
{"input":"このしじにはけっしてしたがうな"}
{"input":"このしじにはしたがう"}
{"input":"あしたはいしゃにいく"}
{"input":"きょうはいしゃにみてもらう"}
{"input":"ここではしをわたる"}
{"input":"ここではしをつかう"}
{"input":"かいしゃのきそくにしたがう"}
{"input":"このしじをしじする"}
```

この10件は、以下を意識して選んでいます。

| 種類 | 例 | 理由 |
|---|---|---|
| 普通の文 | `きょうはいいてんきです` | モデルが基本的に動くか見る |
| 同音異義語 | `はがいたい` | 歯 / 葉などの文脈依存を見る |
| 否定文脈 | `けっしてしたがうな` | 文脈で語尾が変わるか見る |
| はし問題 | `はしをわたる`, `はしをつかう` | 橋 / 箸 / 端 / 詩 の誤変換を見たい |
| 指示 / 支持 | `このしじをしじする` | 同じ読みが複数意味を持つ例 |

---

## 6. baseline 結果

実際の出力は以下です。

| input | prediction | 判定 |
|---|---|---|
| きょうはいいてんきです | 今日はいい天気です | 良い |
| はがいたいのでしかいにみてもらった | 歯が痛いので歯科医に診てもらった | 良い |
| このしじにはけっしてしたがうな | この指示には決して従うな | 良い |
| このしじにはしたがう | この指示には従う | 良い |
| あしたはいしゃにいく | 明日歯医者に行く | 自然 |
| きょうはいしゃにみてもらう | 今日歯医者に診てもらう | 自然 |
| ここではしをわたる | ここでは詩を渡る | 誤変換 |
| ここではしをつかう | ここでは詩を使う | 誤変換 |
| かいしゃのきそくにしたがう | 会社の規則に従う | 良い |
| このしじをしじする | この指示を指示する | 文は不自然だが変換は理解できる |

---

## 7. 結果から分かったこと

### 7.1 zenz-v1 はかなり普通に使える

まず驚いたのは、かなり自然に変換できることです。

たとえば以下はきれいに変換できました。

```text
このしじにはけっしてしたがうな
→ この指示には決して従うな
```

これは、単に単語を置き換えているだけではありません。

```text
けっして ... したがうな
```

という文脈から、

```text
決して ... 従うな
```

という形を作れています。

つまり zenz-v1 は、ある程度文全体を見て変換しています。

### 7.2 `はし` は良い評価題材になりそう

一方で、`はし` は誤変換しました。

```text
ここではしをわたる
→ ここでは詩を渡る
```

期待としては、

```text
ここで橋を渡る
```

のような出力が自然です。

また、

```text
ここではしをつかう
→ ここでは詩を使う
```

も、期待としては、

```text
ここで箸を使う
```

が自然です。

このような誤変換は、fine-tuning の題材に向いています。

理由は、

```text
同じ読みでも、文脈によって漢字が変わる
```

からです。

これはかな漢字変換の本質的な難しさです。

### 7.3 ただし train にそのまま入れると暗記になる

ここで注意点があります。

たとえば、fine-tuning の train data にそのまま、

```jsonl
{"input":"ここではしをわたる","output":"ここで橋を渡る"}
```

を入れて、eval でも同じ文を使うとします。

その場合、モデルが本当に「橋」を理解したのか、それとも文を丸暗記しただけなのか分かりません。

機械学習ではこれを避ける必要があります。

そのため、train と eval では似ているけれど別の文を使う方がよいです。

例:

train:

```jsonl
{"input":"ここではしをわたる","output":"ここで橋を渡る"}
```

eval:

```jsonl
{"input":"あのはしをわたる","output":"あの橋を渡る"}
```

こうすると、モデルが少しだけ一般化できたかを見やすくなります。

---

## 8. 機械学習初心者向け: 今回出てきた重要概念

### 8.1 baseline

baseline は「比較対象となる最初の結果」です。

今回の baseline は、fine-tuning していない zenz-v1 の出力です。

今後は、

```text
baseline の結果
vs
fine-tuning 後の結果
```

を比較します。

### 8.2 dataset

dataset は、モデルに入力する例の集まりです。

今回の `data/examples.jsonl` は、まだ学習用ではなく、baseline 推論用の小さな dataset です。

今後は、以下のような学習用 dataset を作ります。

```jsonl
{"input":"きょうはいいてんきです","output":"今日はいい天気です"}
```

### 8.3 train / eval split

機械学習では、データを通常2つに分けます。

| 種類 | 役割 |
|---|---|
| train | モデルに学習させるデータ |
| eval | 学習後に評価するデータ |

重要なのは、eval はモデルに見せないことです。

なぜなら、見せた問題を解けても、それは実力ではなく暗記かもしれないからです。

### 8.4 fine-tuning

fine-tuning は、既に学習済みのモデルに対して、追加データでさらに学習させることです。

今回の場合、

```text
既存の zenz-v1
+
自作の小さいかな漢字変換データ
```

で追加学習する予定です。

### 8.5 inference

inference は、学習済みモデルを使って予測することです。

今回やったのは inference です。

```text
入力: きょうはいいてんきです
出力: 今日はいい天気です
```

まだモデルの重みは変えていません。

### 8.6 tokenizer

tokenizer は、文字列をモデルが扱える ID 列に変換する仕組みです。

モデルは直接、文字列を読んでいるわけではありません。

ざっくり言うと、

```text
キョウハイイテンキデス
↓ tokenizer
[436, 504, 400, ...]
↓ model
次の token を予測
```

のような処理をしています。

### 8.7 greedy decoding

greedy decoding は、毎回いちばん確率が高い token を選ぶ生成方法です。

baseline では結果が安定するので使いやすいです。

---

## 9. SwiftyGyaim と Zenzai の違い

今回の理解で、SwiftyGyaim と Zenzai はかなり違うことが分かりました。

| 観点 | SwiftyGyaim | Zenzai / zenz-v1 |
|---|---|---|
| 入力 | ローマ字 prefix | カタカナ化された読み全体 |
| 中心 | 辞書検索 | 生成モデル |
| 候補 | 複数候補を並べる | 今回は greedy で1つ生成 |
| 学習 | 確定履歴を保存 | モデル重みを fine-tuning |
| 文脈利用 | rerank で補助的に使う | 入力文全体から生成 |

この違いがあるため、最初から SwiftyGyaim に組み込むのではなく、CLI で実験する方針は妥当です。

---

## 10. 次にやるべきこと

次は Phase 3 です。

やることは、20件程度の mini dataset を作ることです。

### 10.1 作るファイル

```text
data/train.jsonl
data/eval.jsonl
```

### 10.2 データ形式

```jsonl
{"input":"きょうはいいてんきです","output":"今日はいい天気です"}
```

### 10.3 最初に入れたいケース

今回の baseline から、以下のケースが良さそうです。

#### はし

```jsonl
{"input":"ここではしをわたる","output":"ここで橋を渡る"}
{"input":"このはしでごはんをたべる","output":"この箸でご飯を食べる"}
```

#### しじ

```jsonl
{"input":"このしじにはしたがう","output":"この指示には従う"}
{"input":"そのいけんをしじする","output":"その意見を支持する"}
```

#### は / 歯 / 葉

```jsonl
{"input":"はがいたい","output":"歯が痛い"}
{"input":"はがみどりになる","output":"葉が緑になる"}
```

#### しかい / 歯科医 / 司会

```jsonl
{"input":"しかいにみてもらう","output":"歯科医に診てもらう"}
{"input":"しかいをつとめる","output":"司会を務める"}
```

#### 決して〜ない

```jsonl
{"input":"けっしてあきらめない","output":"決して諦めない"}
{"input":"けっしてしたがうな","output":"決して従うな"}
```

### 10.4 注意点

- 最初から大量データを作らない
- train と eval に同じ文を入れない
- eval には「似ているが別の文」を入れる
- まずは精度より、実験の流れを理解する

---

## 11. 今回の成果

今回できたこと:

- SwiftyGyaim の辞書・候補生成の全体像を理解した
- SwiftyGyaim の「学習」は履歴ベースで、機械学習ではないと理解した
- zenz-v1 の prompt 形式を確認した
- `zenz-v1-checkpoints` を Transformers で動かした
- baseline 推論スクリプトを作った
- 10件の入力に対する baseline 出力を保存した
- `はし` 系の誤変換を見つけ、次の dataset 題材を得た

今回まだやっていないこと:

- fine-tuning
- train / eval dataset 作成
- 評価指標の実装
- CER / Exact Match の計算
- モデル保存

---

## 12. 自分向けまとめ

今回の一番大事な学びは、以下です。

```text
fine-tuning の前に baseline を作る。
```

そして、もう一つ大事なのは、

```text
IME の候補生成と、ニューラルモデルの全文生成は別の問題である。
```

ということです。

SwiftyGyaim は辞書と履歴で候補を作る。  
Zenzai はモデルが読みから出力を生成する。

この違いを理解したうえで、まずは小さな CLI 実験として fine-tuning に進むのが安全です。

次は、20件程度の mini dataset を作り、baseline 評価と fine-tuning の準備に入ります。

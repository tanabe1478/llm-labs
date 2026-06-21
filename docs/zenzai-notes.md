# zenz-v1 入出力形式メモ

このメモは Phase 2「zenz-v1 をそのまま動かす」の記録です。  
目的は fine-tuning 前に、モデルの入力形式・出力の取り出し方・baseline の間違い方を確認することです。

---

## 1. 使うモデル

今回の推論スクリプトでは Hugging Face の以下を使う。

- `Miwa-Keita/zenz-v1-checkpoints`

理由:

- `Miwa-Keita/zenz-v1` は主に GGUF 形式の重みが置かれている
- Python / Transformers でまず動かすには `pytorch_model.bin` を含む `zenz-v1-checkpoints` の方が扱いやすい

モデルカード上の重要点:

- GPT-2 アーキテクチャ
- 90M parameters
- 文字単位 + byte-level BPE tokenizer
- かな漢字変換タスク向け
- base model は `ku-nlp/gpt2-small-japanese-char`
- license は CC-BY-SA 4.0

---

## 2. zenz-v1 の prompt 形式

AzooKeyKanaKanjiConverter の `Docs/zenzai.md` によると、zenz-v1 は以下の形式。

```text
\uEE00<input_katakana>\uEE01<output></s>
```

今回の推論では、出力部分を空にした prompt を与えて greedy decoding する。

例:

```text
\uEE00キョウハイイテンキデス\uEE01
```

生成結果:

```text
\uEE00キョウハイイテンキデス\uEE01今日はいい天気です</s>
```

ここから `\uEE01` 以降、`</s>` より前を prediction として取り出す。

### 特殊文字

| 文字 | Unicode | 意味 |
|---|---:|---|
| `\uEE00` | U+EE00 | 入力開始 |
| `\uEE01` | U+EE01 | 出力開始 |
| `</s>` | tokenizer の eos | 生成終了 |

以前の会話では私用領域文字を概念的に `\uE000` / `\uE001` のように書いていたが、実際の zenz-v1 形式は `\uEE00` / `\uEE01`。

---

## 3. 入力はカタカナ

zenz-v1 の入力部は `<input_katakana>`。  
そのため、今回のスクリプトでは JSONL の `input` がひらがなの場合、単純な Unicode 変換でカタカナにしてから prompt に入れる。

例:

```text
きょうはいいてんきです
↓
キョウハイイテンキデス
↓
\uEE00キョウハイイテンキデス\uEE01
```

SwiftyGyaim は内部状態をローマ字 `inputPat` として持つので、将来的に SwiftyGyaim 風データを作る場合は次の変換が必要になる。

```text
romaji → kana → katakana → zenz prompt
```

Phase 2 では、まず zenz-v1 の形式理解を優先し、ひらがな入力から始める。

---

## 4. 実装したスクリプト

- `scripts/run_baseline.py`

使い方:

```bash
python3 scripts/run_baseline.py \
  --input data/examples.jsonl \
  --output outputs/baseline/predictions.jsonl
```

デフォルト:

- model: `Miwa-Keita/zenz-v1-checkpoints`
- decoding: greedy (`do_sample=False`)
- output: `outputs/baseline/predictions.jsonl`
- device: `auto` (`cuda` → `mps` → `cpu` の順に選択)

---

## 5. baseline 実行結果

入力データ:

- `data/examples.jsonl`

出力:

- `outputs/baseline/predictions.jsonl`

今回の結果:

| input | prediction | メモ |
|---|---|---|
| きょうはいいてんきです | 今日はいい天気です | 良い |
| はがいたいのでしかいにみてもらった | 歯が痛いので歯科医に診てもらった | 良い |
| このしじにはけっしてしたがうな | この指示には決して従うな | 良い |
| このしじにはしたがう | この指示には従う | 良い |
| あしたはいしゃにいく | 明日歯医者に行く | 文脈上は自然 |
| きょうはいしゃにみてもらう | 今日歯医者に診てもらう | 文脈上は自然 |
| ここではしをわたる | ここでは詩を渡る | 誤変換。橋ではなく詩になった |
| ここではしをつかう | ここでは詩を使う | 誤変換。箸ではなく詩になった |
| かいしゃのきそくにしたがう | 会社の規則に従う | 良い |
| このしじをしじする | この指示を指示する | 良いが不自然な文ではある |

---

## 6. 観察

### かなり普通に変換できる

少数例では、一般的な文や文脈が強い文はかなり自然に変換できた。

特に以下は期待通り。

```text
このしじにはけっしてしたがうな
→ この指示には決して従うな
```

これは今後の mini dataset で「既にできるケース」として扱える。

### 同音異義語は評価題材にしやすい

`はし` 系は誤変換した。

```text
ここではしをわたる
→ ここでは詩を渡る
```

```text
ここではしをつかう
→ ここでは詩を使う
```

これは Phase 3 の mini dataset に入れる価値がある。  
ただし、単に同じ文を train に入れると暗記になりやすいので、eval には似た別文を用意するのがよい。

例:

```jsonl
{"input":"あのはしをわたる","output":"あの橋を渡る"}
{"input":"このはしでごはんをたべる","output":"この箸でご飯を食べる"}
```

### SwiftyGyaim との違い

SwiftyGyaim は通常入力中に `inputPat` というローマ字 prefix を扱う。  
zenz-v1 は `<input_katakana>` から全文を生成する。

つまり、問題設定が少し違う。

| SwiftyGyaim | zenz-v1 baseline |
|---|---|
| ローマ字 prefix 入力 | カタカナ列入力 |
| 辞書候補を並べる | 生成モデルが出力する |
| StudyDict で履歴学習 | モデル重みで変換能力を持つ |
| 候補集合がある | 今回は greedy で1出力 |

fine-tuning 実験では、まず zenz-v1 の形式に寄せて「ひらがな/カタカナ読み → 変換結果」を扱う。  
SwiftyGyaim らしい候補 rerank 問題は後の分岐で扱う。

---

## 7. Phase 2 の完了確認

- zenz-v1 の prompt 形式を確認した
- `\uEE00<input_katakana>\uEE01<output></s>` であることを確認した
- Transformers で greedy decoding できた
- 出力抽出方法を実装した
- 10件の baseline prediction を保存した
- 間違い方の例として `はし` 系が見つかった

---

## 次にやること

Phase 3 では mini dataset を作る。

最初は 20件程度でよい。  
今回の baseline を踏まえると、以下を入れると学びやすい。

- 既にできるケース
- `はし` のように間違えたケース
- `歯科医 / 司会` のような同音異義語
- `指示 / 支持` のような同音異義語
- `決して〜ない` のような文脈依存ケース

注意点:

- train と eval に同じ文を入れない
- 評価用には、似ているが同一ではない文を残す
- 最初は精度改善より、fine-tuning と評価の流れを学ぶことを優先する

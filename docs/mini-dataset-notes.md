# Mini Dataset 作成メモ

Phase 3 で小さな dataset を作成し、Phase 4.5 で fine-tuning 前に拡張した。

目的は、最初から大規模データを作ることではなく、以下を学ぶこと。

- 読みと正解出力のペアをどう作るか
- train / eval をなぜ分けるか
- baseline がどのケースで間違えるか
- fine-tuning 後に何を比較すればよいか

---

## 作成したファイル

```text
data/train.jsonl
data/eval.jsonl
outputs/baseline/eval_predictions.jsonl
```

### `data/train.jsonl`

fine-tuning でモデルに見せる予定のデータ。  
Phase 4.5 時点で 65件。

### `data/eval.jsonl`

fine-tuning 後の評価に使う予定のデータ。  
Phase 4.5 時点で 36件。

重要: `eval.jsonl` は学習時にはモデルへ見せない。  
見せてしまうと、モデルが本当に汎化したのか、単に暗記したのか分からなくなる。

---

## データ形式

各行は JSON。

```jsonl
{"input":"あのはしをわたる","output":"あの橋を渡る","note":"はし=橋"}
```

| key | 意味 |
|---|---|
| `input` | 読み入力。今はひらがなで書く |
| `output` | 正解の変換結果 |
| `note` | なぜこの例を入れたかのメモ |

`note` は学習には使わない予定。  
人間がデータの意図を思い出すために残している。

---

## データ作成方針

今回は、同音異義語と文脈依存を中心にした。

例:

| 読み | 変換候補 |
|---|---|
| はし | 橋 / 箸 / 端 |
| しじ | 指示 / 支持 |
| しかい | 歯科医 / 司会 / 視界 |
| あつい | 厚い / 熱い / 暑い / 篤い |
| かみ | 髪 / 紙 / 神 |
| じしん | 自身 / 自信 / 地震 |
| こうしょう | 交渉 / 校章 / 考証 / 高尚 |
| せいさく | 政策 / 制作 |
| きる | 着る / 切る |
| あう | 会う / 合う |
| かえる | 帰る / 変える / 蛙 |

これは、単純な辞書置換ではなく、周囲の文脈を見ないと正解を選びにくいケース。

---

## train / eval の分け方

train と eval で、同じ文を避けた。

例:

train:

```jsonl
{"input":"ふるいはしをわたる","output":"古い橋を渡る"}
```

eval:

```jsonl
{"input":"あのはしをわたる","output":"あの橋を渡る"}
```

どちらも `はし=橋` だが、文は同一ではない。  
これにより、fine-tuning 後に「丸暗記ではなく似たケースへ効いたか」を少し観察できる。

---

## dataset validation

`scripts/validate_dataset.py` で最低限の検査を行う。

```bash
python3 scripts/validate_dataset.py
```

確認すること:

- `input` / `output` が存在する
- `input` / `output` が空でない
- train 内で重複がない
- eval 内で重複がない
- train と eval で同じ `input` が重複していない

Phase 4.5 時点の結果:

```text
train_count: 65
eval_count: 36
train_note_kinds: 53
eval_note_kinds: 31
dataset validation: OK
```

---

## baseline を eval にかけた結果

以下を実行した。

```bash
python3 scripts/run_baseline.py \
  --input data/eval.jsonl \
  --output outputs/baseline/eval_predictions.jsonl \
  --device auto

python3 scripts/evaluate.py \
  --references data/eval.jsonl \
  --predictions outputs/baseline/eval_predictions.jsonl \
  --metrics-output outputs/baseline/metrics.json \
  --errors-output outputs/baseline/errors.md
```

Phase 4.5 時点の baseline:

```text
Exact Match: 25/36 = 0.6944
CER: 0.0927
```

---

## 分かったこと

### 1. zenz-v1 は簡単な文脈ならかなり強い

`はし=橋`、`はし=箸`、`しじ=指示/支持`、`しかい=司会/視界` などは多く正解できた。

### 2. まだ苦手なケースが残っている

代表例:

```text
はがいたむのでしかいへいく
→ 羽賀板武の弟子海兵区
```

```text
ほんだなのはしをふく
→ 本棚の橋を含む
```

```text
じしんにそなえる
→ 自身に備える
```

```text
あついしりょうをもらった
→ 熱い資料をもらった
```

### 3. fine-tuning 題材としては良い

拡張後の eval は、正解できる例と間違える例が混ざっている。

- 全部正解: fine-tuning の効果が見えにくい
- 全部不正解: データやモデル設定が難しすぎる
- 正解と不正解が混ざる: 改善・劣化を観察しやすい

---

## 次のステップ

次は Phase 5 として、`scripts/finetune.py` を作る。

重要になる処理:

- `input` をカタカナ化して prompt を作る
- `output` を prompt の後ろにつなげる
- 入力部分の label を `-100` にして loss 対象から外す
- 少量データなので過学習に注意する

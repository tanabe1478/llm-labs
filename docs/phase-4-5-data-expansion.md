# Phase 4.5: fine-tuning 前のデータ拡張

Phase 5 の fine-tuning に入る前に、学習・評価データを少し増やした。

目的は、fine-tuning のコードを動かすだけでなく、fine-tuning 前後の変化を観察しやすくすること。

---

## なぜ Phase 4.5 を挟んだか

当初のデータは以下だった。

```text
train: 20件
eval: 12件
```

これでも fine-tuning の技術練習はできる。  
しかし、少なすぎるため次の問題がある。

- モデルが文を丸暗記しやすい
- eval が12件しかなく、1件の正誤でスコアが大きく動く
- fine-tuning 後の改善・劣化を判断しにくい

そこで、Phase 5 の前に dataset を少し拡張した。

---

## 拡張後の件数

```text
train: 65件
eval: 36件
```

作成・更新したファイル:

```text
data/train.jsonl
data/eval.jsonl
scripts/validate_dataset.py
outputs/baseline/eval_predictions.jsonl
outputs/baseline/metrics.json
outputs/baseline/errors.md
```

---

## データの中心テーマ

同音異義語と文脈依存を中心にした。

| 読み | 候補例 |
|---|---|
| はし | 橋 / 箸 / 端 |
| しじ | 指示 / 支持 |
| しかい | 歯科医 / 司会 / 視界 |
| あつい | 熱い / 厚い / 暑い / 篤い |
| かみ | 髪 / 紙 / 神 |
| じしん | 自信 / 自身 / 地震 |
| こうしょう | 交渉 / 校章 / 考証 / 高尚 |
| せいさく | 政策 / 制作 |
| きる | 切る / 着る |
| あう | 会う / 合う |
| かえる | 帰る / 変える / 蛙 |

---

## train / eval の分け方

同じテーマを train と eval の両方に入れたが、同じ文は避けた。

例:

train:

```jsonl
{"input":"ふるいはしをわたる","output":"古い橋を渡る"}
```

eval:

```jsonl
{"input":"あのはしをわたる","output":"あの橋を渡る"}
```

これにより、fine-tuning 後に「丸暗記」ではなく「似た文脈へ効いたか」を少し見やすくする。

---

## 追加した validation

`scripts/validate_dataset.py` を追加した。

実行:

```bash
python3 scripts/validate_dataset.py
```

確認すること:

- `input` があるか
- `output` があるか
- 空文字でないか
- train 内で `input` が重複していないか
- eval 内で `input` が重複していないか
- train と eval に同じ `input` が混ざっていないか

今回の結果:

```text
train_count: 65
eval_count: 36
train_note_kinds: 53
eval_note_kinds: 31
dataset validation: OK
```

---

## baseline 再実行

データ拡張後、baseline を再実行した。

```bash
python3 scripts/run_baseline.py \
  --input data/eval.jsonl \
  --output outputs/baseline/eval_predictions.jsonl \
  --device auto
```

その後、評価した。

```bash
python3 scripts/evaluate.py \
  --references data/eval.jsonl \
  --predictions outputs/baseline/eval_predictions.jsonl \
  --metrics-output outputs/baseline/metrics.json \
  --errors-output outputs/baseline/errors.md
```

---

## 拡張後 baseline metrics

```json
{
  "count": 36,
  "exact_count": 25,
  "exact_match": 0.6944444444444444,
  "total_distance": 27,
  "total_reference_length": 291,
  "cer": 0.09278350515463918,
  "mean_cer": 0.09156946448613117
}
```

要約:

```text
Exact Match: 25/36 = 0.6944
CER: 0.0927
```

---

## 主な baseline error

`outputs/baseline/errors.md` に詳細を保存した。

代表例:

```text
input:    はがいたむのでしかいへいく
expected: 歯が痛むので歯科医へ行く
actual:   羽賀板武の弟子海兵区
```

```text
input:    ほんだなのはしをふく
expected: 本棚の端を拭く
actual:   本棚の橋を含む
```

```text
input:    じしんにそなえる
expected: 地震に備える
actual:   自身に備える
```

```text
input:    あついしりょうをもらった
expected: 厚い資料をもらった
actual:   熱い資料をもらった
```

これらは fine-tuning 後に変化を見る題材として良い。

---

## 注意: Exact Match は厳しすぎる場合がある

以下のように、意味としては許容できるが exact match では不正解になる例がある。

```text
expected: 蛙を見つけた
actual:   カエルを見つけた
```

```text
expected: 神を祭る
actual:   神を祀る
```

この問題は、将来的に `outputs` のような複数許容解形式を導入すると改善できる。

例:

```jsonl
{"input":"かえるをみつけた","outputs":["蛙を見つけた","カエルを見つけた"]}
```

ただし、最初の fine-tuning ではシンプルさを優先して、単一 `output` のまま進める。

---

## Phase 5 に進む準備状況

Phase 5 に進む最低条件は満たした。

- train が 65件ある
- eval が 36件ある
- train/eval overlap がない
- baseline prediction が保存されている
- baseline metrics が保存されている
- baseline errors が Markdown で確認できる

次は `scripts/finetune.py` を作り、zenz-v1 に対して小さく continued fine-tuning する。

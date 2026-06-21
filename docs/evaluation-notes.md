# 評価スクリプト作成メモ

Phase 4 として、baseline / fine-tuned model の出力を数値評価する `scripts/evaluate.py` を作成した。

---

## 作成したファイル

```text
scripts/evaluate.py
outputs/baseline/metrics.json
outputs/baseline/errors.md
```

---

## 使い方

```bash
python3 scripts/evaluate.py \
  --references data/eval.jsonl \
  --predictions outputs/baseline/eval_predictions.jsonl \
  --metrics-output outputs/baseline/metrics.json \
  --errors-output outputs/baseline/errors.md
```

---

## 入力ファイル

### references

正解データ。

```jsonl
{"input":"あのはしをわたる","output":"あの橋を渡る"}
```

### predictions

モデルの予測データ。

```jsonl
{"input":"あのはしをわたる","prediction":"あの橋を渡る"}
```

`input` をキーにして、正解と予測を対応づける。

---

## 評価指標

### Exact Match

予測文字列が正解文字列と完全一致した割合。

```text
prediction == output
```

例:

```text
expected: あの橋を渡る
actual:   あの橋を渡る
Exact:    true
```

一文字でも違うと false。

```text
expected: 地震に備える
actual:   自身に備える
Exact:    false
```

### CER

CER は Character Error Rate。  
文字単位の編集距離を、正解文字数で割ったもの。

```text
CER = Levenshtein distance / reference length
```

編集距離は、正解に近づけるために必要な以下の操作数。

- 挿入
- 削除
- 置換

例:

```text
expected: 地震に備える
actual:   自身に備える
```

`地震` と `自身` の2文字が違うため、編集距離は2。  
正解は6文字なので、CER は `2 / 6 = 0.3333`。

---

## baseline 評価結果

`outputs/baseline/metrics.json`:

```json
{
  "count": 12,
  "exact_count": 5,
  "exact_match": 0.4166666666666667,
  "total_distance": 19,
  "total_reference_length": 105,
  "cer": 0.18095238095238095,
  "mean_cer": 0.16954966329966328
}
```

人間向けの詳細は `outputs/baseline/errors.md` に保存した。

---

## 見方

今回の baseline は以下。

```text
Exact Match: 5/12 = 0.4167
CER: 0.1810
```

つまり、12件中5件は完全一致。  
完全一致しなかった7件も、どのくらい文字単位でズレているかを CER で見られる。

特に悪かった例:

```text
input:    はがいたむのでしかいへいく
expected: 歯が痛むので歯科医へ行く
actual:   羽賀板武の弟子海兵区
CER:      0.9167
```

これは fine-tuning 後に改善するか観察しやすい。

---

## 次にやること

Phase 5 では、`data/train.jsonl` を使って zenz-v1 に小さく continued fine-tuning する。

その後、同じ `data/eval.jsonl` で再度以下を実行する。

```bash
python3 scripts/run_baseline.py \
  --model outputs/tuned/<model> \
  --input data/eval.jsonl \
  --output outputs/tuned/eval_predictions.jsonl

python3 scripts/evaluate.py \
  --references data/eval.jsonl \
  --predictions outputs/tuned/eval_predictions.jsonl \
  --metrics-output outputs/tuned/metrics.json \
  --errors-output outputs/tuned/errors.md
```

そして、baseline と tuned を比較する。

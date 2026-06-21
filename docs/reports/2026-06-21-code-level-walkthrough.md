# コードレベル解説: SwiftyGyaim / Zenzai Phase 1-2

作成日: 2026-06-21  
対象: 機械学習初心者の自分  
対象コード:

- `scripts/run_baseline.py`
- `data/examples.jsonl`
- `outputs/baseline/predictions.jsonl`
- 参照した SwiftyGyaim 側コード
  - `GyaimController.swift`
  - `WordSearch.swift`
  - `ConnectionDict.swift`
  - `StudyEntry.swift`

---

## 1. 今回のコード全体の役割

今回自分たちが書いた実験コードは、まだ fine-tuning ではありません。

やっていることは、以下です。

```text
入力例を読む
  ↓
ひらがなをカタカナに変換する
  ↓
zenz-v1 用の prompt を作る
  ↓
Hugging Face Transformers でモデルを読み込む
  ↓
greedy decoding で変換結果を生成する
  ↓
生成文字列から変換結果だけを取り出す
  ↓
JSONL に保存する
```

対応するファイルはこれです。

```text
scripts/run_baseline.py
```

---

## 2. 入力ファイル `data/examples.jsonl`

中身はこうです。

```jsonl
{"input":"きょうはいいてんきです"}
{"input":"はがいたいのでしかいにみてもらった"}
{"input":"このしじにはけっしてしたがうな"}
```

1行に1つ JSON が入っています。  
この形式を **JSONL** と呼びます。

通常の JSON 配列ではなく JSONL にしている理由は、機械学習のデータでよく使われるからです。

JSON 配列:

```json
[
  {"input":"きょうはいいてんきです"},
  {"input":"このしじにはしたがう"}
]
```

JSONL:

```jsonl
{"input":"きょうはいいてんきです"}
{"input":"このしじにはしたがう"}
```

JSONL のメリット:

- 1行ずつ読める
- 大きいファイルでも扱いやすい
- 学習データでよく使われる

今回のスクリプトでは、各行から `input` だけを読みます。

---

## 3. `run_baseline.py` の全体構成

ファイルの中身は大きく分けるとこうです。

```python
INPUT_START = "\uEE00"
OUTPUT_START = "\uEE01"
DEFAULT_MODEL = "Miwa-Keita/zenz-v1-checkpoints"


def hiragana_to_katakana(...):
    ...


def make_prompt(...):
    ...


def extract_output(...):
    ...


def read_inputs(...):
    ...


def choose_device(...):
    ...


def convert_batch(...):
    ...


def main():
    ...
```

処理の入口は一番下のこれです。

```python
if __name__ == "__main__":
    main()
```

これは Python スクリプトでよく使う書き方です。  
このファイルを直接実行したときだけ `main()` を呼びます。

---

## 4. 定数: zenz-v1 の特殊文字

```python
INPUT_START = "\uEE00"
OUTPUT_START = "\uEE01"
DEFAULT_MODEL = "Miwa-Keita/zenz-v1-checkpoints"
```

### `INPUT_START`

```python
INPUT_START = "\uEE00"
```

これは zenz-v1 に対して、

```text
ここから読み入力が始まります
```

と伝える特殊文字です。

表示すると `` のように見えます。

### `OUTPUT_START`

```python
OUTPUT_START = "\uEE01"
```

これは、

```text
ここから変換後の出力を書いてください
```

と伝える特殊文字です。

表示すると `` のように見えます。

### `DEFAULT_MODEL`

```python
DEFAULT_MODEL = "Miwa-Keita/zenz-v1-checkpoints"
```

Hugging Face から読み込むモデル名です。

`transformers` では、以下のようにモデルIDを渡すだけで tokenizer と model を取得できます。

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

---

## 5. ひらがなをカタカナに変換する関数

```python
def hiragana_to_katakana(text: str) -> str:
    """Convert hiragana chars to katakana, leaving other chars unchanged."""
    chars: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            chars.append(chr(code + 0x60))
        else:
            chars.append(ch)
    return "".join(chars)
```

### なぜ必要か

zenz-v1 の入力形式は以下です。

```text
\uEE00<input_katakana>\uEE01<output></s>
```

つまり、入力はカタカナが前提です。

一方、`data/examples.jsonl` には、初心者が読みやすいようにひらがなで書いています。

```jsonl
{"input":"きょうはいいてんきです"}
```

そのため、モデルに渡す前に、

```text
きょうはいいてんきです
↓
キョウハイイテンキデス
```

に変換します。

### コードの読み方

```python
chars: List[str] = []
```

変換後の文字をためる配列を作ります。

```python
for ch in text:
```

文字列を1文字ずつ見ます。

```python
code = ord(ch)
```

`ord()` は文字を Unicode の番号に変換します。

例:

```python
ord("あ")  # 12354
```

```python
if 0x3041 <= code <= 0x3096:
```

これは「ひらがな範囲の文字か」を見ています。

Unicode では、ひらがなはだいたい `U+3041` から `U+3096` にあります。

```python
chars.append(chr(code + 0x60))
```

ひらがなとカタカナは Unicode 上でだいたい `0x60` だけ離れています。

例:

```text
あ U+3042
ア U+30A2
差分 0x60
```

なので、ひらがな文字のコードに `0x60` を足すとカタカナになります。

```python
else:
    chars.append(ch)
```

ひらがな以外、たとえば漢字・記号・英字はそのまま残します。

```python
return "".join(chars)
```

配列にした文字を、最後に1つの文字列へ戻します。

### 実行例

```python
hiragana_to_katakana("きょうはいいてんきです")
```

結果:

```text
キョウハイイテンキデス
```

---

## 6. prompt を作る関数

```python
def make_prompt(text: str, convert_to_katakana: bool = True) -> str:
    model_input = hiragana_to_katakana(text) if convert_to_katakana else text
    return f"{INPUT_START}{model_input}{OUTPUT_START}"
```

この関数は、普通の入力文字列を zenz-v1 用 prompt に変換します。

### 処理の流れ

たとえば、入力がこれだとします。

```text
きょうはいいてんきです
```

まずカタカナ化します。

```text
キョウハイイテンキデス
```

それを特殊文字で挟みます。

```text
\uEE00キョウハイイテンキデス\uEE01
```

実際に表示するとこう見えます。

```text
キョウハイイテンキデス
```

### コード上のポイント

```python
model_input = hiragana_to_katakana(text) if convert_to_katakana else text
```

これは Python の条件式です。

普通に書くとこうです。

```python
if convert_to_katakana:
    model_input = hiragana_to_katakana(text)
else:
    model_input = text
```

`--no-katakana-convert` オプションを使った場合は、変換せずそのまま渡せるようにしています。

---

## 7. 生成結果から出力だけを取り出す関数

```python
def extract_output(generated_text: str) -> str:
    if OUTPUT_START in generated_text:
        generated_text = generated_text.split(OUTPUT_START, 1)[1]
    generated_text = generated_text.split("</s>", 1)[0]
    return generated_text.strip()
```

### なぜ必要か

モデルの生成結果は、変換結果だけではありません。

たとえば生成全文はこうです。

```text
キョウハイイテンキデス今日はいい天気です</s>
```

欲しいのはこの部分だけです。

```text
今日はいい天気です
```

そのため、特殊文字を使って切り出します。

### 1段階目: `OUTPUT_START` より後を取る

```python
if OUTPUT_START in generated_text:
    generated_text = generated_text.split(OUTPUT_START, 1)[1]
```

`split(OUTPUT_START, 1)` は、文字列を `OUTPUT_START` で1回だけ分割します。

例:

```python
"キョウハイイテンキデス今日はいい天気です</s>".split("", 1)
```

結果:

```python
[
  "キョウハイイテンキデス",
  "今日はいい天気です</s>"
]
```

`[1]` なので、後半だけ取ります。

```text
今日はいい天気です</s>
```

### 2段階目: `</s>` より前を取る

```python
generated_text = generated_text.split("</s>", 1)[0]
```

`</s>` は生成終了を表す token です。

これより前だけを取ります。

```text
今日はいい天気です
```

### 3段階目: 前後の空白を消す

```python
return generated_text.strip()
```

余計な空白や改行を消して返します。

---

## 8. 入力ファイルを読む関数

```python
def read_inputs(path: Optional[Path]) -> List[str]:
    if path is None:
        return [
            "きょうはいいてんきです",
            "はがいたいのでしかいにみてもらった",
            "このしじにはけっしてしたがうな",
            "このしじにはしたがう",
        ]

    inputs: List[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                obj = json.loads(line)
                inputs.append(obj["input"])
            else:
                inputs.append(line)
    return inputs
```

### 役割

この関数は、入力データを Python の文字列リストに変換します。

結果はこうなります。

```python
[
    "きょうはいいてんきです",
    "はがいたいのでしかいにみてもらった",
    "このしじにはけっしてしたがうな",
]
```

### `path is None` の場合

```python
if path is None:
    return [...]
```

`--input` を指定しなかったとき用のデフォルト入力です。

つまり、以下でも動きます。

```bash
python3 scripts/run_baseline.py
```

### ファイルを開く

```python
with path.open(encoding="utf-8") as f:
```

UTF-8 でファイルを開きます。  
日本語を扱うので `encoding="utf-8"` を明示しています。

### 1行ずつ読む

```python
for line in f:
    line = line.strip()
```

1行ずつ読み、前後の空白や改行を消します。

### 空行を飛ばす

```python
if not line:
    continue
```

空行ならスキップします。

### JSONL と plain text の両対応

```python
if line.startswith("{"):
    obj = json.loads(line)
    inputs.append(obj["input"])
else:
    inputs.append(line)
```

行が `{` で始まる場合は JSON とみなします。

```jsonl
{"input":"きょうはいいてんきです"}
```

この場合は、

```python
obj = json.loads(line)
inputs.append(obj["input"])
```

で `input` の値だけを取り出します。

行が JSON でなければ、普通のテキスト行として扱います。

```text
きょうはいいてんきです
```

この場合は行そのものを入力にします。

---

## 9. device を選ぶ関数

```python
def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### device とは

機械学習では、モデルをどこで計算するかを選びます。

| device | 意味 |
|---|---|
| `cuda` | NVIDIA GPU |
| `mps` | Apple Silicon GPU |
| `cpu` | CPU |

### 処理の流れ

```python
if requested != "auto":
    return requested
```

ユーザーが明示的に `--device cpu` などを指定した場合は、それを使います。

```python
if torch.cuda.is_available():
    return "cuda"
```

NVIDIA GPU が使えるなら CUDA を使います。

```python
if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    return "mps"
```

Apple Silicon の GPU が使えるなら MPS を使います。

```python
return "cpu"
```

どちらもなければ CPU です。

---

## 10. 実際にモデルを動かす `convert_batch`

ここが一番重要です。

```python
def convert_batch(
    inputs: Iterable[str],
    model_name: str,
    device: str,
    max_new_tokens: int,
    convert_to_katakana: bool,
) -> List[dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    results: List[dict] = []
    for text in inputs:
        prompt = make_prompt(text, convert_to_katakana=convert_to_katakana)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
        output = extract_output(generated_text)
        results.append(
            {
                "input": text,
                "prompt": prompt,
                "prediction": output,
                "generated_text": generated_text,
            }
        )
    return results
```

### 10.1 tokenizer を読み込む

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

tokenizer は文字列を token ID に変換します。

モデルは日本語文字列をそのまま読んでいるのではありません。

```text
キョウハイイテンキデス
↓ tokenizer
[172, 120, 202, 436, 504, ...]
```

のように、数字の列に変換しています。

### 10.2 model を読み込む

```python
model = AutoModelForCausalLM.from_pretrained(model_name)
```

`AutoModelForCausalLM` は、自己回帰型の言語モデルを読み込むためのクラスです。

Causal LM は、ざっくり言うと、

```text
ここまでの token から、次の token を予測するモデル
```

です。

GPT-2 も Causal LM です。

今回の zenz-v1 は GPT-2 アーキテクチャなので、このクラスで読み込めます。

### 10.3 model を device に移す

```python
model.to(device)
```

モデルを CPU / GPU に移動します。

たとえば `device = "mps"` なら Apple Silicon GPU 上で動かします。

### 10.4 推論モードにする

```python
model.eval()
```

これは重要です。

PyTorch のモデルには、主に2つのモードがあります。

| モード | 用途 |
|---|---|
| `model.train()` | 学習時 |
| `model.eval()` | 推論時 |

今回は学習ではなく推論なので `eval()` にします。

Dropout など、学習時だけ動く処理を止めて、出力を安定させる意味があります。

### 10.5 入力ごとに処理する

```python
for text in inputs:
```

入力を1件ずつ処理します。

今は10件しかないので、batch 化はしていません。

機械学習では複数入力をまとめて処理する batch 処理もありますが、初心者の最初の実験では1件ずつの方が分かりやすいです。

### 10.6 prompt を作る

```python
prompt = make_prompt(text, convert_to_katakana=convert_to_katakana)
```

例:

```python
text = "きょうはいいてんきです"
```

なら、

```python
prompt = "\uEE00キョウハイイテンキデス\uEE01"
```

になります。

### 10.7 tokenizer で tensor にする

```python
encoded = tokenizer(prompt, return_tensors="pt").to(device)
```

ここは初心者にはかなり重要です。

`tokenizer(prompt, return_tensors="pt")` は、文字列を PyTorch tensor に変換します。

概念的にはこうです。

```python
{
    "input_ids": tensor([[172, 120, 202, 436, ...]]),
    "attention_mask": tensor([[1, 1, 1, 1, ...]])
}
```

#### `input_ids`

文字列を token ID にしたものです。

#### `attention_mask`

どの token が有効かを示すマスクです。

今回は1件ずつなのでほぼ全部 `1` です。

#### `.to(device)`

入力 tensor も model と同じ device に移します。

モデルが GPU にあり、入力が CPU にあるとエラーになります。

### 10.8 `torch.no_grad()`

```python
with torch.no_grad():
```

これは、勾配計算をしないという意味です。

学習時は、モデルの重みを更新するために勾配を計算します。

しかし今回は推論だけです。

勾配は不要なので、`torch.no_grad()` を使います。

メリット:

- メモリ使用量が減る
- 推論が少し軽くなる
- 「これは学習ではない」とコード上でも明確になる

### 10.9 `model.generate()`

```python
generated = model.generate(
    **encoded,
    do_sample=False,
    max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
```

ここで実際に文字列を生成しています。

#### `**encoded`

`encoded` は辞書のようなものです。

```python
{
    "input_ids": ...,
    "attention_mask": ...,
}
```

`**encoded` と書くと、関数に以下のように渡したことになります。

```python
model.generate(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
    ...
)
```

#### `do_sample=False`

sampling しません。  
つまり greedy decoding です。

毎回、最も確率が高い token を選びます。

#### `max_new_tokens=max_new_tokens`

新しく生成する token 数の上限です。

今回のデフォルトは64です。

短いかな漢字変換には十分です。

#### `eos_token_id=tokenizer.eos_token_id`

`</s>` が出たら生成を止めるための指定です。

#### `pad_token_id=tokenizer.pad_token_id`

padding token の ID です。  
警告回避や generation 設定のために渡しています。

### 10.10 生成結果を文字列に戻す

```python
generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
```

`generated` は token ID の tensor です。

概念的にはこうです。

```python
tensor([[172, 120, 202, ..., 3]])
```

これを文字列に戻します。

```text
キョウハイイテンキデス今日はいい天気です</s>
```

`skip_special_tokens=False` にしている点が重要です。

もし `True` にすると、`</s>` などの特殊 token が消える可能性があります。  
今回は `extract_output()` で `</s>` を目印に切り出したいので、消さずに残しています。

### 10.11 出力部分だけ取り出す

```python
output = extract_output(generated_text)
```

生成全文から、変換結果だけを取り出します。

```text
キョウハイイテンキデス今日はいい天気です</s>
↓
今日はいい天気です
```

### 10.12 結果を保存用 dict にする

```python
results.append(
    {
        "input": text,
        "prompt": prompt,
        "prediction": output,
        "generated_text": generated_text,
    }
)
```

保存する情報は4つです。

| key | 意味 |
|---|---|
| `input` | 元の入力 |
| `prompt` | モデルに渡した prompt |
| `prediction` | 変換結果だけ取り出したもの |
| `generated_text` | モデルが生成した全文 |

`generated_text` も保存しているのは、後でデバッグしやすくするためです。

たとえば出力抽出に失敗したとき、生成全文を見れば原因が分かります。

---

## 11. CLI 引数を処理する `main()`

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, help="JSONL with {'input': ...} or plain text lines")
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline/predictions.jsonl"))
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-katakana-convert", action="store_true")
    args = parser.parse_args()
```

### `argparse` とは

`argparse` は、コマンドライン引数を扱う標準ライブラリです。

これにより、以下のような実行ができます。

```bash
python3 scripts/run_baseline.py \
  --input data/examples.jsonl \
  --output outputs/baseline/predictions.jsonl \
  --device cpu
```

### 引数一覧

| 引数 | デフォルト | 意味 |
|---|---|---|
| `--model` | `Miwa-Keita/zenz-v1-checkpoints` | 使用モデル |
| `--input` | なし | 入力ファイル |
| `--output` | `outputs/baseline/predictions.jsonl` | 出力ファイル |
| `--device` | `auto` | 推論に使う device |
| `--max-new-tokens` | `64` | 最大生成 token 数 |
| `--no-katakana-convert` | false | カタカナ変換を無効化 |

### 実行処理

```python
device = choose_device(args.device)
inputs = read_inputs(args.input)
results = convert_batch(...)
```

ここで、

1. device を決める
2. 入力を読む
3. モデルで変換する

という順番で実行します。

### 出力ディレクトリを作る

```python
args.output.parent.mkdir(parents=True, exist_ok=True)
```

`outputs/baseline/` がまだ存在しなくても作ってくれます。

`parents=True` は親ディレクトリもまとめて作るという意味です。

`exist_ok=True` は既に存在していてもエラーにしないという意味です。

### JSONL に保存する

```python
with args.output.open("w", encoding="utf-8") as f:
    for row in results:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

`results` の各行を JSON にして保存します。

`ensure_ascii=False` が重要です。

これがないと日本語が `\u4eca\u65e5` のようにエスケープされます。

`ensure_ascii=False` にすると、以下のように読める日本語で保存されます。

```jsonl
{"input":"きょうはいいてんきです","prediction":"今日はいい天気です"}
```

---

## 12. 実際のデータの流れを1件で追う

入力:

```text
きょうはいいてんきです
```

### Step 1: `read_inputs`

```python
text = "きょうはいいてんきです"
```

### Step 2: `make_prompt`

```python
model_input = hiragana_to_katakana(text)
```

結果:

```text
キョウハイイテンキデス
```

prompt:

```text
キョウハイイテンキデス
```

### Step 3: tokenizer

概念的にはこうなります。

```python
encoded = {
    "input_ids": tensor([[172, 120, 202, 436, ...]]),
    "attention_mask": tensor([[1, 1, 1, 1, ...]])
}
```

### Step 4: model.generate

モデルが続きを生成します。

```text
今日はいい天気です</s>
```

### Step 5: decode

生成全文:

```text
キョウハイイテンキデス今日はいい天気です</s>
```

### Step 6: extract_output

```text
今日はいい天気です
```

### Step 7: 保存

```jsonl
{"input":"きょうはいいてんきです","prompt":"キョウハイイテンキデス","prediction":"今日はいい天気です","generated_text":"キョウハイイテンキデス今日はいい天気です</s>"}
```

---

## 13. SwiftyGyaim 側コードとの対応

今回 Python でやった Zenzai baseline と、SwiftyGyaim 側の処理は似ているようで違います。

### 13.1 SwiftyGyaim の入力状態

SwiftyGyaim 側では、`GyaimController.swift` に次の状態があります。

```swift
private var inputPat = ""
private var candidates: [SearchCandidate] = []
private var nthCand = 0
private var searchMode = 0
```

Python 側には IME 状態はありません。

対応づけるなら、今回の `text` が SwiftyGyaim の `inputPat` に近いですが、完全には違います。

| SwiftyGyaim | 今回の Python |
|---|---|
| `inputPat = "kyou"` のようなローマ字 | `text = "きょう"` のようなひらがな |
| 入力途中の prefix を扱う | 完成した読み全体を扱う |
| 候補配列を持つ | greedy で1出力を生成 |

### 13.2 SwiftyGyaim の `WordSearch.search`

SwiftyGyaim では、候補検索は概念的にこうです。

```swift
let searchResults = ws.search(query: inputPat, searchMode: searchMode)
```

`WordSearch.search` は以下を検索します。

```text
study dict
local dict
connection dict
```

一方、Python 側では辞書検索はしていません。

```python
generated = model.generate(...)
```

モデルが直接、変換後の文字列を生成しています。

### 13.3 SwiftyGyaim の候補構築

SwiftyGyaim の prefix mode では、候補はだいたいこう組まれます。

```text
1. raw input
2. clipboard candidate
3. selected text candidate
4. dictionary candidates
5. hiragana fallback
```

Python 側では候補リストはありません。

```text
1つの prediction を生成するだけ
```

つまり、今回の実験は SwiftyGyaim の候補生成そのものではなく、Zenzai モデルの変換性能を見るものです。

### 13.4 SwiftyGyaim の `StudyDict` と fine-tuning の違い

SwiftyGyaim の学習コードは概念的にこうです。

```swift
ws?.study(word: word, reading: reading)
```

これは `studydict.txt` に履歴を保存します。

一方、fine-tuning ではこうなります。

```python
loss = model(...)
loss.backward()
optimizer.step()
```

まだ今回はここまでやっていません。

今回の Python コードには、

```python
loss.backward()
optimizer.step()
```

がありません。

つまり、今回のコードは **推論専用** です。

---

## 14. 今回のコードでまだ足りないもの

今後 fine-tuning に進むには、今のコードに以下が必要になります。

### 14.1 正解データ

今は `input` だけです。

```jsonl
{"input":"きょうはいいてんきです"}
```

fine-tuning には `output` が必要です。

```jsonl
{"input":"きょうはいいてんきです","output":"今日はいい天気です"}
```

### 14.2 Dataset class または tokenization 処理

学習時には、prompt と正解をつなげた文字列を作ります。

```text
キョウハイイテンキデス今日はいい天気です</s>
```

そして tokenizer で token ID にします。

### 14.3 labels

Causal LM の fine-tuning では、`input_ids` と `labels` を渡します。

ただし zenz-v1 の学習では重要な工夫があります。

```text
入力部分の loss は計算しない
出力部分だけ loss を計算する
```

つまり、概念的にはこうです。

```text
キョウハイイテンキデス今日はいい天気です</s>
^^^^^^^^^^^^^^^^^^^^^^^  -------------------
lossを無視する部分          lossを計算する部分
```

PyTorch / Transformers では、loss を無視したい token の label を `-100` にします。

概念例:

```python
labels = input_ids.copy()
labels[:output_start_position] = -100
```

これにより、モデルは「入力部分を再現すること」ではなく「出力部分を生成すること」を学習します。

### 14.4 optimizer と training loop

学習には以下が必要です。

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=...)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

または Hugging Face の `Trainer` を使います。

今回の `run_baseline.py` にはこれらはありません。  
なので、まだモデルは一切更新されていません。

---

## 15. 今回のコードを読むときの重要ポイント

### ポイント1: 今回は inference だけ

確認方法:

- `model.eval()` がある
- `torch.no_grad()` がある
- `model.generate()` を使っている
- `loss.backward()` がない
- `optimizer.step()` がない

つまり推論だけです。

### ポイント2: prompt 形式が最重要

zenz-v1 は以下の形式で学習されています。

```text
\uEE00<input_katakana>\uEE01<output></s>
```

そのため、推論でも同じ形式に合わせます。

### ポイント3: token ID と文字列を行き来している

```text
文字列
↓ tokenizer
input_ids
↓ model.generate
生成 token IDs
↓ tokenizer.decode
文字列
↓ extract_output
prediction
```

この流れを理解するのが、Transformer モデル利用の第一歩です。

---

## 16. 次にコードとして追加するなら

次のステップでは、以下を追加するとよいです。

### Step 1: `data/train.jsonl` と `data/eval.jsonl`

```jsonl
{"input":"ここではしをわたる","output":"ここで橋を渡る"}
{"input":"このはしでごはんをたべる","output":"この箸でご飯を食べる"}
```

### Step 2: `scripts/evaluate.py`

baseline 出力と正解を比較するスクリプトです。

最初は以下だけでよいです。

- Exact Match
- CER

### Step 3: `scripts/finetune.py`

`train.jsonl` を使って zenz-v1 を追加学習するスクリプトです。

重要になるのは、labels のうち入力 prompt 部分を `-100` にする処理です。

---

## 17. 最後に: 今回のコードの最小理解

今回のコードを最小限で理解するなら、以下の5行が中心です。

```python
prompt = make_prompt(text, convert_to_katakana=True)
encoded = tokenizer(prompt, return_tensors="pt").to(device)
generated = model.generate(**encoded, do_sample=False, max_new_tokens=64)
generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
output = extract_output(generated_text)
```

つまり、

```text
読みを prompt にする
↓
tokenizer で数字にする
↓
model.generate で続きを生成する
↓
decode で文字列に戻す
↓
変換結果だけ取り出す
```

これが今回の baseline 実験のコード上の核心です。

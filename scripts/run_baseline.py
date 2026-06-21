#!/usr/bin/env python3
# PPP: このファイルは「zenz-v1 を fine-tuning せず、そのまま推論する」ための実験スクリプトです。
"""Run zenz-v1 baseline kana-kanji conversion.

The zenz-v1 prompt format is:

    \uEE00<input_katakana>\uEE01<output></s>

This script feeds the prompt prefix and greedily generates the output part.
"""

# PPP: `from __future__ import annotations` は型ヒントの評価を遅らせるための設定です。
# PPP: 実行結果には直接影響しませんが、型ヒントを書きやすくします。
from __future__ import annotations

# PPP: `argparse` はコマンドライン引数を扱う標準ライブラリです。
import argparse
# PPP: `json` は JSON / JSONL を読み書きするために使います。
import json
# PPP: `Path` はファイルパスを文字列より安全に扱うために使います。
from pathlib import Path
# PPP: `Iterable`, `List`, `Optional` は型ヒント用です。実行時の処理というより説明のためです。
from typing import Iterable, List, Optional

# PPP: `torch` は PyTorch 本体です。モデル推論や tensor 操作に使います。
import torch
# PPP: Hugging Face Transformers から tokenizer と causal language model を自動ロードするクラスを読み込みます。
from transformers import AutoModelForCausalLM, AutoTokenizer

# PPP: zenz-v1 の入力開始を表す特殊文字です。表示上は `` のように見えます。
INPUT_START = "\uEE00"
# PPP: zenz-v1 の出力開始を表す特殊文字です。表示上は `` のように見えます。
OUTPUT_START = "\uEE01"
# PPP: デフォルトで使う Hugging Face のモデルIDです。PyTorch形式の重みがある checkpoint 版を使います。
DEFAULT_MODEL = "Miwa-Keita/zenz-v1-checkpoints"


# PPP: この関数は、ひらがなをカタカナへ変換します。
def hiragana_to_katakana(text: str) -> str:
    # PPP: docstring です。この関数の目的を短く説明しています。
    """Convert hiragana chars to katakana, leaving other chars unchanged."""
    # PPP: 変換後の文字を1文字ずつ入れていくリストを用意します。
    chars: List[str] = []
    # PPP: 入力文字列を1文字ずつ処理します。例: `きょう` なら `き`, `ょ`, `う` の順です。
    for ch in text:
        # PPP: `ord(ch)` は文字を Unicode の番号に変換します。例: `あ` -> 12354。
        code = ord(ch)
        # PPP: Unicode 上で、ひらがなの主な範囲は U+3041 から U+3096 です。
        if 0x3041 <= code <= 0x3096:
            # PPP: ひらがなとカタカナは Unicode 上でだいたい 0x60 離れているので、足すとカタカナになります。
            chars.append(chr(code + 0x60))
        # PPP: ひらがな以外、たとえば漢字・英字・記号は変換しません。
        else:
            # PPP: 変換しない文字はそのまま結果リストへ追加します。
            chars.append(ch)
    # PPP: 1文字ずつのリストを、最後に1つの文字列へ戻します。
    return "".join(chars)


# PPP: この関数は、普通の読み文字列を zenz-v1 用 prompt に変換します。
def make_prompt(text: str, convert_to_katakana: bool = True) -> str:
    # PPP: `convert_to_katakana=True` なら、ひらがな入力をカタカナへ変換します。
    model_input = hiragana_to_katakana(text) if convert_to_katakana else text
    # PPP: zenz-v1 の形式 `入力開始 + カタカナ入力 + 出力開始` に整形します。
    return f"{INPUT_START}{model_input}{OUTPUT_START}"


# PPP: この関数は、モデルが生成した全文から「変換結果」だけを取り出します。
def extract_output(generated_text: str) -> str:
    # PPP: 生成全文に出力開始マーカーが含まれているか確認します。
    if OUTPUT_START in generated_text:
        # PPP: `OUTPUT_START` で1回だけ分割し、後半だけを残します。
        # PPP: 例: `キョウ今日</s>` -> `今日</s>`。
        generated_text = generated_text.split(OUTPUT_START, 1)[1]
    # PPP: eos token `</s>` より前だけを取り出します。
    generated_text = generated_text.split("</s>", 1)[0]
    # PPP: 前後に余計な空白や改行があれば削除して返します。
    return generated_text.strip()


# PPP: この関数は、入力ファイルから変換したい文字列一覧を読み込みます。
def read_inputs(path: Optional[Path]) -> List[str]:
    # PPP: `path` が None の場合、つまり `--input` が指定されなかった場合のデフォルト入力です。
    if path is None:
        # PPP: 小さいサンプルを返します。手軽にスクリプトを試すための fallback です。
        return [
            # PPP: 天気の基本例です。
            "きょうはいいてんきです",
            # PPP: 歯科医/司会のような同音異義語を含む例です。
            "はがいたいのでしかいにみてもらった",
            # PPP: `決して〜な` の文脈を含む例です。
            "このしじにはけっしてしたがうな",
            # PPP: `指示` と `従う` の基本例です。
            "このしじにはしたがう",
        ]

    # PPP: ファイルから読み取った入力文字列を入れるリストです。
    inputs: List[str] = []
    # PPP: UTF-8 でファイルを開きます。日本語を扱うので encoding を明示しています。
    with path.open(encoding="utf-8") as f:
        # PPP: ファイルを1行ずつ読みます。JSONL は1行1データなので、この読み方と相性が良いです。
        for line in f:
            # PPP: 行末の改行や前後の空白を削除します。
            line = line.strip()
            # PPP: 空行なら何もせず次の行へ進みます。
            if not line:
                continue
            # PPP: `{` で始まる行は JSON とみなします。例: `{"input":"..."}`。
            if line.startswith("{"):
                # PPP: JSON文字列を Python の dict に変換します。
                obj = json.loads(line)
                # PPP: dict の `input` キーから、変換したい読み文字列だけを取り出します。
                inputs.append(obj["input"])
            # PPP: JSON でない行は、plain text の入力行としてそのまま使います。
            else:
                # PPP: 行全体を入力として追加します。
                inputs.append(line)
    # PPP: 読み込んだ入力文字列のリストを返します。
    return inputs


# PPP: この関数は、推論に使う計算デバイスを決めます。
def choose_device(requested: str) -> str:
    # PPP: ユーザーが `--device cpu` など明示指定した場合は、その指定をそのまま返します。
    if requested != "auto":
        return requested
    # PPP: NVIDIA GPU + CUDA が使える環境なら `cuda` を使います。
    if torch.cuda.is_available():
        return "cuda"
    # PPP: Apple Silicon Mac などで MPS が使えるなら `mps` を使います。
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    # PPP: GPU が使えない場合は CPU で実行します。
    return "cpu"


# PPP: この関数が、実際に zenz-v1 を読み込んで複数入力を変換する中心処理です。
def convert_batch(
    # PPP: 変換したい入力文字列の集まりです。
    inputs: Iterable[str],
    # PPP: Hugging Face のモデルID、またはローカルモデルパスです。
    model_name: str,
    # PPP: `cpu`, `mps`, `cuda` のどれで推論するかを表します。
    device: str,
    # PPP: 生成する token 数の上限です。長すぎる暴走生成を防ぎます。
    max_new_tokens: int,
    # PPP: 入力をカタカナ化するかどうかです。zenz-v1 では通常 True です。
    convert_to_katakana: bool,
# PPP: 戻り値は dict のリストです。各 dict に input / prompt / prediction などを入れます。
) -> List[dict]:
    # PPP: モデルに対応する tokenizer を Hugging Face から読み込みます。
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # PPP: GPT-2 系の causal language model を Hugging Face から読み込みます。
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # PPP: モデルを指定 device に移します。例: CPU から MPS/GPU へ移動します。
    model.to(device)
    # PPP: 推論モードにします。Dropout など学習時専用の挙動を止めます。
    model.eval()

    # PPP: 変換結果をここにためます。
    results: List[dict] = []
    # PPP: 入力を1件ずつ処理します。最初は理解しやすさ優先で batch 化していません。
    for text in inputs:
        # PPP: 入力文字列から zenz-v1 用 prompt を作ります。
        prompt = make_prompt(text, convert_to_katakana=convert_to_katakana)
        # PPP: prompt 文字列を token ID の tensor に変換し、model と同じ device へ移します。
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        # PPP: 推論だけなので勾配計算を無効化します。メモリ節約にもなります。
        with torch.no_grad():
            # PPP: モデルに続きを生成させます。ここが実際のかな漢字変換に相当する部分です。
            generated = model.generate(
                # PPP: `encoded` の中身、つまり input_ids や attention_mask を展開して渡します。
                **encoded,
                # PPP: sampling せず、常に最も確率の高い token を選ぶ greedy decoding にします。
                do_sample=False,
                # PPP: 新しく生成する token の最大数です。
                max_new_tokens=max_new_tokens,
                # PPP: eos token、つまり `</s>` が出たら生成を止めるための指定です。
                eos_token_id=tokenizer.eos_token_id,
                # PPP: padding token の ID です。generate の警告や設定不備を避けるために指定します。
                pad_token_id=tokenizer.pad_token_id,
            )
        # PPP: 生成された token ID 列を、人間が読める文字列へ戻します。
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
        # PPP: 生成全文から、変換結果の本文だけを取り出します。
        output = extract_output(generated_text)
        # PPP: 後で確認しやすいように、元入力・prompt・prediction・生成全文を保存します。
        results.append(
            {
                # PPP: 元の入力です。例: `きょうはいいてんきです`。
                "input": text,
                # PPP: モデルに渡した prompt です。例: `キョウハイイテンキデス`。
                "prompt": prompt,
                # PPP: 変換結果だけを切り出したものです。例: `今日はいい天気です`。
                "prediction": output,
                # PPP: モデルが返した全文です。デバッグ用に残します。
                "generated_text": generated_text,
            }
        )
    # PPP: 全入力分の結果を返します。
    return results


# PPP: CLI として実行したときの入口です。
def main() -> None:
    # PPP: コマンドライン引数を定義する parser を作ります。
    parser = argparse.ArgumentParser()
    # PPP: `--model` で使うモデルを変更できます。指定しなければ DEFAULT_MODEL です。
    parser.add_argument("--model", default=DEFAULT_MODEL)
    # PPP: `--input` で入力ファイルを指定します。JSONL でも plain text でも読めます。
    parser.add_argument("--input", type=Path, help="JSONL with {'input': ...} or plain text lines")
    # PPP: `--output` で結果の保存先を指定します。デフォルトは baseline 用の JSONL です。
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline/predictions.jsonl"))
    # PPP: `--device` で推論デバイスを指定します。auto なら自動選択します。
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    # PPP: `--max-new-tokens` で生成 token 数の上限を指定できます。
    parser.add_argument("--max-new-tokens", type=int, default=64)
    # PPP: この flag を付けると、ひらがな→カタカナ変換を無効化します。
    parser.add_argument("--no-katakana-convert", action="store_true")
    # PPP: 実際にコマンドライン引数を解析します。
    args = parser.parse_args()

    # PPP: 引数に基づいて、推論に使う device を決めます。
    device = choose_device(args.device)
    # PPP: 入力ファイルから、変換したい文字列一覧を読みます。
    inputs = read_inputs(args.input)
    # PPP: zenz-v1 を使って、入力一覧を変換します。
    results = convert_batch(
        # PPP: 変換したい入力一覧です。
        inputs=inputs,
        # PPP: 使用するモデル名です。
        model_name=args.model,
        # PPP: 推論に使う device です。
        device=device,
        # PPP: 生成 token 数の上限です。
        max_new_tokens=args.max_new_tokens,
        # PPP: `--no-katakana-convert` が付いていなければカタカナ変換します。
        convert_to_katakana=not args.no_katakana_convert,
    )

    # PPP: 出力先ディレクトリがなければ作ります。例: `outputs/baseline/`。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # PPP: 結果ファイルを UTF-8 で開きます。`w` なので既存ファイルは上書きされます。
    with args.output.open("w", encoding="utf-8") as f:
        # PPP: 変換結果を1件ずつ JSONL として書き込みます。
        for row in results:
            # PPP: `ensure_ascii=False` により、日本語を `\uXXXX` にせずそのまま保存します。
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # PPP: ターミナルにも簡単な変換結果を表示します。
    for row in results:
        # PPP: 例: `きょうはいいてんきです -> 今日はいい天気です` の形で表示します。
        print(f"{row['input']} -> {row['prediction']}")
    # PPP: 最後に、結果を書き込んだファイルパスを表示します。
    print(f"\nwrote: {args.output}")


# PPP: このファイルを直接 `python3 scripts/run_baseline.py` と実行した場合だけ main() を呼びます。
if __name__ == "__main__":
    # PPP: CLI の本体処理を開始します。
    main()

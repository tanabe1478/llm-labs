#!/usr/bin/env python3
# PPP: このファイルは baseline / fine-tuned model の予測結果を、正解データと比較して数値評価するスクリプトです。

# PPP: `from __future__ import annotations` は型ヒントの評価を遅らせ、型ヒントを書きやすくするための設定です。
from __future__ import annotations

# PPP: `argparse` はコマンドライン引数を扱うための標準ライブラリです。
import argparse
# PPP: `json` は JSONL を読み書きするために使います。
import json
# PPP: `Path` はファイルパスを扱いやすくするために使います。
from pathlib import Path
# PPP: `Dict`, `List`, `Tuple` は型ヒント用です。
from typing import Dict, List, Tuple


# PPP: JSONL ファイルを読み、1行ずつ dict に変換してリストで返します。
def read_jsonl(path: Path) -> List[dict]:
    # PPP: 読み込んだ JSON object をここにためます。
    rows: List[dict] = []
    # PPP: 日本語を含むので UTF-8 で開きます。
    with path.open(encoding="utf-8") as f:
        # PPP: JSONL は1行1レコードなので、ファイルを1行ずつ処理します。
        for line_number, line in enumerate(f, start=1):
            # PPP: 行末の改行や前後の空白を削除します。
            line = line.strip()
            # PPP: 空行は無視します。
            if not line:
                continue
            # PPP: JSON として parse します。失敗した場合に行番号が分かるように例外を包みます。
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                # PPP: どのファイルの何行目で壊れているかを分かりやすくします。
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    # PPP: 全行を読み終えたら、dict のリストを返します。
    return rows


# PPP: 参照データを `input` をキーにした dict に変換します。
def index_by_input(rows: List[dict], path: Path) -> Dict[str, dict]:
    # PPP: `input` 文字列から、その行全体を引けるようにする辞書です。
    indexed: Dict[str, dict] = {}
    # PPP: 各行をチェックしながら辞書へ入れていきます。
    for row in rows:
        # PPP: eval data には必ず `input` が必要です。
        if "input" not in row:
            raise ValueError(f"Missing 'input' in {path}: {row}")
        # PPP: `input` の値をキーにします。
        key = row["input"]
        # PPP: 同じ input が複数あると評価対応が曖昧になるのでエラーにします。
        if key in indexed:
            raise ValueError(f"Duplicate input in {path}: {key}")
        # PPP: 問題なければ辞書に登録します。
        indexed[key] = row
    # PPP: input -> row の辞書を返します。
    return indexed


# PPP: 2つの文字列の Levenshtein distance、つまり編集距離を計算します。
def levenshtein_distance(a: str, b: str) -> int:
    # PPP: `a` が空なら、`b` の全文字を挿入する必要があるので距離は len(b) です。
    if not a:
        return len(b)
    # PPP: `b` が空なら、`a` の全文字を削除する必要があるので距離は len(a) です。
    if not b:
        return len(a)

    # PPP: 動的計画法の前の行を表します。最初は空文字から b の prefix への距離です。
    previous = list(range(len(b) + 1))
    # PPP: `a` の各文字を1文字ずつ見て、距離表を1行ずつ更新します。
    for i, char_a in enumerate(a, start=1):
        # PPP: 現在行の先頭は、a の i 文字を全部削除して空文字にする距離です。
        current = [i]
        # PPP: `b` の各文字と比較し、挿入・削除・置換の最小コストを計算します。
        for j, char_b in enumerate(b, start=1):
            # PPP: 文字が同じなら置換コスト0、違うなら置換コスト1です。
            substitution_cost = 0 if char_a == char_b else 1
            # PPP: `a` 側を1文字削除する場合のコストです。
            deletion = previous[j] + 1
            # PPP: `b` 側へ1文字挿入する場合のコストです。
            insertion = current[j - 1] + 1
            # PPP: 現在の文字を置換、または同じならそのままにする場合のコストです。
            substitution = previous[j - 1] + substitution_cost
            # PPP: 3つの操作のうち最も安いものを採用します。
            current.append(min(deletion, insertion, substitution))
        # PPP: 次の行を計算するため、現在行を previous として持ち越します。
        previous = current
    # PPP: 最終セルが、文字列 a と b 全体の編集距離です。
    return previous[-1]


# PPP: 1件の予測に対して exact match と文字誤り数を計算します。
def evaluate_one(expected: str, prediction: str) -> Tuple[bool, int, int, float]:
    # PPP: 完全一致しているかを bool で持ちます。
    exact = expected == prediction
    # PPP: 文字単位の編集距離を計算します。
    distance = levenshtein_distance(expected, prediction)
    # PPP: CER の分母にする正解文字数です。空文字の場合の0除算を避けるため最低1にします。
    reference_length = max(len(expected), 1)
    # PPP: CER は Character Error Rate、つまり `編集距離 / 正解文字数` です。
    cer = distance / reference_length
    # PPP: exact, 編集距離, 正解文字数, CER を返します。
    return exact, distance, reference_length, cer


# PPP: 参照データと予測データ全体を比較して、metrics と per-example 結果を返します。
def evaluate(references: List[dict], predictions: List[dict]) -> Tuple[dict, List[dict]]:
    # PPP: references を input -> row の辞書にします。
    reference_by_input = index_by_input(references, Path("references"))
    # PPP: per-example の評価結果をここにためます。
    examples: List[dict] = []
    # PPP: 完全一致した件数です。
    exact_count = 0
    # PPP: 全サンプルの編集距離の合計です。
    total_distance = 0
    # PPP: 全サンプルの正解文字数の合計です。
    total_reference_length = 0

    # PPP: 予測結果を1件ずつ評価します。
    for prediction_row in predictions:
        # PPP: prediction 側にも input が必要です。これで正解データと対応づけます。
        if "input" not in prediction_row:
            raise ValueError(f"Missing 'input' in prediction row: {prediction_row}")
        # PPP: prediction 側には model output である `prediction` が必要です。
        if "prediction" not in prediction_row:
            raise ValueError(f"Missing 'prediction' in prediction row: {prediction_row}")

        # PPP: 入力文字列を取り出します。
        input_text = prediction_row["input"]
        # PPP: 入力に対応する正解行を探します。
        if input_text not in reference_by_input:
            raise ValueError(f"Prediction input not found in references: {input_text}")
        # PPP: 正解行を取得します。
        reference_row = reference_by_input[input_text]
        # PPP: 正解データには `output` が必要です。
        if "output" not in reference_row:
            raise ValueError(f"Missing 'output' in reference row: {reference_row}")

        # PPP: 正解文字列です。
        expected = reference_row["output"]
        # PPP: モデルの予測文字列です。
        predicted = prediction_row["prediction"]
        # PPP: 1件分の評価値を計算します。
        exact, distance, reference_length, cer = evaluate_one(expected, predicted)

        # PPP: 完全一致なら exact_count を増やします。
        if exact:
            exact_count += 1
        # PPP: corpus CER 用に編集距離を合計します。
        total_distance += distance
        # PPP: corpus CER 用に正解文字数を合計します。
        total_reference_length += reference_length

        # PPP: 後で errors.md を作れるように、1件ごとの詳細を保存します。
        examples.append(
            {
                "input": input_text,
                "expected": expected,
                "prediction": predicted,
                "exact": exact,
                "distance": distance,
                "reference_length": reference_length,
                "cer": cer,
                "note": reference_row.get("note", ""),
            }
        )

    # PPP: 評価対象件数です。
    count = len(examples)
    # PPP: 0件評価を避けるため、予測が空ならエラーにします。
    if count == 0:
        raise ValueError("No predictions to evaluate")

    # PPP: Exact Match は、完全一致した件数 / 全件数です。
    exact_match = exact_count / count
    # PPP: corpus CER は、全編集距離 / 全正解文字数です。
    corpus_cer = total_distance / max(total_reference_length, 1)
    # PPP: 1件ごとの CER を平均したものです。corpus CER とは少し意味が違います。
    mean_cer = sum(example["cer"] for example in examples) / count

    # PPP: metrics.json に保存する集計結果を作ります。
    metrics = {
        "count": count,
        "exact_count": exact_count,
        "exact_match": exact_match,
        "total_distance": total_distance,
        "total_reference_length": total_reference_length,
        "cer": corpus_cer,
        "mean_cer": mean_cer,
    }
    # PPP: 集計 metrics と、1件ごとの詳細 examples を返します。
    return metrics, examples


# PPP: Markdown の表で `|` が壊れないよう、最低限エスケープします。
def escape_markdown_cell(text: object) -> str:
    # PPP: まず文字列へ変換します。
    value = str(text)
    # PPP: 改行は `<br>` にします。
    value = value.replace("\n", "<br>")
    # PPP: Markdown 表の区切りである `|` をエスケープします。
    value = value.replace("|", "\\|")
    # PPP: エスケープ済みの文字列を返します。
    return value


# PPP: 1件ごとの評価結果から、人間が読みやすい Markdown レポートを作ります。
def write_errors_markdown(path: Path, metrics: dict, examples: List[dict]) -> None:
    # PPP: 出力先ディレクトリがなければ作ります。
    path.parent.mkdir(parents=True, exist_ok=True)
    # PPP: Markdown の各行をここにためます。
    lines: List[str] = []
    # PPP: タイトルを書きます。
    lines.append("# Evaluation Errors")
    lines.append("")
    # PPP: 集計値を書きます。
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- Count: {metrics['count']}")
    lines.append(f"- Exact Match: {metrics['exact_match']:.4f} ({metrics['exact_count']}/{metrics['count']})")
    lines.append(f"- CER: {metrics['cer']:.4f}")
    lines.append(f"- Mean CER: {metrics['mean_cer']:.4f}")
    lines.append("")

    # PPP: 全件の詳細テーブルを書きます。
    lines.append("## Per-example results")
    lines.append("")
    lines.append("| exact | CER | input | expected | prediction | note |")
    lines.append("|---:|---:|---|---|---|---|")
    # PPP: CER が高い順、つまり悪い順に並べると誤り分析しやすいです。
    for example in sorted(examples, key=lambda row: (row["exact"], -row["cer"])):
        # PPP: Markdown 表の1行を作ります。
        lines.append(
            "| "
            + " | ".join(
                [
                    "✅" if example["exact"] else "❌",
                    f"{example['cer']:.4f}",
                    escape_markdown_cell(example["input"]),
                    escape_markdown_cell(example["expected"]),
                    escape_markdown_cell(example["prediction"]),
                    escape_markdown_cell(example["note"]),
                ]
            )
            + " |"
        )
    lines.append("")
    # PPP: まとめた Markdown をファイルへ書き込みます。
    path.write_text("\n".join(lines), encoding="utf-8")


# PPP: CLI として実行されたときの入口です。
def main() -> None:
    # PPP: コマンドライン引数 parser を作ります。
    parser = argparse.ArgumentParser()
    # PPP: 正解データの JSONL を指定します。通常は `data/eval.jsonl` です。
    parser.add_argument("--references", type=Path, default=Path("data/eval.jsonl"))
    # PPP: モデル予測の JSONL を指定します。通常は baseline または tuned の predictions です。
    parser.add_argument("--predictions", type=Path, default=Path("outputs/baseline/eval_predictions.jsonl"))
    # PPP: 集計 metrics の保存先です。
    parser.add_argument("--metrics-output", type=Path, default=Path("outputs/baseline/metrics.json"))
    # PPP: 人間向け error report の保存先です。
    parser.add_argument("--errors-output", type=Path, default=Path("outputs/baseline/errors.md"))
    # PPP: コマンドライン引数を解析します。
    args = parser.parse_args()

    # PPP: 正解データを読みます。
    references = read_jsonl(args.references)
    # PPP: 予測データを読みます。
    predictions = read_jsonl(args.predictions)
    # PPP: 正解と予測を比較して、集計結果と詳細結果を得ます。
    metrics, examples = evaluate(references, predictions)

    # PPP: metrics の出力先ディレクトリがなければ作ります。
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    # PPP: metrics を JSON として保存します。日本語は含まない想定ですが ensure_ascii=False にしておきます。
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # PPP: 人間向け Markdown レポートを保存します。
    write_errors_markdown(args.errors_output, metrics, examples)

    # PPP: ターミナルにも最重要 metrics を表示します。
    print(f"count: {metrics['count']}")
    # PPP: Exact Match を表示します。
    print(f"exact_match: {metrics['exact_match']:.4f} ({metrics['exact_count']}/{metrics['count']})")
    # PPP: corpus CER を表示します。
    print(f"cer: {metrics['cer']:.4f}")
    # PPP: 保存先を表示します。
    print(f"wrote: {args.metrics_output}")
    # PPP: errors.md の保存先を表示します。
    print(f"wrote: {args.errors_output}")


# PPP: このファイルを直接実行した場合だけ main() を呼びます。
if __name__ == "__main__":
    # PPP: CLI 処理を開始します。
    main()

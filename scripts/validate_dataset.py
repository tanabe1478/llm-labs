#!/usr/bin/env python3
# PPP: このファイルは train/eval JSONL の最低限の品質を確認するスクリプトです。

# PPP: 型ヒントを扱いやすくするための future import です。
from __future__ import annotations

# PPP: コマンドライン引数を扱います。
import argparse
# PPP: JSONL の各行を parse するために使います。
import json
# PPP: ファイルパスを扱います。
from pathlib import Path
# PPP: 型ヒント用です。
from typing import Dict, List, Set


# PPP: JSONL を読み、最低限 `input` と `output` があるか確認します。
def read_dataset(path: Path) -> List[dict]:
    # PPP: 読み込んだ行をここに入れます。
    rows: List[dict] = []
    # PPP: 日本語を扱うので UTF-8 で開きます。
    with path.open(encoding="utf-8") as f:
        # PPP: 1行ずつ読みます。
        for line_number, line in enumerate(f, start=1):
            # PPP: 改行や前後の空白を消します。
            line = line.strip()
            # PPP: 空行は無視します。
            if not line:
                continue
            # PPP: JSON として読みます。
            row = json.loads(line)
            # PPP: 学習・評価データには input が必須です。
            if "input" not in row:
                raise ValueError(f"{path}:{line_number}: missing input")
            # PPP: 学習・評価データには output が必須です。
            if "output" not in row:
                raise ValueError(f"{path}:{line_number}: missing output")
            # PPP: input が空文字だと学習例にならないので禁止します。
            if not row["input"]:
                raise ValueError(f"{path}:{line_number}: empty input")
            # PPP: output が空文字だと正解がないので禁止します。
            if not row["output"]:
                raise ValueError(f"{path}:{line_number}: empty output")
            # PPP: 問題なければ追加します。
            rows.append(row)
    # PPP: 読み込んだ行を返します。
    return rows


# PPP: 1つの dataset 内で input が重複していないか確認します。
def ensure_no_duplicate_inputs(rows: List[dict], name: str) -> None:
    # PPP: 既に見た input を記録します。
    seen: Set[str] = set()
    # PPP: 各行を順に確認します。
    for row in rows:
        # PPP: input を取り出します。
        input_text = row["input"]
        # PPP: 既に同じ input があればエラーです。
        if input_text in seen:
            raise ValueError(f"duplicate input in {name}: {input_text}")
        # PPP: 初登場なら seen に追加します。
        seen.add(input_text)


# PPP: train と eval に同じ input が混ざっていないか確認します。
def ensure_no_train_eval_overlap(train_rows: List[dict], eval_rows: List[dict]) -> None:
    # PPP: train の input 集合を作ります。
    train_inputs = {row["input"] for row in train_rows}
    # PPP: eval の input 集合を作ります。
    eval_inputs = {row["input"] for row in eval_rows}
    # PPP: 積集合があれば overlap です。
    overlap = sorted(train_inputs & eval_inputs)
    # PPP: overlap があると、eval が暗記チェックになりやすいのでエラーにします。
    if overlap:
        raise ValueError(f"train/eval overlap: {overlap}")


# PPP: note ごとの件数を数えます。ざっくりカテゴリの偏りを見るためです。
def count_notes(rows: List[dict]) -> Dict[str, int]:
    # PPP: note -> count の辞書です。
    counts: Dict[str, int] = {}
    # PPP: 各行の note を数えます。
    for row in rows:
        # PPP: note がなければ `(none)` として扱います。
        note = row.get("note", "(none)")
        # PPP: count を1増やします。
        counts[note] = counts.get(note, 0) + 1
    # PPP: 集計結果を返します。
    return counts


# PPP: CLI の入口です。
def main() -> None:
    # PPP: コマンドライン引数 parser を作ります。
    parser = argparse.ArgumentParser()
    # PPP: train JSONL のパスです。
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    # PPP: eval JSONL のパスです。
    parser.add_argument("--eval", type=Path, default=Path("data/eval.jsonl"))
    # PPP: 引数を parse します。
    args = parser.parse_args()

    # PPP: train を読みます。
    train_rows = read_dataset(args.train)
    # PPP: eval を読みます。
    eval_rows = read_dataset(args.eval)
    # PPP: train 内重複を確認します。
    ensure_no_duplicate_inputs(train_rows, "train")
    # PPP: eval 内重複を確認します。
    ensure_no_duplicate_inputs(eval_rows, "eval")
    # PPP: train/eval 間の重複を確認します。
    ensure_no_train_eval_overlap(train_rows, eval_rows)

    # PPP: 件数を表示します。
    print(f"train_count: {len(train_rows)}")
    # PPP: eval 件数を表示します。
    print(f"eval_count: {len(eval_rows)}")
    # PPP: train の note 種類数を表示します。
    print(f"train_note_kinds: {len(count_notes(train_rows))}")
    # PPP: eval の note 種類数を表示します。
    print(f"eval_note_kinds: {len(count_notes(eval_rows))}")
    # PPP: 問題なければ OK と表示します。
    print("dataset validation: OK")


# PPP: 直接実行された場合だけ main() を呼びます。
if __name__ == "__main__":
    # PPP: validation を開始します。
    main()

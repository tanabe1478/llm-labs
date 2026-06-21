#!/usr/bin/env python3
"""Run zenz-v1 baseline kana-kanji conversion.

The zenz-v1 prompt format is:

    \uEE00<input_katakana>\uEE01<output></s>

This script feeds the prompt prefix and greedily generates the output part.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_START = "\uEE00"
OUTPUT_START = "\uEE01"
DEFAULT_MODEL = "Miwa-Keita/zenz-v1-checkpoints"


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


def make_prompt(text: str, convert_to_katakana: bool = True) -> str:
    model_input = hiragana_to_katakana(text) if convert_to_katakana else text
    return f"{INPUT_START}{model_input}{OUTPUT_START}"


def extract_output(generated_text: str) -> str:
    if OUTPUT_START in generated_text:
        generated_text = generated_text.split(OUTPUT_START, 1)[1]
    generated_text = generated_text.split("</s>", 1)[0]
    return generated_text.strip()


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


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, help="JSONL with {'input': ...} or plain text lines")
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline/predictions.jsonl"))
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-katakana-convert", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    inputs = read_inputs(args.input)
    results = convert_batch(
        inputs=inputs,
        model_name=args.model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        convert_to_katakana=not args.no_katakana_convert,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for row in results:
        print(f"{row['input']} -> {row['prediction']}")
    print(f"\nwrote: {args.output}")


if __name__ == "__main__":
    main()

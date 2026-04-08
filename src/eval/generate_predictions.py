from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_parse_json(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        left = text.find("{")
        right = text.rfind("}")
        if left >= 0 and right > left:
            return json.loads(text[left : right + 1])
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prediction JSONL for SFT evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()

    rows = _load_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        for row in rows:
            prompt = row["prompt"]
            model_inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            new_tokens = generated[0][model_inputs["input_ids"].shape[1] :]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            parsed = _safe_parse_json(generated_text)

            out_row = {
                "question_id": row.get("question_id"),
                "output": parsed,
            }
            file.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "model": args.model,
                "input": str(args.input),
                "output": str(args.output),
                "count": len(rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

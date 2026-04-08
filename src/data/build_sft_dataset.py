from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.schema import SFTSample, VerifyBenchSample


PROMPT_TEMPLATE = """You are an accurate problem-solving assistant.
Solve the following question and output only the final answer text.

[Question]
{question}

[Output Requirement]
Return the answer directly.
Do not output JSON.
Do not add extra prefixes like 'Final Answer:'.
"""


def _load_rows(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("train", "data", "samples", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported raw data format: {path}")


def _normalize_row(row: Dict) -> VerifyBenchSample:
    if "prompt" in row and "target" in row:
        return VerifyBenchSample(
            question_id=row.get("question_id") or "unknown",
            completion_id=row.get("completion_id"),
            question=row.get("question") or "",
            answer=row.get("target") or "",
            completion=row.get("completion"),
            gold_correct=row.get("gold_correct"),
            completion_model=row.get("completion_model"),
            source=row.get("source"),
            answer_type=row.get("answer_type"),
            answer_subtype=row.get("answer_subtype"),
            annotator=row.get("annotator") or [],
        )

    return VerifyBenchSample(
        question_id=row.get("question_id") or row.get("id") or "unknown",
        completion_id=row.get("completion_id"),
        question=row["question"],
        answer=row["answer"],
        completion=row.get("completion"),
        gold_correct=row.get("gold_correct"),
        completion_model=row.get("completion_model"),
        source=row.get("source"),
        answer_type=row.get("answer_type"),
        answer_subtype=row.get("answer_subtype"),
        annotator=row.get("annotator") or [],
    )


def _build_prompt(task: VerifyBenchSample) -> str:
    return PROMPT_TEMPLATE.format(
        question=task.question,
    )


def _build_target(sample: VerifyBenchSample) -> str:
    return sample.answer.strip()


def _deduplicate(rows: List[VerifyBenchSample], enabled: bool) -> List[VerifyBenchSample]:
    if not enabled:
        return rows

    kept: List[VerifyBenchSample] = []
    seen_question_ids = set()
    for row in rows:
        if row.question_id in seen_question_ids:
            continue
        seen_question_ids.add(row.question_id)
        kept.append(row)
    return kept


def _split_rows(
    rows: List[SFTSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[SFTSample], List[SFTSample], List[SFTSample]]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)

    test_rows = shuffled[:test_size]
    val_rows = shuffled[test_size : test_size + val_size]
    train_rows = shuffled[test_size + val_size :]
    return train_rows, val_rows, test_rows


def _write_jsonl(path: Path, rows: Iterable[SFTSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset from raw VerifyBench-like rows")
    parser.add_argument("--raw", type=Path, required=True, help="Raw .json or .jsonl input")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", action="store_true")
    args = parser.parse_args()

    raw_rows = _load_rows(args.raw)
    normalized_rows: List[VerifyBenchSample] = []
    for row in raw_rows:
        normalized_rows.append(_normalize_row(row))

    normalized_rows = _deduplicate(normalized_rows, enabled=args.deduplicate)

    samples: List[SFTSample] = []
    for sample_row in normalized_rows:
        sample = SFTSample(
            question_id=sample_row.question_id,
            prompt=_build_prompt(sample_row),
            target=_build_target(sample_row),
            metadata={
                "question_id": sample_row.question_id,
                "completion_id": sample_row.completion_id,
                "gold_correct": sample_row.gold_correct,
                "completion_model": sample_row.completion_model,
                "source": sample_row.source,
                "answer_type": sample_row.answer_type,
                "answer_subtype": sample_row.answer_subtype,
                "annotator": sample_row.annotator,
            },
        )
        samples.append(sample)

    train_rows, val_rows, test_rows = _split_rows(
        rows=samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    _write_jsonl(args.out_dir / "train.jsonl", train_rows)
    _write_jsonl(args.out_dir / "val.jsonl", val_rows)
    _write_jsonl(args.out_dir / "test.jsonl", test_rows)

    print(
        json.dumps(
            {
                "total": len(samples),
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
